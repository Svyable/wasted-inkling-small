#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Separate Inkling pre-softmax score drift from softmax implementation drift.

All inputs to this diagnostic are already bounded by earlier probes: layer
RMSNorm, native BF16 Q/K/V/R matrices, K/V short convolution, and Q/K per-head
RMSNorm.

The official logits are reconstructed from official modules and must reproduce
captured official probabilities exactly. The C logits are observed directly:
an evidence-only temporary ``inkling_attention.c`` keeps its score rows intact
and computes exponentials in a separate local array. A baseline-versus-probe
attention-output equality check proves that this instrumentation changes no
numerical result.

The observed logits are then crossed with both softmax implementations to
classify score construction, normalization, or both as the remaining boundary.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from analyze_inkling_router_mismatch import _tensor_metrics
from diagnose_inkling_bf16_attention_components import (
    AttentionComponentError,
    capture_c_attention_components,
    capture_official_attention_probabilities,
)
from diagnose_inkling_bf16_attention_head_norm import (
    AttentionHeadNormError,
    build_head_norm_probe_library,
    transform_attention_head_rms,
)
from diagnose_inkling_native_bf16_backend import (
    NULL_WEIGHT_FIELDS,
    NativeBfloat16BackendError,
    NativeMatrixProvider,
    transform_rms_source,
)
from diagnose_inkling_pre_router import (
    PreRouterDiagnosisError,
    _capture_official_pre_router,
)
from discover_inkling_c_router_experts import _RejectExperts
from discover_inkling_router_experts import input_sha256
from inkling_compact_layer_parity import bind_sparse_layer_compact
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    verify_config_binding,
)
from inkling_layer_parity import (
    FP,
    LAYER_SOURCES,
    LayerParityError,
    LayerScratch,
    LayerState,
    TraceCollector,
    _build_config,
    build_library,
    configure_library,
)
from inkling_release_config import build_transformers_text_config


class AttentionScoreError(RuntimeError):
    """Attention score and softmax components cannot be aligned exactly."""


_LIBM = ctypes.CDLL(None)
_LIBM.expf.argtypes = [ctypes.c_float]
_LIBM.expf.restype = ctypes.c_float

_SOFTMAX_LOOP_OLD = """        double sum = 0.0;
        for (int j = 0; j < kv_len; j++) {
            row[j] = expf(row[j] - max_score);
            sum += row[j];
        }
        if (!(sum > 0.0) || !isfinite(sum)) return -1;
        const float inv_sum = (float)(1.0 / sum);
"""
_SOFTMAX_LOOP_NEW = """        float probability[kv_len];
        double sum = 0.0;
        for (int j = 0; j < kv_len; j++) {
            probability[j] = expf(row[j] - max_score);
            sum += probability[j];
        }
        if (!(sum > 0.0) || !isfinite(sum)) return -1;
        const float inv_sum = (float)(1.0 / sum);
"""
_WEIGHT_OLD = "            const float weight = row[j] * inv_sum;\n"
_WEIGHT_NEW = "            const float weight = probability[j] * inv_sum;\n"


def f32(value: float) -> float:
    return ctypes.c_float(value).value


def transform_attention_preserve_logits(source: str) -> str:
    """Keep score rows as logits while preserving the exact softmax result."""
    transformed = source
    for old, new, label in (
        (_SOFTMAX_LOOP_OLD, _SOFTMAX_LOOP_NEW, "softmax probability loop"),
        (_WEIGHT_OLD, _WEIGHT_NEW, "softmax value weight"),
    ):
        count = transformed.count(old)
        if count != 1:
            raise AttentionScoreError(
                f"expected exactly one {label}; found {count}"
            )
        transformed = transformed.replace(old, new, 1)
    return transformed


def build_score_observation_library(out: Path) -> tuple[Path, dict[str, Any]]:
    """Compile layer/head-RMS fixes plus non-invasive score observation."""
    source_root = Path(__file__).resolve().parents[1] / "inkling" / "src"
    temporary = Path(tempfile.mkdtemp(prefix="inkling-bf16-score-observation-"))
    probe_root = temporary / "src"
    shutil.copytree(source_root, probe_root)

    layer_path = probe_root / "inkling_layer.c"
    layer_original = layer_path.read_text(encoding="utf-8")
    layer_transformed = transform_rms_source(layer_original)
    layer_path.write_text(layer_transformed, encoding="utf-8")

    attention_path = probe_root / "inkling_attention.c"
    attention_original = attention_path.read_text(encoding="utf-8")
    attention_transformed = transform_attention_head_rms(attention_original)
    attention_transformed = transform_attention_preserve_logits(
        attention_transformed
    )
    attention_path.write_text(attention_transformed, encoding="utf-8")

    library = build_library(probe_root, out)
    return library, {
        "production_layer_sha256": hashlib.sha256(
            layer_original.encode()
        ).hexdigest(),
        "probe_layer_sha256": hashlib.sha256(
            layer_transformed.encode()
        ).hexdigest(),
        "production_attention_sha256": hashlib.sha256(
            attention_original.encode()
        ).hexdigest(),
        "probe_attention_sha256": hashlib.sha256(
            attention_transformed.encode()
        ).hexdigest(),
        "production_source_unchanged": (
            (source_root / "inkling_layer.c").read_text(encoding="utf-8")
            == layer_original
            and (source_root / "inkling_attention.c").read_text(encoding="utf-8")
            == attention_original
        ),
        "compiled_sources": list(LAYER_SOURCES),
        "transforms": [
            "layer RMSNorm official BF16 ordering",
            "Q/K head RMSNorm official BF16 ordering",
            "preserve pre-softmax logits while using a separate probability VLA",
        ],
    }


def causal_valid(position: int, key: int, *, is_local: bool, window: int) -> bool:
    if key > position:
        return False
    return not is_local or position - key < window


def reconstruct_official_logits(
    module: Any,
    stages: dict[str, torch.Tensor],
    config: Any,
    layer: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Re-run the official Q/K normalization, relative bias, and score ops."""
    attention = module.self_attn
    tokens = int(stages["q_proj"].shape[0])
    heads = int(attention.num_heads)
    kv_heads = int(attention.num_key_value_heads)
    head_dim = int(attention.head_dim)
    q = stages["q_proj"].to(device=device, dtype=dtype).reshape(
        1, tokens, heads, head_dim
    )
    k = stages["k_sconv"].to(device=device, dtype=dtype).reshape(
        1, tokens, kv_heads, head_dim
    )
    q = attention.q_norm(q).transpose(1, 2)
    k = attention.k_norm(k).transpose(1, 2)
    if heads % kv_heads:
        raise AttentionScoreError("official head geometry is not grouped")
    k = k[:, :, None, :, :].expand(
        1, kv_heads, heads // kv_heads, tokens, head_dim
    ).reshape(1, heads, tokens, head_dim)

    relative = stages["relative_proj_input"].to(
        device=device, dtype=dtype
    ).reshape(1, tokens, heads, -1)
    positions = torch.arange(tokens, device=device)
    bias = attention.rel_logits_proj(relative, positions, positions)
    if not attention.is_sliding and config.log_scaling_n_floor is not None:
        effective_n = (positions + 1).float()
        tau = 1.0 + config.log_scaling_alpha * torch.log(
            (effective_n / config.log_scaling_n_floor).clamp(min=1.0)
        )
        tau = tau.view(1, 1, -1, 1)
        q = (q.float() * tau).to(q.dtype)
        bias = (bias.float() * tau).to(bias.dtype)
    logits = torch.matmul(q, k.transpose(2, 3)) * attention.scaling
    logits = logits + bias
    for position in range(tokens):
        for key in range(tokens):
            if not causal_valid(
                position,
                key,
                is_local=bool(attention.is_sliding),
                window=int(attention.sliding_window or tokens),
            ):
                logits[:, :, position, key] = -torch.inf
    return logits[0].transpose(0, 1).detach().float().cpu().contiguous()


def official_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1, dtype=torch.float32).to(
        torch.bfloat16
    ).float().cpu().contiguous()


def c_softmax(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise AttentionScoreError(
            f"softmax logits must be rank 3, got {tuple(logits.shape)}"
        )
    tokens, heads, keys = logits.shape
    result = torch.zeros(tokens, heads, keys, dtype=torch.float32)
    for position in range(tokens):
        for head in range(heads):
            finite = [
                key
                for key in range(keys)
                if math.isfinite(float(logits[position, head, key]))
            ]
            if not finite:
                raise AttentionScoreError(
                    f"softmax row {position}/{head} has no finite values"
                )
            maximum = max(
                float(logits[position, head, key]) for key in finite
            )
            exponentials = []
            for key in finite:
                difference = f32(float(logits[position, head, key]) - maximum)
                exponentials.append(float(_LIBM.expf(difference)))
            total = sum(exponentials)
            if not total > 0.0 or not math.isfinite(total):
                raise AttentionScoreError("invalid C softmax sum")
            inverse = f32(1.0 / total)
            for key, exponential in zip(finite, exponentials):
                result[position, head, key] = f32(exponential * inverse)
    return result


def finite_logit_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    compare_dtype: torch.dtype,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise AttentionScoreError(
            f"logit shapes differ: {tuple(reference.shape)} vs {tuple(candidate.shape)}"
        )
    reference_finite = torch.isfinite(reference)
    candidate_finite = torch.isfinite(candidate)
    if not torch.equal(reference_finite, candidate_finite):
        raise AttentionScoreError("official and C attention masks differ")
    if not bool(reference_finite.any()):
        raise AttentionScoreError("attention logits contain no finite values")
    metrics = _tensor_metrics(
        reference[reference_finite],
        candidate[candidate_finite],
        compare_dtype=compare_dtype,
    )
    metrics["finite_values"] = int(reference_finite.sum())
    metrics["masked_values"] = int((~reference_finite).sum())
    return metrics


def capture_c_observed_logits(
    lib: Any,
    fixture: Any,
    cfg: dict[str, Any],
    layer: int,
    inputs: list[list[float]],
    provider: NativeMatrixProvider,
) -> dict[str, torch.Tensor]:
    """Run the instrumented C layer and retain observed score rows."""
    compact = bind_sparse_layer_compact(fixture, cfg, layer)
    for field in NULL_WEIGHT_FIELDS:
        setattr(compact.weights, field, FP())
    config = _build_config(lib, cfg)
    hidden = int(cfg["hidden"])
    is_local = layer in set(cfg.get("local_layer_ids", []))
    heads = int(cfg["local_heads"] if is_local else cfg["global_heads"])
    kv_heads = int(
        cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"]
    )
    head_dim = int(
        cfg["local_head_dim"] if is_local else cfg["global_head_dim"]
    )
    conv_k = int(cfg["conv_kernel"])
    capacity = (
        min(max(len(inputs), 1), int(cfg["sliding_window"]))
        if is_local
        else max(len(inputs), 1)
    )
    nfloat = lib.waste_inkling_layer_scratch_floats(
        ctypes.byref(config), layer, capacity
    )
    nint = lib.waste_inkling_layer_scratch_ints(ctypes.byref(config))
    if not nfloat:
        raise AttentionScoreError("scratch sizing failed")
    fbuf = (ctypes.c_float * nfloat)()
    ibuf = (ctypes.c_int * max(nint, 1))()
    scratch = LayerScratch()
    if lib.waste_inkling_layer_scratch_init(
        ctypes.byref(scratch),
        ctypes.byref(config),
        layer,
        capacity,
        fbuf,
        ctypes.c_size_t(nfloat),
        ibuf,
        ctypes.c_size_t(nint),
    ):
        raise AttentionScoreError("scratch init failed")
    kv = (ctypes.c_float * (capacity * kv_heads * head_dim))()
    vv = (ctypes.c_float * (capacity * kv_heads * head_dim))()
    kconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    vconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    aconv = (ctypes.c_float * (hidden * conv_k))()
    mconv = (ctypes.c_float * (hidden * conv_k))()
    state = LayerState()
    if lib.waste_inkling_layer_state_init(
        ctypes.byref(state),
        ctypes.byref(config),
        layer,
        capacity,
        kv,
        vv,
        kconv,
        vconv,
        aconv,
        mconv,
    ):
        raise AttentionScoreError("state init failed")

    collector = TraceCollector()
    reject = _RejectExperts(layer)
    reject_ptr = ctypes.cast(reject.callback, ctypes.c_void_p)
    observed_logits = []
    for position, row in enumerate(inputs):
        if len(row) != hidden:
            raise AttentionScoreError(
                f"input {position} has {len(row)} values; hidden is {hidden}"
            )
        collector.position = position
        x = (ctypes.c_float * hidden)(*row)
        before = len(reject.requests)
        rc = lib.waste_inkling_layer_step_backend_trace(
            ctypes.byref(config),
            layer,
            ctypes.byref(compact.weights),
            ctypes.byref(provider.backend),
            ctypes.byref(state),
            x,
            position,
            ctypes.byref(scratch),
            reject_ptr,
            None,
            ctypes.byref(collector.c_trace),
        )
        if provider.error:
            raise AttentionScoreError(provider.error)
        if rc == 0 or len(reject.requests) != before + 1:
            raise AttentionScoreError(
                f"layer {layer} position {position} did not stop at expert lookup"
            )
        begin = (
            position + 1 - capacity
            if is_local and position + 1 > capacity
            else 0
        )
        kv_len = position - begin + 1
        logits = torch.full(
            (heads, len(inputs)), -torch.inf, dtype=torch.float32
        )
        for head in range(heads):
            offset = head * capacity
            for index in range(kv_len):
                logits[head, begin + index] = float(
                    scratch.scores[offset + index]
                )
        observed_logits.append(logits)

    attention_rows = []
    for position in range(len(inputs)):
        name = f"token.{position}.layer.{layer}.attention_out"
        if name not in collector.values:
            raise AttentionScoreError(f"C trace missed {name}")
        attention_rows.append(collector.values[name])
    return {
        "logits": torch.stack(observed_logits),
        "attention_out": torch.tensor(attention_rows, dtype=torch.float32),
    }


def classify_score_softmax(
    *,
    official_sanity: dict[str, Any],
    c_sanity: dict[str, Any],
    instrumentation_sanity: dict[str, Any],
    score_only: dict[str, Any],
    softmax_only: dict[str, Any],
) -> str:
    if official_sanity["quantized_exact_fraction"] != 1.0:
        return "official_score_reconstruction_failed"
    if c_sanity["max_abs"] > 1e-7:
        return "c_score_observation_failed"
    if instrumentation_sanity["max_abs"] != 0.0:
        return "score_instrumentation_changed_attention"
    score_exact = score_only["quantized_exact_fraction"] == 1.0
    softmax_exact = softmax_only["quantized_exact_fraction"] == 1.0
    if score_exact and softmax_exact:
        return "score_and_softmax_are_individually_equivalent"
    if not score_exact and softmax_exact:
        return "score_construction_is_the_remaining_boundary"
    if score_exact and not softmax_exact:
        return "softmax_is_the_remaining_boundary"
    return "score_construction_and_softmax_both_contribute"


def diagnose_layer(
    baseline_lib: Any,
    observed_lib: Any,
    fixture: Any,
    config: Any,
    c_config: dict[str, Any],
    layer: int,
    inputs: torch.Tensor,
    input_values: list[list[float]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    official, module, _, _ = _capture_official_pre_router(
        fixture,
        config,
        layer,
        inputs,
        device=device,
        dtype=dtype,
    )
    official_probabilities = capture_official_attention_probabilities(
        module,
        config,
        layer,
        inputs,
        device=device,
        dtype=dtype,
    ).float().cpu()

    baseline_provider = NativeMatrixProvider(
        layer,
        module,
        official_norm=official["input_norm"],
    )
    baseline = capture_c_attention_components(
        baseline_lib,
        fixture,
        c_config,
        layer,
        input_values,
        baseline_provider,
    )
    observed_provider = NativeMatrixProvider(
        layer,
        module,
        official_norm=official["input_norm"],
    )
    observed = capture_c_observed_logits(
        observed_lib,
        fixture,
        c_config,
        layer,
        input_values,
        observed_provider,
    )

    official_logits = reconstruct_official_logits(
        module,
        official,
        config,
        layer,
        dtype=dtype,
        device=device,
    )
    c_logits = observed["logits"]
    official_reconstructed = official_softmax(official_logits)
    c_reconstructed = c_softmax(c_logits)
    score_only_probabilities = official_softmax(c_logits)
    softmax_only_probabilities = c_softmax(official_logits)

    official_sanity = _tensor_metrics(
        official_probabilities,
        official_reconstructed,
        compare_dtype=dtype,
    )
    c_sanity = _tensor_metrics(
        baseline["probabilities"],
        c_reconstructed,
        compare_dtype=torch.float32,
    )
    instrumentation_sanity = _tensor_metrics(
        baseline["attention_out"],
        observed["attention_out"],
        compare_dtype=torch.float32,
    )
    score_only = _tensor_metrics(
        official_probabilities,
        score_only_probabilities,
        compare_dtype=dtype,
    )
    softmax_only = _tensor_metrics(
        official_probabilities,
        softmax_only_probabilities,
        compare_dtype=dtype,
    )
    logits = finite_logit_metrics(
        official_logits,
        c_logits,
        compare_dtype=dtype,
    )
    return {
        "classification": classify_score_softmax(
            official_sanity=official_sanity,
            c_sanity=c_sanity,
            instrumentation_sanity=instrumentation_sanity,
            score_only=score_only,
            softmax_only=softmax_only,
        ),
        "logits": logits,
        "official_score_sanity": official_sanity,
        "c_score_sanity": c_sanity,
        "instrumentation_sanity": instrumentation_sanity,
        "score_construction_only": score_only,
        "softmax_only": softmax_only,
        "baseline_backend_calls": baseline_provider.calls,
        "observed_backend_calls": observed_provider.calls,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--c-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    args = parser.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        c_config = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise AttentionScoreError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)

        baseline_library, baseline_source = build_head_norm_probe_library(
            Path(args.out).with_name(Path(args.out).stem + "-baseline.so")
        )
        observed_library, observed_source = build_score_observation_library(
            Path(args.out).with_name(Path(args.out).stem + "-observed.so")
        )
        if not baseline_source["production_source_unchanged"]:
            raise AttentionScoreError("baseline changed production source")
        if not observed_source["production_source_unchanged"]:
            raise AttentionScoreError("observed build changed production source")
        baseline_lib = ctypes.CDLL(str(baseline_library))
        observed_lib = ctypes.CDLL(str(observed_library))
        configure_library(baseline_lib)
        configure_library(observed_lib)

        analyses = {
            str(layer): diagnose_layer(
                baseline_lib,
                observed_lib,
                fixture,
                config,
                c_config,
                layer,
                inputs,
                input_values,
                device=device,
                dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-bfloat16-attention-score-softmax-probe",
            "version": 2,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "baseline_source": baseline_source,
            "observed_source": observed_source,
            "layers": analyses,
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        AttentionScoreError,
        AttentionComponentError,
        AttentionHeadNormError,
        NativeBfloat16BackendError,
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
        PreRouterDiagnosisError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
