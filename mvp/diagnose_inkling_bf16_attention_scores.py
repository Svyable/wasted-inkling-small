#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Separate Inkling pre-softmax score drift from softmax implementation drift.

All inputs to this diagnostic are already bounded by earlier probes: layer
RMSNorm, native BF16 Q/K/V/R matrices, K/V short convolution, and Q/K per-head
RMSNorm. The tool reconstructs both official eager-attention logits and the C
float score loop from the same captured tensors.

Two mandatory sanity checks make the result interpretable:

* reconstructed official logits + official float32 softmax reproduce captured
  official probabilities exactly in BF16;
* reconstructed C logits + the C expf/double-sum normalization reproduce the
  probabilities read from C scratch exactly.

It then crosses the components to classify score construction, softmax, or
both as the remaining boundary.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
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
)
from diagnose_inkling_native_bf16_backend import (
    NativeBfloat16BackendError,
    NativeMatrixProvider,
)
from diagnose_inkling_pre_router import (
    PreRouterDiagnosisError,
    _capture_official_pre_router,
)
from discover_inkling_router_experts import input_sha256
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    verify_config_binding,
)
from inkling_layer_parity import LayerParityError, configure_library
from inkling_release_config import build_transformers_text_config


class AttentionScoreError(RuntimeError):
    """Attention score and softmax components cannot be aligned exactly."""


_LIBM = ctypes.CDLL(None)
_LIBM.sqrtf.argtypes = [ctypes.c_float]
_LIBM.sqrtf.restype = ctypes.c_float
_LIBM.logf.argtypes = [ctypes.c_float]
_LIBM.logf.restype = ctypes.c_float
_LIBM.expf.argtypes = [ctypes.c_float]
_LIBM.expf.restype = ctypes.c_float


def f32(value: float) -> float:
    return ctypes.c_float(value).value


def bf16(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32).to(torch.bfloat16).float())


def c_head_rmsnorm(
    values: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Apply the temporary candidate's exact C + BF16 head RMSNorm ordering."""
    if values.ndim != 1 or weight.ndim != 1 or values.shape != weight.shape:
        raise AttentionScoreError(
            f"head RMSNorm shapes disagree: {tuple(values.shape)} and {tuple(weight.shape)}"
        )
    squared_sum = 0.0
    for value in values.float().tolist():
        squared_sum += float(value) * float(value)
    mean = f32(squared_sum / int(values.numel()))
    denominator = _LIBM.sqrtf(f32(mean + f32(eps)))
    scale = f32(1.0 / float(denominator))
    output = []
    for value, factor in zip(values.float().tolist(), weight.float().tolist()):
        normalized = bf16(f32(float(value) * scale))
        output.append(bf16(f32(normalized * float(factor))))
    return torch.tensor(output, dtype=torch.float32)


def c_log_tau(position: int, floor: int, alpha: float) -> float:
    if position < 0 or floor <= 0:
        raise AttentionScoreError("invalid log-scaling geometry")
    ratio = f32(f32(position + 1) / f32(floor))
    argument = ratio if ratio > 1.0 else 1.0
    logarithm = float(_LIBM.logf(f32(argument)))
    return f32(1.0 + f32(f32(alpha) * logarithm))


def attention_geometry(cfg: dict[str, Any], layer: int) -> dict[str, Any]:
    is_local = layer in set(cfg.get("local_layer_ids", []))
    return {
        "is_local": is_local,
        "heads": int(cfg["local_heads"] if is_local else cfg["global_heads"]),
        "kv_heads": int(cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"]),
        "head_dim": int(cfg["local_head_dim"] if is_local else cfg["global_head_dim"]),
        "extent": int(cfg["sliding_window"] if is_local else cfg["rel_extent"]),
        "d_rel": int(cfg["d_rel"]),
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


def reconstruct_c_logits(
    module: Any,
    stages: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    layer: int,
) -> torch.Tensor:
    """Reproduce the C float score loop over exact projected inputs."""
    geometry = attention_geometry(cfg, layer)
    tokens = int(stages["q_proj"].shape[0])
    heads = geometry["heads"]
    kv_heads = geometry["kv_heads"]
    head_dim = geometry["head_dim"]
    group = heads // kv_heads
    if group < 1 or heads % kv_heads:
        raise AttentionScoreError("C head geometry is not grouped")
    q_values = stages["q_proj"].float().cpu().reshape(tokens, heads, head_dim)
    k_values = stages["k_sconv"].float().cpu().reshape(tokens, kv_heads, head_dim)
    relative = stages["relative_proj_input"].float().cpu().reshape(
        tokens, heads, geometry["d_rel"]
    )
    q_weight = module.self_attn.q_norm.weight.detach().float().cpu()
    k_weight = module.self_attn.k_norm.weight.detach().float().cpu()
    projection = module.self_attn.rel_logits_proj.proj.detach().float().cpu()
    eps = float(cfg["rms_eps"])
    normalized_q = torch.stack(
        [
            torch.stack(
                [c_head_rmsnorm(q_values[p, h], q_weight, eps=eps) for h in range(heads)]
            )
            for p in range(tokens)
        ]
    )
    normalized_k = torch.stack(
        [
            torch.stack(
                [c_head_rmsnorm(k_values[p, h], k_weight, eps=eps) for h in range(kv_heads)]
            )
            for p in range(tokens)
        ]
    )
    logits = torch.full((tokens, heads, tokens), -torch.inf, dtype=torch.float32)
    for position in range(tokens):
        tau = (
            1.0
            if geometry["is_local"]
            else c_log_tau(
                position,
                int(cfg["log_scaling_n_floor"]),
                float(cfg["log_scaling_alpha"]),
            )
        )
        dot_scale = f32(tau / f32(head_dim))
        for head in range(heads):
            kv_head = head // group
            for key in range(tokens):
                if not causal_valid(
                    position,
                    key,
                    is_local=geometry["is_local"],
                    window=geometry["extent"],
                ):
                    continue
                score = f32(0.0)
                for dim in range(head_dim):
                    product = f32(
                        float(normalized_q[position, head, dim])
                        * float(normalized_k[key, kv_head, dim])
                    )
                    score = f32(score + product)
                score = f32(score * dot_scale)
                distance = position - key
                if 0 <= distance < geometry["extent"]:
                    bias = f32(0.0)
                    for index in range(geometry["d_rel"]):
                        product = f32(
                            float(relative[position, head, index])
                            * float(projection[index, distance])
                        )
                        bias = f32(bias + product)
                    score = f32(score + f32(tau * bias))
                logits[position, head, key] = score
    return logits


def official_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1, dtype=torch.float32).to(
        torch.bfloat16
    ).float().cpu().contiguous()


def c_softmax(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise AttentionScoreError(f"softmax logits must be rank 3, got {tuple(logits.shape)}")
    tokens, heads, keys = logits.shape
    result = torch.zeros(tokens, heads, keys, dtype=torch.float32)
    for position in range(tokens):
        for head in range(heads):
            finite = [
                key for key in range(keys) if math.isfinite(float(logits[position, head, key]))
            ]
            if not finite:
                raise AttentionScoreError(
                    f"softmax row {position}/{head} has no finite values"
                )
            maximum = max(float(logits[position, head, key]) for key in finite)
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


def classify_score_softmax(
    *,
    official_sanity: dict[str, Any],
    c_sanity: dict[str, Any],
    score_only: dict[str, Any],
    softmax_only: dict[str, Any],
) -> str:
    if official_sanity["quantized_exact_fraction"] != 1.0:
        return "official_score_reconstruction_failed"
    if c_sanity["max_abs"] > 1e-7:
        return "c_score_reconstruction_failed"
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
    lib: Any,
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
    provider = NativeMatrixProvider(
        layer,
        module,
        official_norm=official["input_norm"],
    )
    candidate = capture_c_attention_components(
        lib,
        fixture,
        c_config,
        layer,
        input_values,
        provider,
    )
    official_logits = reconstruct_official_logits(
        module,
        official,
        config,
        layer,
        dtype=dtype,
        device=device,
    )
    c_logits = reconstruct_c_logits(module, official, c_config, layer)
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
        candidate["probabilities"],
        c_reconstructed,
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
    logits = _tensor_metrics(
        official_logits,
        c_logits,
        compare_dtype=dtype,
    )
    return {
        "classification": classify_score_softmax(
            official_sanity=official_sanity,
            c_sanity=c_sanity,
            score_only=score_only,
            softmax_only=softmax_only,
        ),
        "logits": logits,
        "official_score_sanity": official_sanity,
        "c_score_sanity": c_sanity,
        "score_construction_only": score_only,
        "softmax_only": softmax_only,
        "backend_calls": provider.calls,
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
        library, source = build_head_norm_probe_library(
            Path(args.out).with_suffix(".so")
        )
        if not source["production_source_unchanged"]:
            raise AttentionScoreError("production source changed")
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        analyses = {
            str(layer): diagnose_layer(
                lib,
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
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "source": source,
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
