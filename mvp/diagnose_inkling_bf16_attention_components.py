#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Separate Inkling attention probability drift from value aggregation drift.

The preceding probes make layer RMSNorm, Q/K/V/R projections, K/V short
convolution, and native BF16 matrix boundaries exact. Q/K per-head RMSNorm is
included here because it is independently proven to contribute to the result.

This diagnostic then observes the two remaining halves of eager attention:

1. score construction + float32 softmax -> BF16 probabilities;
2. BF16 probability/value matrix multiplication.

C probabilities are reconstructed from the post-exp scratch rows using the same
double sum and float32 reciprocal/product ordering as ``inkling_attention.c``.
Two cross-combinations identify the responsible component without changing the
production runtime:

* official probabilities through C-style float accumulation;
* C probabilities through the pinned native BF16 value matmul.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

import torch

from analyze_inkling_router_mismatch import _tensor_metrics
from diagnose_inkling_bf16_attention_head_norm import (
    AttentionHeadNormError,
    build_head_norm_probe_library,
)
from diagnose_inkling_native_bf16_backend import (
    NULL_WEIGHT_FIELDS,
    NativeBfloat16BackendError,
    NativeMatrixProvider,
)
from diagnose_inkling_pre_router import (
    PreRouterDiagnosisError,
    _capture_official_pre_router,
)
from discover_inkling_c_router_experts import _RejectExperts
from discover_inkling_router_experts import (
    _CaptureExperts,
    _RouterCaptured,
    _causal_mask,
    input_sha256,
)
from inkling_compact_layer_parity import bind_sparse_layer_compact
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    _import_transformers,
    verify_config_binding,
)
from inkling_layer_parity import (
    FP,
    LayerParityError,
    LayerScratch,
    LayerState,
    TraceCollector,
    _build_config,
    configure_library,
)
from inkling_release_config import build_transformers_text_config


class AttentionComponentError(RuntimeError):
    """Attention components cannot be aligned without guessing."""


def capture_official_attention_probabilities(
    module: Any,
    config: Any,
    layer: int,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Capture eager-attention probabilities as [tokens, heads, keys]."""
    _, _, classes, _ = _import_transformers()
    _, DynamicCache = classes
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _args: Any, output: Any) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise AttentionComponentError("official attention output has no weights")
        weights = output[1]
        if not isinstance(weights, torch.Tensor):
            raise AttentionComponentError("official eager attention returned no weights")
        captured["probabilities"] = weights.detach().clone()

    handle = module.self_attn.register_forward_hook(hook)
    module.mlp.experts = _CaptureExperts()
    prefix = inputs.to(device=device, dtype=dtype).unsqueeze(0)
    try:
        try:
            with torch.no_grad():
                module(
                    prefix,
                    attention_mask=_causal_mask(
                        config,
                        layer,
                        int(inputs.shape[0]),
                        device=device,
                        dtype=dtype,
                    ),
                    conv_mask=None,
                    past_key_values=DynamicCache(config=config),
                )
        except _RouterCaptured:
            pass
        else:
            raise AttentionComponentError(
                "official layer completed without reaching routed experts"
            )
    finally:
        handle.remove()
    weights = captured.get("probabilities")
    if weights is None:
        raise AttentionComponentError("official attention probabilities were not captured")
    if weights.ndim != 4 or weights.shape[0] != 1:
        raise AttentionComponentError(
            f"official probabilities have shape {tuple(weights.shape)}"
        )
    weights = weights[0].transpose(0, 1).contiguous()
    tokens = int(inputs.shape[0])
    if weights.shape[0] != tokens or weights.shape[2] != tokens:
        raise AttentionComponentError(
            f"official probabilities have shape {tuple(weights.shape)}; expected "
            f"[{tokens}, heads, {tokens}]"
        )
    return weights


def reconstruct_c_probabilities(
    scratch: LayerScratch,
    *,
    position: int,
    heads: int,
    capacity: int,
    is_local: bool,
    tokens: int,
) -> torch.Tensor:
    """Reconstruct C probabilities from post-exp scratch rows."""
    begin = position + 1 - capacity if is_local and position + 1 > capacity else 0
    kv_len = position - begin + 1
    result = torch.zeros(heads, tokens, dtype=torch.float32)
    for head in range(heads):
        offset = head * capacity
        exponentials = [float(scratch.scores[offset + index]) for index in range(kv_len)]
        total = sum(exponentials)  # Python float is the C double-sum analogue.
        if not total > 0.0:
            raise AttentionComponentError(
                f"invalid C softmax sum at position {position}, head {head}"
            )
        inverse = ctypes.c_float(1.0 / total).value
        for index, value in enumerate(exponentials):
            probability = ctypes.c_float(value * inverse).value
            result[head, begin + index] = probability
    return result


def capture_c_attention_components(
    lib: Any,
    fixture: Any,
    cfg: dict[str, Any],
    layer: int,
    inputs: list[list[float]],
    provider: NativeMatrixProvider,
) -> dict[str, torch.Tensor]:
    """Run C to expert lookup and retain probabilities, V, and attention output."""
    compact = bind_sparse_layer_compact(fixture, cfg, layer)
    for field in NULL_WEIGHT_FIELDS:
        setattr(compact.weights, field, FP())
    config = _build_config(lib, cfg)
    hidden = int(cfg["hidden"])
    is_local = layer in set(cfg.get("local_layer_ids", []))
    heads = int(cfg["local_heads"] if is_local else cfg["global_heads"])
    kv_heads = int(cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"])
    head_dim = int(cfg["local_head_dim"] if is_local else cfg["global_head_dim"])
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
        raise AttentionComponentError("scratch sizing failed")
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
        raise AttentionComponentError("scratch init failed")
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
        raise AttentionComponentError("state init failed")

    collector = TraceCollector()
    reject = _RejectExperts(layer)
    reject_ptr = ctypes.cast(reject.callback, ctypes.c_void_p)
    probabilities = []
    for position, row in enumerate(inputs):
        if len(row) != hidden:
            raise AttentionComponentError(
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
            raise AttentionComponentError(provider.error)
        if rc == 0 or len(reject.requests) != before + 1:
            raise AttentionComponentError(
                f"layer {layer} position {position} did not stop at expert lookup"
            )
        probabilities.append(
            reconstruct_c_probabilities(
                scratch,
                position=position,
                heads=heads,
                capacity=capacity,
                is_local=is_local,
                tokens=len(inputs),
            )
        )

    def rows(point: str) -> torch.Tensor:
        values = []
        for position in range(len(inputs)):
            name = f"token.{position}.layer.{layer}.{point}"
            if name not in collector.values:
                raise AttentionComponentError(f"C trace missed {name}")
            values.append(collector.values[name])
        return torch.tensor(values, dtype=torch.float32)

    return {
        "probabilities": torch.stack(probabilities),
        "values": rows("v_sconv"),
        "attention_out": rows("attention_out"),
    }


def repeat_values_by_head(
    values: torch.Tensor,
    *,
    heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Convert [tokens, kv_heads*D] to [heads, tokens, D]."""
    if values.ndim != 2 or values.shape[1] != kv_heads * head_dim:
        raise AttentionComponentError(
            f"value shape {tuple(values.shape)} disagrees with KV geometry"
        )
    if heads % kv_heads:
        raise AttentionComponentError("attention heads are not divisible by KV heads")
    grouped = values.reshape(values.shape[0], kv_heads, head_dim).permute(1, 0, 2)
    return grouped.repeat_interleave(heads // kv_heads, dim=0).contiguous()


def native_bfloat16_value_aggregation(
    probabilities: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Apply official-shaped BF16 probability/value matmul."""
    if probabilities.ndim != 3 or values.ndim != 3:
        raise AttentionComponentError("probability and value tensors must be rank 3")
    tokens, heads, keys = probabilities.shape
    if values.shape[0] != heads or values.shape[1] != keys:
        raise AttentionComponentError(
            f"aggregation shapes disagree: {tuple(probabilities.shape)} and {tuple(values.shape)}"
        )
    probs = probabilities.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
    vals = values.unsqueeze(0).to(torch.bfloat16)
    with torch.no_grad():
        output = torch.matmul(probs, vals)[0]
    return output.permute(1, 0, 2).reshape(tokens, -1).float().cpu().contiguous()


def c_style_value_aggregation(
    probabilities: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Apply C float multiply/add ordering to [tokens, heads, keys]."""
    if probabilities.ndim != 3 or values.ndim != 3:
        raise AttentionComponentError("probability and value tensors must be rank 3")
    tokens, heads, keys = probabilities.shape
    if values.shape[0] != heads or values.shape[1] != keys:
        raise AttentionComponentError(
            f"aggregation shapes disagree: {tuple(probabilities.shape)} and {tuple(values.shape)}"
        )
    head_dim = int(values.shape[2])
    output = torch.zeros(tokens, heads, head_dim, dtype=torch.float32)
    for position in range(tokens):
        for head in range(heads):
            for dim in range(head_dim):
                accumulator = ctypes.c_float(0.0).value
                for key in range(keys):
                    product = ctypes.c_float(
                        float(probabilities[position, head, key])
                        * float(values[head, key, dim])
                    ).value
                    accumulator = ctypes.c_float(accumulator + product).value
                output[position, head, dim] = accumulator
    return output.reshape(tokens, heads * head_dim).contiguous()


def classify_components(
    *,
    probability_metrics: dict[str, Any],
    official_native_sanity: dict[str, Any],
    c_style_sanity: dict[str, Any],
    score_softmax_only: dict[str, Any],
    aggregation_only: dict[str, Any],
) -> str:
    if official_native_sanity["quantized_exact_fraction"] != 1.0:
        return "official_native_aggregation_sanity_failed"
    if c_style_sanity["max_abs"] > 1e-6:
        return "c_style_aggregation_sanity_failed"
    probability_exact = probability_metrics["quantized_exact_fraction"] == 1.0
    score_exact = score_softmax_only["quantized_exact_fraction"] == 1.0
    aggregation_exact = aggregation_only["quantized_exact_fraction"] == 1.0
    if probability_exact and score_exact and aggregation_exact:
        return "attention_components_are_exact"
    if aggregation_exact and not score_exact:
        return "score_or_softmax_is_the_remaining_boundary"
    if score_exact and not aggregation_exact:
        return "value_aggregation_is_the_remaining_boundary"
    if not score_exact and not aggregation_exact:
        return "score_softmax_and_value_aggregation_both_contribute"
    return "attention_component_interaction_requires_deeper_trace"


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

    is_local = layer in set(c_config.get("local_layer_ids", []))
    heads = int(c_config["local_heads"] if is_local else c_config["global_heads"])
    kv_heads = int(c_config["local_kv_heads"] if is_local else c_config["global_kv_heads"])
    head_dim = int(c_config["local_head_dim"] if is_local else c_config["global_head_dim"])
    official_values = repeat_values_by_head(
        official["v_sconv"].float().cpu(),
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    candidate_values = repeat_values_by_head(
        candidate["values"],
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    official_output = official["attention_out"].float().cpu()
    candidate_output = candidate["attention_out"]

    official_native = native_bfloat16_value_aggregation(
        official_probabilities, official_values
    )
    c_style_reconstructed = c_style_value_aggregation(
        candidate["probabilities"], candidate_values
    )
    score_softmax_only_output = native_bfloat16_value_aggregation(
        candidate["probabilities"], official_values
    )
    aggregation_only_output = c_style_value_aggregation(
        official_probabilities, official_values
    )

    probability_metrics = _tensor_metrics(
        official_probabilities,
        candidate["probabilities"],
        compare_dtype=dtype,
    )
    official_native_sanity = _tensor_metrics(
        official_output,
        official_native,
        compare_dtype=dtype,
    )
    c_style_sanity = _tensor_metrics(
        candidate_output,
        c_style_reconstructed,
        compare_dtype=torch.float32,
    )
    score_softmax_only = _tensor_metrics(
        official_output,
        score_softmax_only_output,
        compare_dtype=dtype,
    )
    aggregation_only = _tensor_metrics(
        official_output,
        aggregation_only_output,
        compare_dtype=dtype,
    )
    baseline = _tensor_metrics(
        official_output,
        candidate_output,
        compare_dtype=dtype,
    )
    classification = classify_components(
        probability_metrics=probability_metrics,
        official_native_sanity=official_native_sanity,
        c_style_sanity=c_style_sanity,
        score_softmax_only=score_softmax_only,
        aggregation_only=aggregation_only,
    )
    return {
        "classification": classification,
        "baseline_attention_out": baseline,
        "probabilities": probability_metrics,
        "official_native_aggregation_sanity": official_native_sanity,
        "c_style_aggregation_sanity": c_style_sanity,
        "score_softmax_only": score_softmax_only,
        "value_aggregation_only": aggregation_only,
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
            raise AttentionComponentError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        library, source = build_head_norm_probe_library(
            Path(args.out).with_suffix(".so")
        )
        if not source["production_source_unchanged"]:
            raise AttentionComponentError("production source changed")
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
            "format": "inkling-bfloat16-attention-component-probe",
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
