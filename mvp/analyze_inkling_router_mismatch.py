#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Separate Inkling pre-router drift from BF16 router-matmul semantics."""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "inkling" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from discover_inkling_c_router_experts import discover_c_layer_router
from discover_inkling_router_experts import (
    _CaptureExperts,
    _RouterCaptured,
    _causal_mask,
    input_sha256,
)
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    _import_transformers,
    build_layer_from_fixture,
    verify_config_binding,
)
from inkling_layer_parity import (
    LayerParityError,
    build_library,
    configure_library,
)
from inkling_release_config import build_transformers_text_config


class RouterAnalysisError(RuntimeError):
    """Router diagnostics cannot be produced without guessing."""


def _route(
    logits: torch.Tensor,
    *,
    correction_bias: torch.Tensor,
    global_scale: torch.Tensor,
    route_scale: float,
    top_k: int,
    n_shared: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the official 5.14.1 router exactly in the input tensor dtype."""
    scores = logits.sigmoid()
    routed_scores = scores[..., :-n_shared]
    choice = routed_scores + correction_bias
    indices = torch.topk(choice, top_k, dim=-1, sorted=False).indices
    routed_logits = logits[..., :-n_shared]
    shared_logits = logits[..., -n_shared:]
    selected = torch.cat(
        [routed_logits.gather(-1, indices), shared_logits], dim=-1
    )
    log_probs = F.logsigmoid(selected)
    weights = torch.exp(
        log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    )
    weights = weights * route_scale * global_scale
    return indices, weights[..., :top_k].contiguous(), choice


def _sorted_ids(values: torch.Tensor) -> list[list[int]]:
    return [sorted(int(item) for item in row) for row in values.tolist()]


def _variant_summary(
    name: str,
    indices: torch.Tensor,
    weights: torch.Tensor,
    expected_indices: torch.Tensor,
    expected_weights: torch.Tensor,
) -> dict[str, Any]:
    rows = []
    exact = 0
    max_abs = 0.0
    for position in range(indices.shape[0]):
        got_pairs = sorted(
            zip(
                [int(value) for value in indices[position].tolist()],
                [float(value) for value in weights[position].float().tolist()],
            )
        )
        want_pairs = sorted(
            zip(
                [int(value) for value in expected_indices[position].tolist()],
                [float(value) for value in expected_weights[position].float().tolist()],
            )
        )
        ids_ok = [pair[0] for pair in got_pairs] == [pair[0] for pair in want_pairs]
        row_abs = None
        if ids_ok:
            exact += 1
            row_abs = max(
                (
                    abs(got[1] - want[1])
                    for got, want in zip(got_pairs, want_pairs)
                ),
                default=0.0,
            )
            max_abs = max(max_abs, row_abs)
        rows.append(
            {
                "position": position,
                "exact_indices": ids_ok,
                "max_abs_weight": row_abs,
                "expected_ids": [pair[0] for pair in want_pairs],
                "actual_ids": [pair[0] for pair in got_pairs],
            }
        )
    return {
        "name": name,
        "exact_rows": exact,
        "total_rows": int(indices.shape[0]),
        "all_indices_exact": exact == int(indices.shape[0]),
        "max_abs_weight_on_exact_rows": max_abs,
        "rows": rows,
    }


def _tensor_metrics(
    official: torch.Tensor,
    candidate: torch.Tensor,
    *,
    compare_dtype: torch.dtype,
) -> dict[str, Any]:
    if official.shape != candidate.shape:
        raise RouterAnalysisError(
            f"tensor shape mismatch: {tuple(official.shape)} vs {tuple(candidate.shape)}"
        )
    a = official.detach().float().cpu()
    b = candidate.detach().float().cpu()
    delta = (a - b).abs()
    a_q = official.detach().to(compare_dtype).cpu()
    b_q = candidate.detach().to(compare_dtype).cpu()
    exact = a_q == b_q
    per_position = []
    for position in range(a.shape[0]):
        row_delta = delta[position]
        row_exact = exact[position]
        per_position.append(
            {
                "position": position,
                "max_abs": float(row_delta.max()) if row_delta.numel() else 0.0,
                "mean_abs": float(row_delta.mean()) if row_delta.numel() else 0.0,
                "quantized_exact": int(row_exact.sum()),
                "elements": int(row_exact.numel()),
                "quantized_exact_fraction": float(row_exact.float().mean()),
            }
        )
    return {
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "quantized_exact": int(exact.sum()),
        "elements": int(exact.numel()),
        "quantized_exact_fraction": float(exact.float().mean()),
        "per_position": per_position,
    }


def _decision_margins(choice: torch.Tensor, top_k: int) -> list[float]:
    values = torch.topk(choice.float(), top_k + 1, dim=-1, sorted=True).values
    return [
        float(values[position, top_k - 1] - values[position, top_k])
        for position in range(values.shape[0])
    ]


def capture_official_router(
    fixture,
    config,
    layer: int,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return module, full pre-router states, logits, IDs, and routed weights."""
    _, _, classes, _ = _import_transformers()
    _, DynamicCache = classes
    module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    captured: dict[str, torch.Tensor] = {}

    def prehook(_module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise RouterAnalysisError("official gate prehook received no tensor input")
        captured["input"] = args[0].detach().clone()

    handle = module.mlp.gate.register_forward_pre_hook(prehook)
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
                        inputs.shape[0],
                        device=device,
                        dtype=dtype,
                    ),
                    conv_mask=None,
                    past_key_values=DynamicCache(config=config),
                )
        except _RouterCaptured as result:
            official_indices = result.indices
            official_weights = result.weights
        else:
            raise RouterAnalysisError(
                "official layer completed without invoking routed experts"
            )
    finally:
        handle.remove()
    if "input" not in captured:
        raise RouterAnalysisError("official gate input was not captured")
    router_input = captured["input"].reshape(-1, int(config.hidden_size))
    with torch.no_grad():
        router_logits = F.linear(router_input, module.mlp.gate.weight)
    return module, router_input, router_logits, official_indices, official_weights


def analyze_layer(
    module,
    official_input: torch.Tensor,
    official_logits: torch.Tensor,
    official_indices: torch.Tensor,
    official_weights: torch.Tensor,
    c_result: dict[str, Any],
) -> dict[str, Any]:
    gate = module.mlp.gate
    c_input = torch.tensor(c_result["router_inputs"], dtype=torch.float32)
    c_logits = torch.tensor(c_result["router_logits"], dtype=torch.float32)
    c_indices = torch.tensor(c_result["topk_indices"], dtype=torch.int64)
    c_weights = torch.tensor(c_result["topk_weights"], dtype=torch.float32)
    top_k = int(gate.top_k)
    n_shared = int(gate.n_shared_experts)

    def route_variant(
        name: str,
        logits: torch.Tensor,
        *,
        parameter_dtype: torch.dtype,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        indices, weights, choice = _route(
            logits,
            correction_bias=gate.e_score_correction_bias.detach().to(parameter_dtype),
            global_scale=gate.global_scale.detach().to(parameter_dtype),
            route_scale=float(gate.route_scale),
            top_k=top_k,
            n_shared=n_shared,
        )
        return (
            _variant_summary(
                name,
                indices.cpu(),
                weights.cpu(),
                official_indices.cpu(),
                official_weights.cpu(),
            ),
            choice,
        )

    official_variant, official_choice = route_variant(
        "official_recomputed",
        official_logits,
        parameter_dtype=official_logits.dtype,
    )
    c_emitted = _variant_summary(
        "c_emitted",
        c_indices,
        c_weights,
        official_indices.cpu(),
        official_weights.cpu(),
    )
    c_float_variant, c_float_choice = route_variant(
        "c_logits_float32_official_route",
        c_logits,
        parameter_dtype=torch.float32,
    )
    c_rounded_variant, c_rounded_choice = route_variant(
        "c_logits_rounded_bfloat16",
        c_logits.to(torch.bfloat16),
        parameter_dtype=torch.bfloat16,
    )
    with torch.no_grad():
        official_matmul_c_input = F.linear(
            c_input.to(torch.bfloat16),
            gate.weight.detach().to(torch.bfloat16),
        )
    c_input_matmul_variant, c_input_matmul_choice = route_variant(
        "official_bfloat16_matmul_on_c_input",
        official_matmul_c_input,
        parameter_dtype=torch.bfloat16,
    )
    with torch.no_grad():
        float_matmul_official_input = F.linear(
            official_input.float(), gate.weight.detach().float()
        )
    float_official_input_variant, float_official_choice = route_variant(
        "float32_matmul_on_official_input",
        float_matmul_official_input,
        parameter_dtype=torch.float32,
    )

    if c_rounded_variant["all_indices_exact"]:
        classification = "c_router_logit_bfloat16_rounding_is_sufficient"
    elif c_input_matmul_variant["all_indices_exact"]:
        input_exact = _tensor_metrics(
            official_input, c_input, compare_dtype=torch.bfloat16
        )["quantized_exact_fraction"]
        classification = (
            "router_matmul_bfloat16_semantics"
            if input_exact == 1.0
            else "pre_router_drift_does_not_change_official_bfloat16_routing"
        )
    else:
        classification = "pre_router_state_drift_changes_official_bfloat16_routing"

    return {
        "classification": classification,
        "pre_router_state": _tensor_metrics(
            official_input, c_input, compare_dtype=torch.bfloat16
        ),
        "router_logits": {
            "official_vs_c_float": _tensor_metrics(
                official_logits, c_logits, compare_dtype=torch.bfloat16
            ),
            "official_vs_official_matmul_c_input": _tensor_metrics(
                official_logits,
                official_matmul_c_input,
                compare_dtype=torch.bfloat16,
            ),
        },
        "variants": {
            item["name"]: item
            for item in (
                official_variant,
                c_emitted,
                c_float_variant,
                c_rounded_variant,
                c_input_matmul_variant,
                float_official_input_variant,
            )
        },
        "decision_margin": {
            "official": _decision_margins(official_choice, top_k),
            "c_float32": _decision_margins(c_float_choice, top_k),
            "c_rounded_bfloat16": _decision_margins(c_rounded_choice, top_k),
            "official_bfloat16_matmul_c_input": _decision_margins(
                c_input_matmul_choice, top_k
            ),
            "float32_matmul_official_input": _decision_margins(
                float_official_choice, top_k
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--c-config", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    args = ap.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        c_config = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise RouterAnalysisError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        library = build_library(
            REPO / "inkling" / "src", Path(args.out).with_suffix(".so")
        )
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        analyses = {}
        for layer in layers:
            module, official_input, official_logits, indices, weights = (
                capture_official_router(
                    fixture,
                    config,
                    layer,
                    inputs,
                    device=device,
                    dtype=dtype,
                )
            )
            c_result = discover_c_layer_router(
                lib, fixture, c_config, layer, input_values
            )
            analyses[str(layer)] = analyze_layer(
                module,
                official_input,
                official_logits,
                indices,
                weights,
                c_result,
            )
        result = {
            "format": "inkling-router-mismatch-analysis",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "layers": analyses,
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
        RouterAnalysisError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        ap.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
