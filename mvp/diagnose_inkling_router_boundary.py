#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Classify official-vs-C Inkling routing divergence at the BF16 boundary."""
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

from analyze_inkling_router_mismatch import (
    RouterAnalysisError,
    _decision_margins,
    _route,
    _tensor_metrics,
    capture_official_router,
)
from discover_inkling_c_router_experts import discover_c_layer_router
from discover_inkling_router_experts import input_sha256
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    verify_config_binding,
)
from inkling_layer_parity import (
    LayerParityError,
    build_library,
    configure_library,
)
from inkling_release_config import build_transformers_text_config


def _select(
    routed_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if routed_logits.shape[-1] != correction_bias.numel():
        raise RouterAnalysisError(
            f"routed logit width {routed_logits.shape[-1]} differs from bias "
            f"width {correction_bias.numel()}"
        )
    choice = routed_logits.sigmoid() + correction_bias
    indices = torch.topk(choice, top_k, dim=-1, sorted=False).indices
    return indices, choice


def _id_summary(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, Any]:
    rows = []
    exact = 0
    for position in range(actual.shape[0]):
        got = sorted(int(value) for value in actual[position].tolist())
        want = sorted(int(value) for value in expected[position].tolist())
        ok = got == want
        exact += int(ok)
        rows.append(
            {
                "position": position,
                "exact_indices": ok,
                "expected_ids": want,
                "actual_ids": got,
            }
        )
    return {
        "name": name,
        "all_indices_exact": exact == int(actual.shape[0]),
        "exact_rows": exact,
        "total_rows": int(actual.shape[0]),
        "rows": rows,
    }


def _emitted_summary(
    indices: torch.Tensor,
    weights: torch.Tensor,
    expected_indices: torch.Tensor,
    expected_weights: torch.Tensor,
) -> dict[str, Any]:
    base = _id_summary("c_emitted", indices, expected_indices)
    max_abs = 0.0
    for row, item in enumerate(base["rows"]):
        if not item["exact_indices"]:
            item["max_abs_weight"] = None
            continue
        actual = sorted(
            zip(
                [int(value) for value in indices[row].tolist()],
                [float(value) for value in weights[row].tolist()],
            )
        )
        expected = sorted(
            zip(
                [int(value) for value in expected_indices[row].tolist()],
                [float(value) for value in expected_weights[row].tolist()],
            )
        )
        value = max(
            (abs(got[1] - want[1]) for got, want in zip(actual, expected)),
            default=0.0,
        )
        max_abs = max(max_abs, value)
        item["max_abs_weight"] = value
    base["max_abs_weight_on_exact_rows"] = max_abs
    return base


def _full_route_ids(
    logits: torch.Tensor,
    gate,
    *,
    parameter_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices, _weights, choice = _route(
        logits,
        correction_bias=gate.e_score_correction_bias.detach().to(parameter_dtype),
        global_scale=gate.global_scale.detach().to(parameter_dtype),
        route_scale=float(gate.route_scale),
        top_k=int(gate.top_k),
        n_shared=int(gate.n_shared_experts),
    )
    return indices, choice


def diagnose_layer(
    module,
    official_input: torch.Tensor,
    official_logits: torch.Tensor,
    official_indices: torch.Tensor,
    official_weights: torch.Tensor,
    c_result: dict[str, Any],
) -> dict[str, Any]:
    gate = module.mlp.gate
    routed_n = int(gate.num_experts)
    top_k = int(gate.top_k)
    c_input = torch.tensor(c_result["router_inputs"], dtype=torch.float32)
    c_logits = torch.tensor(c_result["router_logits"], dtype=torch.float32)
    c_indices = torch.tensor(c_result["topk_indices"], dtype=torch.int64)
    c_weights = torch.tensor(c_result["topk_weights"], dtype=torch.float32)
    if c_logits.shape[-1] != routed_n:
        raise RouterAnalysisError(
            f"C trace contains {c_logits.shape[-1]} logits; expected {routed_n} routed logits"
        )

    official_recomputed, official_choice = _full_route_ids(
        official_logits,
        gate,
        parameter_dtype=official_logits.dtype,
    )
    if not _id_summary(
        "official_recomputed", official_recomputed.cpu(), official_indices.cpu()
    )["all_indices_exact"]:
        raise RouterAnalysisError("recomputed official routing differs from captured routing")

    c_float_indices, c_float_choice = _select(
        c_logits,
        gate.e_score_correction_bias.detach().float(),
        top_k,
    )
    c_bf16_logits = c_logits.to(torch.bfloat16)
    c_rounded_indices, c_rounded_choice = _select(
        c_bf16_logits,
        gate.e_score_correction_bias.detach().to(torch.bfloat16),
        top_k,
    )
    with torch.no_grad():
        official_matmul_c_input = F.linear(
            c_input.to(torch.bfloat16),
            gate.weight.detach().to(torch.bfloat16),
        )
        float_matmul_official_input = F.linear(
            official_input.float(), gate.weight.detach().float()
        )
    c_input_indices, c_input_choice = _full_route_ids(
        official_matmul_c_input,
        gate,
        parameter_dtype=torch.bfloat16,
    )
    float_official_indices, float_official_choice = _full_route_ids(
        float_matmul_official_input,
        gate,
        parameter_dtype=torch.float32,
    )

    emitted = _emitted_summary(
        c_indices,
        c_weights,
        official_indices.cpu(),
        official_weights.cpu(),
    )
    c_float = _id_summary(
        "c_routed_logits_float32", c_float_indices, official_indices.cpu()
    )
    c_rounded = _id_summary(
        "c_routed_logits_rounded_bfloat16",
        c_rounded_indices.cpu(),
        official_indices.cpu(),
    )
    c_input_matmul = _id_summary(
        "official_bfloat16_matmul_on_c_input",
        c_input_indices.cpu(),
        official_indices.cpu(),
    )
    float_official = _id_summary(
        "float32_matmul_on_official_input",
        float_official_indices.cpu(),
        official_indices.cpu(),
    )

    input_metrics = _tensor_metrics(
        official_input, c_input, compare_dtype=torch.bfloat16
    )
    if c_rounded["all_indices_exact"]:
        classification = "c_router_logit_bfloat16_rounding_is_sufficient"
    elif c_input_matmul["all_indices_exact"]:
        classification = (
            "router_matmul_bfloat16_semantics"
            if input_metrics["quantized_exact_fraction"] == 1.0
            else "pre_router_drift_does_not_change_official_bfloat16_routing"
        )
    else:
        classification = "pre_router_state_drift_changes_official_bfloat16_routing"

    return {
        "classification": classification,
        "pre_router_state": input_metrics,
        "routed_logits": {
            "official_vs_c_float": _tensor_metrics(
                official_logits[..., :routed_n],
                c_logits,
                compare_dtype=torch.bfloat16,
            ),
            "official_vs_official_matmul_c_input": _tensor_metrics(
                official_logits,
                official_matmul_c_input,
                compare_dtype=torch.bfloat16,
            ),
        },
        "variants": {
            value["name"]: value
            for value in (
                emitted,
                c_float,
                c_rounded,
                c_input_matmul,
                float_official,
            )
        },
        "decision_margin": {
            "official": _decision_margins(official_choice, top_k),
            "c_float32": _decision_margins(c_float_choice, top_k),
            "c_rounded_bfloat16": _decision_margins(c_rounded_choice, top_k),
            "official_bfloat16_matmul_c_input": _decision_margins(
                c_input_choice, top_k
            ),
            "float32_matmul_official_input": _decision_margins(
                float_official_choice, top_k
            ),
        },
        "trace_contract": {
            "official_logits": int(official_logits.shape[-1]),
            "c_logits": int(c_logits.shape[-1]),
            "c_scope": "routed_only",
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
            analyses[str(layer)] = diagnose_layer(
                module,
                official_input,
                official_logits,
                indices,
                weights,
                c_result,
            )
        result = {
            "format": "inkling-router-boundary-diagnosis",
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
