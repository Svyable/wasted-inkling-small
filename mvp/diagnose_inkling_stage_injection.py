#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inject or requantize Inkling stages to localize BF16 drift."""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "inkling" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from analyze_inkling_router_mismatch import _tensor_metrics
from diagnose_inkling_pre_router import (
    PreRouterDiagnosisError,
    _capture_c_pre_router,
    _capture_official_pre_router,
    _official_route_on_state,
    _route_summary,
)
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


INJECTION_CASES: dict[str, tuple[str, ...]] = {
    "input_norm": ("input_norm",),
    "projection_outputs": (
        "q_proj",
        "k_proj",
        "v_proj",
        "relative_proj_input",
    ),
    "attention_core_inputs": (
        "q_proj",
        "k_sconv",
        "v_sconv",
        "relative_proj_input",
    ),
    "attention_out": ("attention_out",),
    "attention_branch": ("attention_branch",),
    "post_attention_residual": ("post_attention_residual",),
    "post_attention_norm": ("post_attention_norm",),
}

QUANTIZATION_CASES: dict[str, tuple[str, ...]] = {
    **INJECTION_CASES,
    "branch_and_residual": (
        "attention_branch",
        "post_attention_residual",
    ),
    "attention_out_branch_and_residual": (
        "attention_out",
        "attention_branch",
        "post_attention_residual",
    ),
}


def _validate_injection(
    official: dict[str, torch.Tensor],
    injected: dict[str, torch.Tensor],
    points: tuple[str, ...],
    *,
    dtype: torch.dtype,
) -> dict[str, Any]:
    metrics = {}
    for point in points:
        metric = _tensor_metrics(
            official[point],
            injected[point],
            compare_dtype=dtype,
        )
        if metric["quantized_exact_fraction"] != 1.0:
            raise PreRouterDiagnosisError(
                f"official injection did not replace {point} exactly"
            )
        metrics[point] = metric
    return metrics


def _case_result(
    name: str,
    official: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    c_indices: torch.Tensor,
    c_weights: torch.Tensor,
    gate: Any,
    official_indices: torch.Tensor,
    official_weights: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> dict[str, Any]:
    routed_indices, routed_weights = _official_route_on_state(
        gate,
        candidate["post_attention_norm"],
        dtype=dtype,
    )
    return {
        "post_attention_norm": _tensor_metrics(
            official["post_attention_norm"],
            candidate["post_attention_norm"],
            compare_dtype=dtype,
        ),
        "c_emitted": _route_summary(
            f"{name}:c_emitted",
            c_indices,
            c_weights,
            official_indices,
            official_weights,
        ),
        "official_bfloat16_router": _route_summary(
            f"{name}:official_bfloat16_router",
            routed_indices,
            routed_weights,
            official_indices,
            official_weights,
        ),
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
        c_config = json.loads(
            Path(args.c_config).read_text(encoding="utf-8")
        )
        input_values = json.loads(
            Path(args.inputs).read_text(encoding="utf-8")
        )
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [
            int(value) for value in args.layers.split(",") if value
        ]
        if not layers:
            raise PreRouterDiagnosisError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        library = build_library(
            REPO / "inkling" / "src", Path(args.out).with_suffix(".so")
        )
        lib = ctypes.CDLL(str(library))
        configure_library(lib)

        layer_results = {}
        for layer in layers:
            official, module, official_indices, official_weights = (
                _capture_official_pre_router(
                    fixture,
                    config,
                    layer,
                    inputs,
                    device=device,
                    dtype=dtype,
                )
            )
            gate = module.mlp.gate
            injections = {}
            for case_name, points in INJECTION_CASES.items():
                replacements = {
                    point: official[point] for point in points
                }
                injected, c_indices, c_weights = _capture_c_pre_router(
                    lib,
                    fixture,
                    c_config,
                    layer,
                    input_values,
                    replace_stages=replacements,
                )
                result = _case_result(
                    case_name,
                    official,
                    injected,
                    c_indices,
                    c_weights,
                    gate,
                    official_indices,
                    official_weights,
                    dtype=dtype,
                )
                result["points"] = list(points)
                result["injected_stage_metrics"] = _validate_injection(
                    official,
                    injected,
                    points,
                    dtype=dtype,
                )
                injections[case_name] = result

            quantizations = {}
            for case_name, points in QUANTIZATION_CASES.items():
                quantized, c_indices, c_weights = _capture_c_pre_router(
                    lib,
                    fixture,
                    c_config,
                    layer,
                    input_values,
                    quantize_points=points,
                )
                result = _case_result(
                    case_name,
                    official,
                    quantized,
                    c_indices,
                    c_weights,
                    gate,
                    official_indices,
                    official_weights,
                    dtype=dtype,
                )
                result["points"] = list(points)
                quantizations[case_name] = result

            layer_results[str(layer)] = {
                "cases": injections,
                "quantization_cases": quantizations,
            }

        result = {
            "format": "inkling-stage-counterfactual-diagnosis",
            "version": 2,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "injection_cases": {
                name: list(points)
                for name, points in INJECTION_CASES.items()
            },
            "quantization_cases": {
                name: list(points)
                for name, points in QUANTIZATION_CASES.items()
            },
            "layers": layer_results,
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
        PreRouterDiagnosisError,
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
