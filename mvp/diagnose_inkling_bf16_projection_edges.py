#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Classify rare official-vs-C BF16 projection threshold disagreements."""
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

from analyze_inkling_router_mismatch import _tensor_metrics
from diagnose_inkling_bf16_layer_probe import build_probe_library
from diagnose_inkling_pre_router import (
    PreRouterDiagnosisError,
    _capture_c_pre_router,
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


class ProjectionEdgeError(RuntimeError):
    """Projection threshold cases cannot be classified without guessing."""


def classify_projection_edges(
    official: torch.Tensor,
    candidate: torch.Tensor,
    float32_rounded: torch.Tensor,
    float32_unrounded: torch.Tensor,
    *,
    dtype: torch.dtype,
    limit: int = 32,
) -> dict[str, Any]:
    if not (
        official.shape
        == candidate.shape
        == float32_rounded.shape
        == float32_unrounded.shape
    ):
        raise ProjectionEdgeError("projection tensors have different shapes")
    reference = official.detach().to(dtype).cpu()
    actual = candidate.detach().to(dtype).cpu()
    f32 = float32_rounded.detach().to(dtype).cpu()
    mismatch = actual != reference
    mismatch_count = int(mismatch.sum())
    resolved = mismatch & (f32 == reference)
    resolved_count = int(resolved.sum())
    introduced = (~mismatch) & (f32 != reference)
    introduced_count = int(introduced.sum())

    examples = []
    coordinates = mismatch.nonzero(as_tuple=False)
    for coordinate in coordinates[:limit]:
        position, row = (int(value) for value in coordinate.tolist())
        examples.append(
            {
                "position": position,
                "row": row,
                "official_bfloat16": float(reference[position, row].float()),
                "candidate_bfloat16": float(actual[position, row].float()),
                "float32_rounded_bfloat16": float(f32[position, row].float()),
                "float32_unrounded": float(
                    float32_unrounded[position, row].float()
                ),
                "float32_resolves": bool(resolved[position, row]),
            }
        )

    if mismatch_count == 0:
        classification = "candidate_already_exact"
    elif resolved_count == mismatch_count and introduced_count == 0:
        classification = "float32_accumulation_rounding_is_sufficient"
    elif resolved_count > 0:
        classification = "float32_accumulation_resolves_only_some_edges"
    else:
        classification = "official_gemm_reduction_order_required"
    return {
        "classification": classification,
        "elements": int(reference.numel()),
        "candidate_mismatches": mismatch_count,
        "float32_resolved_mismatches": resolved_count,
        "float32_resolution_fraction": (
            float(resolved_count / mismatch_count) if mismatch_count else 1.0
        ),
        "float32_introduced_mismatches": introduced_count,
        "candidate_metrics": _tensor_metrics(
            official, candidate, compare_dtype=dtype
        ),
        "float32_rounded_metrics": _tensor_metrics(
            official, float32_rounded, compare_dtype=dtype
        ),
        "examples": examples,
    }


def diagnose_layer(
    lib,
    fixture,
    config,
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
    candidate, _, _ = _capture_c_pre_router(
        lib,
        fixture,
        c_config,
        layer,
        input_values,
    )
    with torch.no_grad():
        float32_unrounded = F.linear(
            official["input_norm"].float(),
            module.self_attn.q_proj.weight.detach().float().cpu(),
        )
    float32_rounded = float32_unrounded.to(dtype)
    return classify_projection_edges(
        official["q_proj"],
        candidate["q_proj"],
        float32_rounded,
        float32_unrounded,
        dtype=dtype,
    )


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
            raise ProjectionEdgeError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        library, source = build_probe_library(
            Path(args.out).with_suffix(".so")
        )
        if not source["production_source_unchanged"]:
            raise ProjectionEdgeError("production layer source changed")
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
            "format": "inkling-bfloat16-projection-edge-diagnosis",
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
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
        PreRouterDiagnosisError,
        ProjectionEdgeError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
