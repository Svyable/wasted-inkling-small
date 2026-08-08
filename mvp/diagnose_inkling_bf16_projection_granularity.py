#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare BF16 projection kernels at prefill and token-step granularity.

The official reference projects all fixture tokens in one prefill call, while
WASTE executes a decoder layer one token at a time. BF16 GEMM reduction order
can depend on matrix shape, so a rare disagreement against the batched prefill
kernel is not automatically evidence that the token-step runtime is wrong.

This diagnostic uses the proven temporary RMSNorm/matvec-boundary candidate,
captures the official prefill tensors, and then re-runs the same official BF16
projection modules one row at a time over the exact captured RMSNorm inputs.
Production C source remains unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

import torch

from diagnose_inkling_bf16_layer_probe import (
    Bfloat16LayerProbeError,
    build_probe_library,
)
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


class ProjectionGranularityError(RuntimeError):
    """Projection granularity cannot be classified without exact alignment."""


PROJECTIONS = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "relative_proj_input": "r_proj",
}


def _quantized(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return value.detach().to(dtype=dtype).float().cpu().contiguous()


def _mismatch_positions(a: torch.Tensor, b: torch.Tensor) -> list[int]:
    if a.shape != b.shape:
        raise ProjectionGranularityError(
            f"projection shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    if a.ndim != 2:
        raise ProjectionGranularityError(
            f"projection must be [tokens,width], got {tuple(a.shape)}"
        )
    return [
        index
        for index in range(int(a.shape[0]))
        if not torch.equal(a[index], b[index])
    ]


def classify_projection(
    prefill: torch.Tensor,
    token_step: torch.Tensor,
    candidate: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Classify whether token-step kernel shape explains candidate edges."""
    p = _quantized(prefill, dtype)
    s = _quantized(token_step, dtype)
    c = _quantized(candidate, dtype)
    if p.shape != s.shape or p.shape != c.shape:
        raise ProjectionGranularityError(
            f"unaligned projection shapes: {tuple(p.shape)}, {tuple(s.shape)}, {tuple(c.shape)}"
        )
    candidate_vs_prefill = _mismatch_positions(c, p)
    candidate_vs_token = _mismatch_positions(c, s)
    prefill_vs_token = _mismatch_positions(p, s)
    resolved = sorted(set(candidate_vs_prefill) - set(candidate_vs_token))
    introduced = sorted(set(candidate_vs_token) - set(candidate_vs_prefill))
    total = int(p.numel())
    prefill_exact = int((c == p).sum())
    token_exact = int((c == s).sum())

    if not candidate_vs_prefill and not candidate_vs_token:
        classification = "all_kernels_match_candidate"
    elif candidate_vs_prefill and not candidate_vs_token:
        classification = "token_step_kernel_matches_candidate"
    elif len(candidate_vs_token) < len(candidate_vs_prefill):
        classification = "token_step_kernel_reduces_candidate_mismatch"
    elif len(candidate_vs_token) > len(candidate_vs_prefill):
        classification = "token_step_kernel_worsens_candidate_mismatch"
    else:
        classification = "token_step_kernel_does_not_explain_candidate_mismatch"

    return {
        "classification": classification,
        "shape": list(p.shape),
        "candidate_vs_prefill_mismatch_positions": candidate_vs_prefill,
        "candidate_vs_token_step_mismatch_positions": candidate_vs_token,
        "prefill_vs_token_step_mismatch_positions": prefill_vs_token,
        "resolved_positions": resolved,
        "introduced_positions": introduced,
        "candidate_vs_prefill_exact_fraction": prefill_exact / total,
        "candidate_vs_token_step_exact_fraction": token_exact / total,
        "candidate_vs_prefill_max_abs": float((candidate.float() - prefill.float()).abs().max()),
        "candidate_vs_token_step_max_abs": float((candidate.float() - token_step.float()).abs().max()),
    }


def _rowwise_official_projection(
    module: Any,
    point: str,
    normalized: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    attribute = PROJECTIONS[point]
    projection = getattr(module.self_attn, attribute)
    rows = []
    with torch.no_grad():
        for row in normalized:
            value = projection(
                row.to(device=device, dtype=dtype).reshape(1, -1)
            )
            if value.ndim != 2 or value.shape[0] != 1:
                raise ProjectionGranularityError(
                    f"{point} token-step output has shape {tuple(value.shape)}"
                )
            rows.append(value[0].detach().float().cpu().contiguous())
    return torch.stack(rows)


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
    candidate, _, _ = _capture_c_pre_router(
        lib,
        fixture,
        c_config,
        layer,
        input_values,
    )
    normalized = official["input_norm"]
    projections: dict[str, Any] = {}
    for point in PROJECTIONS:
        token_step = _rowwise_official_projection(
            module,
            point,
            normalized,
            device=device,
            dtype=dtype,
        )
        projections[point] = classify_projection(
            official[point],
            token_step,
            candidate[point],
            dtype=dtype,
        )

    if all(
        value["classification"] in {
            "all_kernels_match_candidate",
            "token_step_kernel_matches_candidate",
        }
        for value in projections.values()
    ):
        classification = "token_step_projection_semantics_are_sufficient"
    elif any(
        value["classification"] in {
            "token_step_kernel_matches_candidate",
            "token_step_kernel_reduces_candidate_mismatch",
        }
        for value in projections.values()
    ):
        classification = "execution_granularity_explains_some_projection_edges"
    else:
        classification = "projection_arithmetic_still_differs_at_token_step_granularity"
    return {
        "classification": classification,
        "projections": projections,
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
            raise ProjectionGranularityError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        library, source = build_probe_library(Path(args.out).with_suffix(".so"))
        if not source["production_source_unchanged"]:
            raise ProjectionGranularityError("production layer source changed")
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
            "format": "inkling-bfloat16-projection-granularity",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "source": source,
            "official_prefill_shape": "[tokens, hidden] projection",
            "official_token_step_shape": "one [1, hidden] projection per token",
            "layers": analyses,
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        Bfloat16LayerProbeError,
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
        PreRouterDiagnosisError,
        ProjectionGranularityError,
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
