#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare official Inkling BF16 primitives with the exact C F32 formulas."""
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
from inkling_layer_parity import (
    LayerParityError,
    build_library,
    configure_library,
)
from inkling_release_config import build_transformers_text_config


class PrimitiveDiagnosisError(RuntimeError):
    """Primitive semantics cannot be compared without guessing."""


def c_rmsnorm_formula(
    values: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reproduce `inkling_layer.c:rmsnorm` including its cast points."""
    x = values.detach().float().cpu()
    w = weight.detach().float().cpu()
    if x.shape[-1] != w.numel():
        raise PrimitiveDiagnosisError(
            f"RMSNorm width mismatch: {x.shape[-1]} vs {w.numel()}"
        )
    ss = (x.double() * x.double()).sum(dim=-1, keepdim=True)
    mean = (ss / x.shape[-1]).float()
    scale = torch.reciprocal(torch.sqrt(mean + torch.tensor(eps, dtype=torch.float32)))
    # C evaluates the two multiplications in float32, left to right.
    return (x * scale) * w


def official_rmsnorm_formula(
    values: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reproduce Transformers 5.14.1 InklingRMSNorm cast order."""
    x = values.detach().to(dtype).cpu()
    w = weight.detach().to(dtype).cpu()
    f32 = x.float()
    variance = f32.pow(2).mean(dim=-1, keepdim=True)
    normalized = f32 * torch.rsqrt(variance + eps)
    return w * normalized.to(dtype)


def c_reduction_official_cast_order(
    values: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Use the C reduction, but the official cast-before-weight ordering."""
    x = values.detach().float().cpu()
    w = weight.detach().to(dtype).cpu()
    ss = (x.double() * x.double()).sum(dim=-1, keepdim=True)
    mean = (ss / x.shape[-1]).float()
    scale = torch.reciprocal(torch.sqrt(mean + torch.tensor(eps, dtype=torch.float32)))
    return w * (x * scale).to(dtype)


def c_linear_sequential(
    values: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Reproduce C matvec's left-to-right double accumulation."""
    x = values.detach().float().cpu()
    w = weight.detach().float().cpu()
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise PrimitiveDiagnosisError(
            f"linear shape mismatch: {tuple(x.shape)} vs {tuple(w.shape)}"
        )
    out = torch.zeros(x.shape[0], w.shape[0], dtype=torch.float64)
    xd = x.double()
    wd = w.double()
    for column in range(x.shape[1]):
        out += xd[:, column : column + 1] * wd[:, column].unsqueeze(0)
    return out.float()


def _sample_rows(rows: int, limit: int = 32) -> list[int]:
    if rows <= limit:
        return list(range(rows))
    # Deterministic coverage of the beginning, interior, and final row.
    return sorted({round(index * (rows - 1) / (limit - 1)) for index in range(limit)})


def _metrics(reference: torch.Tensor, candidate: torch.Tensor, dtype: torch.dtype) -> dict[str, Any]:
    return _tensor_metrics(reference, candidate, compare_dtype=dtype)


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

    eps = float(config.rms_norm_eps)
    norm_weight = module.input_layernorm.weight.detach().cpu()
    c_norm = c_rmsnorm_formula(inputs, norm_weight, eps)
    official_norm = official_rmsnorm_formula(inputs, norm_weight, eps, dtype)
    c_cast_order = c_reduction_official_cast_order(
        inputs, norm_weight, eps, dtype
    )
    rmsnorm = {
        "official_formula_vs_hook": _metrics(
            official["input_norm"], official_norm, dtype
        ),
        "c_formula_vs_trace": _metrics(candidate["input_norm"], c_norm, dtype),
        "c_f32_vs_official": _metrics(official["input_norm"], c_norm, dtype),
        "c_result_rounded_bfloat16_vs_official": _metrics(
            official["input_norm"], c_norm.to(dtype), dtype
        ),
        "c_reduction_official_cast_order_vs_official": _metrics(
            official["input_norm"], c_cast_order, dtype
        ),
    }

    q_weight = module.self_attn.q_proj.weight.detach().cpu()
    row_ids = _sample_rows(int(q_weight.shape[0]))
    sampled_weight = q_weight[row_ids]
    official_q = official["q_proj"][:, row_ids]
    candidate_q = candidate["q_proj"][:, row_ids]
    official_input = official["input_norm"]
    candidate_input = candidate["input_norm"]

    c_on_candidate = c_linear_sequential(candidate_input, sampled_weight)
    c_on_official = c_linear_sequential(official_input, sampled_weight)
    torch_f32_on_official = F.linear(
        official_input.float(), sampled_weight.float()
    )
    official_bf16 = F.linear(
        official_input.to(dtype), sampled_weight.to(dtype)
    )
    linear = {
        "sampled_rows": row_ids,
        "official_bfloat16_formula_vs_hook": _metrics(
            official_q, official_bf16, dtype
        ),
        "c_formula_vs_trace": _metrics(candidate_q, c_on_candidate, dtype),
        "c_double_on_official_input_vs_official": _metrics(
            official_q, c_on_official, dtype
        ),
        "c_double_result_rounded_bfloat16_vs_official": _metrics(
            official_q, c_on_official.to(dtype), dtype
        ),
        "torch_float32_on_official_input_vs_official": _metrics(
            official_q, torch_f32_on_official, dtype
        ),
        "torch_float32_result_rounded_bfloat16_vs_official": _metrics(
            official_q, torch_f32_on_official.to(dtype), dtype
        ),
    }

    def exact_fraction(record: dict[str, Any]) -> float:
        return float(record["quantized_exact_fraction"])

    if exact_fraction(rmsnorm["c_reduction_official_cast_order_vs_official"]) == 1.0:
        rms_class = "cast_before_weight_is_sufficient"
    elif exact_fraction(rmsnorm["official_formula_vs_hook"]) == 1.0:
        rms_class = "reduction_and_cast_order_both_matter"
    else:
        rms_class = "official_rmsnorm_formula_not_reproduced"

    if exact_fraction(linear["c_double_result_rounded_bfloat16_vs_official"]) == 1.0:
        linear_class = "output_rounding_is_sufficient"
    elif exact_fraction(linear["torch_float32_result_rounded_bfloat16_vs_official"]) == 1.0:
        linear_class = "double_accumulation_differs_from_official"
    elif exact_fraction(linear["official_bfloat16_formula_vs_hook"]) == 1.0:
        linear_class = "native_bfloat16_matmul_semantics_required"
    else:
        linear_class = "official_linear_formula_not_reproduced"

    return {
        "rmsnorm_classification": rms_class,
        "linear_classification": linear_class,
        "rmsnorm": rmsnorm,
        "q_projection": linear,
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
            raise PrimitiveDiagnosisError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        library = build_library(
            REPO / "inkling" / "src", Path(args.out).with_suffix(".so")
        )
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
            "format": "inkling-bfloat16-primitive-diagnosis",
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
        PreRouterDiagnosisError,
        PrimitiveDiagnosisError,
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
