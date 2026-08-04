#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Classify the post-attention residual dtype boundary.

The portable attention candidate is BF16-exact through the FP32 attention short
convolution trace, but the next residual is not. This diagnostic distinguishes
three materially different explanations:

* the residual addition needs BF16 operand/result boundaries;
* the branch is only BF16-visible exact while its FP32 values still differ;
* the trace alignment or input reconstruction is wrong.

Both the actual C residual and the official residual are reconstructed before
counterfactuals are interpreted. Production source remains unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

import torch

from diagnose_inkling_native_bf16_backend import (
    NativeBfloat16BackendError,
    NativeMatrixProvider,
    capture_c_with_backend,
)
from diagnose_inkling_portable_bf16_attention import (
    PortableBfloat16AttentionError,
    build_portable_attention_library,
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


class ResidualBoundaryError(RuntimeError):
    """The residual boundary cannot be classified without exact anchors."""


def bf16(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(torch.bfloat16).float().cpu().contiguous()


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu().contiguous()
    candidate = candidate.detach().float().cpu().contiguous()
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ResidualBoundaryError(
            f"shape mismatch {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    if not reference.numel():
        raise ResidualBoundaryError("cannot compare empty residual tensors")
    delta = (reference - candidate).abs()
    raw_exact = reference.eq(candidate)
    reference_bf16 = bf16(reference)
    candidate_bf16 = bf16(candidate)
    bf16_exact = reference_bf16.eq(candidate_bf16)
    return {
        "count": reference.numel(),
        "raw_exact": int(raw_exact.sum().item()),
        "raw_exact_fraction": float(raw_exact.float().mean().item()),
        "raw_max_abs": float(delta.max().item()),
        "raw_mean_abs": float(delta.mean().item()),
        "bfloat16_exact": int(bf16_exact.sum().item()),
        "bfloat16_exact_fraction": float(bf16_exact.float().mean().item()),
        "bfloat16_max_abs": float(
            (reference_bf16 - candidate_bf16).abs().max().item()
        ),
        "bfloat16_mean_abs": float(
            (reference_bf16 - candidate_bf16).abs().mean().item()
        ),
    }


def residual_variants(
    inputs: torch.Tensor,
    branch: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return distinct operand/result policies without collapsing their dtype."""
    input_bf16 = inputs.detach().to(torch.bfloat16)
    branch_float = branch.detach().float()
    branch_bf16 = branch.detach().to(torch.bfloat16)
    return {
        "float32_add_raw_branch": input_bf16.float() + branch_float,
        "float32_add_bfloat16_branch": input_bf16.float() + branch_bf16.float(),
        "bfloat16_result_raw_branch": bf16(input_bf16.float() + branch_float),
        "bfloat16_result_bfloat16_branch": bf16(
            input_bf16.float() + branch_bf16.float()
        ),
        "native_bfloat16_add": (input_bf16 + branch_bf16).float().cpu(),
    }


def classify_boundary(
    branch: dict[str, Any],
    official_anchor: dict[str, Any],
    candidate_variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if official_anchor["raw_exact_fraction"] != 1.0:
        return {
            "classification": "official_residual_reconstruction_failed",
            "sufficient_candidate_policy": None,
        }
    exact = [
        name
        for name, value in candidate_variants.items()
        if value["raw_exact_fraction"] == 1.0
    ]
    if exact:
        return {
            "classification": "candidate_residual_policy_is_raw_exact",
            "sufficient_candidate_policy": exact[0],
        }
    bf16_exact = [
        name
        for name, value in candidate_variants.items()
        if value["bfloat16_exact_fraction"] == 1.0
    ]
    if bf16_exact:
        return {
            "classification": "candidate_residual_policy_is_bfloat16_exact",
            "sufficient_candidate_policy": bf16_exact[0],
        }
    if (
        branch["bfloat16_exact_fraction"] == 1.0
        and branch["raw_exact_fraction"] < 1.0
    ):
        return {
            "classification": "fp32_attention_branch_must_match_before_residual",
            "sufficient_candidate_policy": None,
        }
    return {
        "classification": "residual_boundary_remains_unexplained",
        "sufficient_candidate_policy": None,
    }


def analyze_layer(
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
    provider = NativeMatrixProvider(
        layer,
        module,
        official_norm=official["input_norm"],
    )
    candidate, _, _ = capture_c_with_backend(
        lib,
        fixture,
        c_config,
        layer,
        input_values,
        provider,
    )

    source_inputs = inputs.detach().to(dtype=dtype).float().cpu()
    candidate_anchor_tensor = source_inputs + candidate["attention_branch"].float()
    official_anchor_tensor = source_inputs + official["attention_branch"].float()
    candidate_anchor = metrics(
        candidate["post_attention_residual"],
        candidate_anchor_tensor,
    )
    official_anchor = metrics(
        official["post_attention_residual"],
        official_anchor_tensor,
    )
    if candidate_anchor["raw_exact_fraction"] != 1.0:
        raise ResidualBoundaryError(
            f"layer {layer} C residual reconstruction is not raw-exact"
        )

    branch_metrics = metrics(
        official["attention_branch"],
        candidate["attention_branch"],
    )
    official_variants = {
        name: metrics(official["post_attention_residual"], value)
        for name, value in residual_variants(
            source_inputs,
            official["attention_branch"],
        ).items()
    }
    candidate_variants = {
        name: metrics(official["post_attention_residual"], value)
        for name, value in residual_variants(
            source_inputs,
            candidate["attention_branch"],
        ).items()
    }
    decision = classify_boundary(
        branch_metrics,
        official_anchor,
        candidate_variants,
    )
    return {
        "dtypes": {
            "input": str(inputs.to(dtype=dtype).dtype).removeprefix("torch."),
            "official_attention_branch": str(
                official["attention_branch"].dtype
            ).removeprefix("torch."),
            "official_post_attention_residual": str(
                official["post_attention_residual"].dtype
            ).removeprefix("torch."),
            "candidate_attention_branch": str(
                candidate["attention_branch"].dtype
            ).removeprefix("torch."),
            "candidate_post_attention_residual": str(
                candidate["post_attention_residual"].dtype
            ).removeprefix("torch."),
        },
        "anchors": {
            "candidate_c_reconstruction": candidate_anchor,
            "official_float32_reconstruction": official_anchor,
        },
        "attention_branch": branch_metrics,
        "official_branch_variants": official_variants,
        "candidate_branch_variants": candidate_variants,
        "decision": decision,
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
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
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
            raise ResidualBoundaryError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise ResidualBoundaryError("the C residual probe requires CPU")
        library, source = build_portable_attention_library(
            Path(args.out).with_suffix(".so")
        )
        if not source["production_source_unchanged"]:
            raise ResidualBoundaryError("production source changed")
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        analyses = {
            str(layer): analyze_layer(
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
            "format": "inkling-bfloat16-residual-boundary-probe",
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
        NativeBfloat16BackendError,
        PortableBfloat16AttentionError,
        ResidualBoundaryError,
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
