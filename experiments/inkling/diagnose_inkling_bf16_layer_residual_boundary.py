#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Classify the final sparse-layer residual after the exact composed MoE path.

PR #36 establishes a source-bound position-zero prefix that is BF16-exact
through sparse ``moe_out`` and the causal one-token ``mlp_branch``.  The next
trace, ``layer_out``, is not BF16-exact.  This evidence-only diagnostic keeps
the composed candidate unchanged and asks whether that remaining mismatch is
explained by the final residual operand/result dtype policy.

The probe is fail-closed around three anchors before interpreting variants:

* the C post-attention residual input must be raw-exact to the official input;
* the actual C ``layer_out`` must reconstruct raw-exactly as its current F32 add;
* the official ``layer_out`` must reconstruct raw-exactly as native BF16 add.

Production source and the public matrix-backend ABI remain unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import platform
from pathlib import Path
from typing import Any

import torch

# Importing the runner installs the experiment-only adapters used by #36.
import run_inkling_portable_bf16_composed_moe as composed_runner
import diagnose_inkling_portable_bf16_composed_moe as composed
from diagnose_inkling_pre_router import _capture_official_pre_router
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import load_canonical_routes
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import DTYPES, FixtureReferenceError, verify_config_binding
from inkling_layer_parity import LayerParityError, configure_library
from inkling_release_config import build_transformers_text_config


class LayerResidualBoundaryError(RuntimeError):
    """The final layer residual cannot be classified without exact anchors."""


def bf16(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(torch.bfloat16).float().cpu().contiguous()


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu().contiguous().reshape(-1)
    candidate = candidate.detach().float().cpu().contiguous().reshape(-1)
    if tuple(reference.shape) != tuple(candidate.shape):
        raise LayerResidualBoundaryError(
            f"shape mismatch {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    if not reference.numel():
        raise LayerResidualBoundaryError("cannot compare empty tensors")
    delta = (reference - candidate).abs()
    reference_bf16 = bf16(reference)
    candidate_bf16 = bf16(candidate)
    bf16_delta = (reference_bf16 - candidate_bf16).abs()
    return {
        "count": int(reference.numel()),
        "raw_exact": int(reference.eq(candidate).sum().item()),
        "raw_exact_fraction": float(reference.eq(candidate).float().mean().item()),
        "raw_max_abs": float(delta.max().item()),
        "raw_mean_abs": float(delta.mean().item()),
        "bfloat16_exact": int(reference_bf16.eq(candidate_bf16).sum().item()),
        "bfloat16_exact_fraction": float(
            reference_bf16.eq(candidate_bf16).float().mean().item()
        ),
        "bfloat16_max_abs": float(bf16_delta.max().item()),
        "bfloat16_mean_abs": float(bf16_delta.mean().item()),
    }


def native_bfloat16_add(
    residual: torch.Tensor,
    branch: torch.Tensor,
) -> torch.Tensor:
    return (
        residual.detach().to(torch.bfloat16)
        + branch.detach().to(torch.bfloat16)
    ).float().cpu().contiguous()


def residual_variants(
    residual: torch.Tensor,
    branch: torch.Tensor,
) -> dict[str, torch.Tensor]:
    residual_bf16 = residual.detach().to(torch.bfloat16)
    branch_float = branch.detach().float()
    branch_bf16 = branch.detach().to(torch.bfloat16)
    return {
        "float32_add_raw_branch": residual_bf16.float() + branch_float,
        "float32_add_bfloat16_branch": (
            residual_bf16.float() + branch_bf16.float()
        ),
        "bfloat16_result_raw_branch": bf16(
            residual_bf16.float() + branch_float
        ),
        "bfloat16_result_bfloat16_branch": bf16(
            residual_bf16.float() + branch_bf16.float()
        ),
        "native_bfloat16_add": native_bfloat16_add(
            residual_bf16,
            branch_bf16,
        ),
    }


def classify_boundary(
    residual_input: dict[str, Any],
    candidate_anchor: dict[str, Any],
    official_anchor: dict[str, Any],
    branch: dict[str, Any],
    variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if residual_input["raw_exact_fraction"] != 1.0:
        return {
            "classification": "pre_layer_residual_anchor_failed",
            "sufficient_candidate_policy": None,
            "raw_exact_policies": [],
        }
    if candidate_anchor["raw_exact_fraction"] != 1.0:
        return {
            "classification": "candidate_layer_out_reconstruction_failed",
            "sufficient_candidate_policy": None,
            "raw_exact_policies": [],
        }
    if official_anchor["raw_exact_fraction"] != 1.0:
        return {
            "classification": "official_layer_out_reconstruction_failed",
            "sufficient_candidate_policy": None,
            "raw_exact_policies": [],
        }
    exact = [
        name for name, value in variants.items()
        if value["raw_exact_fraction"] == 1.0
    ]
    if exact:
        preferred = (
            "bfloat16_result_bfloat16_branch"
            if "bfloat16_result_bfloat16_branch" in exact
            else exact[0]
        )
        return {
            "classification": "candidate_layer_residual_policy_is_raw_exact",
            "sufficient_candidate_policy": preferred,
            "raw_exact_policies": exact,
        }
    bf16_exact = [
        name for name, value in variants.items()
        if value["bfloat16_exact_fraction"] == 1.0
    ]
    if bf16_exact:
        return {
            "classification": "candidate_layer_residual_policy_is_bfloat16_exact",
            "sufficient_candidate_policy": bf16_exact[0],
            "raw_exact_policies": [],
        }
    if (
        branch["bfloat16_exact_fraction"] == 1.0
        and branch["raw_exact_fraction"] < 1.0
    ):
        return {
            "classification": "hidden_fp32_mlp_branch_still_matters",
            "sufficient_candidate_policy": None,
            "raw_exact_policies": [],
        }
    return {
        "classification": "layer_residual_boundary_remains_unexplained",
        "sufficient_candidate_policy": None,
        "raw_exact_policies": [],
    }


def torch_fingerprint() -> dict[str, Any]:
    capabilities: dict[str, Any] = {}
    if hasattr(torch.cpu, "get_capabilities"):
        capabilities = dict(torch.cpu.get_capabilities())
    interesting = {
        key: capabilities.get(key)
        for key in (
            "architecture",
            "avx2",
            "avx512_f",
            "avx512_bf16",
            "amx_bf16",
            "amx_tile",
        )
        if key in capabilities
    }
    return {
        "torch_version": torch.__version__,
        "cpu_dispatch": torch.backends.cpu.get_cpu_capability(),
        "cpu_capabilities": interesting,
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _tensor_from_trace(collector: Any, layer: int, point: str) -> torch.Tensor:
    key = f"token.0.layer.{layer}.{point}"
    values = collector.values.get(key)
    if values is None:
        raise LayerResidualBoundaryError(f"candidate trace omitted {key}")
    return torch.tensor(values, dtype=torch.float32)


def analyze_layer(
    lib: Any,
    fixture: Any,
    config: Any,
    cfg: dict[str, Any],
    layer: int,
    inputs: torch.Tensor,
    input_values: list[list[float]],
    route: Any,
    helper: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    pre, module, _, _ = _capture_official_pre_router(
        fixture,
        config,
        layer,
        inputs,
        device=device,
        dtype=dtype,
    )
    official, _ = composed.official_stages(
        module,
        pre["post_attention_norm"][0],
        pre["post_attention_residual"][0],
        route,
    )
    provider = composed.CapturingMoeProvider(layer, module)
    collector = composed_runner.ExactWeightCollector(
        route,
        provider,
        helper,
        n_routed=int(cfg["n_routed_experts"]),
        route_scale=float(cfg["route_scale"]),
        global_scale=float(module.mlp.gate.gate_score_correction_bias.new_tensor(
            1.0
        ).item()) if False else float(cfg.get("router_global_scale", 1.0)),
    )
    # The composed probe obtains the exact global scale from the bound fixture;
    # use the same helper constructor path rather than trusting the fallback above.
    collector = composed_runner.ExactWeightCollector(
        route,
        provider,
        helper,
        n_routed=int(cfg["n_routed_experts"]),
        route_scale=float(cfg["route_scale"]),
        global_scale=float(composed._router_global_scale(module)),
    )
    candidate = composed.run_c_layer(
        lib,
        fixture,
        cfg,
        layer,
        input_values[0],
        route,
        provider,
        collector,
    )

    official_residual = pre["post_attention_residual"][0].detach().float().cpu()
    candidate_residual = _tensor_from_trace(
        collector,
        layer,
        "post_attention_residual",
    )
    official_branch = official["mlp_branch"].detach().float().cpu()
    candidate_branch = candidate["stages"]["mlp_branch"].detach().float().cpu()
    official_out = official["layer_out"].detach().float().cpu()
    candidate_out = candidate["stages"]["layer_out"].detach().float().cpu()

    residual_input = metrics(official_residual, candidate_residual)
    candidate_anchor = metrics(
        candidate_out,
        candidate_residual + candidate_branch,
    )
    official_anchor = metrics(
        official_out,
        native_bfloat16_add(official_residual, official_branch),
    )
    if residual_input["raw_exact_fraction"] != 1.0:
        raise LayerResidualBoundaryError(
            f"layer {layer} residual input is not raw-exact"
        )
    if candidate_anchor["raw_exact_fraction"] != 1.0:
        raise LayerResidualBoundaryError(
            f"layer {layer} C layer_out reconstruction is not raw-exact"
        )
    if official_anchor["raw_exact_fraction"] != 1.0:
        raise LayerResidualBoundaryError(
            f"layer {layer} official BF16 layer_out reconstruction is not raw-exact"
        )

    branch_metrics = metrics(official_branch, candidate_branch)
    candidate_variants = {
        name: metrics(official_out, value)
        for name, value in residual_variants(
            candidate_residual,
            candidate_branch,
        ).items()
    }
    official_variants = {
        name: metrics(official_out, value)
        for name, value in residual_variants(
            official_residual,
            official_branch,
        ).items()
    }
    decision = classify_boundary(
        residual_input,
        candidate_anchor,
        official_anchor,
        branch_metrics,
        candidate_variants,
    )
    return {
        "anchors": {
            "residual_input": residual_input,
            "candidate_current_float32_add": candidate_anchor,
            "official_native_bfloat16_add": official_anchor,
        },
        "mlp_branch": branch_metrics,
        "current_layer_out": metrics(official_out, candidate_out),
        "official_branch_variants": official_variants,
        "candidate_branch_variants": candidate_variants,
        "decision": decision,
        "candidate_route": candidate["candidate_routed_index"],
        "canonical_route": candidate["canonical_routed_index"],
        "backend_calls": candidate["backend_calls"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--c-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    args = parser.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        cfg = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(input_values, list) or len(input_values) != 8:
            raise LayerResidualBoundaryError(
                "--inputs must contain the eight source-bound deterministic rows"
            )
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise LayerResidualBoundaryError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection,
            fixture,
            input_values,
            layers,
        )
        position_zero = {layer: routes[layer][0] for layer in layers}
        for layer, row in position_zero.items():
            declared = set(fixture.experts.get(layer, ()))
            if declared != set(row.indices):
                raise LayerResidualBoundaryError(
                    f"layer {layer} fixture experts {sorted(declared)} differ from "
                    f"position-zero route {sorted(row.indices)}"
                )
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise LayerResidualBoundaryError("the residual probe requires CPU")

        library, source = composed.build_composed_library(
            Path(args.out).with_suffix(".so")
        )
        if not source["production_source_unchanged"]:
            raise LayerResidualBoundaryError("production source changed")
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        helper = composed.FixedWeightHelper(
            composed.build_helper(Path(args.out).with_suffix(".router-helper.so"))
        )
        analyses = {
            str(layer): analyze_layer(
                lib,
                fixture,
                config,
                cfg,
                layer,
                inputs,
                input_values,
                position_zero[layer],
                helper,
                device=device,
                dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-bfloat16-layer-residual-boundary-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "executed_position": 0,
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "backend": torch_fingerprint(),
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
        LayerResidualBoundaryError,
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
