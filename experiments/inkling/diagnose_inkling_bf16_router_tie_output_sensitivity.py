#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure complete-layer sensitivity to valid BF16 router cutoff ties.

PR #40 proves that exact expert IDs at ambiguous BF16 cutoffs are not a
portable value-defined property of the pinned official CPU top-k primitive.
This evidence-only probe asks the next question: how much does the *layer
output* move when the tied subset changes but remains inside the exact same
BF16 cutoff equivalence class?

For source-bound position zero in sparse layer 2 and layer 5, every valid tied
subset is enumerated.  Each route is assigned weights by the unmodified
Transformers 5.14.1 normalization formula and executed through the official
layer.  The current deterministic low-expert-ID route is also executed through
the retained complete temporary C candidate from PR #44 and must match the same
counterfactual official route exactly.

No production source, public ABI, loader, or parity gate is modified.
"""
from __future__ import annotations

import argparse
import ctypes
import itertools
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

import run_inkling_portable_bf16_complete_sparse_layer as complete_runner
import run_inkling_portable_bf16_composed_moe as composed_runner
import diagnose_inkling_portable_bf16_composed_moe as composed
from diagnose_inkling_bf16_layer_residual_boundary import torch_fingerprint
from diagnose_inkling_bf16_router_tie_resolution import (
    cutoff_partition,
    deterministic_id_rule,
    valid_under_partition,
)
from diagnose_inkling_pre_router import _capture_official_pre_router
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import RouteRow, load_canonical_routes
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import DTYPES, FixtureReferenceError, verify_config_binding
from inkling_layer_parity import LayerParityError, configure_library
from inkling_release_config import build_transformers_text_config


def _build_complete_layer_library(out: Path) -> tuple[Path, dict[str, Any]]:
    """Build the retained complete candidate without import-time mutation."""
    library, source = complete_runner.build_complete_sparse_layer_library(out)
    source = dict(source)
    source["layer_residual_policy"] = {
        "residual_operand": "bfloat16_completed",
        "branch_operand": "bfloat16_completed",
        "sum": "float32_of_bfloat16_operands",
        "result": "bfloat16_completed",
    }
    return library, source


class TieOutputSensitivityError(RuntimeError):
    """Tie-output sensitivity cannot be measured without exact source binding."""


STAGES = ("moe_out", "mlp_branch", "layer_out")


def exact_route_row(module: Any, normalized: torch.Tensor, ids: tuple[int, ...]) -> RouteRow:
    """Recompute routed weights exactly from the official gate formula."""
    gate = module.mlp.gate
    normalized = normalized.detach().to(torch.bfloat16).reshape(1, -1)
    full_logits = F.linear(normalized, gate.weight)
    n_routed = int(gate.num_experts)
    n_shared = int(gate.n_shared_experts)
    if full_logits.shape != (1, n_routed + n_shared):
        raise TieOutputSensitivityError(
            f"router logits have unexpected shape {tuple(full_logits.shape)}"
        )
    if len(ids) != int(gate.top_k) or len(set(ids)) != len(ids):
        raise TieOutputSensitivityError("route IDs do not match official top-k geometry")
    index = torch.tensor([ids], device=full_logits.device, dtype=torch.int64)
    routed_logits = full_logits[:, :n_routed]
    shared_logits = full_logits[:, n_routed:]
    selected_logits = torch.cat((routed_logits.gather(1, index), shared_logits), dim=1)
    log_probs = F.logsigmoid(selected_logits)
    normalized_weights = torch.exp(
        log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)
    )
    normalized_weights = (
        normalized_weights * float(gate.route_scale) * gate.global_scale.to(full_logits.dtype)
    )
    routed = normalized_weights[0, : len(ids)].detach().float().cpu()
    return RouteRow(tuple(int(value) for value in ids), tuple(float(value) for value in routed))


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    return composed.metrics(reference, candidate)


def enumerate_routes(partition: dict[str, Any], top_k: int) -> list[tuple[int, ...]]:
    above = sorted(int(value) for value in partition["above"].tolist())
    tied = sorted(int(value) for value in partition["tied"].tolist())
    slots = int(partition["slots"])
    routes = [
        tuple(sorted(above + list(chosen)))
        for chosen in itertools.combinations(tied, slots)
    ]
    if not routes:
        raise TieOutputSensitivityError("cutoff partition produced no valid routes")
    if len(set(routes)) != len(routes):
        raise TieOutputSensitivityError("cutoff route enumeration produced duplicates")
    for route in routes:
        if not valid_under_partition(route, partition, top_k):
            raise TieOutputSensitivityError(
                f"enumerated route {route} falls outside the cutoff class"
            )
    return routes


def classify_impact(alternatives: list[dict[str, Any]]) -> str:
    if not alternatives:
        return "no_cutoff_tie_alternatives"
    outputs = [value["stages"]["layer_out"] for value in alternatives]
    if all(value["raw_exact_fraction"] == 1.0 for value in outputs):
        return "cutoff_tie_alternatives_are_layer_output_exact"
    if all(value["bfloat16_exact_fraction"] == 1.0 for value in outputs):
        return "cutoff_tie_alternatives_are_bfloat16_equivalent"
    return "cutoff_tie_alternatives_change_layer_output"


def summarize(alternatives: list[dict[str, Any]]) -> dict[str, Any]:
    if not alternatives:
        return {
            "alternatives": 0,
            "min_layer_out_bfloat16_exact_fraction": 1.0,
            "max_layer_out_bfloat16_abs": 0.0,
            "max_layer_out_raw_abs": 0.0,
        }
    layer_out = [value["stages"]["layer_out"] for value in alternatives]
    return {
        "alternatives": len(alternatives),
        "min_layer_out_bfloat16_exact_fraction": min(
            value["bfloat16_exact_fraction"] for value in layer_out
        ),
        "max_layer_out_bfloat16_abs": max(value["bfloat16_max_abs"] for value in layer_out),
        "max_layer_out_raw_abs": max(value["raw_max_abs"] for value in layer_out),
    }


def official_route_stages(
    module: Any,
    normalized: torch.Tensor,
    residual: torch.Tensor,
    route: RouteRow,
) -> dict[str, torch.Tensor]:
    stages, _route = composed.official_stages(module, normalized, residual, route)
    return {stage: stages[stage].detach().float().cpu() for stage in STAGES}


def analyze_layer(
    lib: Any,
    helper: Any,
    fixture: Any,
    config: Any,
    cfg: dict[str, Any],
    layer: int,
    inputs: torch.Tensor,
    input_values: list[list[float]],
    committed_route: RouteRow,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    pre, _captured_module, _, _ = _capture_official_pre_router(
        fixture, config, layer, inputs, device=device, dtype=dtype
    )
    module = composed.build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    normalized = pre["post_attention_norm"][0].detach().to(torch.bfloat16)
    residual = pre["post_attention_residual"][0].detach().to(torch.bfloat16)
    gate = module.mlp.gate
    full_logits = F.linear(normalized.reshape(1, -1), gate.weight)
    routed_logits = full_logits[:, : int(gate.num_experts)]
    choice = routed_logits.sigmoid() + gate.e_score_correction_bias.detach().to(
        routed_logits.dtype
    )
    partition = cutoff_partition(choice[0], int(gate.top_k))
    routes = enumerate_routes(partition, int(gate.top_k))
    official_set = set(committed_route.indices)
    if tuple(sorted(committed_route.indices)) not in routes:
        raise TieOutputSensitivityError(
            f"layer {layer} committed route is outside its position-zero cutoff class"
        )

    official_row = exact_route_row(module, normalized, tuple(committed_route.indices))
    if official_row.weights != committed_route.weights:
        max_abs = max(
            abs(a - b) for a, b in zip(official_row.weights, committed_route.weights)
        )
        raise TieOutputSensitivityError(
            f"layer {layer} recomputed official weights differ from archive; max_abs={max_abs}"
        )
    baseline = official_route_stages(module, normalized, residual, official_row)

    evaluations: list[dict[str, Any]] = []
    route_rows: dict[tuple[int, ...], RouteRow] = {}
    for ids in routes:
        row = exact_route_row(module, normalized, ids)
        route_rows[ids] = row
        stages = official_route_stages(module, normalized, residual, row)
        evaluations.append(
            {
                "ids": list(ids),
                "is_official": set(ids) == official_set,
                "weights": list(row.weights),
                "stages": {
                    stage: metrics(baseline[stage], stages[stage]) for stage in STAGES
                },
            }
        )
    alternatives = [value for value in evaluations if not value["is_official"]]

    low_ids = tuple(deterministic_id_rule(partition, highest=False))
    high_ids = tuple(deterministic_id_rule(partition, highest=True))
    if low_ids not in route_rows or high_ids not in route_rows:
        raise TieOutputSensitivityError("deterministic tie route was not enumerated")

    low_row = route_rows[low_ids]
    low_official = official_route_stages(module, normalized, residual, low_row)
    provider = composed.CapturingMoeProvider(layer, module)
    collector = composed_runner.ExactWeightCollector(
        low_row,
        provider,
        helper,
        int(gate.num_experts),
        float(gate.route_scale),
        float(gate.global_scale.detach().float().cpu().reshape(-1)[0]),
    )
    candidate = composed.run_c_layer(
        lib,
        fixture,
        cfg,
        layer,
        input_values[0],
        low_row,
        provider,
        collector,
    )
    candidate_set = set(int(value) for value in candidate["candidate_routed_index"])
    if candidate_set != set(low_ids):
        raise TieOutputSensitivityError(
            f"layer {layer} current C route {sorted(candidate_set)} is not low-ID "
            f"cutoff route {list(low_ids)}"
        )
    c_low = {
        stage: metrics(low_official[stage], candidate["stages"][stage])
        for stage in STAGES
    }
    if c_low["layer_out"]["raw_exact_fraction"] != 1.0:
        raise TieOutputSensitivityError(
            f"layer {layer} C low-ID route does not reproduce official counterfactual"
        )

    return {
        "cutoff": float(partition["cutoff"].float().item()),
        "above_ids": sorted(int(value) for value in partition["above"].tolist()),
        "cutoff_tie_ids": sorted(int(value) for value in partition["tied"].tolist()),
        "slots_from_cutoff_tie": int(partition["slots"]),
        "official_ids": list(committed_route.indices),
        "low_id_route": list(low_ids),
        "high_id_route": list(high_ids),
        "route_count": len(routes),
        "routes": evaluations,
        "decision": {
            "classification": classify_impact(alternatives),
            **summarize(alternatives),
        },
        "c_low_id_validation": {
            "candidate_ids": candidate["candidate_routed_index"],
            "canonical_ids": candidate["canonical_routed_index"],
            "stages": c_low,
            "backend_calls": candidate["backend_calls"],
        },
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
            raise TieOutputSensitivityError("--inputs must contain eight source-bound rows")
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise TieOutputSensitivityError("--layers must be nonempty")
        routes = load_canonical_routes(args.selection, fixture, input_values, layers)
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise TieOutputSensitivityError("the tie-impact probe requires CPU")
        if input_sha256(inputs, dtype) != json.loads(
            Path(args.selection).read_text(encoding="utf-8")
        )["inputs_sha256"]:
            raise TieOutputSensitivityError("input hash differs from route archive")

        library, source = _build_complete_layer_library(
            Path(args.out).with_suffix(".runtime.so")
        )
        if not source["production_source_unchanged"]:
            raise TieOutputSensitivityError("production source changed")
        if source.get("layer_residual_policy", {}).get("result") != "bfloat16_completed":
            raise TieOutputSensitivityError("complete-layer residual policy is not installed")
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        helper = composed.FixedWeightHelper(
            composed.build_helper(Path(args.out).with_suffix(".weights.so"))
        )
        analyses = {
            str(layer): analyze_layer(
                lib,
                helper,
                fixture,
                config,
                cfg,
                layer,
                inputs,
                input_values,
                routes[layer][0],
                device=device,
                dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-bfloat16-router-tie-output-sensitivity",
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
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
        TieOutputSensitivityError,
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