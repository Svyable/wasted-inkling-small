#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compose the proven router denominator with the stateful sparse-layer probe.

The retained stateful harness predates the all-position denominator result and
therefore still injects the earlier #35 router-weight policy.  This runner keeps
that historical harness unchanged, adapts #51's fixed post-reduction policy to
its evidence seam, and reruns the full stateful sparse-MLP ladder.

Production C, the public ABI, loader dispatch, and public inference gates are
unchanged.  Canonical routed IDs remain an evidence adapter and the raw C route
choices remain observable separately.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any, Mapping

import torch

import diagnose_inkling_bf16_stateful_mlp_boundary as stateful_mlp
import diagnose_inkling_portable_bf16_composed_moe as composed
import run_inkling_portable_bf16_complete_sparse_layer as complete_runner
from diagnose_inkling_bf16_router_post_reduction_denom import (
    EXPECTED_FLAGS,
    PostReductionDenomError,
    PostReductionHelper,
    build_helper,
    policy_description,
)
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import (
    CanonicalRouteError,
    load_canonical_routes,
)
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    verify_config_binding,
)
from inkling_layer_parity import LayerParityError, configure_library
from inkling_release_config import build_transformers_text_config
from run_inkling_fixture_reference_canonical import CanonicalOfficialRouteError


class StatefulPostReductionError(RuntimeError):
    """The stateful policy composition cannot proceed without exact contracts."""


def active_router_policy() -> dict[str, Any]:
    """Describe the exact evidence adapter used for every stateful position."""
    return {
        "family": "post_reduction_denominator",
        "policy": policy_description(EXPECTED_FLAGS),
        "denominator_reduction": {
            "logp": "bfloat16_completed",
            "stabilized_delta": "bfloat16_completed",
            "exp_term": "bfloat16_completed",
            "accumulation": "float32",
            "completion": "bfloat16_after_full_reduction",
        },
        "injected_via_trace_adapter": True,
    }


class PostReductionWeightAdapter:
    """Expose #51's helper through the retained stateful collector interface."""

    def __init__(self, helper: PostReductionHelper) -> None:
        self.helper = helper
        self.calls: list[dict[str, Any]] = []

    def evaluate(
        self,
        selected: torch.Tensor,
        route_scale: float,
        global_scale: float,
        legacy_family: int | None = None,
        legacy_flags: int | None = None,
    ) -> torch.Tensor:
        selected = (
            selected.detach().to(torch.bfloat16).float().cpu().contiguous()
            .reshape(-1)
        )
        if not 0 < selected.numel() <= 16:
            raise StatefulPostReductionError(
                f"selected width {selected.numel()} outside adapter bounds"
            )
        stages = self.helper.evaluate(
            selected,
            float(route_scale),
            float(global_scale),
            EXPECTED_FLAGS,
        )
        final = stages.get("final")
        denominator = stages.get("denom")
        if not isinstance(final, torch.Tensor) or final.numel() != selected.numel():
            raise StatefulPostReductionError(
                "post-reduction helper returned an invalid final-weight vector"
            )
        if not isinstance(denominator, torch.Tensor) or denominator.numel() != 1:
            raise StatefulPostReductionError(
                "post-reduction helper returned an invalid denominator"
            )
        if not torch.isfinite(final).all() or not torch.isfinite(denominator).all():
            raise StatefulPostReductionError(
                "post-reduction helper returned a non-finite value"
            )
        self.calls.append(
            {
                "call": len(self.calls),
                "selected_width": int(selected.numel()),
                "denominator": float(denominator.reshape(-1)[0]),
                "policy_flags": EXPECTED_FLAGS,
                "superseded_collector_request": {
                    "family": legacy_family,
                    "flags": legacy_flags,
                },
            }
        )
        return final.detach().float().cpu().contiguous().reshape(-1)


def classify_layers(
    analyses: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify only the tested stateful ladders without generalizing further."""
    if not analyses:
        raise StatefulPostReductionError("stateful analysis is empty")
    exact: list[int] = []
    nonexact: list[int] = []
    first_boundaries: dict[str, dict[str, Any]] = {}
    for layer, value in analyses.items():
        decision = value.get("decision")
        if not isinstance(decision, Mapping):
            raise StatefulPostReductionError(
                f"layer {layer} omitted its stateful decision"
            )
        classification = decision.get("classification")
        if classification == "stateful_sparse_mlp_ladder_exact":
            exact.append(int(layer))
        else:
            nonexact.append(int(layer))
            first_boundaries[str(layer)] = {
                "classification": classification,
                "stage": decision.get("first_boundary_stage"),
                "position": decision.get("first_boundary_position"),
            }
    exact.sort()
    nonexact.sort()
    return {
        "classification": (
            "tested_stateful_sparse_mlp_ladders_exact"
            if not nonexact
            else "tested_stateful_sparse_mlp_ladder_mismatch"
        ),
        "tested_layers": sorted(exact + nonexact),
        "exact_layers": exact,
        "nonexact_layers": nonexact,
        "first_boundaries": first_boundaries,
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
            raise StatefulPostReductionError(
                "--inputs must contain the eight source-bound deterministic rows"
            )
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise StatefulPostReductionError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection, fixture, input_values, layers
        )
        for layer in layers:
            required = {expert for row in routes[layer] for expert in row.indices}
            if set(fixture.experts.get(layer, ())) != required:
                raise StatefulPostReductionError(
                    f"layer {layer} fixture expert union changed"
                )
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise StatefulPostReductionError(
                "the stateful post-reduction probe requires CPU"
            )

        # Do not rely on importing the complete-layer runner to mutate the
        # composed implementation.  Build the candidate through its explicit
        # scoped entry point so the final BF16 residual is part of this exact
        # stateful profile and module import order cannot change the result.
        library, source = complete_runner.build_complete_sparse_layer_library(
            Path(args.out).with_suffix(".runtime.so")
        )
        if not source["production_source_unchanged"]:
            raise StatefulPostReductionError("production source changed")
        source["superseded_router_policy"] = source.pop("router_policy", None)
        source["active_router_policy"] = active_router_policy()
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        helper = PostReductionHelper(
            build_helper(Path(args.out).with_suffix(".post-reduction.so"))
        )
        adapter = PostReductionWeightAdapter(helper)
        analyses = {
            str(layer): stateful_mlp.analyze_layer(
                lib,
                adapter,
                fixture,
                config,
                cfg,
                layer,
                inputs,
                input_values,
                routes[layer],
                device=device,
                dtype=dtype,
            )
            for layer in layers
        }
        expected_calls = len(input_values) * len(layers)
        if len(adapter.calls) != expected_calls:
            raise StatefulPostReductionError(
                f"weight adapter ran {len(adapter.calls)} times; "
                f"expected {expected_calls}"
            )
        result = {
            "format": "inkling-portable-bfloat16-stateful-post-reduction-composition",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "positions": len(input_values),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "execution_profile": composed.collect_execution_profile(),
            "source": source,
            "weight_adapter_calls": adapter.calls,
            "evidence_outcome": classify_layers(analyses),
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
        CanonicalRouteError,
        CanonicalOfficialRouteError,
        PostReductionDenomError,
        StatefulPostReductionError,
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
