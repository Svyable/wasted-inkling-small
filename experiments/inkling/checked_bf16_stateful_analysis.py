#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Side-effect-free stateful sparse analysis for checked-in BF16 execution.

Historical sparse probes selected their candidate implementation by rebinding a
module-global ``run_c_stateful`` function.  That was convenient while exploring
source transforms, but it is the wrong contract for promoted checked-in source:
import order or another probe in the same interpreter must not decide which C
execution path is measured.

This adapter keeps the already-reviewed official reconstruction, canonical-route
collector, and stage classifier, but requires the candidate runner explicitly.
It mutates no imported module state.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

import diagnose_inkling_bf16_stateful_mlp_boundary as boundary
import diagnose_inkling_portable_bf16_composed_moe as composed
import diagnose_inkling_portable_bf16_stateful_sparse_layer as stateful
from diagnose_inkling_pre_router import _capture_official_pre_router
from inkling_canonical_route_layer_parity import RouteRow
from inkling_fixture_reference import build_layer_from_fixture
from run_inkling_fixture_reference_canonical import run_layer_reference_canonical


class CheckedStatefulAnalysisError(RuntimeError):
    """The checked stateful analysis cannot proceed from explicit contracts."""


CandidateRunner = Callable[
    [Any, Any, dict[str, Any], int, list[list[float]], list[RouteRow], Any, Any],
    dict[str, Any],
]


def analyze_layer(
    lib: Any,
    helper: Any,
    fixture: Any,
    config: Any,
    cfg: dict[str, Any],
    layer: int,
    inputs: torch.Tensor,
    input_values: list[list[float]],
    rows: list[RouteRow],
    *,
    device: torch.device,
    dtype: torch.dtype,
    candidate_runner: CandidateRunner,
) -> dict[str, Any]:
    """Compare one stateful sparse layer using an explicit checked candidate."""
    official_pre, _, _, _ = _capture_official_pre_router(
        fixture, config, layer, inputs, device=device, dtype=dtype
    )
    canonical_outputs = run_layer_reference_canonical(
        fixture,
        config,
        layer,
        input_values,
        rows,
        device=device,
        dtype=dtype,
    )
    official = boundary.official_mlp_sequence(
        fixture,
        config,
        layer,
        official_pre["post_attention_norm"],
        official_pre["post_attention_residual"],
        rows,
        canonical_outputs,
        device=device,
        dtype=dtype,
    )

    provider_module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    provider = composed.CapturingMoeProvider(layer, provider_module)
    collector = stateful.StatefulExactWeightCollector(
        rows,
        provider,
        helper,
        int(provider_module.mlp.gate.num_experts),
        float(provider_module.mlp.gate.route_scale),
        float(
            provider_module.mlp.gate.global_scale.detach()
            .float().cpu().reshape(-1)[0]
        ),
    )
    candidate = candidate_runner(
        lib, fixture, cfg, layer, input_values, rows, provider, collector
    )

    stages: dict[str, list[dict[str, Any]]] = {
        "routed_weight": [],
        "shared_weight": [],
        "moe_out": [],
        "mlp_branch": [],
        "layer_out": [],
    }
    for position in range(len(input_values)):
        stages["routed_weight"].append(
            boundary.metrics(
                official["routed_weight"][position],
                boundary._trace_tensor(collector, layer, position, "routed_weight"),
            )
        )
        stages["shared_weight"].append(
            boundary.metrics(
                official["shared_weight"][position],
                boundary._trace_tensor(collector, layer, position, "shared_weight"),
            )
        )
        stages["moe_out"].append(
            boundary.metrics(
                official["moe_out"][position],
                boundary._trace_tensor(collector, layer, position, "moe_out"),
            )
        )
        stages["mlp_branch"].append(
            boundary.metrics(
                official["mlp_branch"][position],
                boundary._trace_tensor(collector, layer, position, "mlp_branch"),
            )
        )
        stages["layer_out"].append(
            boundary.metrics(
                official["layer_out"][position],
                candidate["outputs"][position],
            )
        )

    preexpert = [
        boundary.metrics(
            official_pre["post_attention_norm"][position],
            candidate["post_attention_norm"][position],
        )
        for position in range(len(input_values))
    ]
    if any(value["bfloat16_exact_fraction"] != 1.0 for value in preexpert):
        raise CheckedStatefulAnalysisError(
            f"layer {layer} pre-expert regression invalidates checked analysis"
        )

    return {
        "decision": boundary.classify_ladder(stages),
        "preexpert": preexpert,
        "stages": stages,
        "official_layer_out_anchor": official["layer_out_anchor"],
        "candidate_routes": candidate["candidate_routes"],
        "canonical_routes": candidate["canonical_routes"],
        "backend_calls": candidate["backend_calls"],
    }
