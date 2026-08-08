#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Separate router projection shape sensitivity from normalization policy.

#46 shows the first large layer-2 stateful defect at fixed-ID router weights for
positions 3 and 4. Before changing the normalization policy, this evidence-only
probe compares the official eight-row BF16 router projection against the exact
one-row BF16 ``F.linear`` kernel used by the C matrix backend. It then evaluates
all existing portable normalization policies on both logit sources.

Expert values are never executed. Production source and public APIs remain
unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from diagnose_inkling_bf16_router_weight_lattice import (
    F_FINAL_BF16,
    F_GLOBAL_SCALE_BF16,
    F_LOGP_BF16,
    F_LSE_BF16,
    F_NORMALIZED_BF16,
    F_OUTPUT_DELTA_BF16,
    F_ROUTE_SCALE_BF16,
    FixedWeightHelper,
    RouterWeightLatticeError,
    build_helper,
    enumerate_policies,
    metrics,
    official_manual_weights,
    policy_description,
    selected_logits,
    split_metrics,
)
from diagnose_inkling_native_bf16_backend import native_bfloat16_linear
from diagnose_inkling_pre_router import PreRouterDiagnosisError, _capture_official_pre_router
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import CanonicalRouteError, load_canonical_routes
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import DTYPES, FixtureReferenceError, build_layer_from_fixture, verify_config_binding
from inkling_release_config import build_transformers_text_config
from run_inkling_fixture_reference_canonical import CanonicalOfficialGate, CanonicalOfficialRouteError


class RouterPrefillLatticeError(RuntimeError):
    """Router prefill classification requires exact source-bound anchors."""


LEGACY_FLAGS = (
    F_LOGP_BF16
    | F_LSE_BF16
    | F_OUTPUT_DELTA_BF16
    | F_NORMALIZED_BF16
    | F_ROUTE_SCALE_BF16
    | F_GLOBAL_SCALE_BF16
    | F_FINAL_BF16
)


def _policy_key(record: dict[str, Any]) -> tuple[int, int]:
    return (0 if record["family"] == "ratio" else 1, int(record["flags"]))


def _intersection(exact_by_position: list[list[dict[str, Any]]]) -> list[tuple[int, int]]:
    if not exact_by_position:
        return []
    common = {_policy_key(record) for record in exact_by_position[0]}
    for records in exact_by_position[1:]:
        common.intersection_update(_policy_key(record) for record in records)
    return sorted(common)


def classify(
    positions: list[dict[str, Any]],
    rowwise_intersection: list[tuple[int, int]],
    batched_intersection: list[tuple[int, int]],
) -> dict[str, Any]:
    first_selected_logit_mismatch = next(
        (
            item["position"]
            for item in positions
            if item["selected_logits"]["raw_exact_fraction"] != 1.0
        ),
        None,
    )
    first_legacy_rowwise_weight_mismatch = next(
        (
            item["position"]
            for item in positions
            if item["legacy_rowwise_weights"]["all"]["raw_exact_fraction"] != 1.0
        ),
        None,
    )
    legacy_key = (1, LEGACY_FLAGS)
    if first_selected_logit_mismatch is not None:
        classification = "router_prefill_projection_shape_mismatch"
        preferred = None
    elif first_legacy_rowwise_weight_mismatch is None:
        classification = "legacy_router_weight_policy_generalizes"
        preferred = policy_description(*legacy_key)
    elif rowwise_intersection:
        classification = "router_normalization_policy_generalized"
        family, flags = rowwise_intersection[0]
        preferred = policy_description(family, flags)
    else:
        classification = "router_normalization_requires_unmodeled_arithmetic"
        preferred = None
    return {
        "classification": classification,
        "first_selected_logit_mismatch_position": first_selected_logit_mismatch,
        "first_legacy_rowwise_weight_mismatch_position": first_legacy_rowwise_weight_mismatch,
        "legacy_policy": policy_description(*legacy_key),
        "legacy_exact_on_all_rowwise_positions": legacy_key in rowwise_intersection,
        "legacy_exact_on_all_batched_positions": legacy_key in batched_intersection,
        "preferred_policy": preferred,
        "rowwise_exact_policy_intersection_count": len(rowwise_intersection),
        "batched_exact_policy_intersection_count": len(batched_intersection),
    }


def analyze_layer(
    helper: FixedWeightHelper,
    fixture: Any,
    config: Any,
    layer: int,
    inputs: torch.Tensor,
    rows: list[Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    pre, _, _, _ = _capture_official_pre_router(
        fixture, config, layer, inputs, device=device, dtype=dtype
    )
    module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    normalized = pre["post_attention_norm"].to(device=device, dtype=torch.bfloat16)
    weight = module.mlp.gate.weight.detach().to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        batched_logits = F.linear(normalized, weight).detach().float().cpu()
        rowwise_logits = torch.stack(
            [
                native_bfloat16_linear(
                    weight.detach().cpu(),
                    normalized[position].detach().float().cpu(),
                )
                for position in range(normalized.shape[0])
            ]
        )

    gate = CanonicalOfficialGate(module.mlp.gate, rows)
    _, routed_weights, routed_ids, shared_gammas = gate(normalized)
    if not gate.applied:
        raise RouterPrefillLatticeError("canonical official gate was not applied")

    n_routed = int(module.mlp.gate.num_experts)
    top_k = int(module.mlp.gate.top_k)
    route_scale = float(module.mlp.gate.route_scale)
    global_tensor = module.mlp.gate.global_scale.detach().cpu()
    global_scale = float(global_tensor.float().reshape(-1)[0])

    positions: list[dict[str, Any]] = []
    rowwise_exact: list[list[dict[str, Any]]] = []
    batched_exact: list[list[dict[str, Any]]] = []
    for position, row in enumerate(rows):
        canonical_ids = torch.tensor(row.indices, dtype=torch.int64)
        official = torch.cat(
            (
                routed_weights[position].detach().cpu(),
                shared_gammas[position].detach().cpu(),
            )
        )
        batched_selected = selected_logits(
            batched_logits[position], canonical_ids, n_routed
        )
        rowwise_selected = selected_logits(
            rowwise_logits[position], canonical_ids, n_routed
        )

        manual = official_manual_weights(
            batched_selected, route_scale, global_tensor
        )
        anchor = metrics(official, manual["final"])
        if anchor["raw_exact_fraction"] != 1.0:
            raise RouterPrefillLatticeError(
                f"layer {layer} position {position} official weight anchor failed"
            )
        if routed_ids[position].detach().cpu().tolist() != list(row.indices):
            raise RouterPrefillLatticeError(
                f"layer {layer} position {position} canonical routed IDs changed"
            )

        legacy_batched = helper.evaluate(
            batched_selected, route_scale, global_scale, 1, LEGACY_FLAGS
        )
        legacy_rowwise = helper.evaluate(
            rowwise_selected, route_scale, global_scale, 1, LEGACY_FLAGS
        )
        _top_batched, exact_batched = enumerate_policies(
            official,
            helper,
            batched_selected,
            route_scale,
            global_scale,
            limit=6,
        )
        top_rowwise, exact_rowwise = enumerate_policies(
            official,
            helper,
            rowwise_selected,
            route_scale,
            global_scale,
            limit=8,
        )
        batched_exact.append(exact_batched)
        rowwise_exact.append(exact_rowwise)
        positions.append(
            {
                "position": position,
                "canonical_indices": list(row.indices),
                "all_router_logits": metrics(
                    batched_logits[position], rowwise_logits[position]
                ),
                "selected_logits": metrics(
                    batched_selected, rowwise_selected
                ),
                "official_manual_anchor": anchor,
                "legacy_batched_weights": split_metrics(
                    official, legacy_batched, top_k
                ),
                "legacy_rowwise_weights": split_metrics(
                    official, legacy_rowwise, top_k
                ),
                "rowwise_exact_policy_count": len(exact_rowwise),
                "batched_exact_policy_count": len(exact_batched),
                "top_rowwise_policies": top_rowwise,
            }
        )

    rowwise_intersection = _intersection(rowwise_exact)
    batched_intersection = _intersection(batched_exact)
    decision = classify(positions, rowwise_intersection, batched_intersection)
    return {
        "decision": decision,
        "positions": positions,
        "rowwise_exact_policy_intersection": [
            policy_description(family, flags)
            for family, flags in rowwise_intersection[:32]
        ],
        "batched_exact_policy_intersection": [
            policy_description(family, flags)
            for family, flags in batched_intersection[:32]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
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
        values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(values, list) or len(values) != 8:
            raise RouterPrefillLatticeError(
                "--inputs must contain eight source-bound deterministic rows"
            )
        inputs = torch.tensor(values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise RouterPrefillLatticeError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection, fixture, values, layers
        )
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise RouterPrefillLatticeError("router prefill lattice requires CPU")
        helper = FixedWeightHelper(
            build_helper(Path(args.out).with_suffix(".weights.so"))
        )
        analyses = {
            str(layer): analyze_layer(
                helper,
                fixture,
                config,
                layer,
                inputs,
                routes[layer],
                device=device,
                dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-bfloat16-router-prefill-lattice",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "source_tokens": len(values),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "legacy_policy_flags": LEGACY_FLAGS,
            "production_source_modified": False,
            "layers": analyses,
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        FixtureError,
        FixtureReferenceError,
        PreRouterDiagnosisError,
        CanonicalRouteError,
        CanonicalOfficialRouteError,
        RouterWeightLatticeError,
        RouterPrefillLatticeError,
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
