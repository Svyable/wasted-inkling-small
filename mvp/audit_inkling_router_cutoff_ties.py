#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit whether Inkling routing differences are confined to BF16 cutoff ties."""
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

from analyze_inkling_router_mismatch import capture_official_router
from discover_inkling_c_router_experts import discover_c_layer_router
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


class RouterTieAuditError(RuntimeError):
    """The router cutoff tie audit cannot be completed without guessing."""


def _sorted_ids(values: torch.Tensor) -> list[int]:
    return sorted(int(value) for value in values.tolist())


def expected_candidate_weights(
    logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    n_routed: int,
    n_shared: int,
    route_scale: float,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """Apply official BF16 weight normalization to a candidate expert set."""
    if logits.ndim != 1:
        raise RouterTieAuditError("router logit row must be one-dimensional")
    if logits.numel() != n_routed + n_shared:
        raise RouterTieAuditError(
            f"router logit row has {logits.numel()} values; expected "
            f"{n_routed + n_shared}"
        )
    if candidate_indices.ndim != 1 or not candidate_indices.numel():
        raise RouterTieAuditError("candidate indices must be a nonempty row")
    if torch.any(candidate_indices < 0) or torch.any(candidate_indices >= n_routed):
        raise RouterTieAuditError("candidate expert ID is outside routed range")
    if torch.unique(candidate_indices).numel() != candidate_indices.numel():
        raise RouterTieAuditError("candidate expert IDs are not unique")

    routed_logits = logits[:n_routed]
    shared_logits = logits[n_routed:]
    selected_logits = torch.cat(
        [routed_logits.gather(0, candidate_indices), shared_logits], dim=0
    )
    log_probs = F.logsigmoid(selected_logits)
    weights = torch.exp(log_probs - torch.logsumexp(log_probs, dim=0))
    weights = weights * route_scale * global_scale.to(logits.dtype)
    return weights[: candidate_indices.numel()].contiguous()


def cutoff_tie_row(
    choice: torch.Tensor,
    selected: torch.Tensor,
    candidate: torch.Tensor,
    top_k: int,
) -> dict[str, Any]:
    """Classify one candidate set against the exact top-k cutoff equivalence."""
    if choice.ndim != 1:
        raise RouterTieAuditError("choice row must be one-dimensional")
    if selected.numel() != top_k or candidate.numel() != top_k:
        raise RouterTieAuditError("selected and candidate rows must hold top_k IDs")
    values, _ = torch.topk(choice, top_k, dim=-1, sorted=False)
    cutoff = values.min()
    above = set(
        int(value)
        for value in torch.nonzero(choice > cutoff, as_tuple=False).flatten().tolist()
    )
    tied = set(
        int(value)
        for value in torch.nonzero(choice == cutoff, as_tuple=False).flatten().tolist()
    )
    slots_from_tie = top_k - len(above)
    official = set(int(value) for value in selected.tolist())
    actual = set(int(value) for value in candidate.tolist())
    official_valid = (
        above.issubset(official)
        and official.issubset(above | tied)
        and len(official & tied) == slots_from_tie
    )
    candidate_valid = (
        above.issubset(actual)
        and actual.issubset(above | tied)
        and len(actual & tied) == slots_from_tie
    )
    return {
        "cutoff": float(cutoff.float().item()),
        "choice_dtype": str(choice.dtype).removeprefix("torch."),
        "above_ids": sorted(above),
        "cutoff_tie_ids": sorted(tied),
        "slots_from_cutoff_tie": slots_from_tie,
        "ambiguous_cutoff": len(tied) > slots_from_tie,
        "official_ids": sorted(official),
        "candidate_ids": sorted(actual),
        "official_valid_under_cutoff": official_valid,
        "candidate_valid_under_official_cutoff": candidate_valid,
        "exact_ids": official == actual,
        "differing_ids": sorted(official.symmetric_difference(actual)),
    }


def audit_layer(
    module: Any,
    official_logits: torch.Tensor,
    official_indices: torch.Tensor,
    c_result: dict[str, Any],
    *,
    repeat_topk: int = 16,
    weight_atol: float = 1e-3,
) -> dict[str, Any]:
    gate = module.mlp.gate
    routed_n = int(gate.num_experts)
    shared_n = int(gate.n_shared_experts)
    top_k = int(gate.top_k)
    if official_logits.shape[-1] != routed_n + shared_n:
        raise RouterTieAuditError(
            f"official router logits hold {official_logits.shape[-1]} values; "
            f"expected {routed_n + shared_n}"
        )
    routed_logits = official_logits[..., :routed_n]
    choice = routed_logits.sigmoid() + gate.e_score_correction_bias.detach().to(
        routed_logits.dtype
    )
    c_indices = torch.tensor(c_result["topk_indices"], dtype=torch.int64)
    c_weights = torch.tensor(c_result["topk_weights"], dtype=torch.float32)
    if c_indices.shape != official_indices.cpu().shape:
        raise RouterTieAuditError(
            f"C routed index shape {tuple(c_indices.shape)} differs from official "
            f"{tuple(official_indices.shape)}"
        )
    if c_weights.shape != c_indices.shape:
        raise RouterTieAuditError(
            f"C routed weight shape {tuple(c_weights.shape)} differs from IDs "
            f"{tuple(c_indices.shape)}"
        )

    repeats = [
        torch.topk(choice, top_k, dim=-1, sorted=False).indices.cpu()
        for _ in range(repeat_topk)
    ]
    repeat_stable = all(torch.equal(repeats[0], value) for value in repeats[1:])
    if not torch.equal(repeats[0], official_indices.cpu()):
        raise RouterTieAuditError(
            "captured official IDs differ from topk recomputation on captured logits"
        )

    rows = []
    exact_rows = 0
    tie_equivalent_rows = 0
    ambiguous_rows = 0
    mismatches_outside_tie = []
    weight_failures = []
    max_abs_candidate_weight = 0.0
    for position in range(choice.shape[0]):
        row = cutoff_tie_row(
            choice[position],
            official_indices[position].cpu(),
            c_indices[position],
            top_k,
        )
        row["position"] = position
        expected_weights = expected_candidate_weights(
            official_logits[position],
            c_indices[position].to(official_logits.device),
            n_routed=routed_n,
            n_shared=shared_n,
            route_scale=float(gate.route_scale),
            global_scale=gate.global_scale.detach(),
        ).float().cpu()
        weight_delta = (expected_weights - c_weights[position]).abs()
        row_weight_max = (
            float(weight_delta.max()) if weight_delta.numel() else 0.0
        )
        max_abs_candidate_weight = max(
            max_abs_candidate_weight, row_weight_max
        )
        row["candidate_weight_max_abs_vs_official_logits"] = row_weight_max
        row["candidate_weights_within_atol"] = row_weight_max <= weight_atol
        row["weight_atol"] = weight_atol
        row["expected_candidate_weights"] = [
            float(value) for value in expected_weights.tolist()
        ]
        row["candidate_weights"] = [
            float(value) for value in c_weights[position].tolist()
        ]

        exact_rows += int(row["exact_ids"])
        tie_equivalent_rows += int(row["candidate_valid_under_official_cutoff"])
        ambiguous_rows += int(row["ambiguous_cutoff"])
        if not row["exact_ids"] and not row["candidate_valid_under_official_cutoff"]:
            mismatches_outside_tie.append(position)
        if not row["candidate_weights_within_atol"]:
            weight_failures.append(position)
        rows.append(row)

    return {
        "top_k": top_k,
        "rows": rows,
        "exact_rows": exact_rows,
        "tie_equivalent_rows": tie_equivalent_rows,
        "ambiguous_cutoff_rows": ambiguous_rows,
        "mismatch_positions_outside_official_cutoff_tie": mismatches_outside_tie,
        "all_mismatches_cutoff_tie_equivalent": not mismatches_outside_tie,
        "weight_atol": weight_atol,
        "max_abs_candidate_weight_vs_official_logits": max_abs_candidate_weight,
        "candidate_weight_failure_positions": weight_failures,
        "all_candidate_weights_within_atol": not weight_failures,
        "passed": not mismatches_outside_tie and not weight_failures,
        "topk_repeat_count": repeat_topk,
        "topk_repeat_stable": repeat_stable,
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
    ap.add_argument("--weight-atol", type=float, default=1e-3)
    args = ap.parse_args(argv)
    try:
        if args.weight_atol < 0.0:
            raise RouterTieAuditError("--weight-atol must be nonnegative")
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        c_config = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise RouterTieAuditError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        library = build_library(
            REPO / "inkling" / "src", Path(args.out).with_suffix(".so")
        )
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        analyses = {}
        for layer in layers:
            module, _official_input, official_logits, indices, _weights = (
                capture_official_router(
                    fixture,
                    config,
                    layer,
                    inputs,
                    device=device,
                    dtype=dtype,
                )
            )
            c_result = discover_c_layer_router(
                lib, fixture, c_config, layer, input_values
            )
            analyses[str(layer)] = audit_layer(
                module,
                official_logits,
                indices,
                c_result,
                weight_atol=args.weight_atol,
            )
        result = {
            "format": "inkling-router-cutoff-tie-audit",
            "version": 2,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "weight_atol": args.weight_atol,
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cpu_capability": (
                torch.backends.cpu.get_cpu_capability()
                if hasattr(torch.backends.cpu, "get_cpu_capability")
                else None
            ),
            "layers": analyses,
            "passed": all(value["passed"] for value in analyses.values()),
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
        RouterTieAuditError,
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
