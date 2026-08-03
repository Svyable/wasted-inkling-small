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
) -> dict[str, Any]:
    gate = module.mlp.gate
    routed_n = int(gate.num_experts)
    top_k = int(gate.top_k)
    if official_logits.shape[-1] < routed_n:
        raise RouterTieAuditError("official router logits omit routed experts")
    routed_logits = official_logits[..., :routed_n]
    choice = routed_logits.sigmoid() + gate.e_score_correction_bias.detach().to(
        routed_logits.dtype
    )
    c_indices = torch.tensor(c_result["topk_indices"], dtype=torch.int64)
    if c_indices.shape != official_indices.cpu().shape:
        raise RouterTieAuditError(
            f"C routed index shape {tuple(c_indices.shape)} differs from official "
            f"{tuple(official_indices.shape)}"
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
    for position in range(choice.shape[0]):
        row = cutoff_tie_row(
            choice[position],
            official_indices[position].cpu(),
            c_indices[position],
            top_k,
        )
        row["position"] = position
        exact_rows += int(row["exact_ids"])
        tie_equivalent_rows += int(row["candidate_valid_under_official_cutoff"])
        ambiguous_rows += int(row["ambiguous_cutoff"])
        if not row["exact_ids"] and not row["candidate_valid_under_official_cutoff"]:
            mismatches_outside_tie.append(position)
        rows.append(row)

    return {
        "top_k": top_k,
        "rows": rows,
        "exact_rows": exact_rows,
        "tie_equivalent_rows": tie_equivalent_rows,
        "ambiguous_cutoff_rows": ambiguous_rows,
        "mismatch_positions_outside_official_cutoff_tie": mismatches_outside_tie,
        "all_mismatches_cutoff_tie_equivalent": not mismatches_outside_tie,
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
    args = ap.parse_args(argv)
    try:
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
            )
        result = {
            "format": "inkling-router-cutoff-tie-audit",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cpu_capability": (
                torch.backends.cpu.get_cpu_capability()
                if hasattr(torch.backends.cpu, "get_cpu_capability")
                else None
            ),
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
