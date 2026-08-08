#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Characterize official Inkling BF16 router cutoff-tie selection.

The complete source-bound sparse-layer arithmetic is exact once canonical route
IDs are injected.  Raw C route differences that remain are already known to be
confined to exact BF16 ties at the top-k cutoff.  This evidence-only probe asks
whether the official ``torch.topk`` subset follows a portable expert-ID rule or
is sensitive to storage/index order among exactly tied values.

No C source is modified.  The probe reconstructs the official BF16 choice tensor
from the pinned release, verifies the committed official route archive, and then
compares:

* the current C-style lowest-expert-ID tie rule;
* the opposite highest-expert-ID tie rule;
* ``torch.topk(sorted=True)`` versus the official ``sorted=False`` set;
* several deterministic permutations of expert positions, remapped back to the
  original expert IDs.

Permuted selections are required to remain valid members of the same exact BF16
cutoff equivalence class.  A changed subset is evidence that exact tied IDs are
an implementation/order property, not a model-value-defined choice.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch

from analyze_inkling_router_mismatch import capture_official_router
from discover_inkling_router_experts import input_sha256
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import DTYPES, FixtureReferenceError, verify_config_binding
from inkling_release_config import build_transformers_text_config


class RouterTieResolutionError(RuntimeError):
    """Router tie behavior cannot be characterized without guessing."""


def cutoff_partition(choice: torch.Tensor, top_k: int) -> dict[str, Any]:
    if choice.ndim != 1:
        raise RouterTieResolutionError("choice row must be one-dimensional")
    if top_k < 1 or top_k > choice.numel():
        raise RouterTieResolutionError("top_k is outside the choice row")
    values = torch.topk(choice, top_k, dim=-1, sorted=False).values
    cutoff = values.min()
    above = torch.nonzero(choice > cutoff, as_tuple=False).flatten().to(torch.int64)
    tied = torch.nonzero(choice == cutoff, as_tuple=False).flatten().to(torch.int64)
    slots = top_k - int(above.numel())
    if slots < 1 or slots > int(tied.numel()):
        raise RouterTieResolutionError(
            f"invalid cutoff partition: above={above.numel()} tied={tied.numel()} slots={slots}"
        )
    return {
        "cutoff": cutoff,
        "above": above,
        "tied": tied,
        "slots": slots,
        "ambiguous": int(tied.numel()) > slots,
    }


def _id_set(values: torch.Tensor | list[int] | tuple[int, ...]) -> set[int]:
    if isinstance(values, torch.Tensor):
        return {int(value) for value in values.reshape(-1).tolist()}
    return {int(value) for value in values}


def valid_under_partition(
    selected: torch.Tensor | list[int] | tuple[int, ...],
    partition: dict[str, Any],
    top_k: int,
) -> bool:
    actual = _id_set(selected)
    above = _id_set(partition["above"])
    tied = _id_set(partition["tied"])
    return (
        len(actual) == top_k
        and above.issubset(actual)
        and actual.issubset(above | tied)
        and len(actual & tied) == int(partition["slots"])
    )


def deterministic_id_rule(
    partition: dict[str, Any],
    *,
    highest: bool,
) -> list[int]:
    above = sorted(_id_set(partition["above"]))
    tied = sorted(_id_set(partition["tied"]), reverse=highest)
    chosen = tied[: int(partition["slots"])]
    return sorted(above + chosen)


def expert_permutations(width: int) -> dict[str, torch.Tensor]:
    if width < 2:
        raise RouterTieResolutionError("router width must be at least two")
    base = torch.arange(width, dtype=torch.int64)
    even_odd = torch.cat((base[::2], base[1::2]))
    # 73 is coprime with 256 and, more generally, with any power-of-two router
    # width used by Inkling-Small.  Refuse rather than silently duplicate IDs.
    stride = (base * 73) % width
    permutations = {
        "identity": base,
        "reverse": torch.flip(base, dims=(0,)),
        "rotate_1": torch.roll(base, shifts=1),
        "even_then_odd": even_odd,
        "stride_73": stride,
    }
    for name, value in permutations.items():
        if value.numel() != width or torch.unique(value).numel() != width:
            raise RouterTieResolutionError(
                f"{name} is not a permutation for router width {width}"
            )
    return permutations


def topk_after_permutation(
    choice: torch.Tensor,
    top_k: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    if permutation.ndim != 1 or permutation.numel() != choice.numel():
        raise RouterTieResolutionError("permutation width differs from choice width")
    permuted = choice.index_select(0, permutation.to(choice.device))
    selected_positions = torch.topk(
        permuted, top_k, dim=-1, sorted=False
    ).indices.to(torch.int64).cpu()
    return permutation.cpu().index_select(0, selected_positions)


def classify_layer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ambiguous = [row for row in rows if row["ambiguous_cutoff"]]
    if not ambiguous:
        return {
            "classification": "no_ambiguous_bfloat16_cutoff_rows",
            "ambiguous_rows": 0,
            "original_id_rule": "not_applicable",
            "permutation_sensitive": False,
        }
    low_matches = sum(bool(row["lowest_id_rule_matches"]) for row in ambiguous)
    high_matches = sum(bool(row["highest_id_rule_matches"]) for row in ambiguous)
    if low_matches == len(ambiguous):
        original_rule = "lowest_expert_ids"
    elif high_matches == len(ambiguous):
        original_rule = "highest_expert_ids"
    elif low_matches or high_matches:
        original_rule = "mixed_id_rule"
    else:
        original_rule = "no_simple_id_rule"
    permutation_sensitive = any(
        bool(row["permutation_changed_subset"]) for row in ambiguous
    )
    if permutation_sensitive:
        classification = "official_topk_tie_subset_is_order_sensitive"
    elif original_rule == "lowest_expert_ids":
        classification = "official_cutoff_ties_follow_low_id_rule"
    elif original_rule == "highest_expert_ids":
        classification = "official_cutoff_ties_follow_high_id_rule"
    else:
        classification = "official_topk_tie_subset_is_not_simple_id_rule"
    return {
        "classification": classification,
        "ambiguous_rows": len(ambiguous),
        "original_id_rule": original_rule,
        "permutation_sensitive": permutation_sensitive,
        "lowest_id_matches": low_matches,
        "highest_id_matches": high_matches,
    }


def analyze_row(
    choice: torch.Tensor,
    official_indices: torch.Tensor,
    expected_indices: list[int],
    top_k: int,
    *,
    repeat_topk: int,
) -> dict[str, Any]:
    official = official_indices.detach().to(torch.int64).cpu().contiguous()
    expected = torch.tensor(expected_indices, dtype=torch.int64)
    if not torch.equal(official, expected):
        raise RouterTieResolutionError(
            f"captured official route {official.tolist()} differs from committed "
            f"route {expected.tolist()}"
        )
    partition = cutoff_partition(choice, top_k)
    if not valid_under_partition(official, partition, top_k):
        raise RouterTieResolutionError("official route is outside its own cutoff class")

    repeated = [
        torch.topk(choice, top_k, dim=-1, sorted=False).indices.to(torch.int64).cpu()
        for _ in range(repeat_topk)
    ]
    if not all(torch.equal(official, value) for value in repeated):
        raise RouterTieResolutionError(
            "repeated topk on the unchanged tensor did not reproduce captured IDs"
        )

    official_set = _id_set(official)
    low = deterministic_id_rule(partition, highest=False)
    high = deterministic_id_rule(partition, highest=True)
    sorted_true = torch.topk(choice, top_k, dim=-1, sorted=True).indices.to(torch.int64).cpu()
    permutation_results: dict[str, Any] = {}
    changed = False
    for name, permutation in expert_permutations(int(choice.numel())).items():
        selected = topk_after_permutation(choice, top_k, permutation)
        valid = valid_under_partition(selected, partition, top_k)
        if not valid:
            raise RouterTieResolutionError(
                f"permutation {name} selected IDs outside the exact cutoff class"
            )
        selected_set = _id_set(selected)
        differs = selected_set != official_set
        changed = changed or (name != "identity" and differs)
        permutation_results[name] = {
            "ids": sorted(selected_set),
            "exact_set": not differs,
            "valid_under_cutoff": valid,
        }

    above = sorted(_id_set(partition["above"]))
    tied = sorted(_id_set(partition["tied"]))
    official_tied = sorted(official_set.intersection(tied))
    return {
        "official_ids": official.tolist(),
        "official_set": sorted(official_set),
        "cutoff": float(partition["cutoff"].float().item()),
        "choice_dtype": str(choice.dtype).removeprefix("torch."),
        "above_ids": above,
        "cutoff_tie_ids": tied,
        "slots_from_cutoff_tie": int(partition["slots"]),
        "ambiguous_cutoff": bool(partition["ambiguous"]),
        "official_tie_ids": official_tied,
        "lowest_id_rule_ids": low,
        "highest_id_rule_ids": high,
        "lowest_id_rule_matches": set(low) == official_set,
        "highest_id_rule_matches": set(high) == official_set,
        "sorted_true_ids": sorted(_id_set(sorted_true)),
        "sorted_true_same_set": _id_set(sorted_true) == official_set,
        "repeat_topk": repeat_topk,
        "repeat_topk_stable": True,
        "permutations": permutation_results,
        "permutation_changed_subset": changed,
    }


def torch_fingerprint() -> dict[str, Any]:
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


def analyze_layer(
    fixture: Any,
    config: Any,
    layer: int,
    inputs: torch.Tensor,
    expected_rows: list[list[int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    repeat_topk: int,
) -> dict[str, Any]:
    module, _router_input, router_logits, official_indices, _official_weights = (
        capture_official_router(
            fixture,
            config,
            layer,
            inputs,
            device=device,
            dtype=dtype,
        )
    )
    gate = module.mlp.gate
    routed_n = int(gate.num_experts)
    shared_n = int(gate.n_shared_experts)
    top_k = int(gate.top_k)
    if router_logits.shape != (inputs.shape[0], routed_n + shared_n):
        raise RouterTieResolutionError(
            f"layer {layer} router logits have unexpected shape {tuple(router_logits.shape)}"
        )
    if len(expected_rows) != int(inputs.shape[0]):
        raise RouterTieResolutionError(
            f"layer {layer} committed route row count differs from input positions"
        )
    routed_logits = router_logits[..., :routed_n]
    choice = routed_logits.sigmoid() + gate.e_score_correction_bias.detach().to(
        routed_logits.dtype
    )
    rows = [
        {
            "position": position,
            **analyze_row(
                choice[position],
                official_indices[position],
                expected_rows[position],
                top_k,
                repeat_topk=repeat_topk,
            ),
        }
        for position in range(int(inputs.shape[0]))
    ]
    decision = classify_layer(rows)
    return {
        "top_k": top_k,
        "n_routed_experts": routed_n,
        "rows": rows,
        "decision": decision,
        "all_sorted_true_same_set": all(row["sorted_true_same_set"] for row in rows),
        "all_repeated_topk_stable": all(row["repeat_topk_stable"] for row in rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeat-topk", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    args = parser.parse_args(argv)
    try:
        if args.repeat_topk < 2:
            raise RouterTieResolutionError("--repeat-topk must be at least two")
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        inputs = torch.tensor(input_values, dtype=torch.float32)
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise RouterTieResolutionError("--layers must be nonempty")
        if selection.get("revision") != fixture.source.get("revision"):
            raise RouterTieResolutionError("selection revision differs from fixture")
        if selection.get("config_sha256") != fixture.source.get("config_sha256"):
            raise RouterTieResolutionError("selection config hash differs from fixture")
        if selection.get("index_sha256") != fixture.source.get("index_sha256"):
            raise RouterTieResolutionError("selection index hash differs from fixture")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise RouterTieResolutionError("the tie-resolution probe requires CPU")
        expected_hash = selection.get("inputs_sha256")
        actual_hash = input_sha256(inputs, dtype)
        if actual_hash != expected_hash:
            raise RouterTieResolutionError(
                f"input hash {actual_hash} differs from selection {expected_hash}"
            )
        analyses = {}
        for layer in layers:
            layer_data = selection.get("layers", {}).get(str(layer))
            if not isinstance(layer_data, dict):
                raise RouterTieResolutionError(
                    f"selection archive has no layer {layer}"
                )
            expected_rows = layer_data.get("topk_indices")
            if not isinstance(expected_rows, list):
                raise RouterTieResolutionError(
                    f"selection archive layer {layer} has no topk_indices"
                )
            analyses[str(layer)] = analyze_layer(
                fixture,
                config,
                layer,
                inputs,
                expected_rows,
                device=device,
                dtype=dtype,
                repeat_topk=args.repeat_topk,
            )
        result = {
            "format": "inkling-bfloat16-router-tie-resolution-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": actual_hash,
            "repeat_topk": args.repeat_topk,
            "backend": torch_fingerprint(),
            "layers": analyses,
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        FixtureError,
        FixtureReferenceError,
        RouterTieResolutionError,
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
