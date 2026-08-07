#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run pinned Torch top-k on an exact source-bound BF16 choice-bit archive.

This platform-side auditor is deliberately self-contained: after the Linux
source-reference job emits the archive, no model-side modules or ``mvp`` import
paths are required.  That keeps Windows/macOS results about the pinned Torch
``topk`` primitive itself rather than repository path conventions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch


class ChoiceAuditError(RuntimeError):
    """The cross-platform choice archive cannot be audited safely."""


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def verify_archive_hash(archive: dict[str, Any]) -> str:
    expected = archive.get("archive_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ChoiceAuditError("choice archive has no valid archive_sha256")
    unhashed = dict(archive)
    unhashed.pop("archive_sha256", None)
    actual = hashlib.sha256(_canonical_bytes(unhashed)).hexdigest()
    if actual != expected:
        raise ChoiceAuditError(f"choice archive SHA {actual} != {expected}")
    return actual


def cutoff_partition(choice: torch.Tensor, top_k: int) -> dict[str, Any]:
    """Describe the exact BF16 equivalence class at the top-k cutoff."""
    if choice.ndim != 1:
        raise ChoiceAuditError("choice row must be one-dimensional")
    if top_k < 1 or top_k > choice.numel():
        raise ChoiceAuditError("top_k is outside the choice row")
    values = torch.topk(choice, top_k, dim=-1, sorted=False).values
    cutoff = values.min()
    above = torch.nonzero(choice > cutoff, as_tuple=False).flatten().to(torch.int64)
    tied = torch.nonzero(choice == cutoff, as_tuple=False).flatten().to(torch.int64)
    slots = top_k - int(above.numel())
    if slots < 1 or slots > int(tied.numel()):
        raise ChoiceAuditError(
            f"invalid cutoff partition: above={above.numel()} tied={tied.numel()} slots={slots}"
        )
    return {
        "cutoff": cutoff,
        "above": above,
        "tied": tied,
        "slots": slots,
        "ambiguous": int(tied.numel()) > slots,
    }


def valid_under_partition(
    selected: list[int] | tuple[int, ...] | torch.Tensor,
    partition: dict[str, Any],
    top_k: int,
) -> bool:
    """Require a selected set to differ only inside the exact cutoff tie."""
    if isinstance(selected, torch.Tensor):
        actual = {int(value) for value in selected.reshape(-1).tolist()}
    else:
        actual = {int(value) for value in selected}
    above = {int(value) for value in partition["above"].reshape(-1).tolist()}
    tied = {int(value) for value in partition["tied"].reshape(-1).tolist()}
    return (
        len(actual) == top_k
        and above.issubset(actual)
        and actual.issubset(above | tied)
        and len(actual & tied) == int(partition["slots"])
    )


def platform_fingerprint() -> dict[str, Any]:
    result = {
        "torch_version": torch.__version__,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_compiler": platform.python_compiler(),
        "python_version": platform.python_version(),
    }
    try:
        result["cpu_dispatch"] = torch.backends.cpu.get_cpu_capability()
    except Exception as exc:  # platform/build support differs intentionally.
        result["cpu_dispatch"] = None
        result["cpu_dispatch_error"] = str(exc)
    return result


def audit_layer(layer: str, data: dict[str, Any], repeat_topk: int) -> dict[str, Any]:
    top_k = int(data["top_k"])
    width = int(data["n_routed_experts"])
    bits = torch.tensor(data["choice_bits_int16"], dtype=torch.int16)
    if bits.ndim != 2 or bits.shape[1] != width:
        raise ChoiceAuditError(
            f"layer {layer} bit shape {tuple(bits.shape)} is not (*,{width})"
        )
    choice = bits.contiguous().view(torch.bfloat16)
    expected_rows = data["official_indices"]
    if len(expected_rows) != choice.shape[0]:
        raise ChoiceAuditError(f"layer {layer} route row count differs from archive bits")

    try:
        first = torch.topk(choice, top_k, dim=-1, sorted=False).indices.to(torch.int64).cpu()
    except Exception as exc:
        return {
            "supported": False,
            "error": f"{type(exc).__name__}: {exc}",
            "rows": [],
            "exact_set_rows": 0,
            "exact_order_rows": 0,
            "ambiguous_rows": 0,
            "ambiguous_changed_rows": 0,
            "unambiguous_changed_rows": 0,
        }

    repeated_stable = True
    for _ in range(repeat_topk - 1):
        current = torch.topk(choice, top_k, dim=-1, sorted=False).indices.to(torch.int64).cpu()
        if not torch.equal(first, current):
            repeated_stable = False
            break
    if not repeated_stable:
        raise ChoiceAuditError(f"layer {layer} top-k is unstable on unchanged bits")

    rows: list[dict[str, Any]] = []
    exact_set_rows = 0
    exact_order_rows = 0
    ambiguous_rows = 0
    ambiguous_changed_rows = 0
    unambiguous_changed_rows = 0
    for position in range(choice.shape[0]):
        selected = [int(value) for value in first[position].tolist()]
        expected = [int(value) for value in expected_rows[position]]
        partition = cutoff_partition(choice[position], top_k)
        valid = valid_under_partition(selected, partition, top_k)
        if not valid:
            raise ChoiceAuditError(
                f"layer {layer} position {position} selected outside BF16 cutoff class"
            )
        exact_set = set(selected) == set(expected)
        exact_order = selected == expected
        ambiguous = bool(partition["ambiguous"])
        exact_set_rows += int(exact_set)
        exact_order_rows += int(exact_order)
        ambiguous_rows += int(ambiguous)
        if ambiguous and not exact_set:
            ambiguous_changed_rows += 1
        if not ambiguous and not exact_set:
            unambiguous_changed_rows += 1
        rows.append(
            {
                "position": position,
                "selected_ids": selected,
                "expected_ids": expected,
                "exact_set": exact_set,
                "exact_order": exact_order,
                "ambiguous_cutoff": ambiguous,
                "valid_under_cutoff": valid,
                "cutoff": float(partition["cutoff"].float().item()),
                "above_ids": sorted(int(value) for value in partition["above"].tolist()),
                "tie_ids": sorted(int(value) for value in partition["tied"].tolist()),
                "slots_from_tie": int(partition["slots"]),
            }
        )
    if unambiguous_changed_rows:
        raise ChoiceAuditError(
            f"layer {layer} changed {unambiguous_changed_rows} unambiguous route rows"
        )
    return {
        "supported": True,
        "error": None,
        "repeat_topk": repeat_topk,
        "repeat_topk_stable": True,
        "rows": rows,
        "exact_set_rows": exact_set_rows,
        "exact_order_rows": exact_order_rows,
        "ambiguous_rows": ambiguous_rows,
        "ambiguous_changed_rows": ambiguous_changed_rows,
        "unambiguous_changed_rows": 0,
    }


def classify_platform(layers: dict[str, Any]) -> str:
    supported = [value for value in layers.values() if value["supported"]]
    if not supported:
        return "bfloat16_topk_unsupported"
    if len(supported) != len(layers):
        return "bfloat16_topk_partially_supported"
    if any(value["ambiguous_changed_rows"] for value in supported):
        return "official_tied_route_sets_differ_from_linux_archive"
    return "official_route_sets_match_linux_archive"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--platform-label", required=True)
    parser.add_argument("--repeat-topk", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        if args.repeat_topk < 2:
            raise ChoiceAuditError("--repeat-topk must be at least two")
        archive = json.loads(Path(args.archive).read_text(encoding="utf-8"))
        if archive.get("format") != "inkling-bfloat16-router-choice-bits":
            raise ChoiceAuditError("unexpected choice archive format")
        archive_sha = verify_archive_hash(archive)
        if torch.__version__.split("+", 1)[0] != "2.13.0":
            raise ChoiceAuditError(f"Torch version {torch.__version__} is not 2.13.0")
        layers = {
            layer: audit_layer(layer, data, args.repeat_topk)
            for layer, data in archive["layers"].items()
        }
        result = {
            "format": "inkling-bfloat16-router-topk-platform-audit",
            "version": 1,
            "platform_label": args.platform_label,
            "archive_sha256": archive_sha,
            "model_id": archive.get("model_id"),
            "revision": archive.get("revision"),
            "config_sha256": archive.get("config_sha256"),
            "index_sha256": archive.get("index_sha256"),
            "inputs_sha256": archive.get("inputs_sha256"),
            "platform": platform_fingerprint(),
            "layers": layers,
            "classification": classify_platform(layers),
        }
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (ChoiceAuditError, OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
