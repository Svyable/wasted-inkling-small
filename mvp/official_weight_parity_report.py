#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble official-weight layer comparisons and exact routing evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "inkling" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from inkling_parity import ParityError, read_activation_archive


class OfficialWeightReportError(RuntimeError):
    """The parity report inputs are incomplete or inconsistent."""


def _mapping(values: list[str], option: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for value in values:
        try:
            layer_text, path_text = value.split(":", 1)
            layer = int(layer_text)
        except (ValueError, TypeError) as exc:
            raise OfficialWeightReportError(
                f"{option} needs L:path entries, got {value!r}"
            ) from exc
        if layer < 0 or layer in out or not path_text:
            raise OfficialWeightReportError(
                f"invalid or duplicate {option} layer {layer}"
            )
        out[layer] = Path(path_text)
    if not out:
        raise OfficialWeightReportError(f"{option} must be nonempty")
    return out


def _canonical_pairs(indices: list[int], weights: list[float]) -> tuple[list[int], list[float]]:
    if len(indices) != len(weights) or len(set(indices)) != len(indices):
        raise OfficialWeightReportError("router indices/weights are not unique pairs")
    pairs = sorted(zip(indices, weights), key=lambda pair: pair[0])
    return [int(pair[0]) for pair in pairs], [float(pair[1]) for pair in pairs]


def compare_candidate_routing(
    selection: dict[str, Any],
    layer: int,
    archive: Path | str,
    *,
    atol: float,
) -> dict[str, Any]:
    layer_selection = selection.get("layers", {}).get(str(layer))
    if not isinstance(layer_selection, dict):
        raise OfficialWeightReportError(f"selection has no sparse layer {layer}")
    expected_indices = layer_selection.get("topk_indices")
    expected_weights = layer_selection.get("topk_weights")
    if not isinstance(expected_indices, list) or not isinstance(expected_weights, list):
        raise OfficialWeightReportError(f"selection layer {layer} has no router rows")
    if len(expected_indices) != len(expected_weights):
        raise OfficialWeightReportError(f"selection layer {layer} row count mismatch")

    values, metadata = read_activation_archive(archive)
    rows: list[dict[str, Any]] = []
    exact_indices = True
    max_abs = 0.0
    for position, (want_ids, want_weights) in enumerate(
        zip(expected_indices, expected_weights)
    ):
        index_name = f"token.{position}.layer.{layer}.routed_index"
        weight_name = f"token.{position}.layer.{layer}.routed_weight"
        if index_name not in values or weight_name not in values:
            raise OfficialWeightReportError(
                f"candidate archive lacks router pair {index_name!r}"
            )
        got_ids = [int(value) for value in values[index_name].flatten().tolist()]
        got_weights = [float(value) for value in values[weight_name].flatten().tolist()]
        want_ids_sorted, want_weights_sorted = _canonical_pairs(
            [int(value) for value in want_ids],
            [float(value) for value in want_weights],
        )
        got_ids_sorted, got_weights_sorted = _canonical_pairs(got_ids, got_weights)
        ids_ok = got_ids_sorted == want_ids_sorted
        exact_indices = exact_indices and ids_ok
        if len(got_weights_sorted) != len(want_weights_sorted):
            raise OfficialWeightReportError(
                f"candidate route width differs at layer {layer} position {position}"
            )
        row_max = max(
            (abs(got - want) for got, want in zip(got_weights_sorted, want_weights_sorted)),
            default=0.0,
        )
        max_abs = max(max_abs, row_max)
        rows.append({
            "position": position,
            "exact_indices": ids_ok,
            "max_abs_weight": row_max,
        })
    weights_ok = max_abs <= atol
    return {
        "passed": exact_indices and weights_ok,
        "exact_indices": exact_indices,
        "weights_within_tolerance": weights_ok,
        "max_abs_weight": max_abs,
        "atol": atol,
        "positions": len(rows),
        "rows": rows,
        "candidate_metadata": metadata,
    }


def assemble_report(
    selection: dict[str, Any],
    layer_reports: dict[int, dict[str, Any]],
    candidate_archives: dict[int, Path],
    *,
    route_atol: float,
) -> dict[str, Any]:
    if route_atol < 0:
        raise OfficialWeightReportError("route_atol must be nonnegative")
    if set(layer_reports) != set(candidate_archives):
        raise OfficialWeightReportError(
            "layer reports and candidate archives name different layers"
        )

    layers: dict[str, Any] = {}
    passed = True
    selection_layers = {
        int(key) for key in selection.get("layers", {})
        if isinstance(key, str) and key.isdigit()
    }
    for layer in sorted(layer_reports):
        comparison = layer_reports[layer]
        comparison_passed = comparison.get("passed") is True
        route = None
        if layer in selection_layers:
            route = compare_candidate_routing(
                selection,
                layer,
                candidate_archives[layer],
                atol=route_atol,
            )
        layer_passed = comparison_passed and (route is None or route["passed"])
        passed = passed and layer_passed
        layers[str(layer)] = {
            "passed": layer_passed,
            "activation_comparison": comparison,
            "routing": route,
        }

    return {
        "format": "inkling-official-weight-layer-parity",
        "version": 1,
        "passed": passed,
        "model_id": selection.get("model_id"),
        "revision": selection.get("revision"),
        "config_sha256": selection.get("config_sha256"),
        "index_sha256": selection.get("index_sha256"),
        "inputs_sha256": selection.get("inputs_sha256"),
        "tokens": selection.get("tokens"),
        "input_dtype": selection.get("input_dtype"),
        "route_atol": route_atol,
        "layers": layers,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--layer-report", action="append", default=[])
    ap.add_argument("--candidate-archive", action="append", default=[])
    ap.add_argument("--route-atol", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        report_paths = _mapping(args.layer_report, "--layer-report")
        archive_paths = _mapping(args.candidate_archive, "--candidate-archive")
        reports = {
            layer: json.loads(path.read_text(encoding="utf-8"))
            for layer, path in report_paths.items()
        }
        report = assemble_report(
            selection,
            reports,
            archive_paths,
            route_atol=args.route_atol,
        )
        Path(args.out).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        OSError,
        json.JSONDecodeError,
        OfficialWeightReportError,
        ParityError,
        ValueError,
    ) as exc:
        ap.error(str(exc))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
