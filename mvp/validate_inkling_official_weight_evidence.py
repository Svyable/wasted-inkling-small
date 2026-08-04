#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate an official-weight report as parity or bounded mismatch evidence.

The expensive official-weight workflow has two legitimate numerical outcomes:

* every compared activation passes; or
* execution completes and produces the already-characterized BF16 mismatch.

This validator keeps those outcomes separate from acquisition, binding, archive,
shape, dtype, routing, and report failures. Structural failures always fail.
A numerical mismatch passes only when it is source-bound, complete, and no worse
than the committed evidence envelope.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


class EvidenceValidationError(RuntimeError):
    """The generated evidence is incomplete, inconsistent, or regressed."""


IDENTITY_KEYS = (
    "model_id",
    "revision",
    "config_sha256",
    "index_sha256",
    "inputs_sha256",
    "tokens",
    "input_dtype",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceValidationError(message)


def _finite_nonnegative(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result) and result >= 0.0, f"{label} is invalid")
    return result


def _comparison_results(comparison: dict[str, Any], layer: str) -> dict[str, dict[str, Any]]:
    _require(comparison.get("format") == "inkling-parity-report", f"layer {layer} report format")
    _require(comparison.get("version") == 1, f"layer {layer} report version")
    values = comparison.get("results")
    _require(isinstance(values, list) and values, f"layer {layer} has no comparison results")
    out: dict[str, dict[str, Any]] = {}
    for item in values:
        _require(isinstance(item, dict), f"layer {layer} has malformed result")
        name = item.get("name")
        status = item.get("status")
        _require(isinstance(name, str) and name not in out, f"layer {layer} duplicate result")
        _require(status in {"pass", "mismatch"}, f"layer {layer} structural result {status!r}: {name}")
        if "max_abs" in item:
            _finite_nonnegative(item["max_abs"], f"layer {layer} {name} max_abs")
        out[name] = item
    scope = comparison.get("comparison_scope")
    _require(isinstance(scope, dict), f"layer {layer} lacks comparison scope")
    _require(not scope.get("reference_only_missing"), f"layer {layer} misses official activations")
    return out


def _upper_bound(baseline: float, growth_limit: float) -> float:
    return max(baseline * growth_limit, baseline + 1e-6)


def validate_report(
    report: dict[str, Any],
    evidence: dict[str, Any],
    *,
    growth_limit: float = 1.25,
    require_parity: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable validation summary or raise fail-closed."""
    _require(growth_limit >= 1.0 and math.isfinite(growth_limit), "growth_limit must be finite and >= 1")
    _require(report.get("format") == "inkling-official-weight-layer-parity", "report format")
    _require(report.get("version") == 1, "report version")
    _require(evidence.get("format") == "inkling-official-weight-parity-evidence", "evidence format")
    _require(evidence.get("version") == 1, "evidence version")
    for key in IDENTITY_KEYS:
        _require(report.get(key) == evidence.get(key), f"source identity drift: {key}")

    report_layers = report.get("layers")
    evidence_layers = evidence.get("layers")
    _require(isinstance(report_layers, dict), "report layers are missing")
    _require(isinstance(evidence_layers, dict), "evidence layers are missing")
    _require(set(report_layers) == set(evidence_layers), "report/evidence layer sets differ")

    if report.get("passed") is True:
        for layer, record in report_layers.items():
            _require(isinstance(record, dict) and record.get("passed") is True, f"layer {layer} parity inconsistency")
            comparison = record.get("activation_comparison")
            _require(isinstance(comparison, dict) and comparison.get("passed") is True, f"layer {layer} comparison inconsistency")
            _comparison_results(comparison, layer)
            route = record.get("routing")
            _require(route is None or (isinstance(route, dict) and route.get("passed") is True), f"layer {layer} routing failed")
        return {
            "format": "inkling-official-weight-evidence-validation",
            "version": 1,
            "passed": True,
            "status": "parity",
            "growth_limit": growth_limit,
            "layers": sorted(report_layers, key=int),
        }

    _require(not require_parity, "numerical parity is required but the report failed")
    _require(evidence.get("passed") is False, "committed evidence does not describe a mismatch")
    tokens = int(evidence["tokens"])
    summary: dict[str, Any] = {}

    for layer in sorted(report_layers, key=int):
        record = report_layers[layer]
        baseline = evidence_layers[layer]
        _require(isinstance(record, dict), f"layer {layer} record is malformed")
        _require(record.get("passed") is False, f"layer {layer} unexpectedly marked passed")
        comparison = record.get("activation_comparison")
        _require(isinstance(comparison, dict), f"layer {layer} comparison is missing")
        _require(comparison.get("passed") is False, f"layer {layer} comparison/report disagree")
        results = _comparison_results(comparison, layer)
        _require(any(item["status"] == "mismatch" for item in results.values()), f"layer {layer} has no numerical mismatch")

        expected_route = baseline.get("routing")
        route = record.get("routing")
        if expected_route is None:
            _require(route is None, f"dense layer {layer} unexpectedly has routing")
        else:
            _require(isinstance(route, dict), f"sparse layer {layer} lacks routing")
            _require(route.get("passed") is True, f"sparse layer {layer} routing failed")
            _require(route.get("exact_indices") is True, f"sparse layer {layer} canonical route drift")
            _require(route.get("weights_within_tolerance") is True, f"sparse layer {layer} route weights drift")
            _require(route.get("positions") == tokens, f"sparse layer {layer} route length drift")

        output_pattern = re.compile(rf"^token\.(\d+)\.layer\.{re.escape(layer)}\.output$")
        output_rows: list[float] = []
        for name, item in results.items():
            if output_pattern.match(name):
                output_rows.append(_finite_nonnegative(item.get("max_abs"), f"layer {layer} {name} max_abs"))
        _require(len(output_rows) == tokens, f"layer {layer} output row count drift")
        output_max = max(output_rows)
        baseline_output = _finite_nonnegative(
            baseline.get("max_output_abs_across_positions"),
            f"layer {layer} baseline output",
        )
        _require(
            output_max <= _upper_bound(baseline_output, growth_limit),
            f"layer {layer} output regression: {output_max} > {_upper_bound(baseline_output, growth_limit)}",
        )

        final_position = baseline.get("final_position")
        _require(isinstance(final_position, dict) and final_position, f"layer {layer} baseline points missing")
        point_maxima: dict[str, float] = {}
        for point, baseline_value in final_position.items():
            name = f"token.{tokens - 1}.layer.{layer}.{point}"
            _require(name in results, f"layer {layer} missing evidence point {point}")
            actual = _finite_nonnegative(results[name].get("max_abs"), f"layer {layer} {point} max_abs")
            expected = _finite_nonnegative(baseline_value, f"layer {layer} baseline {point}")
            _require(
                actual <= _upper_bound(expected, growth_limit),
                f"layer {layer} {point} regression: {actual} > {_upper_bound(expected, growth_limit)}",
            )
            point_maxima[point] = actual

        summary[layer] = {
            "output_max_abs": output_max,
            "baseline_output_max_abs": baseline_output,
            "final_position_max_abs": point_maxima,
            "routing_passed": route is None or route.get("passed") is True,
        }

    return {
        "format": "inkling-official-weight-evidence-validation",
        "version": 1,
        "passed": True,
        "status": "expected-bounded-mismatch",
        "growth_limit": growth_limit,
        "layers": summary,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--growth-limit", type=float, default=1.25)
    ap.add_argument("--require-parity", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
        result = validate_report(
            report,
            evidence,
            growth_limit=args.growth_limit,
            require_parity=args.require_parity,
        )
    except (OSError, json.JSONDecodeError, EvidenceValidationError, KeyError, TypeError, ValueError) as exc:
        ap.error(str(exc))
        return 2
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
