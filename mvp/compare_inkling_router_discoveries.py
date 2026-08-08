#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare Inkling router discovery artifacts without depending on top-k slot order."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


class RouterDiscoveryComparisonError(RuntimeError):
    """Discovery artifacts cannot be compared without guessing."""


METADATA_KEYS = (
    "format",
    "version",
    "model_id",
    "revision",
    "config_sha256",
    "index_sha256",
    "tokens",
    "seed",
    "input_dtype",
    "inputs_sha256",
)


def canonicalize_discovery(value: dict[str, Any]) -> dict[str, Any]:
    """Sort every `(expert_id, weight)` pair by expert ID and validate coverage."""
    if not isinstance(value, dict):
        raise RouterDiscoveryComparisonError("router discovery must be an object")
    result = copy.deepcopy(value)
    layers = result.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise RouterDiscoveryComparisonError("router discovery has no layers")

    for layer, record in layers.items():
        if not isinstance(record, dict):
            raise RouterDiscoveryComparisonError(f"layer {layer} is not an object")
        indices = record.get("topk_indices")
        weights = record.get("topk_weights")
        if not isinstance(indices, list) or not isinstance(weights, list):
            raise RouterDiscoveryComparisonError(
                f"layer {layer} is missing top-k indices or weights"
            )
        if len(indices) != len(weights):
            raise RouterDiscoveryComparisonError(
                f"layer {layer} top-k row count mismatch"
            )

        selected: set[int] = set()
        canonical_indices: list[list[int]] = []
        canonical_weights: list[list[float]] = []
        for position, (id_row, weight_row) in enumerate(zip(indices, weights)):
            if not isinstance(id_row, list) or not isinstance(weight_row, list):
                raise RouterDiscoveryComparisonError(
                    f"layer {layer} position {position} is not a list"
                )
            if len(id_row) != len(weight_row):
                raise RouterDiscoveryComparisonError(
                    f"layer {layer} position {position} pair count mismatch"
                )
            pairs = [(int(expert), float(weight)) for expert, weight in zip(id_row, weight_row)]
            ids = [expert for expert, _ in pairs]
            if len(ids) != len(set(ids)):
                raise RouterDiscoveryComparisonError(
                    f"layer {layer} position {position} repeats an expert"
                )
            pairs.sort(key=lambda pair: pair[0])
            canonical_indices.append([expert for expert, _ in pairs])
            canonical_weights.append([weight for _, weight in pairs])
            selected.update(ids)

        declared = record.get("selected_experts")
        if not isinstance(declared, list):
            raise RouterDiscoveryComparisonError(
                f"layer {layer} is missing selected_experts"
            )
        if sorted(int(expert) for expert in declared) != sorted(selected):
            raise RouterDiscoveryComparisonError(
                f"layer {layer} selected_experts does not match routed rows"
            )
        record["selected_experts"] = sorted(selected)
        record["topk_indices"] = canonical_indices
        record["topk_weights"] = canonical_weights
    return result


def compare_discoveries(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Require identical source binding and identical pair-preserving routing."""
    for key in METADATA_KEYS:
        if actual.get(key) != expected.get(key):
            raise RouterDiscoveryComparisonError(
                f"metadata mismatch for {key}: {actual.get(key)!r} != {expected.get(key)!r}"
            )
    canonical_actual = canonicalize_discovery(actual)
    canonical_expected = canonicalize_discovery(expected)
    if canonical_actual != canonical_expected:
        for layer in sorted(
            set(canonical_actual.get("layers", ()))
            | set(canonical_expected.get("layers", ())),
            key=int,
        ):
            if canonical_actual.get("layers", {}).get(layer) != canonical_expected.get("layers", {}).get(layer):
                raise RouterDiscoveryComparisonError(
                    f"pair-preserving routing mismatch in layer {layer}"
                )
        raise RouterDiscoveryComparisonError("router discovery artifacts differ")
    return {
        "passed": True,
        "layers": sorted(canonical_actual["layers"], key=int),
        "tokens": canonical_actual["tokens"],
        "comparison": "expert-weight-pairs-canonicalized-by-expert-id",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", required=True)
    parser.add_argument("--expected", required=True)
    args = parser.parse_args(argv)
    try:
        actual = json.loads(Path(args.actual).read_text(encoding="utf-8"))
        expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
        result = compare_discoveries(actual, expected)
    except (OSError, ValueError, KeyError, RouterDiscoveryComparisonError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
