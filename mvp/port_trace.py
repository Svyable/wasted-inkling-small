#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and compare architecture-neutral chained port evidence traces.

A trace is valid before it is comparable. Validation proves that every boundary
file matches its declared dtype/shape/SHA, layers are consecutive, output bytes
chain exactly into the next layer's input, and optional routing contracts are
well formed. Comparison then stops at the first differing layer/boundary/index
and reports raw bits. No tolerance is applied.

Routing deliberately separates four concepts:
  observed_route          what this implementation selected
  reference_route         the route used as the evidence reference
  downstream_route_used   "observed" or "reference"
  route_equivalence_reason
This lets a port preserve raw routing while comparing downstream arithmetic
under a fixed reference route when an explicitly documented tie contract
requires it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SUPPORTED_DTYPES: dict[str, tuple[int, str]] = {
    "BF16": (2, "H"),
    "F32": (4, "I"),  # raw IEEE-754 bits, never Python float equality
    "U32": (4, "I"),
    "I32": (4, "i"),
    "I64": (8, "q"),
    "U8": (1, "B"),
}
DOWNSTREAM_ROUTE_CHOICES = {"observed", "reference"}


class TraceError(ValueError):
    """The trace is malformed, stale, incompatible, or internally inconsistent."""


@dataclass(frozen=True)
class Boundary:
    name: str
    path: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    nbytes: int


@dataclass(frozen=True)
class RoutingContract:
    observed_route: Any
    reference_route: Any
    downstream_route_used: str
    route_equivalence_reason: str


@dataclass(frozen=True)
class Layer:
    number: int
    structural_class: str
    boundaries: tuple[Boundary, ...]
    routing: RoutingContract | None


@dataclass(frozen=True)
class Trace:
    manifest_path: str
    root: str
    model: str
    revision: str
    name: str
    layers: tuple[Layer, ...]


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _product(shape: tuple[int, ...]) -> int:
    total = 1
    for value in shape:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise TraceError(f"invalid shape dimension {value!r}")
        total *= value
    return total


def _hex_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TraceError(f"{label}: invalid SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise TraceError(f"{label}: invalid SHA-256") from exc
    return value.lower()


def _parse_boundary(root: str, layer: int, raw: Any) -> Boundary:
    if not isinstance(raw, dict):
        raise TraceError(f"layer {layer}: boundary must be an object")
    name = raw.get("name")
    relative = raw.get("file")
    dtype = raw.get("dtype")
    shape_raw = raw.get("shape")
    if not isinstance(name, str) or not name:
        raise TraceError(f"layer {layer}: boundary has invalid name")
    if not isinstance(relative, str) or not relative or os.path.isabs(relative):
        raise TraceError(f"layer {layer} {name}: file must be a relative path")
    normalized = os.path.normpath(relative)
    if normalized == ".." or normalized.startswith(".." + os.sep):
        raise TraceError(f"layer {layer} {name}: file escapes trace root")
    if dtype not in SUPPORTED_DTYPES:
        raise TraceError(f"layer {layer} {name}: unsupported dtype {dtype!r}")
    if not isinstance(shape_raw, list) or not shape_raw:
        raise TraceError(f"layer {layer} {name}: shape must be a non-empty list")
    shape = tuple(shape_raw)
    expected_bytes = _product(shape) * SUPPORTED_DTYPES[dtype][0]
    digest = _hex_digest(raw.get("sha256"), label=f"layer {layer} {name}")
    path = os.path.join(root, normalized)
    if not os.path.isfile(path):
        raise TraceError(f"layer {layer} {name}: missing file {relative}")
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise TraceError(
            f"layer {layer} {name}: {actual_bytes} bytes, expected "
            f"{expected_bytes} for {dtype}{list(shape)}"
        )
    actual_digest = sha256_file(path)
    if actual_digest != digest:
        raise TraceError(
            f"layer {layer} {name}: file SHA {actual_digest} != manifest {digest}"
        )
    return Boundary(name, path, dtype, shape, digest, expected_bytes)


def _validate_route(value: Any, *, label: str) -> Any:
    if not isinstance(value, list):
        raise TraceError(f"{label}: route must be a list")
    for position, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise TraceError(f"{label}: route row {position} must be a non-empty list")
        for expert in row:
            if not isinstance(expert, int) or isinstance(expert, bool) or expert < 0:
                raise TraceError(f"{label}: invalid expert id {expert!r}")
    return value


def _parse_routing(layer: int, raw: Any) -> RoutingContract | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TraceError(f"layer {layer}: routing must be an object")
    observed = _validate_route(
        raw.get("observed_route"), label=f"layer {layer} observed_route"
    )
    reference = _validate_route(
        raw.get("reference_route"), label=f"layer {layer} reference_route"
    )
    if len(observed) != len(reference):
        raise TraceError(f"layer {layer}: observed/reference route row counts differ")
    downstream = raw.get("downstream_route_used")
    if downstream not in DOWNSTREAM_ROUTE_CHOICES:
        raise TraceError(
            f"layer {layer}: downstream_route_used must be one of "
            f"{sorted(DOWNSTREAM_ROUTE_CHOICES)}"
        )
    reason = raw.get("route_equivalence_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise TraceError(f"layer {layer}: route_equivalence_reason must be non-empty")
    return RoutingContract(observed, reference, downstream, reason)


def load_trace(manifest_path: str | os.PathLike[str]) -> Trace:
    manifest_path = os.path.abspath(os.fspath(manifest_path))
    root = os.path.dirname(manifest_path)
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceError(f"cannot read trace manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TraceError("trace manifest must be an object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise TraceError(f"unsupported schema_version {raw.get('schema_version')!r}")
    model = raw.get("model")
    revision = raw.get("revision")
    name = raw.get("trace_name")
    if not isinstance(model, str) or not model:
        raise TraceError("trace model must be a non-empty string")
    if not isinstance(revision, str) or not revision:
        raise TraceError("trace revision must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise TraceError("trace_name must be a non-empty string")

    layers_raw = raw.get("layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        raise TraceError("trace must contain at least one layer")

    layers: list[Layer] = []
    previous_number: int | None = None
    previous_output: Boundary | None = None
    for item in layers_raw:
        if not isinstance(item, dict):
            raise TraceError("layer entry must be an object")
        number = item.get("layer")
        structural_class = item.get("structural_class")
        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            raise TraceError(f"invalid layer number {number!r}")
        if previous_number is not None and number != previous_number + 1:
            raise TraceError(
                f"layers must be consecutive: got {previous_number} then {number}"
            )
        if not isinstance(structural_class, str) or not structural_class:
            raise TraceError(f"layer {number}: structural_class must be non-empty")
        boundaries_raw = item.get("boundaries")
        if not isinstance(boundaries_raw, list) or len(boundaries_raw) < 2:
            raise TraceError(f"layer {number}: need at least input/output boundaries")
        boundaries = tuple(_parse_boundary(root, number, value) for value in boundaries_raw)
        names = [boundary.name for boundary in boundaries]
        if len(set(names)) != len(names):
            raise TraceError(f"layer {number}: duplicate boundary name")
        if names[0] != "input" or names[-1] != "output":
            raise TraceError(
                f"layer {number}: boundary order must start input and end output"
            )
        current_input = boundaries[0]
        current_output = boundaries[-1]
        if previous_output is not None:
            if (
                previous_output.dtype != current_input.dtype
                or previous_output.shape != current_input.shape
            ):
                raise TraceError(
                    f"layer {number}: input geometry does not match previous output"
                )
            if previous_output.sha256 != current_input.sha256:
                raise TraceError(
                    f"layer {number}: chain break; previous output SHA "
                    f"{previous_output.sha256} != input SHA {current_input.sha256}"
                )
        routing = _parse_routing(number, item.get("routing"))
        layers.append(Layer(number, structural_class, boundaries, routing))
        previous_number = number
        previous_output = current_output

    return Trace(manifest_path, root, model, revision, name, tuple(layers))


def ensure_compatible(expected: Trace, actual: Trace) -> None:
    if expected.model != actual.model:
        raise TraceError(f"model mismatch: {expected.model!r} != {actual.model!r}")
    if expected.revision != actual.revision:
        raise TraceError(
            f"revision mismatch: {expected.revision!r} != {actual.revision!r}"
        )
    if len(expected.layers) != len(actual.layers):
        raise TraceError(
            f"layer-count mismatch: {len(expected.layers)} != {len(actual.layers)}"
        )
    for expected_layer, actual_layer in zip(expected.layers, actual.layers):
        if expected_layer.number != actual_layer.number:
            raise TraceError(
                f"layer identity mismatch: {expected_layer.number} != {actual_layer.number}"
            )
        if expected_layer.structural_class != actual_layer.structural_class:
            raise TraceError(
                f"layer {expected_layer.number}: structural class mismatch: "
                f"{expected_layer.structural_class!r} != "
                f"{actual_layer.structural_class!r}"
            )
        if len(expected_layer.boundaries) != len(actual_layer.boundaries):
            raise TraceError(f"layer {expected_layer.number}: boundary-count mismatch")
        for expected_boundary, actual_boundary in zip(
            expected_layer.boundaries, actual_layer.boundaries
        ):
            left = (
                expected_boundary.name,
                expected_boundary.dtype,
                expected_boundary.shape,
            )
            right = (
                actual_boundary.name,
                actual_boundary.dtype,
                actual_boundary.shape,
            )
            if left != right:
                raise TraceError(
                    f"layer {expected_layer.number}: boundary contract mismatch: "
                    f"{left} != {right}"
                )
        eroute = expected_layer.routing
        aroute = actual_layer.routing
        if (eroute is None) != (aroute is None):
            raise TraceError(
                f"layer {expected_layer.number}: routing contract presence mismatch"
            )
        if eroute is not None and aroute is not None:
            if eroute.reference_route != aroute.reference_route:
                raise TraceError(
                    f"layer {expected_layer.number}: reference route mismatch"
                )
            if eroute.downstream_route_used != aroute.downstream_route_used:
                raise TraceError(
                    f"layer {expected_layer.number}: downstream route contract mismatch"
                )
            if eroute.route_equivalence_reason != aroute.route_equivalence_reason:
                raise TraceError(
                    f"layer {expected_layer.number}: route equivalence reason mismatch"
                )


def _first_element_difference(
    expected: Boundary, actual: Boundary
) -> dict[str, Any]:
    item_bytes, fmt = SUPPORTED_DTYPES[expected.dtype]
    with open(expected.path, "rb") as expected_file, open(
        actual.path, "rb"
    ) as actual_file:
        index = 0
        chunk_items = max(1, (1 << 20) // item_bytes)
        while True:
            expected_raw = expected_file.read(chunk_items * item_bytes)
            actual_raw = actual_file.read(chunk_items * item_bytes)
            if not expected_raw and not actual_raw:
                break
            if len(expected_raw) != len(actual_raw):
                raise TraceError("compatible boundary files changed size during comparison")
            if expected_raw == actual_raw:
                index += len(expected_raw) // item_bytes
                continue
            for offset in range(0, len(expected_raw), item_bytes):
                left = expected_raw[offset : offset + item_bytes]
                right = actual_raw[offset : offset + item_bytes]
                if left != right:
                    return {
                        "index": index + offset // item_bytes,
                        "expected_value": struct.unpack("<" + fmt, left)[0],
                        "actual_value": struct.unpack("<" + fmt, right)[0],
                        "expected_hex": left.hex(),
                        "actual_hex": right.hex(),
                    }
            index += len(expected_raw) // item_bytes
    raise TraceError("boundary SHA differs but no differing element was found")


def compare(expected: Trace, actual: Trace) -> dict[str, Any]:
    ensure_compatible(expected, actual)
    for expected_layer, actual_layer in zip(expected.layers, actual.layers):
        for expected_boundary, actual_boundary in zip(
            expected_layer.boundaries, actual_layer.boundaries
        ):
            if expected_boundary.sha256 != actual_boundary.sha256:
                detail = _first_element_difference(
                    expected_boundary, actual_boundary
                )
                return {
                    "status": "diverged",
                    "layer": expected_layer.number,
                    "structural_class": expected_layer.structural_class,
                    "boundary": expected_boundary.name,
                    "dtype": expected_boundary.dtype,
                    "shape": list(expected_boundary.shape),
                    "expected_sha256": expected_boundary.sha256,
                    "actual_sha256": actual_boundary.sha256,
                    **detail,
                }
    return {
        "status": "match",
        "model": expected.model,
        "revision": expected.revision,
        "layers": [layer.number for layer in expected.layers],
        "boundaries_compared": sum(
            len(layer.boundaries) for layer in expected.layers
        ),
    }


def boundary_manifest(
    root: str | os.PathLike[str],
    relative_file: str,
    name: str,
    dtype: str,
    shape: list[int],
    payload: bytes,
) -> dict[str, Any]:
    """Write one boundary payload and return its manifest entry."""
    if dtype not in SUPPORTED_DTYPES:
        raise TraceError(f"unsupported dtype {dtype!r}")
    shape_tuple = tuple(shape)
    expected_bytes = _product(shape_tuple) * SUPPORTED_DTYPES[dtype][0]
    if len(payload) != expected_bytes:
        raise TraceError(
            f"{name}: {len(payload)} payload bytes, expected {expected_bytes}"
        )
    root_path = Path(root)
    path = root_path / relative_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "name": name,
        "file": relative_file.replace(os.sep, "/"),
        "dtype": dtype,
        "shape": list(shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", help="expected trace manifest JSON")
    parser.add_argument("actual", help="actual trace manifest JSON")
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    args = parser.parse_args(argv)
    try:
        expected = load_trace(args.expected)
        actual = load_trace(args.actual)
        result = compare(expected, actual)
    except TraceError as exc:
        result = {"status": "invalid", "error": str(exc)}
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(f"INVALID TRACE: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, sort_keys=True))
    elif result["status"] == "match":
        print(
            f"MATCH {len(result['layers'])} layers / "
            f"{result['boundaries_compared']} boundaries"
        )
    else:
        print(
            "FIRST DIVERGENCE "
            f"layer={result['layer']} class={result['structural_class']} "
            f"boundary={result['boundary']} index={result['index']} "
            f"expected=0x{result['expected_hex']} "
            f"actual=0x{result['actual_hex']}"
        )
    return 0 if result["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
