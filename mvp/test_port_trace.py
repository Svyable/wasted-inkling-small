#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Model-free contracts for the shared port trace localizer."""
from __future__ import annotations

import copy
import json
import shutil
import struct
import tempfile
from pathlib import Path

import port_trace as trace

MODEL = "synthetic/port"
REVISION = "0123456789abcdef"


def _bits(seed: int, count: int) -> list[int]:
    return [
        ((seed * 257 + i * 37) & 0x7FFF) | (0x8000 if i % 5 == 1 else 0)
        for i in range(count)
    ]


def _pack(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}H", *values)


def _write_trace(root: Path, *, observed_offset: int = 0) -> Path:
    l0_input = _bits(1, 16)
    l0_output = _bits(2, 16)
    l1_output = _bits(3, 16)
    layers = []
    for layer, structural_class, inp, out in (
        (0, "dense", l0_input, l0_output),
        (1, "sparse-local", l0_output, l1_output),
    ):
        layer_dir = f"layer-{layer}"
        boundaries = [
            trace.boundary_manifest(
                root,
                f"{layer_dir}/input.bf16.bin",
                "input",
                "BF16",
                [2, 8],
                _pack(inp),
            ),
            trace.boundary_manifest(
                root,
                f"{layer_dir}/mid.bf16.bin",
                "mid",
                "BF16",
                [8],
                _pack(_bits(10 + layer, 8)),
            ),
            trace.boundary_manifest(
                root,
                f"{layer_dir}/output.bf16.bin",
                "output",
                "BF16",
                [2, 8],
                _pack(out),
            ),
        ]
        item = {
            "layer": layer,
            "structural_class": structural_class,
            "boundaries": boundaries,
        }
        if layer == 1:
            reference = [[7, 9], [3, 5]]
            observed = copy.deepcopy(reference)
            observed[0][0] += observed_offset
            item["routing"] = {
                "observed_route": observed,
                "reference_route": reference,
                "downstream_route_used": "reference",
                "route_equivalence_reason": "synthetic cutoff-tie contract",
            }
        layers.append(item)
    manifest = {
        "schema_version": 1,
        "model": MODEL,
        "revision": REVISION,
        "trace_name": root.name,
        "layers": layers,
    }
    path = root / "trace.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _mutate(
    root: Path, layer: int, boundary: str, index: int, *, update_sha: bool
) -> None:
    path = root / f"layer-{layer}/{boundary}.bf16.bin"
    payload = bytearray(path.read_bytes())
    payload[index * 2] ^= 1
    path.write_bytes(payload)
    manifest_path = root / "trace.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(
        value
        for item in manifest["layers"]
        if item["layer"] == layer
        for value in item["boundaries"]
        if value["name"] == boundary
    )
    if update_sha:
        entry["sha256"] = trace.sha256_file(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="port-trace-") as directory:
        base = Path(directory)

        expected_root = base / "expected"
        actual_root = base / "actual"
        expected_root.mkdir()
        expected_path = _write_trace(expected_root)
        shutil.copytree(expected_root, actual_root)
        actual_path = actual_root / "trace.json"
        result = trace.compare(trace.load_trace(expected_path), trace.load_trace(actual_path))
        assert result["status"] == "match"
        assert result["layers"] == [0, 1]
        assert result["boundaries_compared"] == 6

        early = base / "early"
        shutil.copytree(expected_root, early)
        _mutate(early, 1, "output", 7, update_sha=True)
        _mutate(early, 0, "mid", 3, update_sha=True)
        result = trace.compare(
            trace.load_trace(expected_path), trace.load_trace(early / "trace.json")
        )
        assert (result["layer"], result["boundary"], result["index"]) == (
            0,
            "mid",
            3,
        )

        stale = base / "stale"
        shutil.copytree(expected_root, stale)
        _mutate(stale, 1, "mid", 2, update_sha=False)
        try:
            trace.load_trace(stale / "trace.json")
        except trace.TraceError as exc:
            assert "file SHA" in str(exc)
        else:
            raise AssertionError("stale manifest hash was accepted")

        broken = base / "broken"
        shutil.copytree(expected_root, broken)
        _mutate(broken, 1, "input", 0, update_sha=True)
        try:
            trace.load_trace(broken / "trace.json")
        except trace.TraceError as exc:
            assert "chain break" in str(exc)
        else:
            raise AssertionError("broken output->input chain was accepted")

        incompatible = base / "incompatible"
        shutil.copytree(expected_root, incompatible)
        manifest_path = incompatible / "trace.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["layers"][1]["structural_class"] = "global"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        try:
            trace.compare(trace.load_trace(expected_path), trace.load_trace(manifest_path))
        except trace.TraceError as exc:
            assert "structural class mismatch" in str(exc)
        else:
            raise AssertionError("structural-class mismatch was compared")

        route_mismatch = base / "route-mismatch"
        shutil.copytree(expected_root, route_mismatch)
        manifest_path = route_mismatch / "trace.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["layers"][1]["routing"]["reference_route"][0][0] = 99
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        try:
            trace.compare(trace.load_trace(expected_path), trace.load_trace(manifest_path))
        except trace.TraceError as exc:
            assert "reference route mismatch" in str(exc)
        else:
            raise AssertionError("reference-route mismatch was compared")

        # Raw observed routes are intentionally allowed to differ when the
        # reference/downstream contract is unchanged.
        observed = base / "observed-route-diff"
        observed.mkdir()
        observed_path = _write_trace(observed, observed_offset=100)
        result = trace.compare(trace.load_trace(expected_path), trace.load_trace(observed_path))
        assert result["status"] == "match"

    print(
        "PASS shared port trace: integrity, exact chaining, routing contract, "
        "compatibility and first-divergence semantics"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
