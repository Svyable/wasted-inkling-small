#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run compact C layers while canonicalizing ambiguous routed expert choices.

This is evidence tooling, not runtime behavior. The collector preserves the
C-selected pairs under ``candidate_*`` trace names, then substitutes a trusted
official pair set after routing and before expert lookup. Numerical parity can
therefore compare identical expert paths while a separate gate audits whether
the original C choices are exact or valid members of an ambiguous BF16 cutoff.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "inkling" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from discover_inkling_router_experts import input_sha256
from inkling_compact_layer_parity import (
    CompactBinding,
    bind_sparse_layer_compact,
)
from inkling_fixture import Fixture, FixtureError, load_fixture
from inkling_layer_parity import (
    LayerParityError,
    LayerScratch,
    LayerState,
    MatrixBackend,
    TraceCollector,
    _build_config,
    bind_layer,
    build_library,
    configure_library,
    run_layer_trace,
    write_archive,
)


@dataclass(frozen=True)
class RouteRow:
    indices: tuple[int, ...]
    weights: tuple[float, ...]


class CanonicalRouteError(RuntimeError):
    """A canonical route cannot be applied without weakening its binding."""


def load_canonical_routes(
    path: Path | str,
    fixture: Fixture,
    inputs: list[list[float]],
    layers: list[int],
) -> dict[int, list[RouteRow]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "model_id": fixture.model_id,
        "revision": fixture.source.get("revision"),
        "config_sha256": fixture.source.get("config_sha256"),
        "index_sha256": fixture.source.get("index_sha256"),
        "tokens": len(inputs),
        "input_dtype": "bfloat16",
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise CanonicalRouteError(
                f"route override {key}={data.get(key)!r}; expected {value!r}"
            )
    hidden = torch.tensor(inputs, dtype=torch.float32)
    if data.get("inputs_sha256") != input_sha256(hidden, torch.bfloat16):
        raise CanonicalRouteError("route override input hash differs from --inputs")

    out: dict[int, list[RouteRow]] = {}
    layer_data = data.get("layers")
    if not isinstance(layer_data, dict):
        raise CanonicalRouteError("route override has no layer mapping")
    for layer in layers:
        if layer < 0:
            raise CanonicalRouteError(f"invalid layer {layer}")
        if str(layer) not in layer_data:
            raise CanonicalRouteError(f"route override omits layer {layer}")
        value = layer_data[str(layer)]
        ids = value.get("topk_indices")
        weights = value.get("topk_weights")
        if not isinstance(ids, list) or not isinstance(weights, list):
            raise CanonicalRouteError(
                f"route override layer {layer} omits routed pairs"
            )
        if len(ids) != len(inputs) or len(weights) != len(inputs):
            raise CanonicalRouteError(
                f"route override layer {layer} has {len(ids)} ID rows and "
                f"{len(weights)} weight rows; expected {len(inputs)}"
            )
        rows = []
        width = None
        for position, (row_ids, row_weights) in enumerate(zip(ids, weights)):
            if not isinstance(row_ids, list) or not isinstance(row_weights, list):
                raise CanonicalRouteError(
                    f"route override layer {layer} position {position} is not a pair row"
                )
            if len(row_ids) != len(row_weights) or not row_ids:
                raise CanonicalRouteError(
                    f"route override layer {layer} position {position} has unpaired values"
                )
            if width is None:
                width = len(row_ids)
            if len(row_ids) != width:
                raise CanonicalRouteError(
                    f"route override layer {layer} changes top-k width"
                )
            parsed_ids = tuple(int(item) for item in row_ids)
            if len(set(parsed_ids)) != len(parsed_ids):
                raise CanonicalRouteError(
                    f"route override layer {layer} position {position} repeats an expert"
                )
            rows.append(
                RouteRow(parsed_ids, tuple(float(item) for item in row_weights))
            )
        out[layer] = rows
    return out


class CanonicalRouteCollector(TraceCollector):
    """Preserve candidate route traces, then mutate scratch to official pairs."""

    def __init__(self, rows: list[RouteRow]) -> None:
        self.rows = rows
        self.applied_positions: set[int] = set()
        self.error: str | None = None
        super().__init__()

    def _row(self, count: int) -> RouteRow | None:
        if self.position >= len(self.rows):
            self.error = f"canonical route has no position {self.position}"
            return None
        row = self.rows[self.position]
        if len(row.indices) != int(count):
            self.error = (
                f"canonical route position {self.position} has {len(row.indices)} "
                f"pairs; C emitted {int(count)}"
            )
            return None
        return row

    def _emit_int(self, ctx, layer, point, data, count) -> int:
        decoded = point.decode("ascii", "strict")
        if decoded != "routed_index":
            return super()._emit_int(ctx, layer, point, data, count)
        row = self._row(int(count))
        if row is None:
            return 1
        candidate_name = self._name(layer, b"candidate_routed_index")
        self.values[candidate_name] = [int(data[index]) for index in range(count)]
        self.dtypes[candidate_name] = "I32"
        for index, value in enumerate(row.indices):
            data[index] = value
        self.applied_positions.add(self.position)
        return super()._emit_int(ctx, layer, point, data, count)

    def _emit_float(self, ctx, layer, point, data, count) -> int:
        decoded = point.decode("ascii", "strict")
        if decoded != "routed_weight":
            return super()._emit_float(ctx, layer, point, data, count)
        row = self._row(int(count))
        if row is None:
            return 1
        candidate_name = self._name(layer, b"candidate_routed_weight")
        self.values[candidate_name] = [float(data[index]) for index in range(count)]
        self.dtypes[candidate_name] = "F32"
        for index, value in enumerate(row.weights):
            data[index] = value
        return super()._emit_float(ctx, layer, point, data, count)


def run_sparse_layer_canonical(
    lib,
    cfg: dict[str, Any],
    layer: int,
    compact: CompactBinding,
    inputs: list[list[float]],
    rows: list[RouteRow],
) -> dict[str, Any]:
    if compact.provider is None:
        raise CanonicalRouteError("canonical route runner requires sparse experts")
    available = set(compact.provider.experts)
    for position, row in enumerate(rows):
        missing = sorted(set(row.indices).difference(available))
        if missing:
            raise CanonicalRouteError(
                f"route override layer {layer} position {position} requests "
                f"fixture-missing experts {missing}"
            )

    config = _build_config(lib, cfg)
    hidden = int(cfg["hidden"])
    is_local = layer in set(cfg.get("local_layer_ids", []))
    kv_heads = int(
        cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"]
    )
    head_dim = int(
        cfg["local_head_dim"] if is_local else cfg["global_head_dim"]
    )
    conv_k = int(cfg["conv_kernel"])
    cap = (
        min(max(len(inputs), 1), int(cfg["sliding_window"]))
        if is_local
        else max(len(inputs), 1)
    )
    nfloat = lib.waste_inkling_layer_scratch_floats(
        ctypes.byref(config), layer, cap
    )
    nint = lib.waste_inkling_layer_scratch_ints(ctypes.byref(config))
    if not nfloat:
        raise CanonicalRouteError("scratch sizing failed")
    fbuf = (ctypes.c_float * nfloat)()
    ibuf = (ctypes.c_int * max(nint, 1))()
    scratch = LayerScratch()
    if lib.waste_inkling_layer_scratch_init(
        ctypes.byref(scratch),
        ctypes.byref(config),
        layer,
        cap,
        fbuf,
        ctypes.c_size_t(nfloat),
        ibuf,
        ctypes.c_size_t(nint),
    ):
        raise CanonicalRouteError("scratch init failed")
    kv = (ctypes.c_float * (cap * kv_heads * head_dim))()
    vv = (ctypes.c_float * (cap * kv_heads * head_dim))()
    kconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    vconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    aconv = (ctypes.c_float * (hidden * conv_k))()
    mconv = (ctypes.c_float * (hidden * conv_k))()
    state = LayerState()
    if lib.waste_inkling_layer_state_init(
        ctypes.byref(state),
        ctypes.byref(config),
        layer,
        cap,
        kv,
        vv,
        kconv,
        vconv,
        aconv,
        mconv,
    ):
        raise CanonicalRouteError("state init failed")

    collector = CanonicalRouteCollector(rows)
    backend = MatrixBackend()
    outputs = []
    expert_get = ctypes.cast(compact.provider.callback, ctypes.c_void_p)
    for position, values in enumerate(inputs):
        if len(values) != hidden:
            raise CanonicalRouteError(
                f"input {position} has {len(values)} values; hidden is {hidden}"
            )
        collector.position = position
        x = (ctypes.c_float * hidden)(*values)
        rc = lib.waste_inkling_layer_step_backend_trace(
            ctypes.byref(config),
            layer,
            ctypes.byref(compact.weights),
            ctypes.byref(backend),
            ctypes.byref(state),
            x,
            position,
            ctypes.byref(scratch),
            expert_get,
            None,
            ctypes.byref(collector.c_trace),
        )
        if collector.error:
            raise CanonicalRouteError(collector.error)
        if rc:
            raise CanonicalRouteError(
                f"canonical compact layer failed at position {position}: {rc}"
            )
        outputs.append([float(x[index]) for index in range(hidden)])
        name = f"token.{position}.layer.{layer}.output"
        collector.values[name] = outputs[-1]
        collector.dtypes[name] = "F32"
    expected_positions = set(range(len(inputs)))
    if collector.applied_positions != expected_positions:
        raise CanonicalRouteError(
            "canonical route was not applied at every position"
        )
    return {
        "values": collector.values,
        "dtypes": collector.dtypes,
        "outputs": outputs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--route-override", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--library")
    ap.add_argument("--src", default=str(REPO / "inkling" / "src"))
    args = ap.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise CanonicalRouteError("--layers must be nonempty")
        sparse_layers = [
            layer for layer in layers if layer >= int(cfg["dense_layers"])
        ]
        routes = load_canonical_routes(
            args.route_override, fixture, inputs, sparse_layers
        )
        library = Path(args.library) if args.library else build_library(
            Path(args.src), Path(args.out).with_suffix(".so")
        )
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        merged: dict[str, Any] = {"values": {}, "dtypes": {}}
        compact_bytes: dict[str, int] = {}
        for layer in layers:
            if layer < int(cfg["dense_layers"]):
                traced = run_layer_trace(
                    lib, cfg, layer, bind_layer(fixture, cfg, layer), inputs
                )
                compact_bytes[str(layer)] = 0
            else:
                compact = bind_sparse_layer_compact(fixture, cfg, layer)
                traced = run_sparse_layer_canonical(
                    lib, cfg, layer, compact, inputs, routes[layer]
                )
                compact_bytes[str(layer)] = compact.compact_expert_bytes
            merged["values"].update(traced["values"])
            merged["dtypes"].update(traced["dtypes"])
        write_archive(
            args.out,
            merged,
            {
                "runtime": "waste-c-layer-canonical-official-route",
                "layers": layers,
                "positions": len(inputs),
                "fixture": str(args.fixture),
                "route_override": str(args.route_override),
                "compact_expert_bytes": compact_bytes,
            },
        )
    except (
        FixtureError,
        LayerParityError,
        CanonicalRouteError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        ap.error(str(exc))
        return 2
    print(
        f"wrote canonical-route C layer trace for layers {layers} to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
