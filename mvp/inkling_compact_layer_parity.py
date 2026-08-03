#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the C layer harness with compact selected routed experts.

The repository's original fixture binder expands selected experts into three
zero-filled ``n_routed_experts`` float32 banks. That is useful for tiny tests
but is not bounded for Inkling-Small. This MVP runner leaves resident routed
pointers null and serves only fixture-selected experts through the C runtime's
existing ``expert_get`` callback.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "inkling" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from inkling_fixture import Fixture, FixtureError, load_fixture
from inkling_layer_parity import (
    FP,
    IP,
    LayerBinding,
    LayerParityError,
    LayerScratch,
    LayerState,
    MatrixBackend,
    TraceCollector,
    _build_config,
    _carray,
    build_library,
    configure_library,
    run_layer_trace,
    split_fused_gate_up,
    write_archive,
)
from inkling_plan import (
    layer_attention_names,
    layer_sparse_mlp_names,
)


class ExpertWeights(ctypes.Structure):
    _fields_ = [("gate", FP), ("up", FP), ("down", FP)]


ExpertGet = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ExpertWeights),
)


class CompactExpertProvider:
    """Own selected expert arrays and expose them through ``expert_get``."""

    def __init__(self, layer: int) -> None:
        self.layer = layer
        self._experts: dict[int, tuple[ctypes.Array, ctypes.Array, ctypes.Array]] = {}
        self.callback = ExpertGet(self._get)
        self.bytes = 0

    def add(self, expert: int, gate: Any, up: Any, down: Any) -> None:
        if expert in self._experts:
            raise LayerParityError(f"duplicate compact expert {expert}")
        buffers = (_carray(gate), _carray(up), _carray(down))
        self._experts[expert] = buffers
        self.bytes += sum(ctypes.sizeof(value) for value in buffers)

    def _get(self, _ctx, layer: int, expert: int, out) -> int:
        if layer != self.layer or not out or expert not in self._experts:
            return -1
        gate, up, down = self._experts[expert]
        out.contents.gate = ctypes.cast(gate, FP)
        out.contents.up = ctypes.cast(up, FP)
        out.contents.down = ctypes.cast(down, FP)
        return 0

    @property
    def experts(self) -> list[int]:
        return sorted(self._experts)


class CompactBinding:
    def __init__(self, binding: LayerBinding, provider: CompactExpertProvider | None) -> None:
        self.binding = binding
        self.provider = provider

    @property
    def weights(self):
        return self.binding.weights

    @property
    def compact_expert_bytes(self) -> int:
        return self.provider.bytes if self.provider else 0


def _rows(values, rows: int, cols: int, index: int):
    stride = rows * cols
    start = index * stride
    stop = start + stride
    if stop > len(values):
        raise LayerParityError("shared expert slab index is out of range")
    return values[start:stop]


def bind_sparse_layer_compact(
    fixture: Fixture,
    cfg: dict[str, Any],
    layer: int,
    *,
    dialect: str = "provider_raw",
) -> CompactBinding:
    """Bind one sparse layer without allocating absent routed experts."""
    fixture.require_layers([layer])
    if layer < int(cfg["dense_layers"]):
        raise LayerParityError(f"layer {layer} is dense, not sparse")
    hidden = int(cfg["hidden"])
    is_local = layer in set(cfg.get("local_layer_ids", []))
    heads = int(cfg["local_heads"] if is_local else cfg["global_heads"])
    kv_heads = int(cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"])
    head_dim = int(cfg["local_head_dim"] if is_local else cfg["global_head_dim"])
    extent = int(cfg["sliding_window"] if is_local else cfg["rel_extent"])
    d_rel = int(cfg["d_rel"])
    conv_k = int(cfg["conv_kernel"])
    inter = int(cfg["moe_intermediate"])
    routed_n = int(cfg["n_routed_experts"])
    shared_n = int(cfg["n_shared_experts"])

    binding = LayerBinding()

    def take(name: str, expected: int, axis0: int | None = None):
        values = fixture.values(name, axis0)
        if len(values) != expected:
            raise LayerParityError(
                f"{name} holds {len(values)} values, config implies {expected}"
            )
        return values

    attention = layer_attention_names(layer, dialect)
    binding.set("input_norm", take(attention["input_norm"], hidden))
    binding.set("post_attention_norm", take(attention["post_attention_norm"], hidden))
    binding.set("wq", take(attention["q"], heads * head_dim * hidden))
    binding.set("wk", take(attention["k"], kv_heads * head_dim * hidden))
    binding.set("wv", take(attention["v"], kv_heads * head_dim * hidden))
    binding.set("wr", take(attention["r"], heads * d_rel * hidden))
    binding.set("wo", take(attention["o"], hidden * heads * head_dim))
    binding.set("q_norm", take(attention["q_norm"], head_dim))
    binding.set("k_norm", take(attention["k_norm"], head_dim))
    binding.set("relative_proj", take(attention["rel_proj"], d_rel * extent))
    binding.set("k_sconv", take(attention["k_sconv"], kv_heads * head_dim * conv_k))
    binding.set("v_sconv", take(attention["v_sconv"], kv_heads * head_dim * conv_k))
    binding.set("attn_sconv", take(attention["attn_sconv"], hidden * conv_k))
    binding.set("mlp_sconv", take(attention["mlp_sconv"], hidden * conv_k))
    binding.weights.sparse = 1

    sparse = layer_sparse_mlp_names(layer, dialect)
    router = sparse["router"]
    shared = sparse["shared"]
    binding.set(
        "router_weight",
        take(router["weight"], (routed_n + shared_n) * hidden),
    )
    binding.set("router_bias", take(router["correction_bias"], routed_n))
    binding.set("router_global_scale", take(router["global_scale"], 1))

    if "fused_gate_up" in shared:
        fused = take(shared["fused_gate_up"], shared_n * 2 * inter * hidden)
        gate = []
        up = []
        for shared_id in range(shared_n):
            one_gate, one_up = split_fused_gate_up(
                _rows(fused, 2 * inter, hidden, shared_id), inter, hidden
            )
            gate.extend(one_gate)
            up.extend(one_up)
    else:
        gate = take(shared["gate"], shared_n * inter * hidden)
        up = take(shared["up"], shared_n * inter * hidden)
    binding.set("shared_gate", gate)
    binding.set("shared_up", up)
    binding.set("shared_down", take(shared["down"], shared_n * hidden * inter))

    wanted = sorted(set(fixture.experts.get(layer, ())))
    if not wanted:
        raise LayerParityError(f"layer {layer} fixture holds no routed experts")
    fixture.require_experts(layer, wanted)
    provider = CompactExpertProvider(layer)
    routed = sparse["routed"]
    fused_name = routed.get("fused_gate_up")
    if not fused_name:
        raise LayerParityError("compact provider requires provider-raw fused experts")
    for expert in wanted:
        fused = take(fused_name, 2 * inter * hidden, expert)
        expert_gate, expert_up = split_fused_gate_up(fused, inter, hidden)
        expert_down = take(routed["down"], hidden * inter, expert)
        provider.add(expert, expert_gate, expert_up, expert_down)

    # routed_gate/up/down remain NULL. C must consult provider.callback.
    return CompactBinding(binding, provider)


def run_layer_trace_compact(
    lib,
    cfg: dict[str, Any],
    layer: int,
    compact: CompactBinding,
    inputs: list[list[float]],
    *,
    attention_capacity: int = 0,
) -> dict[str, Any]:
    if compact.provider is None:
        return run_layer_trace(
            lib,
            cfg,
            layer,
            compact.binding,
            inputs,
            attention_capacity=attention_capacity,
        )
    config = _build_config(lib, cfg)
    hidden = int(cfg["hidden"])
    is_local = layer in set(cfg.get("local_layer_ids", []))
    kv_heads = int(cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"])
    head_dim = int(cfg["local_head_dim"] if is_local else cfg["global_head_dim"])
    conv_k = int(cfg["conv_kernel"])
    if attention_capacity:
        cap = attention_capacity
    elif is_local:
        cap = min(max(len(inputs), 1), int(cfg["sliding_window"]))
    else:
        cap = max(len(inputs), 1)

    nfloat = lib.waste_inkling_layer_scratch_floats(
        ctypes.byref(config), layer, cap
    )
    nint = lib.waste_inkling_layer_scratch_ints(ctypes.byref(config))
    if not nfloat:
        raise LayerParityError("scratch sizing failed")
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
        raise LayerParityError("scratch init failed")

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
        raise LayerParityError("state init failed")

    collector = TraceCollector()
    backend = MatrixBackend()
    outputs = []
    expert_get = ctypes.cast(compact.provider.callback, ctypes.c_void_p)
    for position, row in enumerate(inputs):
        if len(row) != hidden:
            raise LayerParityError(
                f"input {position} has {len(row)} values, hidden is {hidden}"
            )
        collector.position = position
        x = (ctypes.c_float * hidden)(*row)
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
        if rc:
            raise LayerParityError(f"compact layer step failed at position {position}: {rc}")
        outputs.append([float(x[index]) for index in range(hidden)])
        name = f"token.{position}.layer.{layer}.output"
        collector.values[name] = outputs[-1]
        collector.dtypes[name] = "F32"
    return {
        "values": collector.values,
        "dtypes": collector.dtypes,
        "outputs": outputs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--inputs", required=True)
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
            raise LayerParityError("--layers must be nonempty")
        library = Path(args.library) if args.library else build_library(
            Path(args.src), Path(args.out).with_suffix(".so")
        )
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        merged: dict[str, Any] = {"values": {}, "dtypes": {}}
        compact_bytes: dict[str, int] = {}
        for layer in layers:
            if layer < int(cfg["dense_layers"]):
                from inkling_layer_parity import bind_layer

                compact = CompactBinding(bind_layer(fixture, cfg, layer), None)
            else:
                compact = bind_sparse_layer_compact(fixture, cfg, layer)
            traced = run_layer_trace_compact(lib, cfg, layer, compact, inputs)
            merged["values"].update(traced["values"])
            merged["dtypes"].update(traced["dtypes"])
            compact_bytes[str(layer)] = compact.compact_expert_bytes
        write_archive(args.out, merged, {
            "runtime": "waste-c-layer-compact-experts",
            "layers": layers,
            "positions": len(inputs),
            "fixture": str(args.fixture),
            "compact_expert_bytes": compact_bytes,
        })
    except (FixtureError, LayerParityError, OSError, ValueError, KeyError) as exc:
        ap.error(str(exc))
        return 2
    print(
        f"wrote compact C layer trace for layers {layers} to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
