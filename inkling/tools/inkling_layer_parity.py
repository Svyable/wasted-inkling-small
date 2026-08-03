#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Run one Inkling decoder layer in C from a bounded parity fixture.

This is the candidate side of layer-level parity. Whole-model parity needs the
whole checkpoint, because layer N's activations depend on layers 0..N-1; a
single layer does not, provided both implementations are handed the *same*
input hidden state. That is what makes fixture-scale parity possible at all.

The flow is:

    fixture (bounded, CRC-checked)  ->  waste_inkling_layer_weights
    fixed input hidden states       ->  waste_inkling_layer_step_backend_trace
    named activations               ->  the CRC-protected archive

The archive names match `inkling_trace.py` exactly — `token.{p}.layer.{L}.{point}`
— so `inkling_parity.py --compare-reference` consumes both sides unchanged.

Source tensor names come from `inkling_plan.py`, not from a second copy of the
naming table.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
from array import array
from pathlib import Path
from typing import Any

from inkling_fixture import Fixture, FixtureError, load_fixture
from inkling_plan import layer_attention_names, layer_dense_mlp_names, layer_sparse_mlp_names

FP = ctypes.POINTER(ctypes.c_float)
IP = ctypes.POINTER(ctypes.c_int)


class LayerParityError(RuntimeError):
    pass


# ---- C structures ---------------------------------------------------------
# Field order and types mirror inkling_layer.h, inkling_attention.h, and
# inkling_config.h. A mismatch here is silent memory corruption rather than an
# error, which is why the tests run the same weights through this binding and
# through a direct one and require identical output: a wrong offset moves the
# numbers.

MAX_LAYERS = 128


class LayerCfg(ctypes.Structure):
    _fields_ = [
        ("is_local", ctypes.c_int), ("num_heads", ctypes.c_int),
        ("num_kv_heads", ctypes.c_int), ("head_dim", ctypes.c_int),
        ("relative_extent", ctypes.c_int),
    ]


class Config(ctypes.Structure):
    _fields_ = [
        ("n_layers", ctypes.c_int), ("hidden", ctypes.c_int),
        ("vocab", ctypes.c_int), ("unpadded_vocab", ctypes.c_int),
        ("max_context", ctypes.c_int),
        ("global_heads", ctypes.c_int), ("global_kv_heads", ctypes.c_int),
        ("global_head_dim", ctypes.c_int),
        ("local_heads", ctypes.c_int), ("local_kv_heads", ctypes.c_int),
        ("local_head_dim", ctypes.c_int),
        ("sliding_window", ctypes.c_int), ("d_rel", ctypes.c_int),
        ("rel_extent", ctypes.c_int), ("conv_kernel", ctypes.c_int),
        ("dense_layers", ctypes.c_int), ("dense_intermediate", ctypes.c_int),
        ("moe_intermediate", ctypes.c_int),
        ("n_routed_experts", ctypes.c_int), ("top_k", ctypes.c_int),
        ("n_shared_experts", ctypes.c_int),
        ("rms_eps", ctypes.c_float), ("route_scale", ctypes.c_float),
        ("logits_width_multiplier", ctypes.c_float),
        ("log_scaling_n_floor", ctypes.c_int),
        ("log_scaling_alpha", ctypes.c_float),
        ("layer", LayerCfg * MAX_LAYERS),
    ]


class ConfigArgs(ctypes.Structure):
    _fields_ = Config._fields_[:-1] + [
        ("local_layer_ids", ctypes.POINTER(ctypes.c_int)),
        ("n_local_layers", ctypes.c_int),
    ]


class AttentionState(ctypes.Structure):
    """Must match waste_inkling_attention_state exactly, field for field.

    An earlier version of this declaration was six fields where the header has
    eleven. Every call to waste_inkling_layer_state_init then wrote 88 bytes
    into a 64-byte buffer: the tests passed, the numbers were right, and the
    process segfaulted during shutdown garbage collection. That is why
    test_struct_layout_matches_c compiles a probe and compares sizes rather
    than trusting this block to have been transcribed correctly.
    """

    _fields_ = [
        ("is_local", ctypes.c_int),
        ("num_heads", ctypes.c_int),
        ("num_kv_heads", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("d_rel", ctypes.c_int),
        ("relative_extent", ctypes.c_int),
        ("capacity", ctypes.c_int),
        ("rms_eps", ctypes.c_float),
        ("next_position", ctypes.c_int),
        ("k_cache", FP),
        ("v_cache", FP),
    ]


class LayerWeights(ctypes.Structure):
    _fields_ = [
        ("input_norm", FP), ("post_attention_norm", FP),
        ("wq", FP), ("wk", FP), ("wv", FP), ("wr", FP), ("wo", FP),
        ("q_norm", FP), ("k_norm", FP), ("relative_proj", FP),
        ("k_sconv", FP), ("v_sconv", FP), ("attn_sconv", FP), ("mlp_sconv", FP),
        ("sparse", ctypes.c_int),
        ("dense_gate", FP), ("dense_up", FP), ("dense_down", FP),
        ("dense_global_scale", FP),
        ("router_weight", FP), ("router_bias", FP), ("router_global_scale", FP),
        ("shared_gate", FP), ("shared_up", FP), ("shared_down", FP),
        ("routed_gate", FP), ("routed_up", FP), ("routed_down", FP),
    ]


class LayerState(ctypes.Structure):
    _fields_ = [
        ("attention", AttentionState),
        ("k_conv_state", FP), ("v_conv_state", FP),
        ("attn_conv_state", FP), ("mlp_conv_state", FP),
    ]


class LayerScratch(ctypes.Structure):
    _fields_ = [
        (name, FP) for name in (
            "norm", "q", "k", "v", "relative", "scores", "attn_out", "branch",
            "gate", "up", "ff", "router_logits", "routed_weight", "shared_weight",
        )
    ] + [
        ("routed_index", IP),
        ("float_count", ctypes.c_size_t),
        ("int_count", ctypes.c_size_t),
    ]


TraceFloat = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_char_p, FP, ctypes.c_size_t)
TraceInt = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                            ctypes.c_char_p, IP, ctypes.c_size_t)


class Trace(ctypes.Structure):
    _fields_ = [("emit_float", TraceFloat), ("emit_int", TraceInt),
                ("ctx", ctypes.c_void_p)]


class MatrixBackend(ctypes.Structure):
    _fields_ = [("matvec", ctypes.c_void_p), ("ctx", ctypes.c_void_p)]


LAYER_SOURCES = ("inkling_layer.c", "inkling_attention.c", "inkling_config.c",
                 "inkling.c")


def build_library(src_dir: Path, out: Path) -> Path:
    """Compile the layer subset into a shared library.

    Deliberately not the whole engine: layer parity needs four translation
    units, and a smaller build fails faster and more legibly.
    """
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        raise LayerParityError("no C compiler available")
    cmd = [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
           f"-I{src_dir}"] + [str(src_dir / name) for name in LAYER_SOURCES] + \
          ["-o", str(out), "-lm"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise LayerParityError(f"library build failed:\n{result.stderr}")
    return out


def configure_library(lib: ctypes.CDLL) -> None:
    lib.waste_inkling_config_build.argtypes = [
        ctypes.POINTER(Config), ctypes.POINTER(ConfigArgs)]
    lib.waste_inkling_config_build.restype = ctypes.c_int
    lib.waste_inkling_layer_scratch_floats.argtypes = [
        ctypes.POINTER(Config), ctypes.c_int, ctypes.c_int]
    lib.waste_inkling_layer_scratch_floats.restype = ctypes.c_size_t
    lib.waste_inkling_layer_scratch_ints.argtypes = [ctypes.POINTER(Config)]
    lib.waste_inkling_layer_scratch_ints.restype = ctypes.c_size_t
    lib.waste_inkling_layer_scratch_init.argtypes = [
        ctypes.POINTER(LayerScratch), ctypes.POINTER(Config), ctypes.c_int,
        ctypes.c_int, FP, ctypes.c_size_t, IP, ctypes.c_size_t]
    lib.waste_inkling_layer_scratch_init.restype = ctypes.c_int
    lib.waste_inkling_layer_state_init.argtypes = [
        ctypes.POINTER(LayerState), ctypes.POINTER(Config), ctypes.c_int,
        ctypes.c_int, FP, FP, FP, FP, FP, FP]
    lib.waste_inkling_layer_state_init.restype = ctypes.c_int
    lib.waste_inkling_layer_step_backend_trace.restype = ctypes.c_int
    lib.waste_inkling_layer_step_backend_trace.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(LayerWeights),
        ctypes.POINTER(MatrixBackend), ctypes.POINTER(LayerState),
        FP, ctypes.c_int, ctypes.POINTER(LayerScratch),
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(Trace),
    ]


# ---- fixture -> C weights -------------------------------------------------

def _carray(values) -> ctypes.Array:
    buf = (ctypes.c_float * len(values))()
    buf[:] = values
    return buf


def _rows(values: array, rows: int, cols: int, index: int) -> array:
    """One [rows][cols] slab out of a [n][rows][cols] tensor."""
    stride = rows * cols
    base = index * stride
    if base + stride > len(values):
        raise LayerParityError("expert slab index is out of range")
    return values[base:base + stride]


def split_fused_gate_up(values: array, intermediate: int, hidden: int) -> tuple[array, array]:
    """Official checkpoints ship gate and up fused as [2*inter][hidden] with the
    two interleaved by row: row 2i is gate row i, row 2i+1 is up row i. This is
    the same transform inkling_trunk.py applies when staging, restated here
    because a fixture holds source bytes, not staged artifacts."""
    if len(values) != 2 * intermediate * hidden:
        raise LayerParityError(
            f"fused gate/up has {len(values)} values, expected {2 * intermediate * hidden}")
    gate = array("f", bytes(4 * intermediate * hidden))
    up = array("f", bytes(4 * intermediate * hidden))
    for row in range(intermediate):
        g0 = (2 * row) * hidden
        u0 = (2 * row + 1) * hidden
        d0 = row * hidden
        gate[d0:d0 + hidden] = values[g0:g0 + hidden]
        up[d0:d0 + hidden] = values[u0:u0 + hidden]
    return gate, up


class LayerBinding:
    """Holds the ctypes buffers alive for as long as the weights struct is used.

    ctypes will happily let a temporary array be garbage collected while the C
    struct still points at it. Every buffer is therefore retained here.
    """

    def __init__(self) -> None:
        self.weights = LayerWeights()
        self._buffers: list[ctypes.Array] = []

    def set(self, field: str, values) -> None:
        buf = _carray(values)
        self._buffers.append(buf)
        setattr(self.weights, field, ctypes.cast(buf, FP))


def bind_layer(fixture: Fixture, cfg: dict[str, Any], layer: int, *,
               dialect: str = "provider_raw",
               experts: list[int] | None = None) -> LayerBinding:
    """Build `waste_inkling_layer_weights` for one layer from a fixture.

    Fails closed: a missing tensor names itself, and a shape that disagrees
    with the config is an error rather than a reinterpretation.
    """
    fixture.require_layers([layer])
    hidden = int(cfg["hidden"])
    is_local = layer in set(cfg.get("local_layer_ids", []))
    heads = int(cfg["local_heads"] if is_local else cfg["global_heads"])
    kv_heads = int(cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"])
    head_dim = int(cfg["local_head_dim"] if is_local else cfg["global_head_dim"])
    extent = int(cfg["sliding_window"] if is_local else cfg["rel_extent"])
    d_rel = int(cfg["d_rel"])
    conv_k = int(cfg["conv_kernel"])
    sparse = layer >= int(cfg["dense_layers"])

    names = layer_attention_names(layer, dialect)
    binding = LayerBinding()

    def take(name: str, expected: int, axis0: int | None = None) -> array:
        values = fixture.values(name, axis0)
        if len(values) != expected:
            raise LayerParityError(
                f"{name} holds {len(values)} values, config implies {expected}")
        return values

    binding.set("input_norm", take(names["input_norm"], hidden))
    binding.set("post_attention_norm", take(names["post_attention_norm"], hidden))
    binding.set("wq", take(names["q"], heads * head_dim * hidden))
    binding.set("wk", take(names["k"], kv_heads * head_dim * hidden))
    binding.set("wv", take(names["v"], kv_heads * head_dim * hidden))
    binding.set("wr", take(names["r"], heads * d_rel * hidden))
    binding.set("wo", take(names["o"], hidden * heads * head_dim))
    binding.set("q_norm", take(names["q_norm"], head_dim))
    binding.set("k_norm", take(names["k_norm"], head_dim))
    binding.set("relative_proj", take(names["rel_proj"], d_rel * extent))
    # Conv1d weights are [channels][1][kernel] at source and [channels][kernel]
    # canonically. The payload is identical; only the declared rank differs.
    binding.set("k_sconv", take(names["k_sconv"], kv_heads * head_dim * conv_k))
    binding.set("v_sconv", take(names["v_sconv"], kv_heads * head_dim * conv_k))
    binding.set("attn_sconv", take(names["attn_sconv"], hidden * conv_k))
    binding.set("mlp_sconv", take(names["mlp_sconv"], hidden * conv_k))
    binding.weights.sparse = int(sparse)

    if not sparse:
        inter = int(cfg["dense_intermediate"])
        mlp = layer_dense_mlp_names(layer, dialect)
        if "fused_gate_up" in mlp:
            gate, up = split_fused_gate_up(
                take(mlp["fused_gate_up"], 2 * inter * hidden), inter, hidden)
        else:
            gate = take(mlp["gate"], inter * hidden)
            up = take(mlp["up"], inter * hidden)
        binding.set("dense_gate", gate)
        binding.set("dense_up", up)
        binding.set("dense_down", take(mlp["down"], hidden * inter))
        binding.set("dense_global_scale", take(mlp["global_scale"], 1))
        return binding

    inter = int(cfg["moe_intermediate"])
    routed_n = int(cfg["n_routed_experts"])
    shared_n = int(cfg["n_shared_experts"])
    names_sparse = layer_sparse_mlp_names(layer, dialect)
    router, shared = names_sparse["router"], names_sparse["shared"]

    binding.set("router_weight", take(router["weight"], (routed_n + shared_n) * hidden))
    binding.set("router_bias", take(router["correction_bias"], routed_n))
    binding.set("router_global_scale", take(router["global_scale"], 1))

    if "fused_gate_up" in shared:
        fused = take(shared["fused_gate_up"], shared_n * 2 * inter * hidden)
        gate = array("f")
        up = array("f")
        for s in range(shared_n):
            g, u = split_fused_gate_up(
                _rows(fused, 2 * inter, hidden, s), inter, hidden)
            gate.extend(g)
            up.extend(u)
    else:
        gate = take(shared["gate"], shared_n * inter * hidden)
        up = take(shared["up"], shared_n * inter * hidden)
    binding.set("shared_gate", gate)
    binding.set("shared_up", up)
    binding.set("shared_down", take(shared["down"], shared_n * hidden * inter))

    # Routed experts. A fixture carries selected axis-0 slices, so the resident
    # arrays are built only for the experts it holds; every other slot is
    # zero-filled and must never be selected. The caller states which experts
    # it expects, and require_experts refuses if the fixture disagrees.
    wanted = sorted(set(experts if experts is not None
                        else fixture.experts.get(layer, ())))
    if not wanted:
        raise LayerParityError(
            f"layer {layer} is sparse but the fixture holds no routed experts for it")
    fixture.require_experts(layer, wanted)

    routed_names = names_sparse["routed"]
    fused_name = routed_names.get("fused_gate_up")
    rg = array("f", bytes(4 * routed_n * inter * hidden))
    ru = array("f", bytes(4 * routed_n * inter * hidden))
    rd = array("f", bytes(4 * routed_n * hidden * inter))
    for expert in wanted:
        fused = fixture.values(fused_name, expert)
        if len(fused) != 2 * inter * hidden:
            raise LayerParityError(
                f"routed expert {expert} gate/up holds {len(fused)} values, "
                f"config implies {2 * inter * hidden}")
        g, u = split_fused_gate_up(fused, inter, hidden)
        down = fixture.values(routed_names["down"], expert)
        if len(down) != hidden * inter:
            raise LayerParityError(
                f"routed expert {expert} down holds {len(down)} values, "
                f"config implies {hidden * inter}")
        off_gu = expert * inter * hidden
        off_d = expert * hidden * inter
        rg[off_gu:off_gu + inter * hidden] = g
        ru[off_gu:off_gu + inter * hidden] = u
        rd[off_d:off_d + hidden * inter] = down
    binding.set("routed_gate", rg)
    binding.set("routed_up", ru)
    binding.set("routed_down", rd)
    return binding


# ---- execution ------------------------------------------------------------

class TraceCollector:
    """Same record names as inkling_trace.py, so one comparator reads both."""

    def __init__(self) -> None:
        self.values: dict[str, list[float] | list[int]] = {}
        self.dtypes: dict[str, str] = {}
        self.position = 0
        self._float_cb = TraceFloat(self._emit_float)
        self._int_cb = TraceInt(self._emit_int)
        self.c_trace = Trace(self._float_cb, self._int_cb, None)

    def _name(self, layer: int, point: bytes) -> str:
        scope = "model" if layer < 0 else f"layer.{layer}"
        return f"token.{self.position}.{scope}.{point.decode('ascii', 'strict')}"

    def _emit_float(self, _ctx, layer, point, data, count) -> int:
        name = self._name(layer, point)
        self.values[name] = [float(data[i]) for i in range(count)]
        self.dtypes[name] = "F32"
        return 0

    def _emit_int(self, _ctx, layer, point, data, count) -> int:
        name = self._name(layer, point)
        self.values[name] = [int(data[i]) for i in range(count)]
        self.dtypes[name] = "I32"
        return 0


def _build_config(lib: ctypes.CDLL, cfg: dict[str, Any]) -> ctypes.Array:
    """Build waste_inkling_config through the C builder rather than packing it
    here. The builder is the tested source of per-layer geometry; a second
    Python implementation of it would be a second thing to get wrong."""
    args = ConfigArgs()
    for field, key in (
        ("n_layers", "n_layers"), ("hidden", "hidden"), ("vocab", "vocab"),
        ("unpadded_vocab", "unpadded_vocab"), ("max_context", "max_context"),
        ("global_heads", "global_heads"), ("global_kv_heads", "global_kv_heads"),
        ("global_head_dim", "global_head_dim"), ("local_heads", "local_heads"),
        ("local_kv_heads", "local_kv_heads"), ("local_head_dim", "local_head_dim"),
        ("sliding_window", "sliding_window"), ("d_rel", "d_rel"),
        ("rel_extent", "rel_extent"), ("conv_kernel", "conv_kernel"),
        ("dense_layers", "dense_layers"), ("dense_intermediate", "dense_intermediate"),
        ("moe_intermediate", "moe_intermediate"),
        ("n_routed_experts", "n_routed_experts"), ("top_k", "top_k"),
        ("n_shared_experts", "n_shared_experts"),
    ):
        setattr(args, field, int(cfg[key]))
    args.rms_eps = float(cfg["rms_eps"])
    args.route_scale = float(cfg["route_scale"])
    args.logits_width_multiplier = float(cfg["logits_width_multiplier"])
    args.log_scaling_n_floor = int(cfg["log_scaling_n_floor"])
    args.log_scaling_alpha = float(cfg["log_scaling_alpha"])
    ids = [int(x) for x in cfg.get("local_layer_ids", [])]
    buf = (ctypes.c_int * max(len(ids), 1))(*ids) if ids else (ctypes.c_int * 1)()
    args.local_layer_ids = ctypes.cast(buf, IP)
    args.n_local_layers = len(ids)
    out = Config()
    if lib.waste_inkling_config_build(ctypes.byref(out), ctypes.byref(args)):
        raise LayerParityError("config rejected by waste_inkling_config_build")
    out._keep = buf
    return out


def run_layer_trace(lib: ctypes.CDLL, cfg: dict[str, Any], layer: int,
                    binding: LayerBinding, inputs: list[list[float]], *,
                    attention_capacity: int = 0) -> dict[str, Any]:
    """Run `layer` over successive input hidden states, tracing each step."""
    config = _build_config(lib, cfg)
    hidden = int(cfg["hidden"])
    is_local = layer in set(cfg.get("local_layer_ids", []))
    kv_heads = int(cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"])
    head_dim = int(cfg["local_head_dim"] if is_local else cfg["global_head_dim"])
    conv_k = int(cfg["conv_kernel"])
    # A local layer's attention ring IS its sliding window: the C state uses
    # `capacity` as the window, so defaulting it to the token count silently
    # turns a windowed layer into a full-context one. That produces plausible
    # numbers that diverge from the official implementation only once the
    # sequence outgrows the window — which is exactly when a parity run would
    # start chasing a phantom bug in the C.
    if attention_capacity:
        cap = attention_capacity
    elif is_local:
        cap = min(max(len(inputs), 1), int(cfg["sliding_window"]))
    else:
        cap = max(len(inputs), 1)

    nfloat = lib.waste_inkling_layer_scratch_floats(ctypes.byref(config), layer, cap)
    nint = lib.waste_inkling_layer_scratch_ints(ctypes.byref(config))
    if not nfloat:
        raise LayerParityError("scratch sizing failed")
    fbuf = (ctypes.c_float * nfloat)()
    ibuf = (ctypes.c_int * max(nint, 1))()
    scratch = LayerScratch()
    if lib.waste_inkling_layer_scratch_init(
            ctypes.byref(scratch), ctypes.byref(config), layer, cap,
            fbuf, ctypes.c_size_t(nfloat), ibuf, ctypes.c_size_t(nint)):
        raise LayerParityError("scratch init failed")

    kv = (ctypes.c_float * (cap * kv_heads * head_dim))()
    vv = (ctypes.c_float * (cap * kv_heads * head_dim))()
    kconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    vconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    aconv = (ctypes.c_float * (hidden * conv_k))()
    mconv = (ctypes.c_float * (hidden * conv_k))()
    state = LayerState()
    if lib.waste_inkling_layer_state_init(
            ctypes.byref(state), ctypes.byref(config), layer, cap,
            kv, vv, kconv, vconv, aconv, mconv):
        raise LayerParityError("state init failed")

    collector = TraceCollector()
    backend = MatrixBackend()
    outputs = []
    for position, row in enumerate(inputs):
        if len(row) != hidden:
            raise LayerParityError(
                f"input {position} has {len(row)} values, hidden is {hidden}")
        collector.position = position
        x = (ctypes.c_float * hidden)(*row)
        rc = lib.waste_inkling_layer_step_backend_trace(
            ctypes.byref(config), layer, ctypes.byref(binding.weights),
            ctypes.byref(backend), ctypes.byref(state), x, position,
            ctypes.byref(scratch), None, None, ctypes.byref(collector.c_trace))
        if rc:
            raise LayerParityError(f"layer step failed at position {position}: {rc}")
        outputs.append([float(x[i]) for i in range(hidden)])
        collector.values[f"token.{position}.layer.{layer}.output"] = outputs[-1]
        collector.dtypes[f"token.{position}.layer.{layer}.output"] = "F32"
    return {"values": collector.values, "dtypes": collector.dtypes,
            "outputs": outputs}


def write_archive(out: Path | str, traced: dict[str, Any],
                  metadata: dict[str, Any]) -> dict[str, Any]:
    """Write the CRC-protected archive inkling_parity.py compares.

    torch is imported here and nowhere else in this module: everything above
    runs without it, so a fixture can be bound and executed on a machine that
    has only a C compiler.
    """
    import torch

    from inkling_parity import write_activation_archive

    values = {}
    for name, data in traced["values"].items():
        dtype = torch.int32 if traced["dtypes"][name] == "I32" else torch.float32
        values[name] = torch.tensor(data, dtype=dtype)
    return write_activation_archive(out, values, metadata=metadata)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="run Inkling decoder layers in C from a bounded parity fixture")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--config", required=True,
                    help="JSON with the normalized config the C builder accepts")
    ap.add_argument("--layers", required=True, help="comma-separated layer ids")
    ap.add_argument("--inputs", required=True,
                    help="JSON list of input hidden states, one per position")
    ap.add_argument("--out", required=True)
    ap.add_argument("--library", help="prebuilt shared library; built if omitted")
    ap.add_argument("--src", default=str(Path(__file__).resolve().parents[1] / "src"))
    ap.add_argument("--dialect", default="provider_raw",
                    choices=("provider_raw", "transformers_normalized"))
    args = ap.parse_args(argv)

    try:
        fixture = load_fixture(args.fixture)
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        layers = [int(x) for x in args.layers.split(",") if x]
        library = Path(args.library) if args.library else build_library(
            Path(args.src), Path(args.out).with_suffix(".so"))
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        merged: dict[str, Any] = {"values": {}, "dtypes": {}}
        for layer in layers:
            binding = bind_layer(fixture, cfg, layer, dialect=args.dialect)
            traced = run_layer_trace(lib, cfg, layer, binding, inputs)
            merged["values"].update(traced["values"])
            merged["dtypes"].update(traced["dtypes"])
        write_archive(args.out, merged, {
            "runtime": "waste-c-layer", "layers": layers,
            "positions": len(inputs), "fixture": str(args.fixture),
        })
    except (FixtureError, LayerParityError, OSError, ValueError, KeyError) as exc:
        ap.error(str(exc))
        return 2
    print(f"wrote C layer trace for layers {layers} to {args.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    raise SystemExit(main())
