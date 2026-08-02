#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.

import ctypes
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from test_inkling_attention_c import State as AttentionState, rmsnorm, reference_step
from test_inkling_config_c import Args, Config

FP = ctypes.POINTER(ctypes.c_float)


class Weights(ctypes.Structure):
    _fields_ = [
        ("input_norm", FP),
        ("post_attention_norm", FP),
        ("wq", FP),
        ("wk", FP),
        ("wv", FP),
        ("wr", FP),
        ("wo", FP),
        ("q_norm", FP),
        ("k_norm", FP),
        ("relative_proj", FP),
        ("k_sconv", FP),
        ("v_sconv", FP),
        ("attn_sconv", FP),
        ("mlp_sconv", FP),
        ("sparse", ctypes.c_int),
        ("dense_gate", FP),
        ("dense_up", FP),
        ("dense_down", FP),
        ("dense_global_scale", FP),
        ("router_weight", FP),
        ("router_bias", FP),
        ("router_global_scale", FP),
        ("shared_gate", FP),
        ("shared_up", FP),
        ("shared_down", FP),
        ("routed_gate", FP),
        ("routed_up", FP),
        ("routed_down", FP),
    ]


class ExpertWeights(ctypes.Structure):
    _fields_ = [("gate", FP), ("up", FP), ("down", FP)]


ExpertCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ExpertWeights),
)


class LayerState(ctypes.Structure):
    _fields_ = [
        ("attention", AttentionState),
        ("k_conv_state", FP),
        ("v_conv_state", FP),
        ("attn_conv_state", FP),
        ("mlp_conv_state", FP),
    ]


class Scratch(ctypes.Structure):
    _fields_ = [
        (name, FP)
        for name in (
            "norm",
            "q",
            "k",
            "v",
            "relative",
            "scores",
            "attn_out",
            "branch",
            "gate",
            "up",
            "ff",
            "router_logits",
            "routed_weight",
            "shared_weight",
        )
    ] + [
        ("routed_index", ctypes.POINTER(ctypes.c_int)),
        ("float_count", ctypes.c_size_t),
        ("int_count", ctypes.c_size_t),
    ]


def array(tensor):
    values = torch.as_tensor(tensor, dtype=torch.float32).contiguous().view(-1).tolist()
    return (ctypes.c_float * len(values))(*values)


def tensor_from_array(values, shape):
    return torch.tensor(list(values), dtype=torch.float32).reshape(shape)


def sconv_step(x, state, weight):
    state[:, :-1] = state[:, 1:].clone()
    state[:, -1] = x
    return x + (state * weight).sum(dim=-1)


def expert(gate_w, up_w, down_w, x):
    return down_w @ (F.silu(gate_w @ x) * (up_w @ x))


class InklingLayerCDifferentialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shutil.which("cc")
        if not cc:
            raise unittest.SkipTest("C compiler unavailable")
        cls.td = tempfile.TemporaryDirectory()
        lib_path = Path(cls.td.name) / "libinkling_layer.so"
        subprocess.run(
            [
                cc,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-shared",
                "-fPIC",
                f"-I{REPO / 'src'}",
                str(REPO / "src" / "inkling_layer.c"),
                str(REPO / "src" / "inkling_attention.c"),
                str(REPO / "src" / "inkling_config.c"),
                str(REPO / "src" / "inkling.c"),
                "-lm",
                "-o",
                str(lib_path),
            ],
            check=True,
            capture_output=True,
        )
        cls.lib = ctypes.CDLL(str(lib_path))
        cls.lib.waste_inkling_config_build.argtypes = [ctypes.POINTER(Config), ctypes.POINTER(Args)]
        cls.lib.waste_inkling_config_build.restype = ctypes.c_int
        cls.lib.waste_inkling_layer_scratch_floats.argtypes = [
            ctypes.POINTER(Config), ctypes.c_int, ctypes.c_int
        ]
        cls.lib.waste_inkling_layer_scratch_floats.restype = ctypes.c_size_t
        cls.lib.waste_inkling_layer_scratch_ints.argtypes = [ctypes.POINTER(Config)]
        cls.lib.waste_inkling_layer_scratch_ints.restype = ctypes.c_size_t
        cls.lib.waste_inkling_layer_scratch_init.argtypes = [
            ctypes.POINTER(Scratch), ctypes.POINTER(Config), ctypes.c_int, ctypes.c_int,
            FP, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
        ]
        cls.lib.waste_inkling_layer_scratch_init.restype = ctypes.c_int
        cls.lib.waste_inkling_layer_state_init.argtypes = [
            ctypes.POINTER(LayerState), ctypes.POINTER(Config), ctypes.c_int, ctypes.c_int,
            FP, FP, FP, FP, FP, FP,
        ]
        cls.lib.waste_inkling_layer_state_init.restype = ctypes.c_int
        cls.lib.waste_inkling_layer_step.argtypes = [
            ctypes.POINTER(Config), ctypes.c_int, ctypes.POINTER(Weights),
            ctypes.POINTER(LayerState), FP, ctypes.c_int, ctypes.POINTER(Scratch),
            ExpertCallback, ctypes.c_void_p,
        ]
        cls.lib.waste_inkling_layer_step.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def build_config(self):
        local = (ctypes.c_int * 1)(0)
        a = Args()
        a.n_layers = 2
        a.hidden = 8
        a.vocab = 32
        a.unpadded_vocab = 30
        a.max_context = 16
        a.global_heads = 2
        a.global_kv_heads = 1
        a.global_head_dim = 4
        a.local_heads = 2
        a.local_kv_heads = 1
        a.local_head_dim = 4
        a.sliding_window = 3
        a.d_rel = 2
        a.rel_extent = 5
        a.conv_kernel = 3
        a.dense_layers = 1
        a.dense_intermediate = 12
        a.moe_intermediate = 6
        a.n_routed_experts = 3
        a.top_k = 2
        a.n_shared_experts = 1
        a.rms_eps = 1e-6
        a.route_scale = 2.0
        a.logits_width_multiplier = 2.0
        a.log_scaling_n_floor = 2
        a.log_scaling_alpha = 0.2
        a.local_layer_ids = local
        a.n_local_layers = 1
        cfg = Config()
        self.assertEqual(self.lib.waste_inkling_config_build(ctypes.byref(cfg), ctypes.byref(a)), 0)
        return cfg, local

    def make_common(self, layer, cfg):
        torch.manual_seed(100 + layer)
        l = cfg.layer[layer]
        h = cfg.hidden
        qdim = l.num_heads * l.head_dim
        kdim = l.num_kv_heads * l.head_dim
        rdim = l.num_heads * cfg.d_rel
        extent = l.relative_extent
        k = cfg.conv_kernel
        tensors = {
            "input_norm": torch.rand(h) + 0.5,
            "post_attention_norm": torch.rand(h) + 0.5,
            "wq": torch.randn(qdim, h) * 0.15,
            "wk": torch.randn(kdim, h) * 0.15,
            "wv": torch.randn(kdim, h) * 0.15,
            "wr": torch.randn(rdim, h) * 0.15,
            "wo": torch.randn(h, qdim) * 0.15,
            "q_norm": torch.rand(l.head_dim) + 0.5,
            "k_norm": torch.rand(l.head_dim) + 0.5,
            "relative_proj": torch.randn(cfg.d_rel, extent) * 0.1,
            "k_sconv": torch.randn(kdim, k) * 0.05,
            "v_sconv": torch.randn(kdim, k) * 0.05,
            "attn_sconv": torch.randn(h, k) * 0.05,
            "mlp_sconv": torch.randn(h, k) * 0.05,
        }
        keep = {name: array(value) for name, value in tensors.items()}
        w = Weights()
        for name, value in keep.items():
            setattr(w, name, value)
        return w, tensors, keep

    def init_runtime(self, cfg, layer, capacity):
        l = cfg.layer[layer]
        kdim = l.num_kv_heads * l.head_dim
        kcache = (ctypes.c_float * (capacity * kdim))()
        vcache = (ctypes.c_float * (capacity * kdim))()
        kconv = (ctypes.c_float * (kdim * cfg.conv_kernel))()
        vconv = (ctypes.c_float * (kdim * cfg.conv_kernel))()
        aconv = (ctypes.c_float * (cfg.hidden * cfg.conv_kernel))()
        mconv = (ctypes.c_float * (cfg.hidden * cfg.conv_kernel))()
        state = LayerState()
        self.assertEqual(
            self.lib.waste_inkling_layer_state_init(
                ctypes.byref(state), ctypes.byref(cfg), layer, capacity,
                kcache, vcache, kconv, vconv, aconv, mconv,
            ),
            0,
        )
        nf = self.lib.waste_inkling_layer_scratch_floats(ctypes.byref(cfg), layer, capacity)
        ni = self.lib.waste_inkling_layer_scratch_ints(ctypes.byref(cfg))
        fbuf = (ctypes.c_float * nf)()
        ibuf = (ctypes.c_int * ni)()
        scratch = Scratch()
        self.assertEqual(
            self.lib.waste_inkling_layer_scratch_init(
                ctypes.byref(scratch), ctypes.byref(cfg), layer, capacity,
                fbuf, nf, ibuf, ni,
            ),
            0,
        )
        return state, scratch, (kcache, vcache, kconv, vconv, aconv, mconv, fbuf, ibuf)

    def reference_layer(self, cfg, layer, tensors, extra, x, position, py_state):
        l = cfg.layer[layer]
        norm = rmsnorm(x, tensors["input_norm"], cfg.rms_eps)
        q = (tensors["wq"] @ norm).reshape(l.num_heads, l.head_dim)
        k = (tensors["wk"] @ norm).reshape(l.num_kv_heads, l.head_dim)
        v = (tensors["wv"] @ norm).reshape(l.num_kv_heads, l.head_dim)
        rel = (tensors["wr"] @ norm).reshape(l.num_heads, cfg.d_rel)
        k = sconv_step(k.reshape(-1), py_state["kconv"], tensors["k_sconv"]).reshape_as(k)
        v = sconv_step(v.reshape(-1), py_state["vconv"], tensors["v_sconv"]).reshape_as(v)
        attn = reference_step(
            q, k, v, tensors["q_norm"], tensors["k_norm"], rel,
            tensors["relative_proj"], position, bool(l.is_local),
            py_state["capacity"], py_state["keys"], py_state["values"],
            0 if l.is_local else cfg.log_scaling_n_floor,
            cfg.log_scaling_alpha, cfg.rms_eps,
        ).reshape(-1)
        branch = tensors["wo"] @ attn
        branch = sconv_step(branch, py_state["aconv"], tensors["attn_sconv"])
        x = x + branch
        norm = rmsnorm(x, tensors["post_attention_norm"], cfg.rms_eps)
        if layer < cfg.dense_layers:
            ff = extra["dense_down"] @ (
                F.silu(extra["dense_gate"] @ norm) * (extra["dense_up"] @ norm)
            )
            ff = ff * extra["dense_scale"]
        else:
            logits = extra["router_weight"] @ norm
            choice = logits[: cfg.n_routed_experts].sigmoid() + extra["router_bias"]
            idx = torch.topk(choice, cfg.top_k, sorted=True).indices
            selected = torch.cat([logits[idx], logits[cfg.n_routed_experts :]])
            route = torch.softmax(F.logsigmoid(selected), dim=0)
            route = route * cfg.route_scale * extra["router_scale"]
            ff = torch.zeros(cfg.hidden)
            for slot, expert_idx in enumerate(idx.tolist()):
                ff += route[slot] * expert(
                    extra["routed_gate"][expert_idx],
                    extra["routed_up"][expert_idx],
                    extra["routed_down"][expert_idx], norm,
                )
            for e in range(cfg.n_shared_experts):
                ff += route[cfg.top_k + e] * expert(
                    extra["shared_gate"][e], extra["shared_up"][e],
                    extra["shared_down"][e], norm,
                )
        ff = sconv_step(ff, py_state["mconv"], tensors["mlp_sconv"])
        return x + ff

    def run_layer(self, layer, steps, *, use_callback=False):
        cfg, keep_local = self.build_config()
        capacity = cfg.sliding_window if layer == 0 else 6
        w, tensors, keep = self.make_common(layer, cfg)
        torch.manual_seed(500 + layer)
        if layer < cfg.dense_layers:
            extra = {
                "dense_gate": torch.randn(cfg.dense_intermediate, cfg.hidden) * 0.12,
                "dense_up": torch.randn(cfg.dense_intermediate, cfg.hidden) * 0.12,
                "dense_down": torch.randn(cfg.hidden, cfg.dense_intermediate) * 0.12,
                "dense_scale": torch.tensor(0.8),
            }
            for name in ("dense_gate", "dense_up", "dense_down"):
                keep[name] = array(extra[name])
                setattr(w, name, keep[name])
            keep["dense_scale"] = array(extra["dense_scale"].reshape(1))
            w.dense_global_scale = keep["dense_scale"]
            w.sparse = 0
        else:
            total = cfg.n_routed_experts + cfg.n_shared_experts
            extra = {
                "router_weight": torch.randn(total, cfg.hidden) * 0.12,
                "router_bias": torch.randn(cfg.n_routed_experts) * 0.03,
                "router_scale": torch.tensor(0.9),
                "shared_gate": torch.randn(cfg.n_shared_experts, cfg.moe_intermediate, cfg.hidden) * 0.12,
                "shared_up": torch.randn(cfg.n_shared_experts, cfg.moe_intermediate, cfg.hidden) * 0.12,
                "shared_down": torch.randn(cfg.n_shared_experts, cfg.hidden, cfg.moe_intermediate) * 0.12,
                "routed_gate": torch.randn(cfg.n_routed_experts, cfg.moe_intermediate, cfg.hidden) * 0.12,
                "routed_up": torch.randn(cfg.n_routed_experts, cfg.moe_intermediate, cfg.hidden) * 0.12,
                "routed_down": torch.randn(cfg.n_routed_experts, cfg.hidden, cfg.moe_intermediate) * 0.12,
            }
            for name, value in extra.items():
                keep[name] = array(value.reshape(1) if value.ndim == 0 else value)
                field = "router_global_scale" if name == "router_scale" else name
                setattr(w, field, keep[name])
            w.sparse = 1
        callback = ExpertCallback()
        if layer >= cfg.dense_layers and use_callback:
            gate_arr = keep["routed_gate"]
            up_arr = keep["routed_up"]
            down_arr = keep["routed_down"]
            gate_stride = cfg.moe_intermediate * cfg.hidden * ctypes.sizeof(ctypes.c_float)
            down_stride = cfg.hidden * cfg.moe_intermediate * ctypes.sizeof(ctypes.c_float)

            @ExpertCallback
            def callback(_ctx, callback_layer, expert_idx, out):
                if callback_layer != layer or expert_idx < 0 or expert_idx >= cfg.n_routed_experts:
                    return -1
                out.contents.gate = ctypes.cast(ctypes.byref(gate_arr, expert_idx * gate_stride), FP)
                out.contents.up = ctypes.cast(ctypes.byref(up_arr, expert_idx * gate_stride), FP)
                out.contents.down = ctypes.cast(ctypes.byref(down_arr, expert_idx * down_stride), FP)
                return 0

            w.routed_gate = None
            w.routed_up = None
            w.routed_down = None

        state, scratch, state_keep = self.init_runtime(cfg, layer, capacity)
        l = cfg.layer[layer]
        kdim = l.num_kv_heads * l.head_dim
        py_state = {
            "capacity": capacity,
            "keys": [],
            "values": [],
            "kconv": torch.zeros(kdim, cfg.conv_kernel),
            "vconv": torch.zeros(kdim, cfg.conv_kernel),
            "aconv": torch.zeros(cfg.hidden, cfg.conv_kernel),
            "mconv": torch.zeros(cfg.hidden, cfg.conv_kernel),
        }
        for position in range(steps):
            x = torch.randn(cfg.hidden)
            want = self.reference_layer(cfg, layer, tensors, extra, x.clone(), position, py_state)
            x_c = array(x)
            rc = self.lib.waste_inkling_layer_step(
                ctypes.byref(cfg), layer, ctypes.byref(w), ctypes.byref(state),
                x_c, position, ctypes.byref(scratch), callback, None,
            )
            self.assertEqual(rc, 0)
            got = tensor_from_array(x_c, [cfg.hidden])
            torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-4)
        self.assertIsNotNone((keep_local, keep, state_keep))

    def test_dense_local_layer_matches_torch(self):
        self.run_layer(0, 7)

    def test_sparse_global_layer_matches_torch(self):
        self.run_layer(1, 6)

    def test_sparse_expert_callback_matches_resident_path_reference(self):
        self.run_layer(1, 6, use_callback=True)

    def test_layer_kind_mismatch_fails_closed(self):
        cfg, keep_local = self.build_config()
        layer = 0
        w, _tensors, keep = self.make_common(layer, cfg)
        total = cfg.n_routed_experts + cfg.n_shared_experts
        torch.manual_seed(999)
        values = {
            "router_weight": torch.randn(total, cfg.hidden) * 0.1,
            "router_bias": torch.zeros(cfg.n_routed_experts),
            "router_global_scale": torch.ones(1),
            "shared_gate": torch.randn(cfg.n_shared_experts, cfg.moe_intermediate, cfg.hidden) * 0.1,
            "shared_up": torch.randn(cfg.n_shared_experts, cfg.moe_intermediate, cfg.hidden) * 0.1,
            "shared_down": torch.randn(cfg.n_shared_experts, cfg.hidden, cfg.moe_intermediate) * 0.1,
            "routed_gate": torch.randn(cfg.n_routed_experts, cfg.moe_intermediate, cfg.hidden) * 0.1,
            "routed_up": torch.randn(cfg.n_routed_experts, cfg.moe_intermediate, cfg.hidden) * 0.1,
            "routed_down": torch.randn(cfg.n_routed_experts, cfg.hidden, cfg.moe_intermediate) * 0.1,
        }
        for name, value in values.items():
            keep[name] = array(value)
            setattr(w, name, keep[name])
        w.sparse = 1  # layer 0 is configured dense
        state, scratch, state_keep = self.init_runtime(cfg, layer, cfg.sliding_window)
        x = array(torch.randn(cfg.hidden))
        rc = self.lib.waste_inkling_layer_step(
            ctypes.byref(cfg), layer, ctypes.byref(w), ctypes.byref(state),
            x, 0, ctypes.byref(scratch), ExpertCallback(), None,
        )
        self.assertNotEqual(rc, 0)
        self.assertIsNotNone((keep_local, keep, state_keep))


if __name__ == "__main__":
    unittest.main()
