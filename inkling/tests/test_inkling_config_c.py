#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.

import ctypes
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_LAYERS = 128


class LayerCfg(ctypes.Structure):
    _fields_ = [
        ("is_local", ctypes.c_int),
        ("num_heads", ctypes.c_int),
        ("num_kv_heads", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("relative_extent", ctypes.c_int),
    ]


class Config(ctypes.Structure):
    _fields_ = [
        ("n_layers", ctypes.c_int),
        ("hidden", ctypes.c_int),
        ("vocab", ctypes.c_int),
        ("unpadded_vocab", ctypes.c_int),
        ("max_context", ctypes.c_int),
        ("global_heads", ctypes.c_int),
        ("global_kv_heads", ctypes.c_int),
        ("global_head_dim", ctypes.c_int),
        ("local_heads", ctypes.c_int),
        ("local_kv_heads", ctypes.c_int),
        ("local_head_dim", ctypes.c_int),
        ("sliding_window", ctypes.c_int),
        ("d_rel", ctypes.c_int),
        ("rel_extent", ctypes.c_int),
        ("conv_kernel", ctypes.c_int),
        ("dense_layers", ctypes.c_int),
        ("dense_intermediate", ctypes.c_int),
        ("moe_intermediate", ctypes.c_int),
        ("n_routed_experts", ctypes.c_int),
        ("top_k", ctypes.c_int),
        ("n_shared_experts", ctypes.c_int),
        ("rms_eps", ctypes.c_float),
        ("route_scale", ctypes.c_float),
        ("logits_width_multiplier", ctypes.c_float),
        ("log_scaling_n_floor", ctypes.c_int),
        ("log_scaling_alpha", ctypes.c_float),
        ("layer", LayerCfg * MAX_LAYERS),
    ]


class Args(ctypes.Structure):
    _fields_ = Config._fields_[:-1] + [
        ("local_layer_ids", ctypes.POINTER(ctypes.c_int)),
        ("n_local_layers", ctypes.c_int),
    ]


class Memory(ctypes.Structure):
    _fields_ = [
        ("kv_bytes", ctypes.c_uint64),
        ("conv_bytes", ctypes.c_uint64),
        ("state_bytes", ctypes.c_uint64),
        ("token_vector_bytes", ctypes.c_uint64),
        ("projection_bytes", ctypes.c_uint64),
        ("attention_score_bytes", ctypes.c_uint64),
        ("router_bytes", ctypes.c_uint64),
        ("expert_workspace_bytes", ctypes.c_uint64),
        ("dense_workspace_bytes", ctypes.c_uint64),
        ("shared_workspace_bytes", ctypes.c_uint64),
        ("logits_bytes", ctypes.c_uint64),
        ("decode_scratch_bytes", ctypes.c_uint64),
    ]


def reference(a, local, ctx):
    kv = conv = 0
    max_attention = 0
    max_projection = 0
    for layer in range(a.n_layers):
        is_local = layer in local
        heads = a.local_heads if is_local else a.global_heads
        kv_heads = a.local_kv_heads if is_local else a.global_kv_heads
        dim = a.local_head_dim if is_local else a.global_head_dim
        tokens = min(ctx, a.sliding_window) if is_local else ctx
        kv += tokens * 2 * kv_heads * dim * 4
        conv += (2 * kv_heads * dim + 2 * a.hidden) * a.conv_kernel * 4
        max_attention = max(max_attention, tokens * heads)
        max_projection = max(
            max_projection,
            heads * dim + 2 * kv_heads * dim + heads * a.d_rel,
        )
    fields = {
        "kv_bytes": kv,
        "conv_bytes": conv,
        "state_bytes": kv + conv,
        "token_vector_bytes": 8 * a.hidden * 4,
        "projection_bytes": max_projection * 4,
        "attention_score_bytes": max_attention * 4,
        "router_bytes": (
            (2 * a.n_routed_experts + a.n_shared_experts + 2 * a.top_k) * 4
            + a.top_k * ctypes.sizeof(ctypes.c_int)
        ),
        "expert_workspace_bytes": (
            3 * a.hidden * a.moe_intermediate + 2 * a.moe_intermediate + a.hidden
        )
        * 4,
        "dense_workspace_bytes": (2 * a.dense_intermediate + a.hidden) * 4,
        "shared_workspace_bytes": (2 * a.moe_intermediate + a.hidden) * 4,
        "logits_bytes": a.unpadded_vocab * 4,
    }
    fields["decode_scratch_bytes"] = sum(
        fields[name]
        for name in (
            "token_vector_bytes",
            "projection_bytes",
            "attention_score_bytes",
            "router_bytes",
            "expert_workspace_bytes",
            "dense_workspace_bytes",
            "shared_workspace_bytes",
            "logits_bytes",
        )
    )
    return fields


class InklingConfigCDifferentialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shutil.which("cc")
        if not cc:
            raise unittest.SkipTest("C compiler unavailable")
        cls.td = tempfile.TemporaryDirectory()
        cls.lib_path = Path(cls.td.name) / "libinkling_config.so"
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
                str(REPO / "src" / "inkling_config.c"),
                "-o",
                str(cls.lib_path),
            ],
            check=True,
            capture_output=True,
        )
        cls.lib = ctypes.CDLL(str(cls.lib_path))
        cls.lib.waste_inkling_config_build.argtypes = [ctypes.POINTER(Config), ctypes.POINTER(Args)]
        cls.lib.waste_inkling_config_build.restype = ctypes.c_int
        cls.lib.waste_inkling_plan_decode_memory.argtypes = [
            ctypes.POINTER(Config),
            ctypes.c_uint32,
            ctypes.POINTER(Memory),
        ]
        cls.lib.waste_inkling_plan_decode_memory.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    @staticmethod
    def make_args(rng, local_ids):
        a = Args()
        a.n_layers = rng.randint(2, 12)
        a.hidden = rng.choice([16, 32, 64, 128])
        a.vocab = rng.randint(64, 512)
        a.unpadded_vocab = rng.randint(32, a.vocab)
        a.max_context = rng.choice([128, 256, 512, 1024])
        a.global_kv_heads = rng.choice([1, 2, 4])
        a.global_heads = a.global_kv_heads * rng.choice([1, 2, 4])
        a.global_head_dim = rng.choice([4, 8, 16])
        a.local_kv_heads = rng.choice([1, 2, 4])
        a.local_heads = a.local_kv_heads * rng.choice([1, 2, 4])
        a.local_head_dim = rng.choice([4, 8, 16])
        a.sliding_window = rng.choice([8, 16, 32, 64])
        a.d_rel = rng.choice([2, 4, 8])
        a.rel_extent = rng.choice([32, 64, 128])
        a.conv_kernel = rng.choice([2, 3, 4, 5])
        a.dense_layers = rng.randrange(a.n_layers + 1)
        a.dense_intermediate = a.hidden * rng.choice([2, 4])
        a.moe_intermediate = a.hidden * rng.choice([1, 2])
        a.n_routed_experts = rng.choice([4, 8, 16])
        a.top_k = rng.randint(1, min(4, a.n_routed_experts))
        a.n_shared_experts = rng.choice([1, 2])
        a.rms_eps = 1e-6
        a.route_scale = 8.0
        a.logits_width_multiplier = 2.0
        a.log_scaling_n_floor = 128
        a.log_scaling_alpha = 0.1
        selected = sorted(rng.sample(range(a.n_layers), rng.randrange(a.n_layers + 1)))
        arr = (ctypes.c_int * len(selected))(*selected) if selected else None
        a.local_layer_ids = arr
        a.n_local_layers = len(selected)
        local_ids[:] = selected
        return a, arr

    def test_randomized_config_and_memory_match_python_reference(self):
        rng = random.Random(0x1A2B3C)
        for _ in range(30):
            local = []
            a, keepalive = self.make_args(rng, local)
            cfg = Config()
            self.assertEqual(self.lib.waste_inkling_config_build(ctypes.byref(cfg), ctypes.byref(a)), 0)
            for layer in range(a.n_layers):
                got = cfg.layer[layer]
                self.assertEqual(bool(got.is_local), layer in local)
                self.assertEqual(
                    got.num_kv_heads,
                    a.local_kv_heads if layer in local else a.global_kv_heads,
                )
            ctx = rng.randint(1, a.max_context)
            mem = Memory()
            self.assertEqual(
                self.lib.waste_inkling_plan_decode_memory(ctypes.byref(cfg), ctx, ctypes.byref(mem)),
                0,
            )
            want = reference(a, set(local), ctx)
            for name, value in want.items():
                self.assertEqual(getattr(mem, name), value, name)
            self.assertIsNotNone(keepalive if local else a)

    def test_duplicate_local_layer_and_context_overflow_fail_closed(self):
        rng = random.Random(7)
        local = []
        a, _ = self.make_args(rng, local)
        duplicate = (ctypes.c_int * 2)(0, 0)
        a.local_layer_ids = duplicate
        a.n_local_layers = 2
        cfg = Config()
        self.assertNotEqual(self.lib.waste_inkling_config_build(ctypes.byref(cfg), ctypes.byref(a)), 0)

        local = []
        a, keepalive = self.make_args(rng, local)
        self.assertEqual(self.lib.waste_inkling_config_build(ctypes.byref(cfg), ctypes.byref(a)), 0)
        mem = Memory()
        self.assertNotEqual(
            self.lib.waste_inkling_plan_decode_memory(
                ctypes.byref(cfg), a.max_context + 1, ctypes.byref(mem)
            ),
            0,
        )
        self.assertIsNotNone(keepalive if local else a)


if __name__ == "__main__":
    unittest.main()
