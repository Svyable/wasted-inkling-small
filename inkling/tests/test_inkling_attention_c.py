#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.

import ctypes
import math
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


class State(ctypes.Structure):
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
        ("k_cache", ctypes.POINTER(ctypes.c_float)),
        ("v_cache", ctypes.POINTER(ctypes.c_float)),
    ]


def c_array(values):
    flat = torch.as_tensor(values, dtype=torch.float32).contiguous().view(-1).tolist()
    return (ctypes.c_float * len(flat))(*flat)


def rmsnorm(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


def reference_step(
    q,
    k,
    v,
    q_weight,
    k_weight,
    relative_state,
    relative_proj,
    position,
    is_local,
    capacity,
    kv_cache,
    v_cache,
    log_floor,
    alpha,
    eps,
):
    qn = rmsnorm(q, q_weight, eps)
    kn = rmsnorm(k, k_weight, eps)
    kv_cache.append(kn)
    v_cache.append(v)
    if is_local and len(kv_cache) > capacity:
        del kv_cache[0]
        del v_cache[0]
    keys = torch.stack(kv_cache)
    values = torch.stack(v_cache)
    heads, dim = q.shape
    kv_heads = k.shape[0]
    group = heads // kv_heads
    tau = 1.0
    if log_floor > 0:
        tau = 1.0 + alpha * math.log(max((position + 1) / log_floor, 1.0))
    outputs = []
    begin = position - len(kv_cache) + 1
    for head in range(heads):
        kh = head // group
        scores = []
        for j in range(len(kv_cache)):
            key_pos = begin + j
            score = torch.dot(qn[head], keys[j, kh]) * (tau / dim)
            distance = position - key_pos
            if 0 <= distance < relative_proj.shape[1]:
                score = score + tau * torch.dot(relative_state[head], relative_proj[:, distance])
            scores.append(score)
        weights = torch.softmax(torch.stack(scores).float(), dim=0)
        outputs.append((weights[:, None] * values[:, kh]).sum(dim=0))
    return torch.stack(outputs)


class InklingAttentionCDifferentialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shutil.which("cc")
        if not cc:
            raise unittest.SkipTest("C compiler unavailable")
        cls.td = tempfile.TemporaryDirectory()
        lib_path = Path(cls.td.name) / "libinkling_attention.so"
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
                str(REPO / "src" / "inkling_attention.c"),
                str(REPO / "src" / "inkling.c"),
                "-lm",
                "-o",
                str(lib_path),
            ],
            check=True,
            capture_output=True,
        )
        cls.lib = ctypes.CDLL(str(lib_path))
        cls.lib.waste_inkling_attention_init.argtypes = [
            ctypes.POINTER(State),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        cls.lib.waste_inkling_attention_init.restype = ctypes.c_int
        cls.lib.waste_inkling_attention_step.argtypes = [
            ctypes.POINTER(State),
            *([ctypes.POINTER(ctypes.c_float)] * 7),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        cls.lib.waste_inkling_attention_step.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def run_case(self, *, is_local, capacity, steps, log_floor):
        torch.manual_seed(1234 + int(is_local))
        heads, kv_heads, dim, d_rel = 4, 2, 8, 3
        extent = capacity if is_local else 7
        cache_n = capacity * kv_heads * dim
        k_cache = (ctypes.c_float * cache_n)()
        v_cache = (ctypes.c_float * cache_n)()
        state = State()
        self.assertEqual(
            self.lib.waste_inkling_attention_init(
                ctypes.byref(state),
                int(is_local),
                heads,
                kv_heads,
                dim,
                d_rel,
                extent,
                capacity,
                ctypes.c_float(1e-6),
                k_cache,
                v_cache,
            ),
            0,
        )
        q_weight_t = torch.randn(dim).abs() + 0.5
        k_weight_t = torch.randn(dim).abs() + 0.5
        proj_t = torch.randn(d_rel, extent)
        q_weight = c_array(q_weight_t)
        k_weight = c_array(k_weight_t)
        proj = c_array(proj_t)
        py_k, py_v = [], []
        for position in range(steps):
            q_t = torch.randn(heads, dim)
            k_t = torch.randn(kv_heads, dim)
            v_t = torch.randn(kv_heads, dim)
            rel_t = torch.randn(heads, d_rel)
            want = reference_step(
                q_t,
                k_t,
                v_t,
                q_weight_t,
                k_weight_t,
                rel_t,
                proj_t,
                position,
                is_local,
                capacity,
                py_k,
                py_v,
                log_floor,
                0.2,
                1e-6,
            )
            q, k, v, rel = map(c_array, (q_t, k_t, v_t, rel_t))
            scores = (ctypes.c_float * (heads * capacity))()
            out = (ctypes.c_float * (heads * dim))()
            rc = self.lib.waste_inkling_attention_step(
                ctypes.byref(state),
                q,
                k,
                v,
                q_weight,
                k_weight,
                rel,
                proj,
                position,
                log_floor,
                ctypes.c_float(0.2),
                scores,
                out,
            )
            self.assertEqual(rc, 0)
            got = torch.tensor(list(out)).reshape(heads, dim)
            torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)

    def test_local_ring_cache_matches_torch(self):
        self.run_case(is_local=True, capacity=4, steps=11, log_floor=0)

    def test_global_cache_and_log_scaling_match_torch(self):
        self.run_case(is_local=False, capacity=9, steps=9, log_floor=2)

    def test_nonsequential_and_full_global_cache_fail_closed(self):
        cache = (ctypes.c_float * 2)()
        state = State()
        self.assertEqual(
            self.lib.waste_inkling_attention_init(
                ctypes.byref(state), 0, 1, 1, 1, 1, 1, 2, 1e-6, cache, cache
            ),
            0,
        )
        one = (ctypes.c_float * 1)(1.0)
        scores = (ctypes.c_float * 2)()
        out = (ctypes.c_float * 1)()
        self.assertNotEqual(
            self.lib.waste_inkling_attention_step(
                ctypes.byref(state), one, one, one, one, one, one, one, 1, 0, 0.0, scores, out
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
