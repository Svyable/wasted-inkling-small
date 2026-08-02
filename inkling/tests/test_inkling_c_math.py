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


class InklingCMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shutil.which("cc")
        if not cc:
            raise unittest.SkipTest("no C compiler")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.lib_path = Path(cls.tmp.name) / "libinkling.so"
        subprocess.run(
            [
                cc,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fPIC",
                "-shared",
                "-I",
                str(REPO / "src"),
                str(REPO / "src" / "inkling.c"),
                "-lm",
                "-o",
                str(cls.lib_path),
            ],
            check=True,
        )
        cls.lib = ctypes.CDLL(str(cls.lib_path))
        fp = ctypes.POINTER(ctypes.c_float)
        ip = ctypes.POINTER(ctypes.c_int)
        cls.lib.waste_inkling_route.argtypes = [
            fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float, ip, fp, fp,
        ]
        cls.lib.waste_inkling_route.restype = ctypes.c_int
        cls.lib.waste_inkling_sconv_step.argtypes = [
            fp, fp, fp, ctypes.c_int, ctypes.c_int, fp,
        ]
        cls.lib.waste_inkling_sconv_step.restype = ctypes.c_int
        cls.lib.waste_inkling_relative_bias.argtypes = [
            fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, fp,
        ]
        cls.lib.waste_inkling_relative_bias.restype = ctypes.c_int
        cls.lib.waste_inkling_log_tau.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
        cls.lib.waste_inkling_log_tau.restype = ctypes.c_float

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def fp(tensor):
        if tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise TypeError("ctypes inputs must be contiguous float32 tensors")
        return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))

    def test_router_matches_official_equation(self):
        torch.manual_seed(7)
        n_routed, n_shared, top_k = 11, 2, 4
        logits = torch.randn(n_routed + n_shared, dtype=torch.float32)
        bias = torch.randn(n_routed, dtype=torch.float32) * 0.05
        route_scale, global_scale = 8.0, 0.875

        choice = logits[:n_routed].sigmoid() + bias
        ref_idx = torch.topk(choice, top_k, sorted=False).indices
        ref_logits = torch.cat([logits[ref_idx], logits[n_routed:]])
        ref_weights = torch.softmax(F.logsigmoid(ref_logits), dim=0) * route_scale * global_scale
        ref_routed = {int(i): float(w) for i, w in zip(ref_idx, ref_weights[:top_k])}
        ref_shared = ref_weights[top_k:]

        out_idx = torch.empty(top_k, dtype=torch.int32)
        out_routed = torch.empty(top_k, dtype=torch.float32)
        out_shared = torch.empty(n_shared, dtype=torch.float32)
        rc = self.lib.waste_inkling_route(
            self.fp(logits), self.fp(bias), n_routed, n_shared, top_k,
            route_scale, global_scale,
            ctypes.cast(out_idx.data_ptr(), ctypes.POINTER(ctypes.c_int)),
            self.fp(out_routed), self.fp(out_shared),
        )
        self.assertEqual(rc, 0)
        self.assertEqual(set(out_idx.tolist()), set(ref_idx.tolist()))
        for idx, weight in zip(out_idx.tolist(), out_routed.tolist()):
            self.assertAlmostEqual(weight, ref_routed[idx], places=6)
        torch.testing.assert_close(out_shared, ref_shared, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(
            float(out_routed.sum() + out_shared.sum()),
            route_scale * global_scale,
            places=5,
        )

    def test_short_convolution_cached_step_matches_torch(self):
        torch.manual_seed(11)
        channels, kernel = 5, 4
        x = torch.randn(channels, dtype=torch.float32)
        state = torch.randn(channels, kernel, dtype=torch.float32)
        weight = torch.randn(channels, kernel, dtype=torch.float32)
        state_ref = torch.cat([state[:, 1:], x[:, None]], dim=1)
        out_ref = x + (state_ref * weight).sum(dim=1)

        out = torch.empty_like(x)
        state_c = state.clone()
        rc = self.lib.waste_inkling_sconv_step(
            self.fp(x), self.fp(state_c), self.fp(weight), channels, kernel, self.fp(out)
        )
        self.assertEqual(rc, 0)
        torch.testing.assert_close(state_c, state_ref, rtol=0, atol=0)
        torch.testing.assert_close(out, out_ref, rtol=1e-6, atol=1e-6)

    def test_relative_bias_matches_dense_reference(self):
        torch.manual_seed(13)
        heads, d_rel, extent = 3, 4, 5
        query_pos, key_pos0, kv_len = 7, 1, 9
        relative = torch.randn(heads, d_rel, dtype=torch.float32)
        projection = torch.randn(d_rel, extent, dtype=torch.float32)
        ref = torch.zeros(heads, kv_len, dtype=torch.float32)
        for key in range(kv_len):
            distance = query_pos - (key_pos0 + key)
            if 0 <= distance < extent:
                ref[:, key] = relative @ projection[:, distance]

        out = torch.empty_like(ref)
        rc = self.lib.waste_inkling_relative_bias(
            self.fp(relative), self.fp(projection), heads, d_rel, extent,
            query_pos, key_pos0, kv_len, self.fp(out)
        )
        self.assertEqual(rc, 0)
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)

    def test_global_log_scaling(self):
        floor, alpha = 128000, 0.1
        self.assertEqual(self.lib.waste_inkling_log_tau(0, floor, alpha), 1.0)
        self.assertEqual(self.lib.waste_inkling_log_tau(floor - 1, floor, alpha), 1.0)
        position = 511999
        ref = 1.0 + alpha * math.log((position + 1) / floor)
        self.assertAlmostEqual(self.lib.waste_inkling_log_tau(position, floor, alpha), ref, places=6)


if __name__ == "__main__":
    unittest.main()
