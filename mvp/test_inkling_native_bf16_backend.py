#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes
import types
import unittest

import torch
from torch.nn import functional as F

from diagnose_inkling_native_bf16_backend import (
    MAT_K,
    MAT_O,
    MAT_Q,
    MAT_R,
    MAT_ROUTER,
    MAT_V,
    NativeBfloat16BackendError,
    NativeMatrixProvider,
    _exact_prefix,
    native_bfloat16_linear,
    transform_rms_source,
)
from inkling_layer_parity import FP


class NativeBfloat16BackendTest(unittest.TestCase):
    @staticmethod
    def source() -> str:
        return (
            '#include "inkling.h"\n\n'
            '#include <math.h>\n#include <stddef.h>\n#include <string.h>\n\n'
            'static void matvec(float *out)\n{\n'
            '        out[r] = (float)sum;\n'
            '}\n'
            'static void rmsnorm(float *out, const float *x, const float *weight, int n)\n{\n'
            '    for (int i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];\n'
            '}\n'
        )

    @staticmethod
    def module() -> types.SimpleNamespace:
        def linear(rows: int, cols: int, offset: float) -> types.SimpleNamespace:
            values = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
            return types.SimpleNamespace(weight=(values / 32.0 + offset).to(torch.bfloat16))

        attention = types.SimpleNamespace(
            q_proj=linear(3, 4, 0.1),
            k_proj=linear(2, 4, 0.2),
            v_proj=linear(2, 4, 0.3),
            r_proj=linear(2, 4, 0.4),
            o_proj=linear(4, 3, 0.5),
        )
        mlp = types.SimpleNamespace(gate=linear(5, 4, 0.6))
        return types.SimpleNamespace(self_attn=attention, mlp=mlp)

    def test_transform_changes_only_rmsnorm_contract(self):
        original = self.source()
        transformed = transform_rms_source(original)
        self.assertIn("bf16_round_probe", transformed)
        self.assertIn("const float normalized = bf16_round_probe", transformed)
        self.assertIn("out[r] = (float)sum;", transformed)
        self.assertNotIn(
            "out[i] = x[i] * scale * weight[i]", transformed
        )

    def test_transform_fails_closed_on_reapply_or_drift(self):
        transformed = transform_rms_source(self.source())
        with self.assertRaisesRegex(NativeBfloat16BackendError, "include anchor"):
            transform_rms_source(transformed)
        with self.assertRaisesRegex(NativeBfloat16BackendError, "RMSNorm output"):
            transform_rms_source(
                self.source().replace(
                    "out[i] = x[i] * scale * weight[i]",
                    "out[i] = x[i]",
                )
            )

    def test_native_linear_matches_direct_pinned_kernel(self):
        weight = torch.tensor(
            [[0.5, -0.25, 0.125], [0.75, 0.5, -0.5]],
            dtype=torch.float32,
        )
        vector = torch.tensor([0.25, -0.75, 0.5], dtype=torch.float32)
        got = native_bfloat16_linear(weight, vector)
        expected = F.linear(
            vector.to(torch.bfloat16).reshape(1, -1),
            weight.to(torch.bfloat16),
        )[0].float()
        self.assertTrue(torch.equal(got, expected))

    def test_native_linear_rejects_bad_geometry(self):
        with self.assertRaisesRegex(NativeBfloat16BackendError, "geometry"):
            native_bfloat16_linear(torch.zeros(2, 3), torch.zeros(4))

    def test_exact_prefix_obeys_declared_stage_order(self):
        stages = {
            point: {"quantized_exact_fraction": 1.0}
            for point in (
                "input_norm",
                "q_proj",
                "k_proj",
                "v_proj",
                "relative_proj_input",
                "k_sconv",
                "v_sconv",
                "attention_out",
                "attention_branch",
                "post_attention_residual",
                "post_attention_norm",
            )
        }
        stages["v_proj"]["quantized_exact_fraction"] = 0.99
        self.assertEqual(
            _exact_prefix({"stages": stages}),
            ["input_norm", "q_proj", "k_proj"],
        )

    def test_callback_serves_native_matrix_and_records_call(self):
        module = self.module()
        official_norm = torch.tensor([[0.5, -0.25, 0.75, -0.5]])
        provider = NativeMatrixProvider(
            2,
            module,
            official_norm=official_norm,
        )
        x = (ctypes.c_float * 4)(*official_norm[0].tolist())
        out = (ctypes.c_float * 3)()
        rc = provider.callback(
            None,
            2,
            MAT_Q,
            0,
            ctypes.cast(x, FP),
            ctypes.cast(out, FP),
            3,
            4,
        )
        self.assertEqual(rc, 0, provider.error)
        expected = native_bfloat16_linear(module.self_attn.q_proj.weight, official_norm[0])
        self.assertEqual(list(out), expected.tolist())
        self.assertEqual(provider.calls[0]["mode"], "native-token-step")
        self.assertEqual(provider.calls[0]["position"], 0)

    def test_callback_oracle_requires_exact_bfloat16_norm(self):
        module = self.module()
        official_norm = torch.tensor([[0.5, -0.25, 0.75, -0.5]])
        oracle = {"q_proj": torch.tensor([[1.0, 2.0, 3.0]])}
        provider = NativeMatrixProvider(
            2,
            module,
            official_norm=official_norm,
            projection_oracle=oracle,
        )
        wrong = (ctypes.c_float * 4)(0.0, -0.25, 0.75, -0.5)
        out = (ctypes.c_float * 3)()
        rc = provider.callback(
            None,
            2,
            MAT_Q,
            0,
            ctypes.cast(wrong, FP),
            ctypes.cast(out, FP),
            3,
            4,
        )
        self.assertEqual(rc, -1)
        self.assertIn("non-official RMSNorm input", provider.error or "")

    def test_callback_rejects_unknown_matrix_contract(self):
        module = self.module()
        provider = NativeMatrixProvider(
            2,
            module,
            official_norm=torch.zeros(1, 4),
        )
        x = (ctypes.c_float * 4)()
        out = (ctypes.c_float * 3)()
        rc = provider.callback(
            None,
            2,
            99,
            0,
            ctypes.cast(x, FP),
            ctypes.cast(out, FP),
            3,
            4,
        )
        self.assertEqual(rc, -1)
        self.assertIn("unexpected backend request", provider.error or "")

    def test_matrix_kind_constants_remain_bound_to_header_order(self):
        self.assertEqual(
            (MAT_Q, MAT_K, MAT_V, MAT_R, MAT_O, MAT_ROUTER),
            (0, 1, 2, 3, 4, 8),
        )


if __name__ == "__main__":
    unittest.main()
