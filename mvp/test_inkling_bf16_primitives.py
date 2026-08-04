#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import torch

from diagnose_inkling_bf16_primitives import (
    c_linear_sequential,
    c_reduction_official_cast_order,
    c_rmsnorm_formula,
    official_rmsnorm_formula,
)


class Bfloat16PrimitiveDiagnosisTest(unittest.TestCase):
    def test_c_rmsnorm_matches_declared_double_reduction(self) -> None:
        values = torch.tensor(
            [[0.5, -0.25, 0.75], [-1.0, 0.125, 0.25]],
            dtype=torch.float32,
        )
        weight = torch.tensor([0.75, 1.25, 0.5], dtype=torch.float32)
        eps = 1e-6
        got = c_rmsnorm_formula(values, weight, eps)

        rows = []
        for row in values.tolist():
            ss = sum(float(value) * float(value) for value in row)
            mean = torch.tensor(ss / len(row), dtype=torch.float32)
            scale = torch.reciprocal(torch.sqrt(mean + eps)).item()
            rows.append(
                [
                    torch.tensor(value * scale, dtype=torch.float32).item()
                    * float(scale_weight)
                    for value, scale_weight in zip(row, weight.tolist())
                ]
            )
        expected = torch.tensor(rows, dtype=torch.float32)
        self.assertTrue(torch.allclose(got, expected, atol=1e-7, rtol=1e-7))

    def test_official_rmsnorm_formula_has_cast_before_weight(self) -> None:
        values = torch.tensor(
            [[0.33333334, -0.7777778, 1.125]], dtype=torch.float32
        )
        weight = torch.tensor([0.8125, 1.1875, 0.9375], dtype=torch.bfloat16)
        eps = 1e-6
        got = official_rmsnorm_formula(
            values, weight, eps, torch.bfloat16
        )
        c_order = c_reduction_official_cast_order(
            values, weight, eps, torch.bfloat16
        )
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertEqual(c_order.dtype, torch.bfloat16)
        self.assertEqual(tuple(got.shape), tuple(values.shape))

    def test_c_linear_uses_left_to_right_double_accumulation(self) -> None:
        values = torch.tensor(
            [[0.5, -0.25, 0.75], [-1.0, 0.125, 0.25]],
            dtype=torch.float32,
        )
        weight = torch.tensor(
            [[0.25, 0.5, -0.75], [1.0, -0.125, 0.375]],
            dtype=torch.float32,
        )
        got = c_linear_sequential(values, weight)
        expected_rows = []
        for row in values.tolist():
            outputs = []
            for matrix_row in weight.tolist():
                total = 0.0
                for left, right in zip(row, matrix_row):
                    total += float(left) * float(right)
                outputs.append(torch.tensor(total, dtype=torch.float32).item())
            expected_rows.append(outputs)
        expected = torch.tensor(expected_rows, dtype=torch.float32)
        self.assertTrue(torch.equal(got, expected))

    def test_shape_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "linear shape mismatch"):
            c_linear_sequential(torch.ones(2, 3), torch.ones(4, 2))
        with self.assertRaisesRegex(RuntimeError, "RMSNorm width mismatch"):
            c_rmsnorm_formula(torch.ones(2, 3), torch.ones(2), 1e-6)


if __name__ == "__main__":
    unittest.main()
