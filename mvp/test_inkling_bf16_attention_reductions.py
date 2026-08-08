#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import torch

from diagnose_inkling_bf16_attention_reductions import (
    AttentionReductionError,
    bf16_product,
    bf16_scalar,
    classify_operation,
    completed_bf16_dot,
    reduce_dot,
    row_mismatches,
    software_value_sum,
)


class AttentionReductionTest(unittest.TestCase):
    def test_bfloat16_scalar_and_product_match_torch(self):
        value = 1.00390625
        self.assertEqual(
            bf16_scalar(value),
            float(torch.tensor(value).to(torch.bfloat16).float()),
        )
        self.assertEqual(
            bf16_product(0.75, 0.375),
            float(
                (
                    torch.tensor(0.75, dtype=torch.bfloat16)
                    * torch.tensor(0.375, dtype=torch.bfloat16)
                ).float()
            ),
        )

    def test_float_and_double_reductions_are_explicit(self):
        left = torch.tensor([1.0e20, 1.0, -1.0e20], dtype=torch.float32)
        right = torch.ones(3, dtype=torch.float32)
        self.assertEqual(reduce_dot(left, right, "float32_left_to_right"), 0.0)
        self.assertEqual(reduce_dot(left, right, "double_left_to_right"), 0.0)
        with self.assertRaisesRegex(AttentionReductionError, "unknown"):
            reduce_dot(left, right, "pairwise")
        with self.assertRaisesRegex(AttentionReductionError, "geometry"):
            reduce_dot(left, torch.ones(2), "float32_left_to_right")

    def test_completed_reduction_is_bfloat16_bound(self):
        left = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        right = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32)
        value = completed_bf16_dot(left, right, "float32_left_to_right")
        self.assertEqual(value, bf16_scalar(value))

    def test_software_value_sum_rejects_bad_geometry(self):
        probabilities = torch.tensor([0.25, 0.75], dtype=torch.float32)
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        result = software_value_sum(
            probabilities,
            values,
            "float32_left_to_right",
        )
        self.assertEqual(tuple(result.shape), (2,))
        self.assertTrue(torch.equal(result, result.to(torch.bfloat16).float()))
        with self.assertRaisesRegex(AttentionReductionError, "row mismatch"):
            software_value_sum(
                torch.ones(3),
                values,
                "float32_left_to_right",
            )

    def test_row_mismatch_records_position_and_head(self):
        reference = [
            (0, 0, torch.tensor([1.0, 2.0])),
            (1, 3, torch.tensor([3.0, 4.0])),
        ]
        candidate = [
            (0, 0, torch.tensor([1.0, 2.0])),
            (1, 3, torch.tensor([3.0, 4.25])),
        ]
        result = row_mismatches(reference, candidate)
        self.assertEqual(result["mismatch_positions"], [1])
        self.assertEqual(result["mismatch_rows"], [{"position": 1, "head": 3}])
        with self.assertRaisesRegex(AttentionReductionError, "identity"):
            row_mismatches(reference, [(0, 0, reference[0][2]), (1, 2, reference[1][2])])

    def test_classification_prefers_portable_float_then_double(self):
        variants = {
            "float32_left_to_right": {"quantized_exact_fraction": 1.0},
            "double_left_to_right": {"quantized_exact_fraction": 1.0},
        }
        self.assertEqual(
            classify_operation(variants),
            "float32_left_to_right_matches_native_bf16",
        )
        variants["float32_left_to_right"]["quantized_exact_fraction"] = 0.9
        self.assertEqual(
            classify_operation(variants),
            "double_left_to_right_matches_native_bf16",
        )
        variants["double_left_to_right"]["quantized_exact_fraction"] = 0.9
        self.assertEqual(
            classify_operation(variants),
            "native_bf16_kernel_required",
        )


if __name__ == "__main__":
    unittest.main()
