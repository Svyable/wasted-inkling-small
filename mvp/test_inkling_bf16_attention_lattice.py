#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import torch

from diagnose_inkling_bf16_attention_lattice import (
    AttentionLatticeError,
    c_dot,
    c_head_norm,
    c_softmax,
    classify_lattice,
    f32,
    official_softmax,
    tensor_metrics,
    valid_keys,
)


class AttentionArithmeticLatticeTest(unittest.TestCase):
    def test_f32_matches_float32_assignment(self):
        value = 1.0 + 2.0**-24
        self.assertEqual(f32(value), float(torch.tensor(value, dtype=torch.float32)))
        self.assertNotEqual(f32(1.0 + 2.0**-23), 1.0)

    def test_c_dot_is_left_to_right_float32(self):
        left = torch.tensor([1.0e20, 1.0, -1.0e20], dtype=torch.float32)
        right = torch.ones(3, dtype=torch.float32)
        expected = f32(
            f32(f32(0.0 + f32(1.0e20)) + f32(1.0)) + f32(-1.0e20)
        )
        self.assertEqual(c_dot(left, right), expected)
        with self.assertRaisesRegex(AttentionLatticeError, "geometry"):
            c_dot(left, torch.ones(2))

    def test_c_softmax_is_positive_and_normalized(self):
        value = c_softmax([0.25, -0.5, 1.0])
        self.assertTrue(torch.all(value > 0.0))
        self.assertAlmostEqual(float(value.sum()), 1.0, places=6)
        with self.assertRaisesRegex(AttentionLatticeError, "empty"):
            c_softmax([])

    def test_official_softmax_rounds_probabilities_to_bfloat16(self):
        scores = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
        actual = official_softmax(scores)
        expected = (
            torch.softmax(scores.to(torch.bfloat16), dim=-1, dtype=torch.float32)
            .to(torch.bfloat16)
            .float()
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_head_norm_preserves_geometry_and_rejects_bad_weight(self):
        values = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 0.5], [2.0, -3.0]]],
            dtype=torch.float32,
        )
        weight = torch.tensor([0.5, 1.5], dtype=torch.float32)
        result = c_head_norm(values, weight, 1.0e-6)
        self.assertEqual(tuple(result.shape), tuple(values.shape))
        self.assertTrue(torch.isfinite(result).all())
        with self.assertRaisesRegex(AttentionLatticeError, "geometry"):
            c_head_norm(values, torch.ones(3), 1.0e-6)

    def test_valid_keys_matches_global_and_sliding_causality(self):
        self.assertEqual(list(valid_keys(4, sliding=False, window=2)), [0, 1, 2, 3, 4])
        self.assertEqual(list(valid_keys(4, sliding=True, window=3)), [2, 3, 4])
        self.assertEqual(list(valid_keys(1, sliding=True, window=3)), [0, 1])
        with self.assertRaisesRegex(AttentionLatticeError, "nonnegative"):
            list(valid_keys(-1, sliding=False, window=3))

    def test_tensor_metrics_compares_bfloat16_bits(self):
        reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
        candidate = reference + torch.tensor([1.0e-4, 1.0e-2])
        result = tensor_metrics(reference, candidate)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["quantized_exact"], 1)
        self.assertEqual(result["quantized_exact_fraction"], 0.5)
        with self.assertRaisesRegex(AttentionLatticeError, "shape mismatch"):
            tensor_metrics(reference, torch.ones(3))

    def test_classification_uses_first_exact_and_leave_one_necessity(self):
        def metric(exact: float):
            return {
                "quantized_exact_fraction": exact,
                "max_abs": 0.0 if exact == 1.0 else 0.1,
                "mean_abs": 0.0 if exact == 1.0 else 0.01,
            }

        cumulative = {
            "c_baseline": {"attention_out": metric(0.5)},
            "head_norm_bf16": {"attention_out": metric(0.7)},
            "qk_score_bf16": {"attention_out": metric(1.0)},
            "relative_bias_bf16": {"attention_out": metric(1.0)},
            "softmax_bf16": {"attention_out": metric(1.0)},
            "all_official": {"attention_out": metric(1.0)},
        }
        ablations = {
            "head_norm": {"attention_out": metric(0.8)},
            "qk_score": {"attention_out": metric(0.9)},
            "relative_bias": {"attention_out": metric(1.0)},
            "softmax": {"attention_out": metric(1.0)},
            "value": {"attention_out": metric(1.0)},
        }
        result = classify_lattice(cumulative, ablations)
        self.assertEqual(result["first_exact_cumulative_variant"], "qk_score_bf16")
        self.assertEqual(
            result["necessary_boundaries_by_leave_one_c"],
            ["head_norm", "qk_score"],
        )
        self.assertEqual(
            result["classification"],
            "cumulative_attention_boundary_identified",
        )


if __name__ == "__main__":
    unittest.main()
