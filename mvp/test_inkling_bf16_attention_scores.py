#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import unittest

import torch

from diagnose_inkling_bf16_attention_scores import (
    AttentionScoreError,
    bf16,
    c_head_rmsnorm,
    c_log_tau,
    c_softmax,
    causal_valid,
    classify_score_softmax,
    f32,
    official_softmax,
)


class AttentionScoreSoftmaxTest(unittest.TestCase):
    @staticmethod
    def metric(exact: float, maximum: float = 0.0):
        return {
            "quantized_exact_fraction": exact,
            "max_abs": maximum,
            "mean_abs": maximum / 2.0,
        }

    def test_c_head_rmsnorm_applies_bfloat16_before_weight(self):
        values = torch.tensor([0.5, -1.25, 0.75], dtype=torch.float32)
        weight = torch.tensor([1.0, 0.5, -0.75], dtype=torch.bfloat16).float()
        got = c_head_rmsnorm(values, weight, eps=1e-6)
        squared = sum(float(value) ** 2 for value in values.tolist())
        scale = f32(1.0 / math.sqrt(f32(f32(squared / 3) + f32(1e-6))))
        expected = torch.tensor(
            [
                bf16(f32(bf16(f32(float(value) * scale)) * float(factor)))
                for value, factor in zip(values.tolist(), weight.tolist())
            ]
        )
        self.assertTrue(torch.equal(got, expected))

    def test_c_head_rmsnorm_rejects_geometry(self):
        with self.assertRaisesRegex(AttentionScoreError, "shapes disagree"):
            c_head_rmsnorm(torch.zeros(2), torch.zeros(3), eps=1e-6)

    def test_log_tau_respects_floor(self):
        self.assertEqual(c_log_tau(0, 16, 0.5), 1.0)
        self.assertEqual(c_log_tau(15, 16, 0.5), 1.0)
        self.assertGreater(c_log_tau(31, 16, 0.5), 1.0)
        with self.assertRaisesRegex(AttentionScoreError, "geometry"):
            c_log_tau(0, 0, 0.5)

    def test_c_softmax_matches_exp_double_sum_order(self):
        logits = torch.tensor(
            [[[1.0, 0.0, -torch.inf], [2.0, 2.0, -torch.inf]]],
            dtype=torch.float32,
        )
        got = c_softmax(logits)
        self.assertAlmostEqual(float(got[0, 1, 0]), 0.5)
        self.assertAlmostEqual(float(got[0, 1, 1]), 0.5)
        self.assertEqual(float(got[0, 0, 2]), 0.0)
        self.assertAlmostEqual(float(got[0, 0].sum()), 1.0, places=6)

    def test_official_softmax_quantizes_to_bfloat16(self):
        logits = torch.tensor([[[0.25, -0.5]]], dtype=torch.float32)
        got = official_softmax(logits)
        expected = torch.softmax(logits, -1, dtype=torch.float32).to(
            torch.bfloat16
        ).float()
        self.assertTrue(torch.equal(got, expected))

    def test_causal_and_local_window_validity(self):
        self.assertTrue(causal_valid(3, 0, is_local=False, window=2))
        self.assertFalse(causal_valid(3, 4, is_local=False, window=2))
        self.assertTrue(causal_valid(3, 2, is_local=True, window=2))
        self.assertFalse(causal_valid(3, 1, is_local=True, window=2))

    def test_classifies_score_boundary(self):
        result = classify_score_softmax(
            official_sanity=self.metric(1.0),
            c_sanity=self.metric(1.0, 0.0),
            score_only=self.metric(0.8),
            softmax_only=self.metric(1.0),
        )
        self.assertEqual(result, "score_construction_is_the_remaining_boundary")

    def test_classifies_softmax_boundary(self):
        result = classify_score_softmax(
            official_sanity=self.metric(1.0),
            c_sanity=self.metric(1.0, 0.0),
            score_only=self.metric(1.0),
            softmax_only=self.metric(0.8),
        )
        self.assertEqual(result, "softmax_is_the_remaining_boundary")

    def test_classifies_both_and_sanity_failures(self):
        both = classify_score_softmax(
            official_sanity=self.metric(1.0),
            c_sanity=self.metric(1.0, 0.0),
            score_only=self.metric(0.8),
            softmax_only=self.metric(0.9),
        )
        self.assertEqual(both, "score_construction_and_softmax_both_contribute")
        official_fail = classify_score_softmax(
            official_sanity=self.metric(0.99),
            c_sanity=self.metric(1.0, 0.0),
            score_only=self.metric(1.0),
            softmax_only=self.metric(1.0),
        )
        self.assertEqual(official_fail, "official_score_reconstruction_failed")
        c_fail = classify_score_softmax(
            official_sanity=self.metric(1.0),
            c_sanity=self.metric(1.0, 1e-4),
            score_only=self.metric(1.0),
            softmax_only=self.metric(1.0),
        )
        self.assertEqual(c_fail, "c_score_reconstruction_failed")

    def test_c_softmax_rejects_empty_finite_row(self):
        with self.assertRaisesRegex(AttentionScoreError, "no finite"):
            c_softmax(torch.full((1, 1, 2), -torch.inf))


if __name__ == "__main__":
    unittest.main()
