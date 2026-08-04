#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import torch

from diagnose_inkling_bf16_attention_scores import (
    AttentionScoreError,
    c_softmax,
    causal_valid,
    classify_score_softmax,
    finite_logit_metrics,
    official_softmax,
    transform_attention_preserve_logits,
)


class AttentionScoreSoftmaxTest(unittest.TestCase):
    @staticmethod
    def metric(exact: float, maximum: float = 0.0):
        return {
            "quantized_exact_fraction": exact,
            "max_abs": maximum,
            "mean_abs": maximum / 2.0,
        }

    @staticmethod
    def source() -> str:
        return """        double sum = 0.0;
        for (int j = 0; j < kv_len; j++) {
            row[j] = expf(row[j] - max_score);
            sum += row[j];
        }
        if (!(sum > 0.0) || !isfinite(sum)) return -1;
        const float inv_sum = (float)(1.0 / sum);
        float *oh = out + (size_t)h * D;
        for (int d = 0; d < D; d++) oh[d] = 0.0f;
        for (int j = 0; j < kv_len; j++) {
            const float weight = row[j] * inv_sum;
            for (int d = 0; d < D; d++) oh[d] += weight * vc[d];
        }
"""

    def test_score_observation_preserves_rows_and_uses_probability_vla(self):
        transformed = transform_attention_preserve_logits(self.source())
        self.assertIn("float probability[kv_len]", transformed)
        self.assertIn("probability[j] = expf(row[j] - max_score)", transformed)
        self.assertIn("const float weight = probability[j] * inv_sum", transformed)
        self.assertNotIn("row[j] = expf", transformed)

    def test_score_observation_fails_closed_on_reapply_or_drift(self):
        transformed = transform_attention_preserve_logits(self.source())
        with self.assertRaisesRegex(AttentionScoreError, "probability loop"):
            transform_attention_preserve_logits(transformed)
        with self.assertRaisesRegex(AttentionScoreError, "value weight"):
            transform_attention_preserve_logits(
                self.source().replace(
                    "const float weight = row[j] * inv_sum;",
                    "const float weight = inv_sum;",
                )
            )

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

    def test_finite_logit_metrics_ignore_shared_mask(self):
        reference = torch.tensor([[1.0, -torch.inf, 2.0]])
        candidate = torch.tensor([[1.0, -torch.inf, 2.03125]])
        metrics = finite_logit_metrics(
            reference,
            candidate,
            compare_dtype=torch.bfloat16,
        )
        self.assertEqual(metrics["finite_values"], 2)
        self.assertEqual(metrics["masked_values"], 1)
        self.assertEqual(metrics["quantized_exact_fraction"], 0.5)

    def test_finite_logit_metrics_reject_mask_drift(self):
        with self.assertRaisesRegex(AttentionScoreError, "masks differ"):
            finite_logit_metrics(
                torch.tensor([[1.0, -torch.inf]]),
                torch.tensor([[1.0, 0.0]]),
                compare_dtype=torch.bfloat16,
            )

    def test_classifies_score_boundary(self):
        result = classify_score_softmax(
            official_sanity=self.metric(1.0),
            c_sanity=self.metric(1.0, 0.0),
            instrumentation_sanity=self.metric(1.0, 0.0),
            score_only=self.metric(0.8),
            softmax_only=self.metric(1.0),
        )
        self.assertEqual(result, "score_construction_is_the_remaining_boundary")

    def test_classifies_softmax_boundary_and_both(self):
        softmax = classify_score_softmax(
            official_sanity=self.metric(1.0),
            c_sanity=self.metric(1.0, 0.0),
            instrumentation_sanity=self.metric(1.0, 0.0),
            score_only=self.metric(1.0),
            softmax_only=self.metric(0.8),
        )
        self.assertEqual(softmax, "softmax_is_the_remaining_boundary")
        both = classify_score_softmax(
            official_sanity=self.metric(1.0),
            c_sanity=self.metric(1.0, 0.0),
            instrumentation_sanity=self.metric(1.0, 0.0),
            score_only=self.metric(0.8),
            softmax_only=self.metric(0.9),
        )
        self.assertEqual(both, "score_construction_and_softmax_both_contribute")

    def test_classifies_all_sanity_failures(self):
        official_fail = classify_score_softmax(
            official_sanity=self.metric(0.99),
            c_sanity=self.metric(1.0, 0.0),
            instrumentation_sanity=self.metric(1.0, 0.0),
            score_only=self.metric(1.0),
            softmax_only=self.metric(1.0),
        )
        self.assertEqual(official_fail, "official_score_reconstruction_failed")
        c_fail = classify_score_softmax(
            official_sanity=self.metric(1.0),
            c_sanity=self.metric(1.0, 1e-4),
            instrumentation_sanity=self.metric(1.0, 0.0),
            score_only=self.metric(1.0),
            softmax_only=self.metric(1.0),
        )
        self.assertEqual(c_fail, "c_score_observation_failed")
        instrumentation_fail = classify_score_softmax(
            official_sanity=self.metric(1.0),
            c_sanity=self.metric(1.0, 0.0),
            instrumentation_sanity=self.metric(1.0, 1e-4),
            score_only=self.metric(1.0),
            softmax_only=self.metric(1.0),
        )
        self.assertEqual(
            instrumentation_fail,
            "score_instrumentation_changed_attention",
        )

    def test_c_softmax_rejects_empty_finite_row(self):
        with self.assertRaisesRegex(AttentionScoreError, "no finite"):
            c_softmax(torch.full((1, 1, 2), -torch.inf))


if __name__ == "__main__":
    unittest.main()
