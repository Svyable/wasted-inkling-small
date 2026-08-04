#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes
import unittest

import torch

from diagnose_inkling_bf16_attention_components import (
    AttentionComponentError,
    c_style_value_aggregation,
    classify_components,
    native_bfloat16_value_aggregation,
    reconstruct_c_probabilities,
    repeat_values_by_head,
)
from inkling_layer_parity import FP, LayerScratch


class AttentionComponentProbeTest(unittest.TestCase):
    @staticmethod
    def metric(exact: float, maximum: float = 0.0):
        return {
            "quantized_exact_fraction": exact,
            "max_abs": maximum,
            "mean_abs": maximum / 2.0,
        }

    def test_reconstructs_c_softmax_probability_order(self):
        scores = (ctypes.c_float * 6)(2.0, 1.0, 0.0, 4.0, 2.0, 0.0)
        scratch = LayerScratch()
        scratch.scores = ctypes.cast(scores, FP)
        result = reconstruct_c_probabilities(
            scratch,
            position=1,
            heads=2,
            capacity=3,
            is_local=False,
            tokens=3,
        )
        inv0 = ctypes.c_float(1.0 / 3.0).value
        inv1 = ctypes.c_float(1.0 / 6.0).value
        expected = torch.tensor(
            [
                [ctypes.c_float(2.0 * inv0).value, ctypes.c_float(inv0).value, 0.0],
                [ctypes.c_float(4.0 * inv1).value, ctypes.c_float(2.0 * inv1).value, 0.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(result, expected))

    def test_reconstructs_local_window_positions(self):
        scores = (ctypes.c_float * 2)(1.0, 3.0)
        scratch = LayerScratch()
        scratch.scores = ctypes.cast(scores, FP)
        result = reconstruct_c_probabilities(
            scratch,
            position=3,
            heads=1,
            capacity=2,
            is_local=True,
            tokens=5,
        )
        self.assertEqual(result[0, :2].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(float(result[0, 2]), 0.25)
        self.assertAlmostEqual(float(result[0, 3]), 0.75)
        self.assertEqual(float(result[0, 4]), 0.0)

    def test_repeats_kv_values_by_attention_group(self):
        values = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        )
        repeated = repeat_values_by_head(
            values,
            heads=4,
            kv_heads=2,
            head_dim=2,
        )
        self.assertEqual(tuple(repeated.shape), (4, 2, 2))
        self.assertTrue(torch.equal(repeated[0], repeated[1]))
        self.assertTrue(torch.equal(repeated[2], repeated[3]))
        self.assertFalse(torch.equal(repeated[0], repeated[2]))

    def test_native_bfloat16_aggregation_matches_full_matmul(self):
        probabilities = torch.tensor(
            [[[1.0, 0.0]], [[0.25, 0.75]]], dtype=torch.float32
        )
        values = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32
        )
        got = native_bfloat16_value_aggregation(probabilities, values)
        expected = torch.matmul(
            probabilities.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16),
            values.unsqueeze(0).to(torch.bfloat16),
        )[0].permute(1, 0, 2).reshape(2, 2).float()
        self.assertTrue(torch.equal(got, expected))

    def test_c_style_aggregation_uses_float_multiply_add(self):
        probabilities = torch.tensor([[[0.25, 0.75]]], dtype=torch.float32)
        values = torch.tensor([[[1.25], [2.5]]], dtype=torch.float32)
        got = c_style_value_aggregation(probabilities, values)
        first = ctypes.c_float(0.25 * 1.25).value
        second = ctypes.c_float(0.75 * 2.5).value
        expected = ctypes.c_float(ctypes.c_float(0.0 + first).value + second).value
        self.assertEqual(float(got[0, 0]), expected)

    def test_classifies_score_softmax_boundary(self):
        result = classify_components(
            probability_metrics=self.metric(0.5),
            official_native_sanity=self.metric(1.0),
            c_style_sanity=self.metric(1.0, 0.0),
            score_softmax_only=self.metric(0.7),
            aggregation_only=self.metric(1.0),
        )
        self.assertEqual(result, "score_or_softmax_is_the_remaining_boundary")

    def test_classifies_value_aggregation_boundary(self):
        result = classify_components(
            probability_metrics=self.metric(1.0),
            official_native_sanity=self.metric(1.0),
            c_style_sanity=self.metric(1.0, 0.0),
            score_softmax_only=self.metric(1.0),
            aggregation_only=self.metric(0.8),
        )
        self.assertEqual(result, "value_aggregation_is_the_remaining_boundary")

    def test_classifies_both_and_sanity_failures(self):
        both = classify_components(
            probability_metrics=self.metric(0.5),
            official_native_sanity=self.metric(1.0),
            c_style_sanity=self.metric(1.0, 0.0),
            score_softmax_only=self.metric(0.7),
            aggregation_only=self.metric(0.8),
        )
        self.assertEqual(both, "score_softmax_and_value_aggregation_both_contribute")
        official_fail = classify_components(
            probability_metrics=self.metric(0.5),
            official_native_sanity=self.metric(0.99),
            c_style_sanity=self.metric(1.0, 0.0),
            score_softmax_only=self.metric(0.7),
            aggregation_only=self.metric(0.8),
        )
        self.assertEqual(official_fail, "official_native_aggregation_sanity_failed")
        c_fail = classify_components(
            probability_metrics=self.metric(0.5),
            official_native_sanity=self.metric(1.0),
            c_style_sanity=self.metric(1.0, 1e-4),
            score_softmax_only=self.metric(0.7),
            aggregation_only=self.metric(0.8),
        )
        self.assertEqual(c_fail, "c_style_aggregation_sanity_failed")

    def test_rejects_bad_aggregation_geometry(self):
        with self.assertRaisesRegex(AttentionComponentError, "disagree"):
            native_bfloat16_value_aggregation(
                torch.zeros(2, 3, 2),
                torch.zeros(4, 2, 5),
            )
        with self.assertRaisesRegex(AttentionComponentError, "divisible"):
            repeat_values_by_head(
                torch.zeros(2, 6),
                heads=3,
                kv_heads=2,
                head_dim=3,
            )


if __name__ == "__main__":
    unittest.main()
