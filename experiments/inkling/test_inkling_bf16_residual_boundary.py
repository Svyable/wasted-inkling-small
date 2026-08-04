#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import torch

from diagnose_inkling_bf16_residual_boundary import (
    ResidualBoundaryError,
    classify_boundary,
    metrics,
    residual_variants,
)


class ResidualBoundaryTest(unittest.TestCase):
    @staticmethod
    def metric(raw: float, bf16: float):
        return {
            "raw_exact_fraction": raw,
            "bfloat16_exact_fraction": bf16,
        }

    def test_metrics_distinguishes_raw_and_bfloat16_exactness(self):
        reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
        candidate = reference + torch.tensor([1.0e-4, 1.0e-2])
        result = metrics(reference, candidate)
        self.assertEqual(result["raw_exact_fraction"], 0.0)
        self.assertEqual(result["bfloat16_exact_fraction"], 0.5)
        with self.assertRaisesRegex(ResidualBoundaryError, "shape mismatch"):
            metrics(reference, torch.ones(3))

    def test_variants_keep_distinct_branch_and_result_policies(self):
        inputs = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        branch = torch.tensor([[0.0039, -0.0079]], dtype=torch.float32)
        variants = residual_variants(inputs, branch)
        self.assertEqual(
            set(variants),
            {
                "float32_add_raw_branch",
                "float32_add_bfloat16_branch",
                "bfloat16_result_raw_branch",
                "bfloat16_result_bfloat16_branch",
                "native_bfloat16_add",
            },
        )
        for value in variants.values():
            self.assertEqual(tuple(value.shape), (1, 2))

    def test_classifies_raw_exact_candidate_policy(self):
        decision = classify_boundary(
            self.metric(0.5, 1.0),
            self.metric(1.0, 1.0),
            {
                "float32_add_raw_branch": self.metric(1.0, 1.0),
                "native_bfloat16_add": self.metric(0.5, 1.0),
            },
        )
        self.assertEqual(
            decision["classification"],
            "candidate_residual_policy_is_raw_exact",
        )
        self.assertEqual(
            decision["sufficient_candidate_policy"],
            "float32_add_raw_branch",
        )

    def test_classifies_bfloat16_exact_candidate_policy(self):
        decision = classify_boundary(
            self.metric(0.5, 1.0),
            self.metric(1.0, 1.0),
            {
                "float32_add_raw_branch": self.metric(0.5, 0.8),
                "native_bfloat16_add": self.metric(0.5, 1.0),
            },
        )
        self.assertEqual(
            decision["classification"],
            "candidate_residual_policy_is_bfloat16_exact",
        )
        self.assertEqual(
            decision["sufficient_candidate_policy"],
            "native_bfloat16_add",
        )

    def test_classifies_hidden_fp32_branch_requirement(self):
        decision = classify_boundary(
            self.metric(0.75, 1.0),
            self.metric(1.0, 1.0),
            {
                "float32_add_raw_branch": self.metric(0.5, 0.8),
                "native_bfloat16_add": self.metric(0.5, 0.9),
            },
        )
        self.assertEqual(
            decision["classification"],
            "fp32_attention_branch_must_match_before_residual",
        )
        self.assertIsNone(decision["sufficient_candidate_policy"])

    def test_rejects_untrusted_official_anchor(self):
        decision = classify_boundary(
            self.metric(1.0, 1.0),
            self.metric(0.9, 1.0),
            {"native_bfloat16_add": self.metric(1.0, 1.0)},
        )
        self.assertEqual(
            decision["classification"],
            "official_residual_reconstruction_failed",
        )


if __name__ == "__main__":
    unittest.main()
