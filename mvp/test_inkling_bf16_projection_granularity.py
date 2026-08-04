#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import torch

from diagnose_inkling_bf16_projection_granularity import (
    ProjectionGranularityError,
    classify_projection,
)


class ProjectionGranularityTest(unittest.TestCase):
    def test_token_step_can_fully_explain_prefill_edges(self):
        candidate = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        prefill = torch.tensor([[1.0, 2.0], [3.0, 4.03125]])
        token_step = candidate.clone()
        result = classify_projection(
            prefill,
            token_step,
            candidate,
            dtype=torch.bfloat16,
        )
        self.assertEqual(result["classification"], "token_step_kernel_matches_candidate")
        self.assertEqual(result["candidate_vs_prefill_mismatch_positions"], [1])
        self.assertEqual(result["candidate_vs_token_step_mismatch_positions"], [])
        self.assertEqual(result["resolved_positions"], [1])

    def test_token_step_can_reduce_without_eliminating_edges(self):
        candidate = torch.tensor([[1.0], [2.0], [3.0]])
        prefill = torch.tensor([[1.03125], [2.03125], [3.0]])
        token_step = torch.tensor([[1.0], [2.03125], [3.0]])
        result = classify_projection(
            prefill,
            token_step,
            candidate,
            dtype=torch.bfloat16,
        )
        self.assertEqual(
            result["classification"],
            "token_step_kernel_reduces_candidate_mismatch",
        )
        self.assertEqual(result["resolved_positions"], [0])
        self.assertEqual(result["candidate_vs_token_step_mismatch_positions"], [1])

    def test_token_step_can_fail_to_explain_edges(self):
        candidate = torch.tensor([[1.0], [2.0]])
        prefill = torch.tensor([[1.03125], [2.0]])
        token_step = torch.tensor([[1.03125], [2.0]])
        result = classify_projection(
            prefill,
            token_step,
            candidate,
            dtype=torch.bfloat16,
        )
        self.assertEqual(
            result["classification"],
            "token_step_kernel_does_not_explain_candidate_mismatch",
        )

    def test_rejects_unaligned_shapes(self):
        with self.assertRaisesRegex(ProjectionGranularityError, "unaligned"):
            classify_projection(
                torch.zeros(2, 2),
                torch.zeros(2, 3),
                torch.zeros(2, 2),
                dtype=torch.bfloat16,
            )

    def test_all_equal_is_reported_separately(self):
        value = torch.tensor([[1.0], [2.0]])
        result = classify_projection(
            value,
            value,
            value,
            dtype=torch.bfloat16,
        )
        self.assertEqual(result["classification"], "all_kernels_match_candidate")


if __name__ == "__main__":
    unittest.main()
