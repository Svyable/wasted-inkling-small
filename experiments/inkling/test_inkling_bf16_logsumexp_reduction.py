#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from diagnose_inkling_bf16_logsumexp_reduction import (
    PrimitiveHelper,
    any_bf16_tree_sum,
    bf16_round_scalar,
    build_helper,
    classify_position,
)


def metric(exact: bool = True):
    return {"raw_exact_fraction": 1.0 if exact else 0.0}


class LogsumexpReductionTest(unittest.TestCase):
    def test_primitive_helper_builds_and_returns_finite_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = PrimitiveHelper(
                build_helper(Path(directory) / "logsumexp-primitives.so")
            )
            value = helper.evaluate(
                torch.tensor([-0.5, -1.0, -2.0], dtype=torch.bfloat16)
            )
            self.assertEqual(tuple(value["max"].shape), (1,))
            self.assertEqual(tuple(value["delta"].shape), (3,))
            self.assertEqual(tuple(value["terms"].shape), (3,))
            for tensor in value.values():
                self.assertTrue(torch.isfinite(tensor).all())

    def test_bf16_tree_search_finds_associated_sum(self):
        values = torch.tensor([1.0, 0.5, 0.25, 0.25], dtype=torch.float32)
        search = any_bf16_tree_sum(values, 2.0)
        self.assertTrue(search["reachable"])
        self.assertGreaterEqual(search["possible_result_count"], 1)
        self.assertIsNotNone(search["witness"])
        miss = any_bf16_tree_sum(values, bf16_round_scalar(2.5))
        self.assertFalse(miss["reachable"])

    def test_classifier_distinguishes_exp_reduction_log_and_final_add(self):
        base = {
            "torch_staged_lse": metric(),
            "scalar_max": metric(),
            "scalar_delta": metric(),
            "scalar_terms": metric(),
            "scalar_left_sum": metric(),
            "scalar_f32_sum": metric(),
            "scalar_logdenom": metric(),
            "scalar_lse": metric(),
        }
        torch_values = {}
        scalar = {}

        exp_case = dict(base)
        exp_case["scalar_terms"] = metric(False)
        self.assertEqual(
            classify_position(
                torch_values, scalar, exp_case,
                {"reachable": False},
            )["classification"],
            "logsumexp_exp_primitive_mismatch",
        )

        reduction_case = dict(base)
        reduction_case["scalar_left_sum"] = metric(False)
        reduction_case["scalar_f32_sum"] = metric(False)
        self.assertEqual(
            classify_position(
                torch_values, scalar, reduction_case,
                {"reachable": True},
            )["classification"],
            "logsumexp_reduction_order_mismatch",
        )

        log_case = dict(base)
        log_case["scalar_logdenom"] = metric(False)
        self.assertEqual(
            classify_position(
                torch_values, scalar, log_case,
                {"reachable": False},
            )["classification"],
            "logsumexp_log_primitive_mismatch",
        )

        add_case = dict(base)
        add_case["scalar_lse"] = metric(False)
        self.assertEqual(
            classify_position(
                torch_values, scalar, add_case,
                {"reachable": False},
            )["classification"],
            "logsumexp_final_add_mismatch",
        )

    def test_classifier_refuses_nonexact_torch_decomposition_anchor(self):
        comparisons = {
            "torch_staged_lse": metric(False),
            "scalar_max": metric(),
            "scalar_delta": metric(),
            "scalar_terms": metric(),
            "scalar_left_sum": metric(),
            "scalar_f32_sum": metric(),
            "scalar_logdenom": metric(),
            "scalar_lse": metric(),
        }
        decision = classify_position({}, {}, comparisons, {"reachable": False})
        self.assertEqual(
            decision["classification"],
            "torch_logsumexp_internal_kernel_boundary",
        )


if __name__ == "__main__":
    unittest.main()
