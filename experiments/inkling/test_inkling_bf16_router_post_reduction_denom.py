#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from diagnose_inkling_bf16_router_post_reduction_denom import (
    EXPECTED_FLAGS,
    PostReductionHelper,
    build_helper,
    classify,
)


def metric(exact: bool = True):
    return {"raw_exact_fraction": 1.0 if exact else 0.0}


def position(index: int, *, denom_exact: bool = True, exact_policy_count: int = 1):
    return {
        "position": index,
        "denom_anchor": metric(denom_exact),
        "exact_policy_count": exact_policy_count,
    }


class PostReductionDenomTest(unittest.TestCase):
    def test_helper_builds_and_completes_denominator_after_reduction(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = PostReductionHelper(
                build_helper(Path(directory) / "post-reduction-denom.so")
            )
            selected = torch.tensor(
                [-3.984375, -3.875, -3.78125, -2.328125,
                 -2.390625, -4.171875, -6.625, -6.90625],
                dtype=torch.bfloat16,
            )
            value = helper.evaluate(
                selected,
                route_scale=2.5,
                global_scale=4.0,
                flags=EXPECTED_FLAGS,
            )
            self.assertEqual(tuple(value["denom"].shape), (1,))
            self.assertEqual(tuple(value["final"].shape), (8,))
            for tensor in value.values():
                self.assertTrue(torch.isfinite(tensor).all())

    def test_classifier_prefers_expected_policy_when_it_generalizes(self):
        positions = [position(i) for i in range(8)]
        decision = classify(positions, [3, EXPECTED_FLAGS])
        self.assertEqual(
            decision["classification"],
            "expected_post_reduction_policy_is_raw_exact",
        )
        self.assertEqual(decision["preferred_policy"]["flags"], EXPECTED_FLAGS)
        self.assertIsNone(decision["first_failure_position"])

    def test_classifier_accepts_alternate_exact_policy_without_guessing(self):
        positions = [position(i) for i in range(8)]
        decision = classify(positions, [5])
        self.assertEqual(
            decision["classification"],
            "alternate_post_reduction_policy_is_raw_exact",
        )
        self.assertEqual(decision["preferred_policy"]["flags"], 5)

    def test_classifier_fails_closed_on_denominator_regression(self):
        positions = [position(i) for i in range(8)]
        positions[3] = position(3, denom_exact=False, exact_policy_count=0)
        decision = classify(positions, [])
        self.assertEqual(
            decision["classification"],
            "post_reduction_denominator_policy_failed",
        )
        self.assertEqual(decision["first_failure_position"], 3)
        self.assertIsNone(decision["preferred_policy"])

    def test_classifier_reports_unexplained_empty_intersection(self):
        positions = [position(i) for i in range(8)]
        positions[4] = position(4, exact_policy_count=0)
        decision = classify(positions, [])
        self.assertEqual(
            decision["classification"],
            "post_reduction_pipeline_still_unexplained",
        )
        self.assertEqual(decision["first_failure_position"], 4)


if __name__ == "__main__":
    unittest.main()
