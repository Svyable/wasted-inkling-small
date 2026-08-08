#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from diagnose_inkling_bf16_router_normalization_stages import (
    StageHelper,
    build_stage_helper,
    classify_positions,
    stage_prefix_lattice,
)


class _FakeHelper:
    def evaluate(self, logits, route_scale, global_scale, flags):
        del logits, route_scale, global_scale
        lse = 0.0 if (flags & 1) else 1.0
        return {
            "logsigmoid": torch.zeros(2, dtype=torch.float32),
            "logsumexp": torch.tensor([lse], dtype=torch.float32),
            "normalized": torch.ones(2, dtype=torch.float32),
            "after_route_scale": torch.ones(2, dtype=torch.float32),
            "final": torch.ones(2, dtype=torch.float32),
        }


class RouterNormalizationStagesTest(unittest.TestCase):
    def test_stage_helper_builds_and_returns_finite_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = StageHelper(
                build_stage_helper(Path(directory) / "router-stages.so")
            )
            value = helper.evaluate(
                torch.tensor([0.5, -1.0], dtype=torch.float32),
                route_scale=2.5,
                global_scale=4.0,
                flags=0,
            )
            self.assertEqual(tuple(value["logsigmoid"].shape), (2,))
            self.assertEqual(tuple(value["logsumexp"].shape), (1,))
            self.assertEqual(tuple(value["normalized"].shape), (2,))
            self.assertEqual(tuple(value["after_route_scale"].shape), (2,))
            self.assertEqual(tuple(value["final"].shape), (2,))
            for tensor in value.values():
                self.assertTrue(torch.isfinite(tensor).all())

    def test_prefix_lattice_reports_first_stage_where_survivors_die(self):
        official = {
            "logsigmoid": torch.zeros(2, dtype=torch.float32),
            "logsumexp": torch.zeros(1, dtype=torch.float32),
            "normalized": torch.zeros(2, dtype=torch.float32),
            "after_route_scale": torch.zeros(2, dtype=torch.float32),
            "final": torch.zeros(2, dtype=torch.float32),
        }
        result = stage_prefix_lattice(
            _FakeHelper(),
            torch.zeros(2, dtype=torch.float32),
            official,
            route_scale=1.0,
            global_scale=1.0,
        )
        self.assertEqual(
            result["stages"]["logsigmoid"]["prefix_exact_policy_count"],
            2048,
        )
        self.assertEqual(
            result["stages"]["logsumexp"]["prefix_exact_policy_count"],
            1024,
        )
        self.assertEqual(
            result["stages"]["normalized"]["prefix_exact_policy_count"],
            0,
        )
        self.assertEqual(result["first_zero_prefix_stage"], "normalized")
        self.assertEqual(result["final_exact_policy_count"], 0)

    def test_classifier_maps_first_failed_prefix_to_primitive(self):
        positions = [
            {
                "position": 0,
                "final_exact_policy_count": 10,
                "first_zero_prefix_stage": None,
            },
            {
                "position": 1,
                "final_exact_policy_count": 2,
                "first_zero_prefix_stage": None,
            },
            {
                "position": 2,
                "final_exact_policy_count": 4,
                "first_zero_prefix_stage": None,
            },
            {
                "position": 3,
                "final_exact_policy_count": 0,
                "first_zero_prefix_stage": "logsumexp",
            },
        ]
        decision = classify_positions(positions)
        self.assertEqual(
            decision["classification"],
            "router_logsumexp_reduction_mismatch",
        )
        self.assertEqual(decision["first_unmodeled_position"], 3)
        self.assertEqual(decision["first_unmodeled_stage"], "logsumexp")


if __name__ == "__main__":
    unittest.main()
