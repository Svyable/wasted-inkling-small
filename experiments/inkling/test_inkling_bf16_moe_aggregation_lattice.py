#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from diagnose_inkling_bf16_moe_aggregation_lattice import (
    MoeAggregationError,
    classify_lattice,
    metrics,
    routed_variants,
    shared_variants,
)
from diagnose_inkling_portable_bf16_single_token_moe_activation import (
    build_activation_library,
)


class MoeAggregationLatticeTest(unittest.TestCase):
    @staticmethod
    def metric(raw: float = 1.0, bf16: float = 1.0):
        return {
            "raw_exact_fraction": raw,
            "bfloat16_exact_fraction": bf16,
        }

    def test_classifies_complete_policy(self):
        anchors = {
            "c": self.metric(),
            "routed": self.metric(),
            "shared": self.metric(),
            "final": self.metric(),
        }
        routed = {
            "route_order_bfloat16_sequential": self.metric(),
            "expert_order_bfloat16_sequential": self.metric(),
            "expert_order_native_index_add": self.metric(),
        }
        shared = {
            "official_gamma_after_down_bfloat16_contribution_float32_sum_bfloat16": self.metric(0.5, 1.0),
            "official_gamma_before_down_float32_sum_bfloat16": self.metric(),
            "official_gamma_before_down_bfloat16_sequential": self.metric(),
        }
        final = {
            "float32_add_result_bfloat16": self.metric(),
            "native_bfloat16_add": self.metric(),
        }
        decision = classify_lattice(
            anchors, routed, shared, final, self.metric(0.5, 1.0)
        )
        self.assertEqual(
            decision["classification"],
            "moe_weighting_and_accumulation_policy_identified",
        )
        self.assertEqual(
            decision["routed_policy"],
            "route_order_bfloat16_sequential",
        )
        self.assertEqual(
            decision["shared_policy"],
            "official_gamma_before_down_float32_sum_bfloat16",
        )
        self.assertTrue(decision["candidate_shared_weights_bfloat16_exact"])

    def test_rejects_failed_anchor(self):
        decision = classify_lattice(
            {"c": self.metric(0.9, 1.0)},
            {},
            {},
            {},
            self.metric(),
        )
        self.assertEqual(
            decision["classification"],
            "aggregation_anchor_failed",
        )
        self.assertEqual(decision["failed_anchors"], ["c"])

    def test_metrics_rejects_shape_drift(self):
        reference = torch.tensor([1.0, 2.0])
        candidate = reference + torch.tensor([1.0e-4, 1.0e-2])
        result = metrics(reference, candidate)
        self.assertEqual(result["raw_exact_fraction"], 0.0)
        self.assertEqual(result["bfloat16_exact_fraction"], 0.5)
        with self.assertRaisesRegex(MoeAggregationError, "shape mismatch"):
            metrics(reference, torch.ones(3))

    def test_routed_variants_preserve_all_policies(self):
        ids = [4, 1]
        downs = [
            torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            torch.tensor([3.0, -1.0], dtype=torch.bfloat16),
        ]
        weights = torch.tensor([0.25, 0.75], dtype=torch.bfloat16)
        variants = routed_variants(ids, downs, weights)
        self.assertEqual(
            set(variants),
            {
                "route_order_float32",
                "route_order_bfloat16_contribution_float32_sum",
                "route_order_bfloat16_contribution_float32_sum_bfloat16",
                "route_order_bfloat16_sequential",
                "expert_order_bfloat16_sequential",
                "expert_order_native_index_add",
            },
        )
        for value in variants.values():
            self.assertEqual(tuple(value.shape), (2,))

    def test_shared_variants_preserve_gamma_placement(self):
        components = {
            "official_gammas": torch.tensor([0.25, 0.75], dtype=torch.bfloat16),
            "candidate_gammas": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "unweighted_downs": [
                torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
                torch.tensor([3.0, -1.0], dtype=torch.bfloat16),
            ],
            "official_before_downs": [
                torch.tensor([0.25, 0.5], dtype=torch.bfloat16),
                torch.tensor([2.25, -0.75], dtype=torch.bfloat16),
            ],
            "candidate_before_downs": [
                torch.tensor([0.25, 0.5], dtype=torch.bfloat16),
                torch.tensor([2.25, -0.75], dtype=torch.bfloat16),
            ],
        }
        variants = shared_variants(components)
        self.assertEqual(len(variants), 6)
        for value in variants.values():
            self.assertEqual(tuple(value.shape), (2,))

    def test_parent_candidate_still_compiles_without_production_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            library, source = build_activation_library(
                Path(directory) / "libinkling-aggregation-parent.so"
            )
            self.assertTrue(library.is_file())
            self.assertTrue(source["production_source_unchanged"])


if __name__ == "__main__":
    unittest.main()
