#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import torch

from diagnose_inkling_bf16_router_tie_resolution import (
    classify_layer,
    cutoff_partition,
    deterministic_id_rule,
    expert_permutations,
    topk_after_permutation,
    valid_under_partition,
)


class RouterTieResolutionTest(unittest.TestCase):
    def test_cutoff_partition_and_id_rules(self):
        choice = torch.tensor([9.0, 8.0, 7.0, 7.0, 7.0], dtype=torch.bfloat16)
        partition = cutoff_partition(choice, 3)
        self.assertEqual(partition["above"].tolist(), [0, 1])
        self.assertEqual(partition["tied"].tolist(), [2, 3, 4])
        self.assertEqual(partition["slots"], 1)
        self.assertTrue(partition["ambiguous"])
        self.assertEqual(
            deterministic_id_rule(partition, highest=False),
            [0, 1, 2],
        )
        self.assertEqual(
            deterministic_id_rule(partition, highest=True),
            [0, 1, 4],
        )

    def test_cutoff_validity_accepts_only_exact_equivalence_class(self):
        choice = torch.tensor([9.0, 8.0, 7.0, 7.0, 7.0], dtype=torch.bfloat16)
        partition = cutoff_partition(choice, 3)
        self.assertTrue(valid_under_partition([0, 1, 2], partition, 3))
        self.assertTrue(valid_under_partition([0, 1, 4], partition, 3))
        self.assertFalse(valid_under_partition([0, 2, 3], partition, 3))
        self.assertFalse(valid_under_partition([0, 1, 2, 3], partition, 3))

    def test_permutations_are_bijections_and_remap_to_original_ids(self):
        choice = torch.tensor([5.0, 4.0, 3.0, 3.0, 3.0, 1.0, 0.0, -1.0])
        partition = cutoff_partition(choice, 3)
        for name, permutation in expert_permutations(choice.numel()).items():
            self.assertEqual(torch.unique(permutation).numel(), choice.numel(), name)
            selected = topk_after_permutation(choice, 3, permutation)
            self.assertTrue(
                valid_under_partition(selected, partition, 3),
                (name, selected.tolist()),
            )

    def test_classifier_prioritizes_permutation_sensitivity(self):
        rows = [
            {
                "ambiguous_cutoff": True,
                "lowest_id_rule_matches": True,
                "highest_id_rule_matches": False,
                "permutation_changed_subset": True,
            },
            {
                "ambiguous_cutoff": True,
                "lowest_id_rule_matches": True,
                "highest_id_rule_matches": False,
                "permutation_changed_subset": False,
            },
        ]
        decision = classify_layer(rows)
        self.assertEqual(decision["original_id_rule"], "lowest_expert_ids")
        self.assertTrue(decision["permutation_sensitive"])
        self.assertEqual(
            decision["classification"],
            "official_topk_tie_subset_is_order_sensitive",
        )

    def test_classifier_reports_high_id_rule_when_stable(self):
        rows = [
            {
                "ambiguous_cutoff": True,
                "lowest_id_rule_matches": False,
                "highest_id_rule_matches": True,
                "permutation_changed_subset": False,
            }
        ]
        decision = classify_layer(rows)
        self.assertEqual(decision["original_id_rule"], "highest_expert_ids")
        self.assertFalse(decision["permutation_sensitive"])
        self.assertEqual(
            decision["classification"],
            "official_cutoff_ties_follow_high_id_rule",
        )

    def test_classifier_handles_unambiguous_rows(self):
        decision = classify_layer(
            [
                {
                    "ambiguous_cutoff": False,
                    "lowest_id_rule_matches": True,
                    "highest_id_rule_matches": True,
                    "permutation_changed_subset": False,
                }
            ]
        )
        self.assertEqual(
            decision["classification"],
            "no_ambiguous_bfloat16_cutoff_rows",
        )
        self.assertEqual(decision["ambiguous_rows"], 0)


if __name__ == "__main__":
    unittest.main()
