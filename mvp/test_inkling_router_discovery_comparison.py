#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import unittest

from compare_inkling_router_discoveries import (
    RouterDiscoveryComparisonError,
    canonicalize_discovery,
    compare_discoveries,
)


def artifact() -> dict:
    return {
        "format": "inkling-router-discovery",
        "version": 1,
        "model_id": "fixture",
        "revision": "a" * 40,
        "config_sha256": "b" * 64,
        "index_sha256": "c" * 64,
        "tokens": 2,
        "seed": 19,
        "input_dtype": "bfloat16",
        "inputs_sha256": "d" * 64,
        "layers": {
            "2": {
                "seed_fixture_experts": [0],
                "selected_experts": [3, 7, 11],
                "topk_indices": [[7, 3], [11, 7]],
                "topk_weights": [[0.25, 0.5], [0.75, 0.125]],
            }
        },
    }


class RouterDiscoveryComparisonTest(unittest.TestCase):
    def test_accepts_different_topk_slot_order_with_pairs_intact(self) -> None:
        expected = artifact()
        actual = copy.deepcopy(expected)
        actual["layers"]["2"]["topk_indices"][0] = [3, 7]
        actual["layers"]["2"]["topk_weights"][0] = [0.5, 0.25]
        result = compare_discoveries(actual, expected)
        self.assertTrue(result["passed"])

    def test_rejects_weight_detached_from_its_expert(self) -> None:
        expected = artifact()
        actual = copy.deepcopy(expected)
        actual["layers"]["2"]["topk_indices"][0] = [3, 7]
        with self.assertRaisesRegex(
            RouterDiscoveryComparisonError, "pair-preserving routing mismatch"
        ):
            compare_discoveries(actual, expected)

    def test_rejects_duplicate_expert_or_false_selected_union(self) -> None:
        duplicate = artifact()
        duplicate["layers"]["2"]["topk_indices"][0] = [7, 7]
        with self.assertRaisesRegex(RouterDiscoveryComparisonError, "repeats"):
            canonicalize_discovery(duplicate)

        false_union = artifact()
        false_union["layers"]["2"]["selected_experts"] = [3, 7]
        with self.assertRaisesRegex(
            RouterDiscoveryComparisonError, "selected_experts"
        ):
            canonicalize_discovery(false_union)


if __name__ == "__main__":
    unittest.main()
