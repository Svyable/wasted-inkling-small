#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

from diagnose_inkling_bf16_router_prefill_lattice import (
    LEGACY_FLAGS,
    _intersection,
    _joint_intersection,
    classify,
)


def policy(family: str, flags: int):
    return {"family": family, "flags": flags}


def weight_metric(raw: float = 1.0):
    return {"all": {"raw_exact_fraction": raw}}


def position(
    index: int,
    *,
    selected_raw: float = 1.0,
    legacy_batched_raw: float = 1.0,
    legacy_rowwise_raw: float = 1.0,
):
    return {
        "position": index,
        "selected_logits": {"raw_exact_fraction": selected_raw},
        "legacy_batched_weights": weight_metric(legacy_batched_raw),
        "legacy_rowwise_weights": weight_metric(legacy_rowwise_raw),
    }


class RouterPrefillLatticeTest(unittest.TestCase):
    def test_intersection_retains_only_all_position_policies(self):
        records = [
            [policy("logsumexp", LEGACY_FLAGS), policy("ratio", 0x001)],
            [policy("logsumexp", LEGACY_FLAGS), policy("logsumexp", 0x021)],
            [policy("logsumexp", LEGACY_FLAGS)],
        ]
        self.assertEqual(_intersection(records), [(1, LEGACY_FLAGS)])
        self.assertEqual(
            _joint_intersection([(1, LEGACY_FLAGS), (1, 0x021)], [(1, 0x021)]),
            [(1, 0x021)],
        )

    def test_legacy_policy_generalizes_when_exact_on_both_shapes(self):
        positions = [position(i, selected_raw=0.75 if i == 3 else 1.0) for i in range(8)]
        decision = classify(
            positions,
            [(1, LEGACY_FLAGS)],
            [(1, LEGACY_FLAGS)],
            [(1, LEGACY_FLAGS)],
        )
        self.assertEqual(
            decision["classification"],
            "legacy_router_weight_policy_generalizes",
        )
        self.assertTrue(decision["legacy_exact_on_all_rowwise_positions"])
        self.assertTrue(decision["legacy_exact_on_all_batched_positions"])

    def test_joint_policy_wins_over_harmless_projection_shape_difference(self):
        positions = [
            position(
                i,
                selected_raw=0.75 if i == 3 else 1.0,
                legacy_batched_raw=0.5 if i == 3 else 1.0,
                legacy_rowwise_raw=0.5 if i == 3 else 1.0,
            )
            for i in range(8)
        ]
        joint = (1, 0x761)
        decision = classify(
            positions,
            [joint],
            [joint],
            [joint],
        )
        self.assertEqual(
            decision["classification"],
            "router_prefill_policy_generalized_across_shapes",
        )
        self.assertEqual(decision["preferred_policy"]["flags"], joint[1])
        self.assertEqual(decision["first_selected_logit_mismatch_position"], 3)

    def test_projection_shape_is_hard_boundary_when_no_joint_policy_exists(self):
        positions = [
            position(
                i,
                selected_raw=0.5 if i == 3 else 1.0,
                legacy_batched_raw=1.0,
                legacy_rowwise_raw=0.5 if i == 3 else 1.0,
            )
            for i in range(8)
        ]
        decision = classify(
            positions,
            [(1, 0x761)],
            [(1, LEGACY_FLAGS)],
            [],
        )
        self.assertEqual(
            decision["classification"],
            "router_prefill_projection_shape_mismatch",
        )
        self.assertEqual(decision["first_selected_logit_mismatch_position"], 3)
        self.assertIsNone(decision["preferred_policy"])


if __name__ == "__main__":
    unittest.main()
