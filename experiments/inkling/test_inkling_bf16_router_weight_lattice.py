#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from diagnose_inkling_bf16_router_weight_lattice import (
    F_FINAL_BF16,
    F_GLOBAL_SCALE_BF16,
    F_LOGP_BF16,
    F_LSE_BF16,
    F_NORMALIZED_BF16,
    F_OUTPUT_DELTA_BF16,
    F_ROUTE_SCALE_BF16,
    FixedWeightHelper,
    RouterWeightLatticeError,
    build_helper,
    classify,
    force_bias_for_order,
    metrics,
    official_manual_weights,
    policy_description,
)


class RouterWeightLatticeTest(unittest.TestCase):
    @staticmethod
    def metric(raw: float, bf16: float = 1.0):
        return {
            "raw_exact_fraction": raw,
            "bfloat16_exact_fraction": bf16,
        }

    def test_force_bias_preserves_requested_order(self):
        ids = torch.tensor([150, 67, 69, 90, 13, 238])
        bias = force_bias_for_order(256, ids)
        selected = torch.topk(bias, 6, sorted=True).indices
        self.assertTrue(torch.equal(selected, ids))
        self.assertLess(float(bias[0]), float(bias[13]))

    def test_official_manual_path_retains_bfloat16(self):
        selected = torch.tensor(
            [0.5, -0.25, 1.0, -2.0, 0.125, 0.75, -1.0, 0.0],
            dtype=torch.bfloat16,
        )
        values = official_manual_weights(
            selected,
            2.5,
            torch.tensor([4.0], dtype=torch.bfloat16),
        )
        for name in (
            "selected_logits",
            "logsigmoid",
            "logsumexp",
            "normalized",
            "after_route_scale",
            "final",
        ):
            self.assertEqual(values[name].dtype, torch.bfloat16)
        self.assertEqual(tuple(values["final"].shape), (8,))

    def test_classifies_route_only_and_portable_policy(self):
        route_only = classify(
            self.metric(1.0),
            {
                "all": self.metric(1.0),
                "routed": self.metric(1.0),
                "shared": self.metric(1.0),
            },
            [],
        )
        self.assertEqual(
            route_only["classification"],
            "router_weight_mismatch_is_route_choice_only",
        )
        exact = [
            {
                **policy_description(
                    1,
                    F_LOGP_BF16
                    | F_LSE_BF16
                    | F_OUTPUT_DELTA_BF16
                    | F_NORMALIZED_BF16
                    | F_ROUTE_SCALE_BF16
                    | F_GLOBAL_SCALE_BF16
                    | F_FINAL_BF16,
                ),
                "metrics": self.metric(1.0),
            }
        ]
        portable = classify(
            self.metric(1.0),
            {
                "all": self.metric(0.5),
                "routed": self.metric(0.5),
                "shared": self.metric(0.5),
            },
            exact,
        )
        self.assertEqual(
            portable["classification"],
            "portable_router_weight_policy_identified",
        )
        self.assertEqual(
            portable["preferred_policy"]["family"],
            "logsumexp",
        )

    def test_rejects_anchor_and_shape_failures(self):
        failed = classify(
            self.metric(0.9),
            {
                "all": self.metric(1.0),
                "routed": self.metric(1.0),
                "shared": self.metric(1.0),
            },
            [],
        )
        self.assertEqual(
            failed["classification"],
            "router_weight_anchor_failed",
        )
        with self.assertRaisesRegex(RouterWeightLatticeError, "shape mismatch"):
            metrics(torch.ones(2), torch.ones(3))

    def test_standalone_helper_compiles_and_executes_policy_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            library = build_helper(Path(directory) / "librouter-weights.so")
            self.assertTrue(library.is_file())
            helper = FixedWeightHelper(library)
            selected = torch.tensor(
                [0.5, -0.25, 1.0, -2.0, 0.125, 0.75, -1.0, 0.0],
                dtype=torch.bfloat16,
            )
            current = helper.evaluate(selected, 2.5, 4.0, 0, 0)
            official_like = helper.evaluate(
                selected,
                2.5,
                4.0,
                1,
                F_LOGP_BF16
                | F_LSE_BF16
                | F_OUTPUT_DELTA_BF16
                | F_NORMALIZED_BF16
                | F_ROUTE_SCALE_BF16
                | F_GLOBAL_SCALE_BF16
                | F_FINAL_BF16,
            )
            self.assertEqual(tuple(current.shape), (8,))
            self.assertEqual(tuple(official_like.shape), (8,))
            self.assertFalse(torch.equal(current, official_like))


if __name__ == "__main__":
    unittest.main()
