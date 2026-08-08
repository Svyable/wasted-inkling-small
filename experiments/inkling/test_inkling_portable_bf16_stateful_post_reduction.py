#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import torch

from run_inkling_portable_bf16_stateful_post_reduction import (
    EXPECTED_FLAGS,
    PostReductionWeightAdapter,
    StatefulPostReductionError,
    active_router_policy,
    classify_layers,
)


class _Helper:
    def __init__(self, *, bad: str | None = None) -> None:
        self.bad = bad
        self.calls: list[tuple[int, float, float, int]] = []

    def evaluate(self, selected, route_scale, global_scale, flags):
        self.calls.append(
            (int(selected.numel()), float(route_scale), float(global_scale), flags)
        )
        final = torch.arange(selected.numel(), dtype=torch.float32) / 8.0
        denominator = torch.tensor([2.5], dtype=torch.float32)
        if self.bad == "width":
            final = final[:-1]
        elif self.bad == "denom":
            denominator = torch.tensor([], dtype=torch.float32)
        elif self.bad == "finite":
            final[0] = float("nan")
        return {"final": final, "denom": denominator}


def analysis(classification: str, stage=None, position=None):
    return {
        "decision": {
            "classification": classification,
            "first_boundary_stage": stage,
            "first_boundary_position": position,
        }
    }


class StatefulPostReductionTest(unittest.TestCase):
    def test_adapter_fixes_expected_policy_and_preserves_call_provenance(self):
        helper = _Helper()
        adapter = PostReductionWeightAdapter(helper)
        selected = torch.linspace(-4.0, 1.0, 8, dtype=torch.float32)
        result = adapter.evaluate(
            selected,
            route_scale=2.5,
            global_scale=4.0,
            legacy_family=1,
            legacy_flags=123,
        )
        self.assertEqual(tuple(result.shape), (8,))
        self.assertEqual(helper.calls, [(8, 2.5, 4.0, EXPECTED_FLAGS)])
        self.assertEqual(adapter.calls[0]["denominator"], 2.5)
        self.assertEqual(adapter.calls[0]["policy_flags"], EXPECTED_FLAGS)
        self.assertEqual(
            adapter.calls[0]["superseded_collector_request"],
            {"family": 1, "flags": 123},
        )

    def test_adapter_fails_closed_on_invalid_helper_results(self):
        selected = torch.ones(8)
        for bad in ("width", "denom", "finite"):
            with self.subTest(bad=bad):
                adapter = PostReductionWeightAdapter(_Helper(bad=bad))
                with self.assertRaises(StatefulPostReductionError):
                    adapter.evaluate(selected, 2.5, 4.0)
                self.assertEqual(adapter.calls, [])
        with self.assertRaisesRegex(StatefulPostReductionError, "outside"):
            PostReductionWeightAdapter(_Helper()).evaluate(
                torch.empty(0), 2.5, 4.0
            )

    def test_classifier_scopes_exactness_to_tested_ladders(self):
        result = classify_layers(
            {"2": analysis("stateful_sparse_mlp_ladder_exact")}
        )
        self.assertEqual(
            result["classification"],
            "tested_stateful_sparse_mlp_ladders_exact",
        )
        self.assertEqual(result["tested_layers"], [2])
        self.assertEqual(result["exact_layers"], [2])
        self.assertEqual(result["nonexact_layers"], [])

    def test_classifier_preserves_first_mismatch(self):
        result = classify_layers(
            {
                "2": analysis(
                    "stateful_moe_out_mismatch", "moe_out", 3
                ),
                "5": analysis("stateful_sparse_mlp_ladder_exact"),
            }
        )
        self.assertEqual(
            result["classification"],
            "tested_stateful_sparse_mlp_ladder_mismatch",
        )
        self.assertEqual(result["exact_layers"], [5])
        self.assertEqual(result["nonexact_layers"], [2])
        self.assertEqual(
            result["first_boundaries"]["2"],
            {
                "classification": "stateful_moe_out_mismatch",
                "stage": "moe_out",
                "position": 3,
            },
        )

    def test_policy_metadata_names_the_new_denominator_boundary(self):
        policy = active_router_policy()
        self.assertEqual(policy["policy"]["flags"], EXPECTED_FLAGS)
        self.assertEqual(
            policy["denominator_reduction"]["completion"],
            "bfloat16_after_full_reduction",
        )
        self.assertTrue(policy["injected_via_trace_adapter"])


if __name__ == "__main__":
    unittest.main()
