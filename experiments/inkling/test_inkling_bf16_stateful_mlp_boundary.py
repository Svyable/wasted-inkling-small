#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import diagnose_inkling_portable_bf16_composed_moe as composed
from diagnose_inkling_bf16_stateful_mlp_boundary import classify_ladder


def metric(raw: float = 1.0, bf16: float = 1.0):
    return {
        "raw_exact_fraction": raw,
        "bfloat16_exact_fraction": bf16,
    }


def exact_stages(count: int = 8):
    return {
        "routed_weight": [metric() for _ in range(count)],
        "shared_weight": [metric() for _ in range(count)],
        "moe_out": [metric() for _ in range(count)],
        "mlp_branch": [metric(raw=0.1, bf16=1.0) for _ in range(count)],
        "layer_out": [metric() for _ in range(count)],
    }


class StatefulMlpBoundaryTest(unittest.TestCase):
    def test_classifier_prefers_earliest_ladder_stage(self):
        stages = exact_stages()
        stages["routed_weight"][4] = metric(raw=0.5, bf16=0.5)
        stages["moe_out"][3] = metric(raw=0.5, bf16=0.5)
        decision = classify_ladder(stages)
        self.assertEqual(decision["first_boundary_stage"], "routed_weight")
        self.assertEqual(decision["first_boundary_position"], 4)

    def test_mlp_branch_contract_is_bfloat16_visible(self):
        stages = exact_stages()
        self.assertEqual(
            classify_ladder(stages)["classification"],
            "stateful_sparse_mlp_ladder_exact",
        )
        stages["mlp_branch"][3] = metric(raw=0.0, bf16=0.75)
        decision = classify_ladder(stages)
        self.assertEqual(decision["first_boundary_stage"], "mlp_branch")
        self.assertEqual(decision["first_boundary_position"], 3)

    def test_complete_candidate_compiles_and_preserves_production(self):
        with tempfile.TemporaryDirectory() as directory:
            library, source = composed.build_composed_library(
                Path(directory) / "libinkling-stateful-mlp-boundary.so"
            )
            self.assertTrue(library.is_file())
            self.assertTrue(source["production_source_unchanged"])


if __name__ == "__main__":
    unittest.main()
