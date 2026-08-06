#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from diagnose_inkling_portable_bf16_single_token_moe import (
    MAT_ROUTED_DOWN,
    MAT_ROUTED_GATE,
    MAT_ROUTED_UP,
    STAGES,
    SingleTokenMoeError,
    build_single_token_moe_library,
    classify_stages,
    metrics,
    transform_moe_source,
)


class SingleTokenMoeTest(unittest.TestCase):
    @staticmethod
    def stage_metrics(exact: int):
        return {
            stage: {
                "bfloat16_exact_fraction": 1.0 if index < exact else 0.5
            }
            for index, stage in enumerate(STAGES)
        }

    def test_private_matrix_kinds_do_not_overlap_public_abi(self):
        self.assertEqual((MAT_ROUTED_GATE, MAT_ROUTED_UP, MAT_ROUTED_DOWN), (12, 13, 14))
        self.assertGreater(MAT_ROUTED_GATE, 11)

    def test_metrics_separates_raw_and_bfloat16_exactness(self):
        reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
        candidate = reference + torch.tensor([1.0e-4, 1.0e-2])
        result = metrics(reference, candidate)
        self.assertEqual(result["raw_exact_fraction"], 0.0)
        self.assertEqual(result["bfloat16_exact_fraction"], 0.5)
        with self.assertRaisesRegex(SingleTokenMoeError, "shape mismatch"):
            metrics(reference, torch.ones(3))

    def test_classification_stops_at_first_mismatch(self):
        partial = classify_stages(self.stage_metrics(3))
        self.assertEqual(
            partial["classification"],
            "single_token_moe_mismatch_isolated",
        )
        self.assertEqual(partial["first_nonexact_stage"], "routed_down0")
        complete = classify_stages(self.stage_metrics(len(STAGES)))
        self.assertEqual(
            complete["classification"],
            "single_token_sparse_moe_is_bfloat16_exact",
        )
        self.assertIsNone(complete["first_nonexact_stage"])

    def test_source_transform_is_fail_closed(self):
        source = """static float silu(float x) { return x / (1.0f + expf(-x)); }
            waste_inkling_expert_weights ew;
            if (get_routed(cfg, w, layer, s->routed_index[k],
                           expert_get, expert_ctx, &ew) ||
                expert_eval(s->branch, s->gate, s->up, &ew,
                            s->norm, hidden, inter)) return -1;
            const float gamma = s->routed_weight[k];
            for (int i = 0; i < hidden; i++) s->ff[i] += gamma * s->branch[i];
                if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_GATE, e,
                                 s->gate, NULL, s->norm, inter, hidden) ||
                    apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_UP, e,
                                 s->up, NULL, s->norm, inter, hidden)) return -1;
                for (int j = 0; j < inter; j++)
                    s->gate[j] = silu(s->gate[j]) * s->up[j];
                if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_DOWN, e,
                                 s->branch, NULL, s->gate, hidden, inter)) return -1;
"""
        transformed = transform_moe_source(source)
        self.assertIn("WASTE_IK_MAT_ROUTED_GATE_PROBE", transformed)
        self.assertIn('"routed_gated0"', transformed)
        self.assertIn('"shared_down0"', transformed)
        with self.assertRaisesRegex(SingleTokenMoeError, "private kind anchor"):
            transform_moe_source(transformed)

    def test_temporary_candidate_compiles_and_preserves_production(self):
        with tempfile.TemporaryDirectory() as directory:
            library, source = build_single_token_moe_library(
                Path(directory) / "libinkling-single-token-moe.so"
            )
            self.assertTrue(library.is_file())
            self.assertTrue(source["production_source_unchanged"])
            self.assertNotEqual(
                source["production_layer_sha256"],
                source["probe_layer_sha256"],
            )
            self.assertEqual(
                source["private_matrix_kinds"],
                {"routed_gate": 12, "routed_up": 13, "routed_down": 14},
            )


if __name__ == "__main__":
    unittest.main()
