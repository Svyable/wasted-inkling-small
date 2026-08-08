#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

# Install #39's complete temporary-source transform before importing composed.
import run_inkling_portable_bf16_complete_sparse_layer  # noqa: F401
import diagnose_inkling_portable_bf16_composed_moe as composed
from diagnose_inkling_bf16_router_tie_resolution import cutoff_partition
from diagnose_inkling_bf16_router_tie_output_sensitivity import (
    classify_impact,
    enumerate_routes,
    exact_route_row,
)


class TieOutputSensitivityTest(unittest.TestCase):
    def test_enumerates_every_valid_position_zero_style_subset(self):
        choice = torch.tensor(
            [9.0, 8.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            dtype=torch.bfloat16,
        )
        partition = cutoff_partition(choice, 4)
        routes = enumerate_routes(partition, 4)
        self.assertEqual(len(routes), 10)  # choose 2 of 5 tied experts
        self.assertEqual(len(set(routes)), 10)
        self.assertTrue(all(route[:2] == (0, 1) for route in routes))

    def test_enumerates_layer5_style_four_routes(self):
        choice = torch.tensor(
            [9.0, 8.0, 7.5, 7.0, 7.0, 7.0, 7.0],
            dtype=torch.bfloat16,
        )
        partition = cutoff_partition(choice, 6)
        routes = enumerate_routes(partition, 6)
        self.assertEqual(len(routes), 4)  # choose 3 of 4 tied experts

    def test_classifier_distinguishes_exact_bf16_and_material_change(self):
        def alt(raw: float, bf16: float):
            return {
                "stages": {
                    "layer_out": {
                        "raw_exact_fraction": raw,
                        "bfloat16_exact_fraction": bf16,
                    }
                }
            }

        self.assertEqual(
            classify_impact([alt(1.0, 1.0), alt(1.0, 1.0)]),
            "cutoff_tie_alternatives_are_layer_output_exact",
        )
        self.assertEqual(
            classify_impact([alt(0.5, 1.0), alt(0.7, 1.0)]),
            "cutoff_tie_alternatives_are_bfloat16_equivalent",
        )
        self.assertEqual(
            classify_impact([alt(0.5, 0.9)]),
            "cutoff_tie_alternatives_change_layer_output",
        )

    def test_exact_route_row_reproduces_official_normalization_formula(self):
        weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=torch.bfloat16,
        )
        gate = SimpleNamespace(
            weight=weight,
            num_experts=2,
            n_shared_experts=1,
            top_k=1,
            route_scale=2.0,
            global_scale=torch.tensor([0.5], dtype=torch.bfloat16),
        )
        module = SimpleNamespace(mlp=SimpleNamespace(gate=gate))
        hidden = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        row = exact_route_row(module, hidden, (1,))

        logits = F.linear(hidden.reshape(1, -1), weight)
        selected = torch.cat((logits[:, 1:2], logits[:, 2:3]), dim=1)
        log_probs = F.logsigmoid(selected)
        expected = torch.exp(log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True))
        expected = expected * 2.0 * gate.global_scale
        self.assertEqual(row.indices, (1,))
        self.assertEqual(row.weights, (float(expected[0, 0].float()),))

    def test_complete_candidate_compiles_without_production_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            library, source = composed.build_composed_library(
                Path(directory) / "libinkling-tie-impact.so"
            )
            self.assertTrue(library.is_file())
            self.assertTrue(source["production_source_unchanged"])
            self.assertEqual(
                source["layer_residual_policy"]["result"],
                "bfloat16_completed",
            )
            self.assertNotEqual(
                source["production_layer_sha256"], source["probe_layer_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
