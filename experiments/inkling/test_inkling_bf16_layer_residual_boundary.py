#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from diagnose_inkling_bf16_layer_residual_boundary import (
    classify_boundary,
    metrics,
    native_bfloat16_add,
    residual_variants,
    torch_fingerprint,
)
import diagnose_inkling_portable_bf16_composed_moe as composed


class LayerResidualBoundaryTest(unittest.TestCase):
    @staticmethod
    def exact_metric() -> dict[str, float]:
        return {
            "raw_exact_fraction": 1.0,
            "bfloat16_exact_fraction": 1.0,
        }

    def test_official_anchor_uses_native_bfloat16_add(self):
        residual = torch.tensor([1.0], dtype=torch.float32)
        branch = torch.tensor([0.00390625], dtype=torch.float32)
        out = native_bfloat16_add(residual, branch)
        self.assertEqual(float(out[0]), 1.0)
        self.assertNotEqual(
            float((residual.to(torch.bfloat16).float() + branch.to(torch.bfloat16).float())[0]),
            float(out[0]),
        )

    def test_hidden_fp32_branch_noise_requires_explicit_bfloat16_boundary(self):
        residual = torch.tensor([1.0], dtype=torch.float32)
        official_branch = torch.tensor([0.00390625], dtype=torch.float32)
        candidate_branch = official_branch + 1.0e-6
        self.assertEqual(
            metrics(official_branch, candidate_branch)["bfloat16_exact_fraction"],
            1.0,
        )
        official_out = native_bfloat16_add(residual, official_branch)
        variants = residual_variants(residual, candidate_branch)
        self.assertNotEqual(
            metrics(official_out, variants["float32_add_raw_branch"])["raw_exact_fraction"],
            1.0,
        )
        self.assertNotEqual(
            metrics(official_out, variants["float32_add_bfloat16_branch"])["raw_exact_fraction"],
            1.0,
        )
        self.assertNotEqual(
            metrics(official_out, variants["bfloat16_result_raw_branch"])["raw_exact_fraction"],
            1.0,
        )
        self.assertEqual(
            metrics(
                official_out,
                variants["bfloat16_result_bfloat16_branch"],
            )["raw_exact_fraction"],
            1.0,
        )
        self.assertEqual(
            metrics(official_out, variants["native_bfloat16_add"])["raw_exact_fraction"],
            1.0,
        )

    def test_classifier_prefers_explicit_bfloat16_policy(self):
        exact = self.exact_metric()
        branch = {
            "raw_exact_fraction": 0.1,
            "bfloat16_exact_fraction": 1.0,
        }
        variants = {
            "float32_add_raw_branch": {
                "raw_exact_fraction": 0.2,
                "bfloat16_exact_fraction": 0.8,
            },
            "bfloat16_result_bfloat16_branch": exact,
            "native_bfloat16_add": exact,
        }
        decision = classify_boundary(exact, exact, exact, branch, variants)
        self.assertEqual(
            decision["classification"],
            "candidate_layer_residual_policy_is_raw_exact",
        )
        self.assertEqual(
            decision["sufficient_candidate_policy"],
            "bfloat16_result_bfloat16_branch",
        )
        self.assertEqual(
            decision["raw_exact_policies"],
            ["bfloat16_result_bfloat16_branch", "native_bfloat16_add"],
        )

    def test_classifier_fails_closed_on_residual_input_drift(self):
        drift = {
            "raw_exact_fraction": 0.999,
            "bfloat16_exact_fraction": 1.0,
        }
        exact = self.exact_metric()
        decision = classify_boundary(drift, exact, exact, exact, {})
        self.assertEqual(
            decision["classification"],
            "pre_layer_residual_anchor_failed",
        )
        self.assertIsNone(decision["sufficient_candidate_policy"])

    def test_backend_fingerprint_is_explicit(self):
        fingerprint = torch_fingerprint()
        self.assertEqual(fingerprint["torch_version"], torch.__version__)
        self.assertTrue(fingerprint["cpu_dispatch"])
        self.assertIn("num_threads", fingerprint)
        self.assertIn("cpu_capabilities", fingerprint)

    def test_composed_candidate_compiles_and_preserves_production(self):
        with tempfile.TemporaryDirectory() as directory:
            library, source = composed.build_composed_library(
                Path(directory) / "libinkling-layer-residual.so"
            )
            self.assertTrue(library.is_file())
            self.assertTrue(source["production_source_unchanged"])
            self.assertEqual(
                source["aggregation_policy"]["routed_order"],
                "ascending_expert_id",
            )
            self.assertEqual(
                source["router_policy"],
                {"family": "logsumexp", "flags": 0x7E1},
            )


if __name__ == "__main__":
    unittest.main()
