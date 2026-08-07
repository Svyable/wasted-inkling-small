#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import diagnose_inkling_portable_bf16_composed_moe as implementation
from run_inkling_portable_bf16_complete_layer import (
    transform_final_residual_source,
    validate_complete_result,
)


class CompleteLayerTest(unittest.TestCase):
    def test_final_residual_transform_is_exactly_scoped(self):
        source = """    if (trace_f(trace, layer, \"mlp_branch\", s->ff, (size_t)hidden)) return -1;
    for (int i = 0; i < hidden; i++) x[i] += s->ff[i];
    if (trace_f(trace, layer, \"layer_out\", x, (size_t)hidden)) return -1;
"""
        transformed = transform_final_residual_source(source)
        self.assertIn("const float residual = bf16_round_probe(x[i]);", transformed)
        self.assertIn("const float branch = bf16_round_probe(s->ff[i]);", transformed)
        self.assertIn("x[i] = bf16_round_probe(residual + branch);", transformed)
        self.assertNotIn("x[i] += s->ff[i]", transformed)

    def test_final_residual_transform_fails_closed(self):
        with self.assertRaisesRegex(
            implementation.ComposedMoeError,
            "expected exactly one final layer residual; found 0",
        ):
            transform_final_residual_source("return 0;\n")
        duplicate = (
            "    for (int i = 0; i < hidden; i++) x[i] += s->ff[i];\n" * 2
        )
        with self.assertRaisesRegex(
            implementation.ComposedMoeError,
            "expected exactly one final layer residual; found 2",
        ):
            transform_final_residual_source(duplicate)

    def test_complete_result_requires_raw_exact_layer_out(self):
        result = {
            "source": {
                "production_source_unchanged": True,
                "layer_residual_policy": {"result": "bfloat16_completed"},
            },
            "layers": {
                "2": {
                    "decision": {
                        "classification": "single_token_sparse_layer_is_bfloat16_exact",
                        "first_nonexact_stage": None,
                    },
                    "stages": {
                        "layer_out": {
                            "raw_exact_fraction": 1.0,
                            "bfloat16_exact_fraction": 1.0,
                        }
                    },
                }
            },
        }
        validate_complete_result(result)
        result["layers"]["2"]["stages"]["layer_out"]["raw_exact_fraction"] = 0.9
        with self.assertRaisesRegex(
            implementation.ComposedMoeError,
            "layer 2 layer_out is not raw-exact",
        ):
            validate_complete_result(result)

    def test_complete_result_rejects_longer_but_incomplete_prefix(self):
        result = {
            "source": {
                "production_source_unchanged": True,
                "layer_residual_policy": {"result": "bfloat16_completed"},
            },
            "layers": {
                "5": {
                    "decision": {
                        "classification": "composed_moe_mismatch_isolated",
                        "first_nonexact_stage": "layer_out",
                    },
                    "stages": {
                        "layer_out": {
                            "raw_exact_fraction": 0.8,
                            "bfloat16_exact_fraction": 0.9,
                        }
                    },
                }
            },
        }
        with self.assertRaisesRegex(
            implementation.ComposedMoeError,
            "layer 5 did not become BF16-exact",
        ):
            validate_complete_result(result)

    def test_complete_candidate_compiles_and_preserves_production(self):
        with tempfile.TemporaryDirectory() as directory:
            library, source = implementation.build_composed_library(
                Path(directory) / "libinkling-complete-layer.so"
            )
            self.assertTrue(library.is_file())
            self.assertTrue(source["production_source_unchanged"])
            self.assertEqual(
                source["layer_residual_policy"],
                {
                    "residual_operand": "bfloat16_completed",
                    "branch_operand": "bfloat16_completed",
                    "sum": "float32_of_bfloat16_operands",
                    "result": "bfloat16_completed",
                },
            )
            self.assertNotEqual(
                source["production_layer_sha256"],
                source["probe_layer_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
