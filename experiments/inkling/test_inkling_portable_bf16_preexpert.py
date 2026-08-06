#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from diagnose_inkling_native_bf16_backend import transform_rms_source
from diagnose_inkling_portable_bf16_preexpert import (
    PortableBfloat16PreExpertError,
    build_portable_preexpert_library,
    classify_preexpert,
    transform_residual_source,
)


class PortableBfloat16PreExpertTest(unittest.TestCase):
    @staticmethod
    def comparison(exact_points: int):
        points = (
            "input_norm",
            "q_proj",
            "k_proj",
            "v_proj",
            "relative_proj_input",
            "k_sconv",
            "v_sconv",
            "attention_out",
            "attention_branch",
            "post_attention_residual",
            "post_attention_norm",
        )
        return {
            "stages": {
                point: {
                    "quantized_exact_fraction": 1.0 if index < exact_points else 0.5
                }
                for index, point in enumerate(points)
            }
        }

    def test_residual_transform_applies_once_and_reapplication_fails(self):
        source = """#include <math.h>
#include <stddef.h>
#include <string.h>

static void rmsnorm(float *out, const float *x, const float *weight,
                    int n, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

void step(float *x, int hidden)
{
    for (int i = 0; i < hidden; i++) x[i] += s->branch[i];
}
"""
        transformed = transform_residual_source(transform_rms_source(source))
        self.assertIn("bf16_round_probe(s->branch[i])", transformed)
        self.assertIn("x[i] = bf16_round_probe(residual + branch);", transformed)
        with self.assertRaisesRegex(
            PortableBfloat16PreExpertError,
            "post-attention residual anchor",
        ):
            transform_residual_source(transformed)

    def test_transform_requires_one_bfloat16_helper(self):
        with self.assertRaisesRegex(
            PortableBfloat16PreExpertError,
            "exactly one BF16 helper",
        ):
            transform_residual_source(
                "    for (int i = 0; i < hidden; i++) x[i] += s->branch[i];\n"
            )

    def test_classifies_complete_preexpert_path(self):
        decision = classify_preexpert(
            self.comparison(11),
            {"passed": True},
        )
        self.assertEqual(
            decision["classification"],
            "complete_preexpert_path_is_bfloat16_exact",
        )
        self.assertIsNone(decision["first_nonexact_stage"])

    def test_classifies_activation_and_route_failures(self):
        activation = classify_preexpert(
            self.comparison(9),
            {"passed": True},
        )
        self.assertEqual(
            activation["classification"],
            "preexpert_activation_mismatch_remains",
        )
        self.assertEqual(
            activation["first_nonexact_stage"],
            "post_attention_residual",
        )
        route = classify_preexpert(
            self.comparison(11),
            {"passed": False},
        )
        self.assertEqual(
            route["classification"],
            "preexpert_activations_exact_but_route_audit_failed",
        )

    def test_temporary_candidate_compiles_without_changing_production(self):
        with tempfile.TemporaryDirectory() as directory:
            library, source = build_portable_preexpert_library(
                Path(directory) / "libinkling-preexpert.so"
            )
            self.assertTrue(library.is_file())
            self.assertTrue(source["production_source_unchanged"])
            self.assertNotEqual(
                source["production_layer_sha256"],
                source["probe_layer_sha256"],
            )
            self.assertNotEqual(
                source["production_attention_sha256"],
                source["probe_attention_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
