#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

from diagnose_inkling_bf16_attention_head_norm import (
    AttentionHeadNormError,
    classify_attention_boundary,
    transform_attention_head_rms,
)


class AttentionHeadNormProbeTest(unittest.TestCase):
    @staticmethod
    def source() -> str:
        return (
            '#include "inkling_attention.h"\n\n'
            '#include <math.h>\n#include <stddef.h>\n#include <string.h>\n\n'
            'static void rmsnorm_head(float *out, const float *x, '
            'const float *weight, int dim, float eps)\n{\n'
            '    double ss = 0.0;\n'
            '    const float scale = 1.0f;\n'
            '    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * weight[i];\n'
            '}\n'
            'static void score(void) { score += tau * bias; }\n'
        )

    @staticmethod
    def variant(exact: float, maximum: float, mean: float):
        return {
            "stage_comparison": {
                "stages": {
                    "attention_out": {
                        "quantized_exact_fraction": exact,
                        "max_abs": maximum,
                        "mean_abs": mean,
                    }
                }
            },
            "exact_bfloat16_prefix": ["input_norm", "q_proj"],
            "next_stage_after_exact_prefix": "attention_out",
        }

    def test_transform_changes_only_head_rmsnorm(self):
        transformed = transform_attention_head_rms(self.source())
        self.assertIn("bf16_round_attention_probe", transformed)
        self.assertIn("const float normalized", transformed)
        self.assertIn("score += tau * bias", transformed)
        self.assertNotIn(
            "out[i] = x[i] * scale * weight[i]", transformed
        )

    def test_transform_fails_closed_on_reapply_or_drift(self):
        transformed = transform_attention_head_rms(self.source())
        with self.assertRaisesRegex(AttentionHeadNormError, "include anchor"):
            transform_attention_head_rms(transformed)
        with self.assertRaisesRegex(AttentionHeadNormError, "head RMSNorm"):
            transform_attention_head_rms(
                self.source().replace(
                    "out[i] = x[i] * scale * weight[i]",
                    "out[i] = x[i]",
                )
            )

    def test_classifies_sufficient_head_norm(self):
        result = classify_attention_boundary(
            self.variant(0.5, 0.2, 0.02),
            self.variant(1.0, 0.0, 0.0),
        )
        self.assertEqual(result["classification"], "head_rmsnorm_is_sufficient")

    def test_classifies_monotonic_improvement(self):
        result = classify_attention_boundary(
            self.variant(0.5, 0.2, 0.02),
            self.variant(0.8, 0.1, 0.01),
        )
        self.assertEqual(
            result["classification"],
            "head_rmsnorm_improves_but_attention_math_still_differs",
        )
        self.assertGreater(result["exact_fraction_delta"], 0.0)
        self.assertLess(result["max_abs_delta"], 0.0)

    def test_classifies_no_effect(self):
        baseline = self.variant(0.5, 0.2, 0.02)
        result = classify_attention_boundary(baseline, self.variant(0.5, 0.2, 0.02))
        self.assertEqual(result["classification"], "head_rmsnorm_has_no_effect")

    def test_classifies_nonmonotonic_change(self):
        result = classify_attention_boundary(
            self.variant(0.5, 0.2, 0.02),
            self.variant(0.7, 0.3, 0.01),
        )
        self.assertEqual(
            result["classification"],
            "head_rmsnorm_changes_attention_without_monotonic_improvement",
        )

    def test_rejects_invalid_metrics(self):
        with self.assertRaisesRegex(AttentionHeadNormError, "invalid candidate"):
            classify_attention_boundary(
                self.variant(0.5, 0.2, 0.02),
                self.variant(1.1, 0.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
