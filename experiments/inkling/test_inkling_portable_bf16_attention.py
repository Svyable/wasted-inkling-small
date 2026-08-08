#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from diagnose_inkling_portable_bf16_attention import (
    PortableBfloat16AttentionError,
    _exact_prefix,
    build_portable_attention_library,
    transform_attention_source,
)


REPO = Path(__file__).resolve().parents[2]


class PortableBfloat16AttentionTest(unittest.TestCase):
    def test_transform_applies_every_boundary_once(self):
        path = REPO / "inkling" / "src" / "inkling_attention.c"
        original = path.read_text(encoding="utf-8")
        transformed = transform_attention_source(original)
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        self.assertNotEqual(transformed, original)
        self.assertIn("static float bf16_round_probe", transformed)
        self.assertIn("const float source = bf16_round_probe(x[i]);", transformed)
        self.assertIn("score = bf16_round_probe(score * dot_scale);", transformed)
        self.assertIn("score = bf16_round_probe(score + bias);", transformed)
        self.assertIn(
            "const float weight = bf16_round_probe(row[j] * inv_sum);",
            transformed,
        )
        self.assertIn("oh[d] = bf16_round_probe(oh[d]);", transformed)
        self.assertNotIn("memcpy(vdst, v, kv_stride * sizeof(float));", transformed)
        with self.assertRaisesRegex(
            PortableBfloat16AttentionError, "expected exactly one"
        ):
            transform_attention_source(transformed)

    def test_temporary_candidate_compiles_without_changing_production(self):
        layer = REPO / "inkling" / "src" / "inkling_layer.c"
        attention = REPO / "inkling" / "src" / "inkling_attention.c"
        before_layer = layer.read_bytes()
        before_attention = attention.read_bytes()
        with tempfile.TemporaryDirectory() as td:
            library, metadata = build_portable_attention_library(
                Path(td) / "portable-attention.so"
            )
            self.assertTrue(library.is_file())
            self.assertTrue(metadata["production_source_unchanged"])
            self.assertNotEqual(
                metadata["production_layer_sha256"],
                metadata["probe_layer_sha256"],
            )
            self.assertNotEqual(
                metadata["production_attention_sha256"],
                metadata["probe_attention_sha256"],
            )
        self.assertEqual(layer.read_bytes(), before_layer)
        self.assertEqual(attention.read_bytes(), before_attention)

    def test_exact_prefix_stops_at_first_nonexact_stage(self):
        import diagnose_inkling_portable_bf16_attention as module

        stages = {
            point: {"quantized_exact_fraction": 1.0}
            for point in module.POINTS
        }
        stages[module.POINTS[3]]["quantized_exact_fraction"] = 0.5
        result = _exact_prefix({"stages": stages})
        self.assertEqual(result, list(module.POINTS[:3]))


if __name__ == "__main__":
    unittest.main()
