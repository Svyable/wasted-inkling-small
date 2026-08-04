#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from diagnose_inkling_bf16_layer_probe import (
    Bfloat16LayerProbeError,
    build_probe_library,
    transform_layer_source,
)


REPO = Path(__file__).resolve().parents[1]


class Bfloat16LayerProbeTest(unittest.TestCase):
    def test_transform_is_exact_and_does_not_modify_production_source(self) -> None:
        source_path = REPO / "inkling" / "src" / "inkling_layer.c"
        original = source_path.read_text(encoding="utf-8")
        transformed = transform_layer_source(original)

        self.assertEqual(source_path.read_text(encoding="utf-8"), original)
        self.assertNotEqual(transformed, original)
        self.assertIn("static float bf16_round_probe", transformed)
        self.assertIn("bf16_round_probe((float)sum)", transformed)
        self.assertIn(
            "const float normalized = bf16_round_probe(x[i] * scale);",
            transformed,
        )
        self.assertNotIn("out[r] = (float)sum;", transformed)
        self.assertNotIn(
            "out[i] = x[i] * scale * weight[i];", transformed
        )

        with self.assertRaisesRegex(
            Bfloat16LayerProbeError, "expected exactly one"
        ):
            transform_layer_source(transformed)

    def test_probe_library_compiles_from_temporary_copy(self) -> None:
        source_path = REPO / "inkling" / "src" / "inkling_layer.c"
        before = source_path.read_bytes()
        with tempfile.TemporaryDirectory() as td:
            library, metadata = build_probe_library(Path(td) / "probe.so")
            self.assertTrue(library.is_file())
            self.assertTrue(metadata["production_source_unchanged"])
            self.assertNotEqual(
                metadata["production_layer_sha256"],
                metadata["probe_layer_sha256"],
            )
        self.assertEqual(source_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
