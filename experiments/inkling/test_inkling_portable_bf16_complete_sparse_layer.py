#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import diagnose_inkling_portable_bf16_composed_moe as implementation
from run_inkling_portable_bf16_complete_sparse_layer import (
    _FINAL_RESIDUAL_NEW,
    _FINAL_RESIDUAL_OLD,
    transform_complete_sparse_layer_source,
)


class CompleteSparseLayerTest(unittest.TestCase):
    def test_final_residual_transform_is_explicit_and_fail_closed(self):
        source = _FINAL_RESIDUAL_OLD
        transformed = transform_complete_sparse_layer_source(source)
        self.assertIn("bf16_round_probe(x[i])", transformed)
        self.assertIn("bf16_round_probe(s->ff[i])", transformed)
        self.assertIn("bf16_round_probe(residual + branch)", transformed)
        with self.assertRaisesRegex(
            implementation.ComposedMoeError,
            "final layer residual",
        ):
            transform_complete_sparse_layer_source(transformed)

    def test_final_residual_transform_does_not_touch_attention_residual(self):
        source = (
            "    for (int i = 0; i < hidden; i++) x[i] += s->branch[i];\n"
            + _FINAL_RESIDUAL_OLD
        )
        transformed = transform_complete_sparse_layer_source(source)
        self.assertIn("x[i] += s->branch[i]", transformed)
        self.assertIn(_FINAL_RESIDUAL_NEW, transformed)

    def test_complete_candidate_compiles_and_preserves_production(self):
        with tempfile.TemporaryDirectory() as directory:
            library, source = implementation.build_composed_library(
                Path(directory) / "libinkling-complete-sparse-layer.so"
            )
            self.assertTrue(library.is_file())
            self.assertTrue(source["production_source_unchanged"])
            self.assertNotEqual(
                source["production_layer_sha256"],
                source["probe_layer_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
