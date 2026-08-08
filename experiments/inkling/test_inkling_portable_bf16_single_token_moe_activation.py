#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from diagnose_inkling_portable_bf16_single_token_moe_activation import (
    MoeActivationError,
    build_activation_library,
    transform_activation_source,
)


class MoeActivationTest(unittest.TestCase):
    def test_transform_applies_both_completion_boundaries_once(self):
        source = """                for (int i = 0; i < inter; i++)
                    s->gate[i] = silu(s->gate[i]) * s->up[i];
                for (int j = 0; j < inter; j++)
                    s->gate[j] = silu(s->gate[j]) * s->up[j];
"""
        transformed = transform_activation_source(source)
        self.assertEqual(
            transformed.count("bf16_round_probe(silu("),
            2,
        )
        self.assertEqual(
            transformed.count("activated * bf16_round_probe(s->up["),
            2,
        )
        with self.assertRaisesRegex(MoeActivationError, "routed activation loop"):
            transform_activation_source(transformed)

    def test_transform_rejects_missing_shared_loop(self):
        source = """                for (int i = 0; i < inter; i++)
                    s->gate[i] = silu(s->gate[i]) * s->up[i];
"""
        with self.assertRaisesRegex(MoeActivationError, "shared activation loop"):
            transform_activation_source(source)

    def test_temporary_candidate_compiles_and_preserves_production(self):
        with tempfile.TemporaryDirectory() as directory:
            library, source = build_activation_library(
                Path(directory) / "libinkling-moe-activation.so"
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
            self.assertEqual(
                source["activation_policy"],
                {
                    "silu_completion": "bfloat16",
                    "up_operand_completion": "bfloat16",
                    "product_completion": "bfloat16",
                },
            )


if __name__ == "__main__":
    unittest.main()
