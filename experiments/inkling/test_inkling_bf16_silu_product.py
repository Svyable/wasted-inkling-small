#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from diagnose_inkling_bf16_silu_product import (
    CVariants,
    SiluProductError,
    build_helper,
    classify,
    metrics,
)


class SiluProductTest(unittest.TestCase):
    @staticmethod
    def metric(exact: float):
        return {"bfloat16_exact_fraction": exact}

    def test_classifies_cast_before_multiply_policy(self):
        decision = classify(
            {
                "silu_bfloat16": self.metric(1.0),
                "product_silu_bfloat16_result_bfloat16": self.metric(1.0),
                "product_result_bfloat16": self.metric(0.8),
            }
        )
        self.assertEqual(
            decision["classification"],
            "bfloat16_complete_silu_before_multiply_is_sufficient",
        )

    def test_classifies_silu_and_product_failures(self):
        silu = classify(
            {
                "silu_bfloat16": self.metric(0.9),
                "product_silu_bfloat16_result_bfloat16": self.metric(0.8),
                "product_result_bfloat16": self.metric(0.8),
            }
        )
        self.assertEqual(
            silu["classification"],
            "portable_expf_does_not_reproduce_native_bfloat16_silu",
        )
        product = classify(
            {
                "silu_bfloat16": self.metric(1.0),
                "product_silu_bfloat16_result_bfloat16": self.metric(0.9),
                "product_result_bfloat16": self.metric(0.8),
            }
        )
        self.assertEqual(
            product["classification"],
            "bfloat16_multiply_boundary_remains_unexplained",
        )

    def test_metrics_rejects_shape_drift(self):
        reference = torch.tensor([1.0, 2.0])
        candidate = reference + torch.tensor([1.0e-4, 1.0e-2])
        result = metrics(reference, candidate)
        self.assertEqual(result["raw_exact_fraction"], 0.0)
        self.assertEqual(result["bfloat16_exact_fraction"], 0.5)
        with self.assertRaisesRegex(SiluProductError, "shape mismatch"):
            metrics(reference, torch.ones(3))

    def test_c_helper_compiles_and_variants_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            library = build_helper(Path(directory) / "libsilu-product.so")
            self.assertTrue(library.is_file())
            helper = CVariants(library)
            gate = torch.tensor([0.3, -2.0, 5.0], dtype=torch.bfloat16)
            up = torch.tensor([3.0, -1.0, 0.25], dtype=torch.bfloat16)
            values = helper.evaluate(gate, up)
            self.assertEqual(
                set(values),
                {
                    "silu_raw",
                    "silu_bfloat16",
                    "product_raw",
                    "product_result_bfloat16",
                    "product_silu_bfloat16_result_bfloat16",
                },
            )
            for value in values.values():
                self.assertEqual(tuple(value.shape), (3,))
            self.assertFalse(
                torch.equal(values["silu_raw"], values["product_raw"])
            )


if __name__ == "__main__":
    unittest.main()
