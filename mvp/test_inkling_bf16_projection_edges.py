#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import torch

from diagnose_inkling_bf16_projection_edges import (
    ProjectionEdgeError,
    classify_projection_edges,
)


class Bfloat16ProjectionEdgeTest(unittest.TestCase):
    def test_float32_accumulation_can_resolve_all_candidate_edges(self) -> None:
        official = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        candidate = torch.tensor([[1.0078125, 2.0]], dtype=torch.bfloat16)
        float32_rounded = official.clone()
        result = classify_projection_edges(
            official,
            candidate,
            float32_rounded,
            float32_rounded.float(),
            dtype=torch.bfloat16,
        )
        self.assertEqual(
            result["classification"],
            "float32_accumulation_rounding_is_sufficient",
        )
        self.assertEqual(result["candidate_mismatches"], 1)
        self.assertEqual(result["float32_resolved_mismatches"], 1)
        self.assertEqual(result["float32_introduced_mismatches"], 0)

    def test_partial_resolution_and_introduced_edges_are_disclosed(self) -> None:
        official = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)
        candidate = torch.tensor(
            [[1.0078125, 2.015625, 3.0]], dtype=torch.bfloat16
        )
        float32_rounded = torch.tensor(
            [[1.0, 2.015625, 3.015625]], dtype=torch.bfloat16
        )
        result = classify_projection_edges(
            official,
            candidate,
            float32_rounded,
            float32_rounded.float(),
            dtype=torch.bfloat16,
        )
        self.assertEqual(
            result["classification"],
            "float32_accumulation_resolves_only_some_edges",
        )
        self.assertEqual(result["candidate_mismatches"], 2)
        self.assertEqual(result["float32_resolved_mismatches"], 1)
        self.assertEqual(result["float32_introduced_mismatches"], 1)

    def test_shape_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ProjectionEdgeError, "different shapes"):
            classify_projection_edges(
                torch.ones(1, 2),
                torch.ones(1, 3),
                torch.ones(1, 2),
                torch.ones(1, 2),
                dtype=torch.bfloat16,
            )


if __name__ == "__main__":
    unittest.main()
