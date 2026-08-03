#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "inkling" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from inkling_parity import write_activation_archive
from compare_inkling_layer_archives import compare_pairwise
from inkling_release_config import (
    ReleaseConfigError,
    build_transformers_text_config,
    canonicalize_router_pairs,
    normalize_release_text_config,
)


class ReleaseSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        path = REPO / "inkling" / "tests" / "data" / "inkling-small-config.json"
        self.release = json.loads(path.read_text(encoding="utf-8"))

    def test_maps_the_released_small_geometry(self) -> None:
        normalized = normalize_release_text_config(self.release)
        self.assertEqual(normalized["intermediate_size"], 16384)
        self.assertEqual(normalized["moe_intermediate_size"], 2048)
        self.assertNotIn("dense_intermediate_size", normalized)
        self.assertEqual(
            self.release["text_config"]["intermediate_size"], 2048,
            "normalization must not mutate release JSON",
        )

    def test_refuses_conflicting_explicit_moe_width(self) -> None:
        value = json.loads(json.dumps(self.release))
        value["text_config"]["moe_intermediate_size"] = 3072
        with self.assertRaisesRegex(ReleaseConfigError, "conflicting routed widths"):
            normalize_release_text_config(value)

    def test_refuses_missing_geometry(self) -> None:
        value = {"text_config": {"dense_intermediate_size": 16384}}
        with self.assertRaisesRegex(ReleaseConfigError, "intermediate_size"):
            normalize_release_text_config(value)

    def test_transformers_constructs_16384_dense_2048_routed(self) -> None:
        try:
            config = build_transformers_text_config(self.release)
        except ReleaseConfigError as exc:
            if "Transformers with" in str(exc):
                self.skipTest(str(exc))
            raise
        self.assertEqual(config.intermediate_size, 16384)
        self.assertEqual(config.moe_intermediate_size, 2048)


class RouterPairTest(unittest.TestCase):
    def test_canonicalization_keeps_index_weight_pairs(self) -> None:
        values = {
            "token.0.layer.2.routed_index": torch.tensor([7, 2, 5], dtype=torch.int32),
            "token.0.layer.2.routed_weight": torch.tensor([0.2, 0.7, 0.1]),
        }
        canonicalize_router_pairs(values)
        self.assertEqual(values["token.0.layer.2.routed_index"].tolist(), [2, 5, 7])
        self.assertEqual(
            values["token.0.layer.2.routed_weight"].tolist(),
            torch.tensor([0.7, 0.1, 0.2]).tolist(),
        )

    def test_pairwise_comparator_accepts_different_topk_order(self) -> None:
        with tempfile.TemporaryDirectory() as rd, tempfile.TemporaryDirectory() as cd:
            ref = {
                "token.0.layer.2.routed_index": torch.tensor([7, 2, 5], dtype=torch.int32),
                "token.0.layer.2.routed_weight": torch.tensor([0.2, 0.7, 0.1]),
            }
            got = {
                "token.0.layer.2.routed_index": torch.tensor([2, 5, 7], dtype=torch.int32),
                "token.0.layer.2.routed_weight": torch.tensor([0.7, 0.1, 0.2]),
            }
            write_activation_archive(rd, ref)
            write_activation_archive(cd, got)
            report = compare_pairwise(rd, cd)
            self.assertTrue(report["passed"])

    def test_reference_scope_discloses_candidate_extras(self) -> None:
        with tempfile.TemporaryDirectory() as rd, tempfile.TemporaryDirectory() as cd:
            write_activation_archive(rd, {"official": torch.tensor([1.0])})
            write_activation_archive(cd, {
                "official": torch.tensor([1.0]),
                "candidate.only": torch.tensor([2.0]),
            })
            strict = compare_pairwise(rd, cd)
            self.assertFalse(strict["passed"])
            scoped = compare_pairwise(rd, cd, allow_candidate_extras=True)
            self.assertTrue(scoped["passed"])
            self.assertEqual(
                scoped["comparison_scope"]["candidate_only_ignored"],
                ["candidate.only"],
            )

    def test_reference_scope_never_ignores_missing_official_name(self) -> None:
        with tempfile.TemporaryDirectory() as rd, tempfile.TemporaryDirectory() as cd:
            write_activation_archive(rd, {
                "required": torch.tensor([1.0]),
                "also.required": torch.tensor([2.0]),
            })
            write_activation_archive(cd, {"required": torch.tensor([1.0])})
            report = compare_pairwise(rd, cd, allow_candidate_extras=True)
            self.assertFalse(report["passed"])
            self.assertEqual(
                report["comparison_scope"]["reference_only_missing"],
                ["also.required"],
            )

    def test_refuses_unpaired_or_duplicate_router_entries(self) -> None:
        with self.assertRaisesRegex(ReleaseConfigError, "no paired weights"):
            canonicalize_router_pairs({"x.routed_index": torch.tensor([1])})
        with self.assertRaisesRegex(ReleaseConfigError, "duplicate expert"):
            canonicalize_router_pairs({
                "x.routed_index": torch.tensor([1, 1]),
                "x.routed_weight": torch.tensor([0.5, 0.5]),
            })


if __name__ == "__main__":
    unittest.main()
