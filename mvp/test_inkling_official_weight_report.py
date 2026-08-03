#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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
from official_weight_parity_report import (
    OfficialWeightReportError,
    assemble_report,
    compare_candidate_routing,
)


SELECTION = {
    "model_id": "synthetic",
    "revision": "abcdef1",
    "config_sha256": "0" * 64,
    "index_sha256": "1" * 64,
    "inputs_sha256": "2" * 64,
    "tokens": 2,
    "input_dtype": "bfloat16",
    "layers": {
        "2": {
            "topk_indices": [[7, 2], [3, 5]],
            "topk_weights": [[0.2, 0.8], [0.4, 0.6]],
        }
    },
}


class OfficialWeightReportTest(unittest.TestCase):
    def _archive(self, root: Path, *, wrong_id: bool = False) -> Path:
        values = {
            "token.0.layer.2.routed_index": torch.tensor(
                [2, 7 if not wrong_id else 8], dtype=torch.int32
            ),
            "token.0.layer.2.routed_weight": torch.tensor([0.8002, 0.1998]),
            "token.1.layer.2.routed_index": torch.tensor([5, 3], dtype=torch.int32),
            "token.1.layer.2.routed_weight": torch.tensor([0.6001, 0.3999]),
        }
        write_activation_archive(root, values, metadata={"positions": 2})
        return root

    def test_routing_compares_pairs_with_exact_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = compare_candidate_routing(
                SELECTION, 2, self._archive(Path(td)), atol=1e-3
            )
            self.assertTrue(result["passed"])
            self.assertTrue(result["exact_indices"])
            self.assertLess(result["max_abs_weight"], 1e-3)

    def test_routing_refuses_one_wrong_expert(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = compare_candidate_routing(
                SELECTION,
                2,
                self._archive(Path(td), wrong_id=True),
                atol=1e-3,
            )
            self.assertFalse(result["passed"])
            self.assertFalse(result["exact_indices"])

    def test_assembled_report_requires_activations_and_routes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            archive = self._archive(Path(td))
            passed = assemble_report(
                SELECTION,
                {2: {"passed": True, "results": []}},
                {2: archive},
                route_atol=1e-3,
            )
            self.assertTrue(passed["passed"])
            failed = assemble_report(
                SELECTION,
                {2: {"passed": False, "results": []}},
                {2: archive},
                route_atol=1e-3,
            )
            self.assertFalse(failed["passed"])

    def test_layer_sets_must_match(self) -> None:
        with self.assertRaisesRegex(OfficialWeightReportError, "different layers"):
            assemble_report(
                SELECTION,
                {0: {"passed": True}},
                {},
                route_atol=1e-3,
            )


if __name__ == "__main__":
    unittest.main()
