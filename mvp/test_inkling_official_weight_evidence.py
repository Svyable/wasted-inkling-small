#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class OfficialWeightEvidenceTest(unittest.TestCase):
    def test_result_is_bound_to_plan_selection_and_failed_numerically(self) -> None:
        result = json.loads(
            (REPO / "docs" / "OFFICIAL-WEIGHT-PARITY-RESULT.json").read_text()
        )
        plan = json.loads(
            (REPO / "docs" / "OFFICIAL-FIXTURE-PLAN.json").read_text()
        )
        selection = json.loads(
            (REPO / "docs" / "OFFICIAL-ROUTER-SELECTION.json").read_text()
        )

        self.assertEqual(result["format"], "inkling-official-weight-parity-evidence")
        self.assertEqual(result["version"], 1)
        for key in ("model_id", "revision", "config_sha256", "index_sha256"):
            self.assertEqual(result[key], plan[key], key)
        self.assertEqual(result["inputs_sha256"], selection["inputs_sha256"])
        self.assertEqual(result["tokens"], selection["tokens"])
        self.assertEqual(result["input_dtype"], selection["input_dtype"])
        self.assertEqual(result["fixture"]["entries"], plan["entries"])
        self.assertEqual(result["fixture"]["payload_bytes"], plan["payload_bytes"])
        self.assertEqual(
            result["fixture"]["verified_bytes"], result["fixture"]["payload_bytes"]
        )
        self.assertTrue(result["fixture"]["all_crcs_match"])
        self.assertTrue(result["comparison"]["canonical_official_route_override"])
        self.assertTrue(result["comparison"]["candidate_routes_audited_separately"])
        self.assertFalse(result["passed"])

        self.assertEqual(set(result["layers"]), {"0", "2", "5"})
        self.assertEqual(result["layers"]["0"]["class"], "dense-local")
        for layer in ("0", "2", "5"):
            record = result["layers"][layer]
            self.assertTrue(record["execution_completed"], layer)
            self.assertFalse(record["activations_passed"], layer)
            self.assertGreater(record["max_output_abs_across_positions"], 1e-3)
            self.assertGreater(record["final_position"]["input_norm"], 1e-3)
            self.assertGreater(record["final_position"]["attention_out"], 1e-3)
            self.assertGreater(record["final_position"]["output"], 1e-3)

        self.assertIsNone(result["layers"]["0"]["routing"])
        for layer in ("2", "5"):
            routing = result["layers"][layer]["routing"]
            self.assertTrue(routing["passed"])
            self.assertTrue(routing["exact_indices"])
            self.assertEqual(routing["max_abs_weight"], 0.0)
            self.assertEqual(routing["positions"], result["tokens"])


if __name__ == "__main__":
    unittest.main()
