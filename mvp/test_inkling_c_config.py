#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import unittest
from pathlib import Path

from inkling_c_config import CConfigError, normalized_c_layer_config


REPO = Path(__file__).resolve().parents[1]


class CLayerConfigTest(unittest.TestCase):
    def test_maps_recorded_release_config_exactly(self) -> None:
        raw = json.loads(
            (REPO / "inkling/tests/data/inkling-small-config.json").read_text()
        )
        got = normalized_c_layer_config(raw)
        self.assertEqual(got["n_layers"], 42)
        self.assertEqual(got["hidden"], 4096)
        self.assertEqual(got["dense_layers"], 2)
        self.assertEqual(got["dense_intermediate"], 16384)
        self.assertEqual(got["moe_intermediate"], 2048)
        self.assertEqual(got["global_heads"], 32)
        self.assertEqual(got["local_heads"], 32)
        self.assertEqual(got["sliding_window"], 512)
        self.assertEqual(got["top_k"], 6)
        self.assertIn(2, got["local_layer_ids"])
        self.assertNotIn(5, got["local_layer_ids"])

    def test_release_routed_width_does_not_become_dense_width(self) -> None:
        raw = json.loads(
            (REPO / "inkling/tests/data/inkling-small-config.json").read_text()
        )
        got = normalized_c_layer_config(raw["text_config"])
        self.assertNotEqual(got["dense_intermediate"], got["moe_intermediate"])
        self.assertEqual(got["dense_intermediate"], 8 * got["moe_intermediate"])

    def test_refuses_missing_or_inconsistent_geometry(self) -> None:
        raw = json.loads(
            (REPO / "inkling/tests/data/inkling-small-config.json").read_text()
        )["text_config"]
        missing = dict(raw)
        missing.pop("swa_head_dim")
        with self.assertRaisesRegex(CConfigError, "swa_head_dim"):
            normalized_c_layer_config(missing)

        duplicate = dict(raw)
        duplicate["local_layer_ids"] = [0, 0]
        with self.assertRaisesRegex(CConfigError, "local_layer_ids"):
            normalized_c_layer_config(duplicate)

        too_many = dict(raw)
        too_many["num_experts_per_tok"] = 257
        with self.assertRaisesRegex(CConfigError, "exceeds"):
            normalized_c_layer_config(too_many)


if __name__ == "__main__":
    unittest.main()
