#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from inkling_fixture_reference import (
    FixtureReferenceError,
    FixtureSparseExperts,
    official_gate_up,
    split_provider_gate_up,
    verify_config_binding,
)


class GateUpTransformTest(unittest.TestCase):
    def test_splits_provider_row_interleaving(self) -> None:
        fused = torch.tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
        )
        gate, up = split_provider_gate_up(fused)
        self.assertTrue(torch.equal(gate, torch.tensor([[10.0, 11.0], [30.0, 31.0]])))
        self.assertTrue(torch.equal(up, torch.tensor([[20.0, 21.0], [40.0, 41.0]])))
        self.assertTrue(
            torch.equal(
                official_gate_up(fused),
                torch.tensor(
                    [[10.0, 11.0], [30.0, 31.0], [20.0, 21.0], [40.0, 41.0]]
                ),
            )
        )

    def test_refuses_odd_row_count(self) -> None:
        with self.assertRaises(FixtureReferenceError):
            split_provider_gate_up(torch.zeros(3, 2))


class SparseExpertAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.gate_up = {1: torch.randn(6, 4), 3: torch.randn(6, 4)}
        self.down = {1: torch.randn(4, 3), 3: torch.randn(4, 3)}
        self.adapter = FixtureSparseExperts(
            num_experts=4, gate_up=self.gate_up, down=self.down
        )

    def test_matches_direct_selected_expert_math(self) -> None:
        hidden = torch.randn(3, 4)
        indices = torch.tensor([[1, 3], [3, 1], [1, 3]])
        weights = torch.tensor([[0.75, 0.25], [0.4, 0.6], [0.2, 0.8]])
        got = self.adapter(hidden, indices, weights)
        expected = torch.zeros_like(hidden)
        for token in range(hidden.shape[0]):
            for slot in range(indices.shape[1]):
                expert = int(indices[token, slot])
                gate, up = F.linear(
                    hidden[token : token + 1], self.gate_up[expert]
                ).chunk(2, -1)
                value = F.linear(F.silu(gate) * up, self.down[expert])[0]
                expected[token] += weights[token, slot] * value
        self.assertTrue(torch.allclose(got, expected, atol=1e-6, rtol=1e-6))

    def test_names_missing_router_expert(self) -> None:
        hidden = torch.randn(1, 4)
        with self.assertRaises(FixtureReferenceError) as ctx:
            self.adapter(hidden, torch.tensor([[2]]), torch.tensor([[1.0]]))
        self.assertIn("2", str(ctx.exception))
        self.assertIn("re-extract", str(ctx.exception))

    def test_requires_matching_expert_banks(self) -> None:
        with self.assertRaises(FixtureReferenceError):
            FixtureSparseExperts(
                num_experts=4, gate_up={1: torch.ones(2, 2)}, down={}
            )


class ConfigBindingTest(unittest.TestCase):
    def test_accepts_exact_bytes_and_refuses_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            raw = b'{"text_config":{"hidden_size":4}}\n'
            path.write_bytes(raw)
            fixture = SimpleNamespace(
                source={"config_sha256": hashlib.sha256(raw).hexdigest()}
            )
            self.assertEqual(
                verify_config_binding(fixture, path)["text_config"]["hidden_size"], 4
            )
            path.write_text(
                json.dumps({"text_config": {"hidden_size": 5}}), encoding="utf-8"
            )
            with self.assertRaises(FixtureReferenceError) as ctx:
                verify_config_binding(fixture, path)
            self.assertIn("hash mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
