#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from discover_inkling_router_experts import (
    deterministic_hidden_states,
    discover_layer_router,
    input_sha256,
)
from inkling_fixture import load_fixture
from inkling_fixture_reference import run_layer_reference
from inkling_release_config import build_transformers_text_config
from test_inkling_official_layer_parity import (
    H,
    RELEASE,
    make_tensors,
    write_fixture,
)


class RouterDiscoveryTest(unittest.TestCase):
    def test_capture_matches_official_last_token(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_fixture(root, 1, make_tensors(1))
            fixture = load_fixture(root)
            config = build_transformers_text_config(RELEASE)
            inputs = deterministic_hidden_states(6, H, 19)

            found = discover_layer_router(
                fixture,
                config,
                1,
                inputs,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            reference = run_layer_reference(
                fixture,
                config,
                1,
                inputs.tolist(),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            expected = reference["token.5.layer.1.routed_index"].to(torch.int64)
            actual = torch.tensor(found["topk_indices"][-1], dtype=torch.int64)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(
                found["selected_experts"],
                sorted({value for row in found["topk_indices"] for value in row}),
            )

    def test_inputs_are_repeatable_and_bfloat16_bound(self) -> None:
        a = deterministic_hidden_states(4, 8, 19)
        b = deterministic_hidden_states(4, 8, 19)
        c = deterministic_hidden_states(4, 8, 20)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertTrue(torch.equal(a, a.to(torch.bfloat16).float()))
        self.assertEqual(
            input_sha256(a, torch.bfloat16),
            input_sha256(b, torch.bfloat16),
        )

    def test_dense_layer_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_fixture(root, 0, make_tensors(0))
            fixture = load_fixture(root)
            config = build_transformers_text_config(RELEASE)
            inputs = deterministic_hidden_states(3, H, 19)
            with self.assertRaisesRegex(RuntimeError, "not sparse"):
                discover_layer_router(
                    fixture,
                    config,
                    0,
                    inputs,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )


if __name__ == "__main__":
    unittest.main()
