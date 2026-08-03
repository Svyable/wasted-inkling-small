#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes
import tempfile
import unittest
from pathlib import Path

import torch

from discover_inkling_c_router_experts import discover_c_layer_router
from inkling_fixture import load_fixture
from inkling_layer_parity import (
    bind_layer,
    build_library,
    configure_library,
    run_layer_trace,
)
from test_inkling_official_layer_parity import (
    C_CONFIG,
    H,
    make_tensors,
    write_fixture,
)


REPO = Path(__file__).resolve().parents[1]


class CRouterDiscoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.td = tempfile.TemporaryDirectory()
        cls.lib = ctypes.CDLL(str(build_library(
            REPO / "inkling" / "src",
            Path(cls.td.name) / "router-discovery.so",
        )))
        configure_library(cls.lib)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.td.cleanup()

    def test_router_only_trace_matches_complete_sparse_layer(self) -> None:
        root = Path(tempfile.mkdtemp(dir=self.td.name))
        write_fixture(root, 1, make_tensors(1))
        fixture = load_fixture(root)
        torch.manual_seed(41)
        inputs = torch.randn(6, H).tolist()

        discovered = discover_c_layer_router(
            self.lib, fixture, C_CONFIG, 1, inputs
        )
        complete = run_layer_trace(
            self.lib,
            C_CONFIG,
            1,
            bind_layer(fixture, C_CONFIG, 1),
            inputs,
        )
        for position in range(len(inputs)):
            prefix = f"token.{position}.layer.1."
            self.assertEqual(
                discovered["topk_indices"][position],
                complete["values"][prefix + "routed_index"],
            )
            self.assertEqual(
                discovered["topk_weights"][position],
                complete["values"][prefix + "routed_weight"],
            )
            self.assertEqual(
                discovered["router_logits"][position],
                complete["values"][prefix + "router_logits"],
            )
        self.assertEqual(len(discovered["first_rejected_expert"]), len(inputs))

    def test_dense_layer_is_refused(self) -> None:
        root = Path(tempfile.mkdtemp(dir=self.td.name))
        write_fixture(root, 0, make_tensors(0))
        fixture = load_fixture(root)
        with self.assertRaisesRegex(RuntimeError, "not sparse"):
            discover_c_layer_router(
                self.lib,
                fixture,
                C_CONFIG,
                0,
                torch.zeros(3, H).tolist(),
            )


if __name__ == "__main__":
    unittest.main()
