#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes
import tempfile
import unittest
from pathlib import Path

import torch

from inkling_compact_layer_parity import (
    bind_sparse_layer_compact,
    run_layer_trace_compact,
)
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
    MOE_INTER,
    NROUTED,
    make_tensors,
    write_fixture,
)


REPO = Path(__file__).resolve().parents[1]


class CompactLayerParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.td = tempfile.TemporaryDirectory()
        cls.lib = ctypes.CDLL(str(build_library(
            REPO / "inkling" / "src",
            Path(cls.td.name) / "compact-layer.so",
        )))
        configure_library(cls.lib)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.td.cleanup()

    def test_compact_expert_get_matches_resident_fixture(self) -> None:
        root = Path(tempfile.mkdtemp(dir=self.td.name))
        write_fixture(root, 1, make_tensors(1))
        fixture = load_fixture(root)
        compact = bind_sparse_layer_compact(fixture, C_CONFIG, 1)
        resident = bind_layer(fixture, C_CONFIG, 1)
        self.assertFalse(bool(compact.weights.routed_gate))
        self.assertFalse(bool(compact.weights.routed_up))
        self.assertFalse(bool(compact.weights.routed_down))
        expected_bytes = NROUTED * (
            2 * MOE_INTER * H + H * MOE_INTER
        ) * 4
        self.assertEqual(compact.compact_expert_bytes, expected_bytes)
        self.assertEqual(compact.provider.experts, list(range(NROUTED)))

        torch.manual_seed(31)
        inputs = torch.randn(6, H).tolist()
        got = run_layer_trace_compact(
            self.lib, C_CONFIG, 1, compact, inputs
        )
        want = run_layer_trace(
            self.lib, C_CONFIG, 1, resident, inputs
        )
        self.assertEqual(got["dtypes"], want["dtypes"])
        self.assertEqual(got["values"].keys(), want["values"].keys())
        for name in got["values"]:
            self.assertEqual(got["values"][name], want["values"][name], name)

    def test_dense_layer_is_not_accepted_by_sparse_binder(self) -> None:
        root = Path(tempfile.mkdtemp(dir=self.td.name))
        write_fixture(root, 0, make_tensors(0))
        fixture = load_fixture(root)
        with self.assertRaisesRegex(RuntimeError, "dense"):
            bind_sparse_layer_compact(fixture, C_CONFIG, 0)


if __name__ == "__main__":
    unittest.main()
