#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes
import tempfile
import unittest
from pathlib import Path

import torch

from diagnose_inkling_pre_router import (
    POINTS,
    capture_c_pre_router,
    capture_official_pre_router,
    compare_pre_router,
)
from inkling_fixture import load_fixture
from inkling_layer_parity import build_library, configure_library
from inkling_release_config import build_transformers_text_config
from test_inkling_official_layer_parity import (
    C_CONFIG,
    H,
    RELEASE,
    make_tensors,
    write_fixture,
)


REPO = Path(__file__).resolve().parents[1]


class PreRouterDiagnosisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.td = tempfile.TemporaryDirectory()
        cls.lib = ctypes.CDLL(
            str(
                build_library(
                    REPO / "inkling" / "src",
                    Path(cls.td.name) / "pre-router.so",
                )
            )
        )
        configure_library(cls.lib)
        cls.config = build_transformers_text_config(RELEASE)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.td.cleanup()

    def test_sparse_global_trace_points_align_with_official_float32(self) -> None:
        root = Path(tempfile.mkdtemp(dir=self.td.name))
        write_fixture(root, 1, make_tensors(1))
        fixture = load_fixture(root)
        torch.manual_seed(99)
        inputs = torch.randn(6, H, dtype=torch.float32)

        official = capture_official_pre_router(
            fixture,
            self.config,
            1,
            inputs,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        candidate = capture_c_pre_router(
            self.lib,
            fixture,
            C_CONFIG,
            1,
            inputs.tolist(),
        )

        self.assertEqual(set(official), set(POINTS))
        self.assertEqual(set(candidate), set(POINTS))
        for point in POINTS:
            with self.subTest(point=point):
                self.assertEqual(official[point].shape, candidate[point].shape)
                max_abs = float(
                    (official[point].float() - candidate[point].float())
                    .abs()
                    .max()
                )
                self.assertLess(max_abs, 2e-5)

    def test_first_nonexact_stage_uses_declared_trace_order(self) -> None:
        official = {
            point: torch.zeros(2, 3, dtype=torch.float32) for point in POINTS
        }
        candidate = {point: value.clone() for point, value in official.items()}
        candidate["q_proj"][1, 2] = 0.125
        candidate["post_attention_norm"][0, 0] = 1.0

        result = compare_pre_router(
            official,
            candidate,
            compare_dtype=torch.float32,
        )

        self.assertEqual(result["first_nonexact_bfloat16_stage"], "q_proj")
        self.assertFalse(result["all_stages_bfloat16_exact"])
        self.assertEqual(
            result["stages"]["input_norm"]["quantized_exact_fraction"],
            1.0,
        )
        self.assertLess(
            result["stages"]["q_proj"]["quantized_exact_fraction"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
