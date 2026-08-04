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
    _capture_c_pre_router,
    _round_float32_pointer_to_bfloat16,
    capture_c_pre_router,
    capture_official_pre_router,
    compare_pre_router,
)
from diagnose_inkling_stage_injection import CUMULATIVE_INJECTION_CASES
from inkling_fixture import load_fixture
from inkling_layer_parity import build_library, configure_library
from inkling_release_config import build_transformers_text_config
from test_inkling_official_layer_parity import (
    C_CONFIG,
    H,
    RELEASE,
    TOPK,
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

    def _fixture_and_inputs(self):
        root = Path(tempfile.mkdtemp(dir=self.td.name))
        write_fixture(root, 1, make_tensors(1))
        fixture = load_fixture(root)
        torch.manual_seed(99)
        inputs = torch.randn(6, H, dtype=torch.float32)
        return fixture, inputs

    def test_sparse_global_trace_points_align_with_official_float32(self) -> None:
        fixture, inputs = self._fixture_and_inputs()

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

    def test_bfloat16_pointer_rounding_matches_torch(self) -> None:
        values = [
            -123.75,
            -1.00390625,
            -0.0,
            0.0,
            1.0e-30,
            0.3333333432674408,
            1.00390625,
            65504.0,
        ]
        storage = (ctypes.c_float * len(values))(*values)
        _round_float32_pointer_to_bfloat16(storage, len(values))
        got = torch.tensor(
            [float(storage[index]) for index in range(len(values))],
            dtype=torch.float32,
        )
        expected = torch.tensor(values, dtype=torch.float32).to(
            torch.bfloat16
        ).float()
        self.assertTrue(torch.equal(got, expected))

    def test_trace_boundary_counterfactual_mutates_to_bfloat16(self) -> None:
        fixture, inputs = self._fixture_and_inputs()
        captured, indices, weights = _capture_c_pre_router(
            self.lib,
            fixture,
            C_CONFIG,
            1,
            inputs.tolist(),
            quantize_points=POINTS,
        )

        self.assertEqual(tuple(indices.shape), (len(inputs), TOPK))
        self.assertEqual(tuple(weights.shape), (len(inputs), TOPK))
        for point in POINTS:
            with self.subTest(point=point):
                self.assertTrue(
                    torch.equal(
                        captured[point],
                        captured[point].to(torch.bfloat16).float(),
                    )
                )

    def test_official_stage_injection_replaces_writable_buffers(self) -> None:
        fixture, inputs = self._fixture_and_inputs()
        replacements = {
            "attention_branch": torch.full(
                (len(inputs), H), 0.125, dtype=torch.float32
            ),
            "post_attention_norm": torch.full(
                (len(inputs), H), -0.25, dtype=torch.float32
            ),
        }
        captured, indices, weights = _capture_c_pre_router(
            self.lib,
            fixture,
            C_CONFIG,
            1,
            inputs.tolist(),
            replace_stages=replacements,
        )

        for point, expected in replacements.items():
            with self.subTest(point=point):
                self.assertTrue(torch.equal(captured[point], expected))
        self.assertEqual(tuple(indices.shape), (len(inputs), TOPK))
        self.assertEqual(tuple(weights.shape), (len(inputs), TOPK))

    def test_cumulative_cases_are_exact_declared_prefixes(self) -> None:
        self.assertEqual(len(CUMULATIVE_INJECTION_CASES), len(POINTS))
        for index, point in enumerate(POINTS):
            with self.subTest(point=point):
                self.assertEqual(
                    CUMULATIVE_INJECTION_CASES[f"through_{point}"],
                    POINTS[: index + 1],
                )

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
