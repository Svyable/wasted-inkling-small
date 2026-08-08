#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes
import tempfile
import unittest
from pathlib import Path

import torch

import diagnose_inkling_portable_bf16_composed_moe as composed
from diagnose_inkling_portable_bf16_stateful_sparse_layer import (
    StatefulExactWeightCollector,
    classify_positions,
)
from inkling_canonical_route_layer_parity import RouteRow


class _Provider:
    def __init__(self) -> None:
        self.router_logits = torch.arange(6, dtype=torch.float32)


class _Helper:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, selected, route_scale, global_scale, family, flags):
        del selected, route_scale, global_scale, family, flags
        self.calls += 1
        base = float(self.calls)
        return torch.tensor(
            [base, base + 0.25, base + 0.5, base + 0.75],
            dtype=torch.float32,
        )


def _metric(raw: float = 1.0, bf16: float = 1.0):
    return {
        "raw_exact_fraction": raw,
        "bfloat16_exact_fraction": bf16,
    }


class StatefulSparseLayerTest(unittest.TestCase):
    def test_classifier_distinguishes_preexpert_and_output_boundaries(self):
        exact = [_metric(), _metric()]
        self.assertEqual(
            classify_positions(exact, exact)["classification"],
            "stateful_sparse_layer_outputs_raw_exact",
        )
        pre = [_metric(), _metric(bf16=0.5)]
        decision = classify_positions(pre, exact)
        self.assertEqual(decision["classification"], "stateful_preexpert_mismatch")
        self.assertEqual(decision["first_preexpert_nonexact_position"], 1)
        outputs = [_metric(), _metric(raw=0.5, bf16=1.0)]
        self.assertEqual(
            classify_positions(exact, outputs)["classification"],
            "stateful_sparse_layer_outputs_bfloat16_exact",
        )

    def test_exact_weight_collector_is_position_scoped(self):
        rows = [
            RouteRow((2, 0), (0.0, 0.0)),
            RouteRow((3, 1), (0.0, 0.0)),
        ]
        provider = _Provider()
        helper = _Helper()
        collector = StatefulExactWeightCollector(
            rows, provider, helper, n_routed=4,
            route_scale=2.5, global_scale=4.0,
        )
        for position, expected in ((0, [1.0, 1.25]), (1, [2.0, 2.25])):
            collector.position = position
            routed = (ctypes.c_float * 2)(0.9, 0.1)
            shared = (ctypes.c_float * 2)(0.2, 0.3)
            self.assertEqual(
                collector._emit_float(None, 2, b"routed_weight", routed, 2), 0
            )
            self.assertEqual(
                collector._emit_float(None, 2, b"shared_weight", shared, 2), 0
            )
            self.assertEqual([float(value) for value in routed], expected)
        self.assertEqual(helper.calls, 2)
        self.assertIsNone(collector.error)

    def test_complete_candidate_compiles_and_preserves_production(self):
        with tempfile.TemporaryDirectory() as directory:
            library, source = composed.build_composed_library(
                Path(directory) / "libinkling-stateful-sparse.so"
            )
            self.assertTrue(library.is_file())
            self.assertTrue(source["production_source_unchanged"])


if __name__ == "__main__":
    unittest.main()
