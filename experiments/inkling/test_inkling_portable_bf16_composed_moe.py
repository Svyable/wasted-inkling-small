#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes
import tempfile
import unittest
from pathlib import Path

import torch

from diagnose_inkling_portable_bf16_composed_moe import (
    ComposedMoeError,
    ROUTER_POLICY_FLAGS,
    STAGES,
    build_composed_library,
    classify_stages,
    transform_aggregation_source,
)
from inkling_canonical_route_layer_parity import RouteRow
from run_inkling_portable_bf16_composed_moe import ExactWeightCollector


class _Provider:
    def __init__(self) -> None:
        self.router_logits = torch.tensor(
            [0.5, -0.25, 1.0, -2.0, 0.125, 0.75],
            dtype=torch.float32,
        )


class _Helper:
    def evaluate(self, selected, route_scale, global_scale, family, flags):
        del selected, route_scale, global_scale, family
        if flags != ROUTER_POLICY_FLAGS:
            raise AssertionError(flags)
        return torch.tensor(
            [0.5, 0.25, 0.125, 0.0625],
            dtype=torch.float32,
        )


class ComposedMoeTest(unittest.TestCase):
    @staticmethod
    def stage_metrics(exact: int):
        return {
            stage: {
                "bfloat16_exact_fraction": 1.0 if index < exact else 0.5
            }
            for index, stage in enumerate(STAGES)
        }

    def test_router_policy_is_the_shared_exact_policy(self):
        self.assertEqual(ROUTER_POLICY_FLAGS, 0x7E1)

    def test_stage_classifier_advances_through_the_complete_layer(self):
        partial = classify_stages(self.stage_metrics(9))
        self.assertEqual(
            partial["classification"],
            "composed_moe_mismatch_isolated",
        )
        self.assertEqual(partial["first_nonexact_stage"], "mlp_branch")
        complete = classify_stages(self.stage_metrics(len(STAGES)))
        self.assertEqual(
            complete["classification"],
            "single_token_sparse_layer_is_bfloat16_exact",
        )
        self.assertIsNone(complete["first_nonexact_stage"])

    def test_aggregation_transform_is_fail_closed(self):
        source = """        for (int i = 0; i < hidden; i++) s->ff[i] = 0.0f;
        for (int k = 0; k < cfg->top_k; k++) {
            const float gamma = s->routed_weight[k];
            for (int i = 0; i < hidden; i++) s->ff[i] += gamma * s->branch[i];
        const size_t gh = (size_t)inter * hidden;
        const size_t dh = (size_t)hidden * inter;
                if (e == 0 && trace_f(trace, layer, "shared_gated0",
                                      s->gate, (size_t)inter)) return -1;
                if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_DOWN, e,
                                 s->branch, NULL, s->gate, hidden, inter)) return -1;
                if (e == 0 && trace_f(trace, layer, "shared_down0",
                                      s->branch, (size_t)hidden)) return -1;
            const float gamma = s->shared_weight[e];
            for (int i = 0; i < hidden; i++) s->ff[i] += gamma * s->branch[i];
        }
        if (trace_f(trace, layer, "moe_out", s->ff, (size_t)hidden)) return -1;
"""
        transformed = transform_aggregation_source(source)
        self.assertIn("s->routed_index[b] < s->routed_index[a]", transformed)
        self.assertIn("shared_gamma", transformed)
        self.assertIn("bf16_round_probe(routed + shared)", transformed)
        with self.assertRaisesRegex(
            ComposedMoeError,
            "routed expert-order loop",
        ):
            transform_aggregation_source(transformed)

    def test_exact_weight_collector_preserves_and_injects_weights(self):
        collector = ExactWeightCollector(
            RouteRow((2, 0), (0.0, 0.0)),
            _Provider(),
            _Helper(),
            n_routed=4,
            route_scale=2.5,
            global_scale=4.0,
        )
        collector.position = 0
        routed = (ctypes.c_float * 2)(0.9, 0.1)
        shared = (ctypes.c_float * 2)(0.2, 0.3)
        self.assertEqual(
            collector._emit_float(None, 2, b"routed_weight", routed, 2),
            0,
        )
        self.assertEqual(
            collector._emit_float(None, 2, b"shared_weight", shared, 2),
            0,
        )
        self.assertEqual([float(value) for value in routed], [0.5, 0.25])
        self.assertEqual([float(value) for value in shared], [0.125, 0.0625])
        self.assertEqual(
            collector.values["token.0.layer.2.candidate_routed_weight"],
            [0.8999999761581421, 0.10000000149011612],
        )
        self.assertEqual(
            collector.values["token.0.layer.2.candidate_shared_weight"],
            [0.20000000298023224, 0.30000001192092896],
        )
        self.assertIsNone(collector.error)

    def test_collector_fails_closed_when_logits_are_missing(self):
        provider = _Provider()
        provider.router_logits = None
        collector = ExactWeightCollector(
            RouteRow((2, 0), (0.0, 0.0)),
            provider,
            _Helper(),
            n_routed=4,
            route_scale=2.5,
            global_scale=4.0,
        )
        collector.position = 0
        routed = (ctypes.c_float * 2)(0.9, 0.1)
        self.assertEqual(
            collector._emit_float(None, 2, b"routed_weight", routed, 2),
            1,
        )
        self.assertIn("router logits were not captured", collector.error)

    def test_temporary_candidate_compiles_and_preserves_production(self):
        with tempfile.TemporaryDirectory() as directory:
            library, source = build_composed_library(
                Path(directory) / "libinkling-composed-moe.so"
            )
            self.assertTrue(library.is_file())
            self.assertTrue(source["production_source_unchanged"])
            self.assertNotEqual(
                source["production_layer_sha256"],
                source["probe_layer_sha256"],
            )
            self.assertNotEqual(
                source["production_attention_sha256"],
                source["probe_attention_sha256"],
            )
            self.assertEqual(
                source["router_policy"],
                {"family": "logsumexp", "flags": 0x7E1},
            )
            self.assertEqual(
                source["aggregation_policy"]["routed_order"],
                "ascending_expert_id",
            )


if __name__ == "__main__":
    unittest.main()
