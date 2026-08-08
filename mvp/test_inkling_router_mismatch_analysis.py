#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from analyze_inkling_router_mismatch import _route, analyze_layer


class RouterMismatchAnalysisTest(unittest.TestCase):
    def test_identical_bfloat16_path_is_classified_and_exact(self) -> None:
        gate = SimpleNamespace(
            top_k=1,
            n_shared_experts=1,
            route_scale=2.0,
            weight=torch.tensor(
                [[0.5, -0.25], [-0.125, 0.75], [0.25, 0.125]],
                dtype=torch.bfloat16,
            ),
            e_score_correction_bias=torch.tensor(
                [0.01, -0.02], dtype=torch.bfloat16
            ),
            global_scale=torch.tensor([0.75], dtype=torch.bfloat16),
        )
        module = SimpleNamespace(mlp=SimpleNamespace(gate=gate))
        official_input = torch.tensor(
            [[0.5, -0.25], [-0.75, 0.125]], dtype=torch.bfloat16
        )
        official_logits = F.linear(official_input, gate.weight)
        indices, weights, _ = _route(
            official_logits,
            correction_bias=gate.e_score_correction_bias,
            global_scale=gate.global_scale,
            route_scale=gate.route_scale,
            top_k=gate.top_k,
            n_shared=gate.n_shared_experts,
        )
        result = analyze_layer(
            module,
            official_input,
            official_logits,
            indices,
            weights,
            {
                "router_inputs": official_input.float().tolist(),
                "router_logits": official_logits.float().tolist(),
                "topk_indices": indices.tolist(),
                "topk_weights": weights.float().tolist(),
            },
        )
        self.assertEqual(
            result["classification"],
            "c_router_logit_bfloat16_rounding_is_sufficient",
        )
        self.assertEqual(
            result["pre_router_state"]["quantized_exact_fraction"], 1.0
        )
        for variant in (
            "official_recomputed",
            "c_emitted",
            "c_logits_float32_official_route",
            "c_logits_rounded_bfloat16",
            "official_bfloat16_matmul_on_c_input",
        ):
            self.assertTrue(result["variants"][variant]["all_indices_exact"])

    def test_malformed_tensor_shape_fails_closed(self) -> None:
        gate = SimpleNamespace(
            top_k=1,
            n_shared_experts=1,
            route_scale=1.0,
            weight=torch.ones(3, 2, dtype=torch.bfloat16),
            e_score_correction_bias=torch.zeros(2, dtype=torch.bfloat16),
            global_scale=torch.ones(1, dtype=torch.bfloat16),
        )
        module = SimpleNamespace(mlp=SimpleNamespace(gate=gate))
        official_input = torch.ones(1, 2, dtype=torch.bfloat16)
        logits = F.linear(official_input, gate.weight)
        indices, weights, _ = _route(
            logits,
            correction_bias=gate.e_score_correction_bias,
            global_scale=gate.global_scale,
            route_scale=1.0,
            top_k=1,
            n_shared=1,
        )
        with self.assertRaisesRegex(
            RuntimeError, "shape mismatch|cannot be multiplied"
        ):
            analyze_layer(
                module,
                official_input,
                logits,
                indices,
                weights,
                {
                    "router_inputs": [[1.0, 1.0, 1.0]],
                    "router_logits": logits.float().tolist(),
                    "topk_indices": indices.tolist(),
                    "topk_weights": weights.float().tolist(),
                },
            )


if __name__ == "__main__":
    unittest.main()
