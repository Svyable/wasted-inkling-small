#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import types
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

import inkling_fixture_reference as implementation
from inkling_canonical_route_layer_parity import RouteRow
from run_inkling_fixture_reference_canonical import (
    CanonicalOfficialGate,
    CanonicalOfficialRouteError,
    run_layer_reference_canonical,
)


class SyntheticGate(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.num_experts = 3
        self.n_shared_experts = 2
        self.route_scale = 1.25
        self.global_scale = nn.Parameter(
            torch.tensor(0.75, dtype=logits.dtype), requires_grad=False
        )

    def forward(self, _hidden: torch.Tensor):
        tokens = self.logits.shape[0]
        return (
            self.logits,
            torch.zeros(tokens, 2, dtype=self.logits.dtype),
            torch.zeros(tokens, 2, dtype=torch.int64),
            torch.zeros(tokens, 2, dtype=self.logits.dtype),
        )


class OfficialCanonicalRouteTest(unittest.TestCase):
    @staticmethod
    def rows(logits: torch.Tensor) -> list[RouteRow]:
        templates = [[0, 2], [1, 2]]
        if logits.shape[0] > len(templates):
            raise AssertionError("synthetic route template is too short")
        ids = torch.tensor(templates[: logits.shape[0]], dtype=torch.int64)
        selected = torch.cat(
            (logits[:, :3].gather(1, ids), logits[:, 3:]), dim=1
        )
        log_probs = F.logsigmoid(selected)
        weights = torch.exp(
            log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)
        ) * 1.25 * torch.tensor(0.75, dtype=logits.dtype)
        return [
            RouteRow(
                tuple(int(value) for value in ids[row].tolist()),
                tuple(float(value) for value in weights[row, :2].tolist()),
            )
            for row in range(logits.shape[0])
        ]

    def test_gate_substitutes_bound_pairs_and_recomputes_shared(self):
        logits = torch.tensor(
            [
                [0.25, -0.5, 0.75, 0.125, -0.25],
                [-0.125, 0.5, 0.25, -0.75, 0.375],
            ],
            dtype=torch.bfloat16,
        )
        rows = self.rows(logits)
        wrapper = CanonicalOfficialGate(SyntheticGate(logits), rows)
        output = wrapper(torch.zeros(2, 4, dtype=torch.bfloat16))
        self.assertTrue(wrapper.applied)
        self.assertEqual(output[2].tolist(), [list(row.indices) for row in rows])
        self.assertEqual(
            output[1].float().tolist(),
            [[float(value) for value in row.weights] for row in rows],
        )
        self.assertEqual(tuple(output[3].shape), (2, 2))
        total = torch.cat((output[1], output[3]), dim=1).sum(dim=1)
        expected_total = torch.full_like(total, 1.25 * 0.75)
        self.assertTrue(torch.allclose(total, expected_total, atol=2e-3, rtol=0.0))

    def test_gate_rejects_committed_weight_drift(self):
        logits = torch.tensor(
            [[0.25, -0.5, 0.75, 0.125, -0.25]],
            dtype=torch.bfloat16,
        )
        row = self.rows(logits)[0]
        wrong = [RouteRow(row.indices, (row.weights[0] + 0.01, row.weights[1]))]
        wrapper = CanonicalOfficialGate(SyntheticGate(logits), wrong)
        with self.assertRaisesRegex(
            CanonicalOfficialRouteError, "do not reproduce"
        ):
            wrapper(torch.zeros(1, 4, dtype=torch.bfloat16))

    def test_gate_rejects_row_count_and_duplicate_ids(self):
        logits = torch.tensor(
            [[0.25, -0.5, 0.75, 0.125, -0.25]],
            dtype=torch.bfloat16,
        )
        rows = self.rows(logits)
        with self.assertRaisesRegex(CanonicalOfficialRouteError, "produced 1 rows"):
            CanonicalOfficialGate(SyntheticGate(logits), rows + rows)(
                torch.zeros(1, 4, dtype=torch.bfloat16)
            )
        duplicate = [RouteRow((0, 0), rows[0].weights)]
        with self.assertRaisesRegex(CanonicalOfficialRouteError, "repeats"):
            CanonicalOfficialGate(SyntheticGate(logits), duplicate)(
                torch.zeros(1, 4, dtype=torch.bfloat16)
            )

    def test_runner_installs_and_restores_gate_wrapper(self):
        logits = torch.tensor(
            [[0.25, -0.5, 0.75, 0.125, -0.25]],
            dtype=torch.bfloat16,
        )
        rows = self.rows(logits)
        module = types.SimpleNamespace(
            mlp=types.SimpleNamespace(gate=SyntheticGate(logits))
        )
        original_build = implementation.build_layer_from_fixture

        def fake_build(*_args, **_kwargs):
            return module

        def fake_run(*_args, **_kwargs):
            built = implementation.build_layer_from_fixture(None, None, 2)
            gate_output = built.mlp.gate(
                torch.zeros(1, 4, dtype=torch.bfloat16)
            )
            return {
                "token.0.layer.2.routed_index": gate_output[2][0].to(torch.int32),
                "token.0.layer.2.routed_weight": gate_output[1][0].float(),
            }

        with mock.patch.object(implementation, "build_layer_from_fixture", fake_build), mock.patch.object(
            implementation, "run_layer_reference", fake_run
        ), mock.patch(
            "run_inkling_fixture_reference_canonical.canonicalize_router_pairs",
            lambda values: values,
        ):
            values = run_layer_reference_canonical(
                object(),
                object(),
                2,
                [[0.0] * 4],
                rows,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )
            self.assertIn("token.0.layer.2.routed_index", values)
            self.assertIs(
                implementation.build_layer_from_fixture,
                fake_build,
            )
        self.assertIs(implementation.build_layer_from_fixture, original_build)


if __name__ == "__main__":
    unittest.main()
