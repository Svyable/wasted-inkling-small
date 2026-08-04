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
    """Mirror official schema: full matrix, routed-only returned logits."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.num_experts = 3
        self.n_shared_experts = 2
        self.n_total_experts = 5
        self.hidden_dim = 5
        self.route_scale = 1.25
        self.weight = nn.Parameter(
            torch.eye(5, dtype=dtype), requires_grad=False
        )
        self.global_scale = nn.Parameter(
            torch.tensor(0.75, dtype=dtype), requires_grad=False
        )

    def forward(self, hidden: torch.Tensor):
        flat = hidden.reshape(-1, self.hidden_dim)
        full = F.linear(flat, self.weight)
        tokens = full.shape[0]
        return (
            full[:, : self.num_experts],
            torch.zeros(tokens, 2, dtype=full.dtype),
            torch.zeros(tokens, 2, dtype=torch.int64),
            torch.zeros(tokens, self.n_shared_experts, dtype=full.dtype),
        )


class OfficialCanonicalRouteTest(unittest.TestCase):
    @staticmethod
    def rows(full_logits: torch.Tensor) -> list[RouteRow]:
        templates = [[0, 2], [1, 2]]
        if full_logits.shape[0] > len(templates):
            raise AssertionError("synthetic route template is too short")
        ids = torch.tensor(templates[: full_logits.shape[0]], dtype=torch.int64)
        selected = torch.cat(
            (full_logits[:, :3].gather(1, ids), full_logits[:, 3:]), dim=1
        )
        log_probs = F.logsigmoid(selected)
        weights = torch.exp(
            log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)
        ) * 1.25 * torch.tensor(0.75, dtype=full_logits.dtype)
        return [
            RouteRow(
                tuple(int(value) for value in ids[row].tolist()),
                tuple(float(value) for value in weights[row, :2].tolist()),
            )
            for row in range(full_logits.shape[0])
        ]

    @staticmethod
    def logits(rows: int = 2) -> torch.Tensor:
        values = torch.tensor(
            [
                [0.25, -0.5, 0.75, 0.125, -0.25],
                [-0.125, 0.5, 0.25, -0.75, 0.375],
            ],
            dtype=torch.bfloat16,
        )
        return values[:rows]

    def test_gate_substitutes_bound_pairs_and_recomputes_shared(self):
        logits = self.logits()
        rows = self.rows(logits)
        wrapper = CanonicalOfficialGate(SyntheticGate(), rows)
        output = wrapper(logits)
        self.assertTrue(wrapper.applied)
        self.assertEqual(tuple(output[0].shape), (2, 3))
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
        logits = self.logits(1)
        row = self.rows(logits)[0]
        wrong = [RouteRow(row.indices, (row.weights[0] + 0.01, row.weights[1]))]
        wrapper = CanonicalOfficialGate(SyntheticGate(), wrong)
        with self.assertRaisesRegex(
            CanonicalOfficialRouteError, "do not reproduce"
        ):
            wrapper(logits)

    def test_gate_rejects_row_count_and_duplicate_ids(self):
        logits = self.logits(1)
        rows = self.rows(logits)
        with self.assertRaisesRegex(CanonicalOfficialRouteError, "produced 1 rows"):
            CanonicalOfficialGate(SyntheticGate(), rows + rows)(logits)
        duplicate = [RouteRow((0, 0), rows[0].weights)]
        with self.assertRaisesRegex(CanonicalOfficialRouteError, "repeats"):
            CanonicalOfficialGate(SyntheticGate(), duplicate)(logits)

    def test_gate_rejects_reconstructed_routed_logit_drift(self):
        logits = self.logits(1)
        rows = self.rows(logits)
        gate = SyntheticGate()
        original = gate.forward

        def drifted(hidden: torch.Tensor):
            output = list(original(hidden))
            output[0] = output[0].clone()
            output[0][0, 0] += torch.tensor(0.125, dtype=output[0].dtype)
            return tuple(output)

        gate.forward = drifted
        with self.assertRaisesRegex(CanonicalOfficialRouteError, "routed logits differ"):
            CanonicalOfficialGate(gate, rows)(logits)

    def test_runner_installs_and_restores_gate_wrapper(self):
        logits = self.logits(1)
        rows = self.rows(logits)
        module = types.SimpleNamespace(
            mlp=types.SimpleNamespace(gate=SyntheticGate())
        )
        original_build = implementation.build_layer_from_fixture

        def fake_build(*_args, **_kwargs):
            return module

        def fake_run(*_args, **_kwargs):
            built = implementation.build_layer_from_fixture(None, None, 2)
            gate_output = built.mlp.gate(logits)
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
                [[0.0] * 5],
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
