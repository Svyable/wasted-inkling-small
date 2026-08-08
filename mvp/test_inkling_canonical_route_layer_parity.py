#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes
import json
import tempfile
import unittest
from pathlib import Path

import torch

from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import (
    CanonicalRouteError,
    RouteRow,
    load_canonical_routes,
    run_sparse_layer_canonical,
)
from inkling_compact_layer_parity import (
    bind_sparse_layer_compact,
    run_layer_trace_compact,
)
from inkling_fixture import load_fixture
from inkling_layer_parity import build_library, configure_library
from test_inkling_official_layer_parity import (
    C_CONFIG,
    H,
    NROUTED,
    TOPK,
    make_tensors,
    write_fixture,
)


REPO = Path(__file__).resolve().parents[1]


class CanonicalRouteLayerParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.td = tempfile.TemporaryDirectory()
        cls.lib = ctypes.CDLL(
            str(
                build_library(
                    REPO / "inkling" / "src",
                    Path(cls.td.name) / "canonical-route.so",
                )
            )
        )
        configure_library(cls.lib)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.td.cleanup()

    def _case(self):
        root = Path(tempfile.mkdtemp(dir=self.td.name))
        write_fixture(root, 1, make_tensors(1))
        fixture = load_fixture(root)
        compact = bind_sparse_layer_compact(fixture, C_CONFIG, 1)
        torch.manual_seed(73)
        inputs = torch.randn(6, H).tolist()
        baseline = run_layer_trace_compact(
            self.lib, C_CONFIG, 1, compact, inputs
        )
        rows = []
        for position in range(len(inputs)):
            ids = list(
                baseline["values"][f"token.{position}.layer.1.routed_index"]
            )
            weights = list(
                baseline["values"][f"token.{position}.layer.1.routed_weight"]
            )
            rows.append(RouteRow(tuple(ids), tuple(weights)))
        return fixture, compact, inputs, baseline, rows

    def test_override_preserves_candidate_trace_and_changes_execution_route(self) -> None:
        fixture, compact, inputs, baseline, rows = self._case()
        first = rows[0]
        replacement = next(
            expert for expert in range(NROUTED) if expert not in first.indices
        )
        changed = list(rows)
        changed[0] = RouteRow(
            (replacement,) + first.indices[1:],
            first.weights,
        )

        traced = run_sparse_layer_canonical(
            self.lib,
            C_CONFIG,
            1,
            compact,
            inputs,
            changed,
        )

        for position, row in enumerate(changed):
            candidate_name = (
                f"token.{position}.layer.1.candidate_routed_index"
            )
            canonical_name = f"token.{position}.layer.1.routed_index"
            self.assertEqual(
                traced["values"][candidate_name],
                baseline["values"][canonical_name],
            )
            self.assertEqual(traced["values"][canonical_name], list(row.indices))
            self.assertEqual(
                traced["values"][
                    f"token.{position}.layer.1.routed_weight"
                ],
                list(row.weights),
            )
        self.assertNotEqual(traced["outputs"][0], baseline["outputs"][0])

    def test_missing_override_expert_fails_before_layer_execution(self) -> None:
        _fixture, compact, inputs, _baseline, rows = self._case()
        invalid = list(rows)
        invalid[0] = RouteRow(
            (NROUTED,) + invalid[0].indices[1:],
            invalid[0].weights,
        )
        with self.assertRaisesRegex(CanonicalRouteError, "fixture-missing"):
            run_sparse_layer_canonical(
                self.lib,
                C_CONFIG,
                1,
                compact,
                inputs,
                invalid,
            )

    def test_route_file_is_bound_to_fixture_and_inputs(self) -> None:
        fixture, _compact, inputs, baseline, _rows = self._case()
        layers = {
            "1": {
                "topk_indices": [
                    baseline["values"][
                        f"token.{position}.layer.1.routed_index"
                    ]
                    for position in range(len(inputs))
                ],
                "topk_weights": [
                    baseline["values"][
                        f"token.{position}.layer.1.routed_weight"
                    ]
                    for position in range(len(inputs))
                ],
            }
        }
        data = {
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": len(inputs),
            "input_dtype": "bfloat16",
            "inputs_sha256": input_sha256(
                torch.tensor(inputs, dtype=torch.float32), torch.bfloat16
            ),
            "layers": layers,
        }
        path = Path(tempfile.mkdtemp(dir=self.td.name)) / "routes.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        loaded = load_canonical_routes(path, fixture, inputs, [1])
        self.assertEqual(len(loaded[1]), len(inputs))
        self.assertEqual(len(loaded[1][0].indices), TOPK)

        data["inputs_sha256"] = "0" * 64
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(CanonicalRouteError, "input hash"):
            load_canonical_routes(path, fixture, inputs, [1])


if __name__ == "__main__":
    unittest.main()
