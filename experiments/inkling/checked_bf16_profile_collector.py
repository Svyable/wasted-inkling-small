#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Side-effect-free adapters for the checked-in BF16 promotion probe.

The historical composed-MoE diagnostic predates the final router denominator
result and also assumes wrapper-provided collector helpers.  The checked-in
promotion probe must use neither hidden import state nor the superseded #35
weight policy.  These adapters preserve raw candidate weights, inject the
source-bound canonical route, and evaluate exact downstream weights with the
retained #51 post-reduction denominator contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import diagnose_inkling_portable_bf16_composed_moe as composed
from diagnose_inkling_bf16_router_post_reduction_denom import (
    EXPECTED_FLAGS,
    PostReductionHelper,
    build_helper as build_post_reduction_helper,
)
from inkling_layer_parity import TraceCollector


class CheckedPostReductionWeightHelper:
    """Expose #51 weights through the historical fixed-helper call shape."""

    def __init__(self, library: Path) -> None:
        self.helper = PostReductionHelper(library)

    def evaluate(
        self,
        selected: Any,
        route_scale: float,
        global_scale: float,
        family: int | None = None,
        flags: int | None = None,
    ):
        # ``family``/``flags`` are deliberately ignored: callers inherited the
        # old lattice signature, but this promotion gate has one reviewed policy.
        stages = self.helper.evaluate(
            selected,
            float(route_scale),
            float(global_scale),
            EXPECTED_FLAGS,
        )
        final = stages["final"]
        if final.numel() != selected.numel():
            raise composed.ComposedMoeError(
                "post-reduction helper changed the selected-weight width"
            )
        return final


def build_checked_weight_helper(out: Path) -> Path:
    return build_post_reduction_helper(out)


class CheckedExactWeightCollector(composed.ExactWeightCollector):
    """Preserve raw weights and inject #51 exact weights without global mutation."""

    def _store_candidate(
        self,
        layer: int,
        point: bytes,
        data: Any,
        count: int,
    ) -> None:
        name = self._name(layer, point)
        self.values[name] = [float(data[index]) for index in range(count)]
        self.dtypes[name] = "F32"

    def _emit_float(self, ctx, layer, point, data, count) -> int:
        try:
            decoded = point.decode("ascii", "strict")
            count = int(count)
            if decoded == "routed_weight":
                if count != len(self.rows[0].indices):
                    raise composed.ComposedMoeError("routed-weight width changed")
                self._store_candidate(
                    layer,
                    b"candidate_routed_weight",
                    data,
                    count,
                )
                weights = self._compute_weights()
                if weights.numel() < count:
                    raise composed.ComposedMoeError(
                        "exact routed-weight vector is too short"
                    )
                for index in range(count):
                    data[index] = float(weights[index])
            elif decoded == "shared_weight":
                self._store_candidate(
                    layer,
                    b"candidate_shared_weight",
                    data,
                    count,
                )
                weights = self._compute_weights()
                start = len(self.rows[0].indices)
                if weights.numel() != start + count:
                    raise composed.ComposedMoeError("shared-weight width changed")
                for index in range(count):
                    data[index] = float(weights[start + index])
            return TraceCollector._emit_float(
                self,
                ctx,
                layer,
                point,
                data,
                count,
            )
        except Exception as exc:  # ctypes callbacks cannot propagate safely.
            self.error = str(exc)
            return 1
