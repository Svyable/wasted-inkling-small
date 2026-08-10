#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exception-safe weight adapter for the checked-in BF16 promotion probe.

The historical base ``ExactWeightCollector`` in the composed-MoE diagnostic
assumes a private ``_preserve`` helper supplied by its wrapper entrypoint.  The
checked-in promotion probe intentionally does not import that wrapper because
it mutates several shared diagnostic globals at import time.  This adapter
provides the same candidate-weight preservation explicitly and is safe to use
without those historical side effects.
"""
from __future__ import annotations

from typing import Any

import diagnose_inkling_portable_bf16_composed_moe as composed
from inkling_layer_parity import TraceCollector


class CheckedExactWeightCollector(composed.ExactWeightCollector):
    """Preserve raw weights and inject exact weights without global mutation."""

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
