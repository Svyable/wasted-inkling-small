#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the composed MoE diagnostic with a fail-closed ctypes collector.

The main diagnostic is intentionally kept focused on arithmetic composition.
This small entrypoint replaces its collector callback with an exception-safe
implementation that preserves raw candidate weights and injects only the exact
fixed-ID weights computed by the committed portable router policy.
"""
from __future__ import annotations

from typing import Any

import diagnose_inkling_portable_bf16_composed_moe as implementation
from inkling_layer_parity import TraceCollector


class ExactWeightCollector(implementation.ExactWeightCollector):
    """Preserve raw weights and inject exact weights without leaking callbacks."""

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
                    raise implementation.ComposedMoeError(
                        "routed-weight width changed"
                    )
                self._store_candidate(
                    layer,
                    b"candidate_routed_weight",
                    data,
                    count,
                )
                weights = self._compute_weights()
                if weights.numel() < count:
                    raise implementation.ComposedMoeError(
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
                    raise implementation.ComposedMoeError(
                        "shared-weight width changed"
                    )
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


implementation.ExactWeightCollector = ExactWeightCollector


def main(argv: list[str] | None = None) -> int:
    return implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
