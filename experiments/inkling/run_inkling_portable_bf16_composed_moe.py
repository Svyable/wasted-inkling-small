#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the composed MoE diagnostic with fail-closed experiment adapters.

The arithmetic implementation stays focused on policy composition. This small
entrypoint replaces experiment-only adapters:

* a scratch transform that reuses the inactive post-attention output buffer as
  the shared FP32 accumulator, guarded by ``qdim >= hidden``;
* an exception-safe collector that preserves raw candidate weights and injects
  only exact fixed-ID weights from the committed portable router policy;
* the complete ordered pre-router trace ahead of every expert-local stage, so
  composition regressions name the first actual compared boundary.
"""
from __future__ import annotations

from typing import Any

import torch

import diagnose_inkling_portable_bf16_composed_moe as implementation
from inkling_layer_parity import TraceCollector


_original_transform_aggregation_source = implementation.transform_aggregation_source
_original_official_stages = implementation.official_stages
PROBE_STAGES = tuple(implementation.PRE_ROUTER_POINTS) + implementation.STAGES


def transform_aggregation_source(source: str) -> str:
    """Reuse ``attn_out`` only after proving it can hold one hidden row."""
    transformed = _original_transform_aggregation_source(source)
    replacements = (
        (
            "        for (int i = 0; i < hidden; i++) s->attn[i] = 0.0f;\n",
            "        if (qdim < hidden) return -1;\n"
            "        for (int i = 0; i < hidden; i++) s->attn_out[i] = 0.0f;\n",
        ),
        (
            "            for (int i = 0; i < hidden; i++) s->attn[i] += s->branch[i];\n",
            "            for (int i = 0; i < hidden; i++) "
            "s->attn_out[i] += s->branch[i];\n",
        ),
        (
            "            const float shared = bf16_round_probe(s->attn[i]);\n",
            "            const float shared = "
            "bf16_round_probe(s->attn_out[i]);\n",
        ),
    )
    for old, new in replacements:
        count = transformed.count(old)
        if count != 1:
            raise implementation.ComposedMoeError(
                f"expected exactly one shared scratch adapter anchor; found {count}"
            )
        transformed = transformed.replace(old, new, 1)
    return transformed


def official_stages(module, normalized, residual, row):
    """Retain the pre-expert row while the base probe adds the full trace."""
    anchor = normalized.detach().to(torch.bfloat16).float().reshape(-1)
    stages, route = _original_official_stages(module, normalized, residual, row)
    return {"post_attention_norm": anchor, **stages}, route


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


implementation.transform_aggregation_source = transform_aggregation_source
implementation.official_stages = official_stages
implementation.ExactWeightCollector = ExactWeightCollector
implementation.STAGES = PROBE_STAGES


def main(argv: list[str] | None = None) -> int:
    return implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
