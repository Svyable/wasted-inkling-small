#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply the proven final BF16 residual rule to the composed sparse-layer probe.

This is evidence-only. It reuses #36's composed temporary source and changes
only the final decoder-layer residual in that copied source: complete both
operands to BF16, add, and retain a BF16 result. Production source and the
public matrix-backend ABI remain unchanged.
"""
from __future__ import annotations

import run_inkling_portable_bf16_composed_moe as composed_runner
import diagnose_inkling_portable_bf16_composed_moe as implementation


_base_transform_aggregation_source = implementation.transform_aggregation_source

_FINAL_RESIDUAL_OLD = """    for (int i = 0; i < hidden; i++) x[i] += s->ff[i];
"""
_FINAL_RESIDUAL_NEW = """    for (int i = 0; i < hidden; i++) {
        const float residual = bf16_round_probe(x[i]);
        const float branch = bf16_round_probe(s->ff[i]);
        x[i] = bf16_round_probe(residual + branch);
    }
"""


def transform_complete_sparse_layer_source(source: str) -> str:
    transformed = _base_transform_aggregation_source(source)
    count = transformed.count(_FINAL_RESIDUAL_OLD)
    if count != 1:
        raise implementation.ComposedMoeError(
            f"expected exactly one final layer residual; found {count}"
        )
    transformed = transformed.replace(
        _FINAL_RESIDUAL_OLD,
        _FINAL_RESIDUAL_NEW,
        1,
    )
    return transformed


implementation.transform_aggregation_source = transform_complete_sparse_layer_source
implementation.ExactWeightCollector = composed_runner.ExactWeightCollector


def main(argv: list[str] | None = None) -> int:
    return implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
