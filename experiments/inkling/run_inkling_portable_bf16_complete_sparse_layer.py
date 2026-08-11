#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply the proven final BF16 residual rule to the composed sparse-layer probe.

This is evidence-only. It reuses #36's composed temporary source and changes
only the final decoder-layer residual in that copied source: complete both
operands to BF16, add, and retain a BF16 result. Production source and the
public matrix-backend ABI remain unchanged.

Unlike the historical runner, importing this module does not permanently
rebind the shared composed-MoE implementation. The complete-layer overrides
exist only while this runner is building or executing, and are restored even
when the underlying probe raises.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import run_inkling_portable_bf16_composed_moe as composed_runner
import diagnose_inkling_portable_bf16_composed_moe as implementation


_base_transform_aggregation_source = composed_runner.transform_aggregation_source

_FINAL_RESIDUAL_OLD = """    for (int i = 0; i < hidden; i++) x[i] += s->ff[i];
"""
_FINAL_RESIDUAL_NEW = """    for (int i = 0; i < hidden; i++) {
        const float residual = bf16_round_probe(x[i]);
        const float branch = bf16_round_probe(s->ff[i]);
        x[i] = bf16_round_probe(residual + branch);
    }
"""


def apply_final_residual_source(source: str) -> str:
    count = source.count(_FINAL_RESIDUAL_OLD)
    if count != 1:
        raise implementation.ComposedMoeError(
            f"expected exactly one final layer residual; found {count}"
        )
    return source.replace(_FINAL_RESIDUAL_OLD, _FINAL_RESIDUAL_NEW, 1)


def transform_complete_sparse_layer_source(source: str) -> str:
    return apply_final_residual_source(
        _base_transform_aggregation_source(source)
    )


@contextmanager
def complete_sparse_layer_overrides() -> Iterator[None]:
    """Install the complete-layer adapters without leaking module state."""
    original_transform = implementation.transform_aggregation_source
    original_collector = implementation.ExactWeightCollector
    try:
        implementation.transform_aggregation_source = (
            transform_complete_sparse_layer_source
        )
        implementation.ExactWeightCollector = composed_runner.ExactWeightCollector
        yield
    finally:
        implementation.transform_aggregation_source = original_transform
        implementation.ExactWeightCollector = original_collector


def build_complete_sparse_layer_library(out):
    """Build the complete-layer candidate with scoped source transforms."""
    with complete_sparse_layer_overrides():
        return implementation.build_composed_library(out)


def main(argv: list[str] | None = None) -> int:
    with complete_sparse_layer_overrides():
        return implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
