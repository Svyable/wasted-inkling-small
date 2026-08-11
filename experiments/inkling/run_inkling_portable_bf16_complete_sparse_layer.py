#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply the proven final BF16 residual rule to the composed sparse-layer probe.

This is evidence-only. It reuses #36's composed temporary source and changes
only the final decoder-layer residual in that copied source: complete both
operands to BF16, add, and retain a BF16 result. Production source and the
public matrix-backend ABI remain unchanged.

The checked-in layer now also contains a promoted BF16_REFERENCE sparse path.
The retained historical probe still has a different job: independently rebuild
that policy from the preserved F32 path using its source transforms. During the
scoped build only, the compatibility adapter below selects the F32 arm from the
temporary source copy before applying those historical transforms. The checked-
in production file is never rewritten.

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
_base_transform_moe_source = implementation.transform_moe_source

_FINAL_RESIDUAL_OLD = """    for (int i = 0; i < hidden; i++) x[i] += s->ff[i];
"""
_FINAL_RESIDUAL_NEW = """    for (int i = 0; i < hidden; i++) {
        const float residual = bf16_round_probe(x[i]);
        const float branch = bf16_round_probe(s->ff[i]);
        x[i] = bf16_round_probe(residual + branch);
    }
"""

_PROFILE_SPARSE_BEGIN = """        if (bf16) {
            /* Official expert reductions are ordered by expert id, not router
"""
# Include the preceding newline so a more deeply indented nested ``else`` cannot
# satisfy this marker by matching halfway through its leading spaces.
_PROFILE_SPARSE_ELSE = """
        } else {
"""
_PROFILE_SPARSE_END = """        }
        if (trace_f(trace, layer, "moe_out", s->ff, (size_t)hidden)) return -1;
"""
# The preserved F32 implementation acquired this harmless line wrap when the
# checked-in profile was promoted. Reconstruct the historical text shape in the
# temporary copy so the old evidence transform keeps testing the same math.
_PROFILE_SHARED_DOWN_CURRENT = """                if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_DOWN, e,
                                 s->branch, NULL, s->gate,
                                 hidden, inter)) return -1;
"""
_PROFILE_SHARED_DOWN_LEGACY = """                if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_DOWN, e,
                                 s->branch, NULL, s->gate, hidden, inter)) return -1;
"""


def _legacy_f32_sparse_source(source: str) -> str:
    """Select the preserved F32 sparse arm from a temporary profile-aware copy.

    Historical composed-MoE transforms predate ``BF16_REFERENCE`` and are
    deliberately anchored to the old F32 sparse block.  Once that BF16 policy
    was promoted into production, the same F32 block moved four spaces deeper
    under ``else``.  Re-indenting every old anchor would make the evidence
    depend on the promoted implementation it is meant to reconstruct.

    Instead, collapse exactly one profile branch in the copied source and feed
    the original F32 body, at its historical indentation, to the old transforms.
    The markers fail closed if production control flow changes again.
    """
    begin_count = source.count(_PROFILE_SPARSE_BEGIN)
    if begin_count != 1:
        raise implementation.ComposedMoeError(
            f"expected exactly one BF16 sparse-profile start; found {begin_count}"
        )
    begin = source.index(_PROFILE_SPARSE_BEGIN)

    # The layer contains several unrelated if/else pairs. Only an outer split
    # whose closing brace begins at this exact indentation is accepted here.
    end_count = source[begin:].count(_PROFILE_SPARSE_END)
    if end_count != 1:
        raise implementation.ComposedMoeError(
            f"expected exactly one sparse-profile end after its start; found {end_count}"
        )
    end = source.index(_PROFILE_SPARSE_END, begin + len(_PROFILE_SPARSE_BEGIN))

    else_line = source.find(
        _PROFILE_SPARSE_ELSE,
        begin + len(_PROFILE_SPARSE_BEGIN),
        end,
    )
    if else_line < 0:
        raise implementation.ComposedMoeError(
            "BF16 sparse-profile region has no F32 else arm"
        )
    duplicate_else = source.find(
        _PROFILE_SPARSE_ELSE,
        else_line + len(_PROFILE_SPARSE_ELSE),
        end,
    )
    if duplicate_else >= 0:
        raise implementation.ComposedMoeError(
            "BF16 sparse-profile region has more than one outer F32 split"
        )

    body_start = else_line + len(_PROFILE_SPARSE_ELSE)
    f32_body = source[body_start:end]

    dedented: list[str] = []
    for line in f32_body.splitlines(keepends=True):
        if line.strip():
            if not line.startswith("    "):
                raise implementation.ComposedMoeError(
                    "profile F32 sparse body lost its expected indentation"
                )
            line = line[4:]
        dedented.append(line)

    # Drop only the copied profile wrapper's closing brace. Keep the existing
    # moe_out trace immediately after it so historical anchors remain intact.
    tail = end + len("        }\n")
    return source[:begin] + "".join(dedented) + source[tail:]


def transform_complete_moe_source(source: str) -> str:
    """Run the historical MoE transform against the preserved F32 arm."""
    legacy = _legacy_f32_sparse_source(source)
    count = legacy.count(_PROFILE_SHARED_DOWN_CURRENT)
    if count != 1:
        raise implementation.ComposedMoeError(
            f"expected exactly one wrapped F32 shared-down block; found {count}"
        )
    legacy = legacy.replace(
        _PROFILE_SHARED_DOWN_CURRENT,
        _PROFILE_SHARED_DOWN_LEGACY,
        1,
    )
    return _base_transform_moe_source(legacy)


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
    original_moe_transform = implementation.transform_moe_source
    original_collector = implementation.ExactWeightCollector
    try:
        implementation.transform_moe_source = transform_complete_moe_source
        implementation.transform_aggregation_source = (
            transform_complete_sparse_layer_source
        )
        implementation.ExactWeightCollector = composed_runner.ExactWeightCollector
        yield
    finally:
        implementation.transform_moe_source = original_moe_transform
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
