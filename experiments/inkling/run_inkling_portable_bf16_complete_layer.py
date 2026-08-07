#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply the proven final BF16 residual rule to the composed temporary layer.

This wrapper deliberately reuses the complete evidence-only composition from
PR #36 and changes one additional copied-source statement: the final decoder
residual.  PR #38 proves that the candidate MLP branch may differ in hidden
FP32 values while remaining BF16-exact, and that exact official ``layer_out``
requires both BF16 branch completion and a BF16 result boundary.

Production source and the public matrix-backend ABI remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Importing this runner installs #36's fail-closed experiment adapters first.
import run_inkling_portable_bf16_composed_moe as composed_runner
import diagnose_inkling_portable_bf16_composed_moe as implementation
from diagnose_inkling_bf16_layer_residual_boundary import torch_fingerprint


_FINAL_RESIDUAL_OLD = (
    "    for (int i = 0; i < hidden; i++) x[i] += s->ff[i];\n"
)
_FINAL_RESIDUAL_NEW = """    for (int i = 0; i < hidden; i++) {
        const float residual = bf16_round_probe(x[i]);
        const float branch = bf16_round_probe(s->ff[i]);
        x[i] = bf16_round_probe(residual + branch);
    }
"""

_original_transform_aggregation_source = implementation.transform_aggregation_source
_original_build_composed_library = implementation.build_composed_library


def transform_final_residual_source(source: str) -> str:
    """Apply exactly one explicit BF16 branch/add/result boundary."""
    count = source.count(_FINAL_RESIDUAL_OLD)
    if count != 1:
        raise implementation.ComposedMoeError(
            f"expected exactly one final layer residual; found {count}"
        )
    return source.replace(_FINAL_RESIDUAL_OLD, _FINAL_RESIDUAL_NEW, 1)


def transform_complete_layer_source(source: str) -> str:
    """Compose #36's transforms, then apply only the #38 residual rule."""
    return transform_final_residual_source(
        _original_transform_aggregation_source(source)
    )


def build_complete_layer_library(out: Path) -> tuple[Path, dict[str, Any]]:
    """Build #36's copied source after the single final-residual transform."""
    library, source = _original_build_composed_library(out)
    source = dict(source)
    source["layer_residual_policy"] = {
        "residual_operand": "bfloat16_completed",
        "branch_operand": "bfloat16_completed",
        "sum": "float32_of_bfloat16_operands",
        "result": "bfloat16_completed",
    }
    return library, source


implementation.transform_aggregation_source = transform_complete_layer_source
implementation.build_composed_library = build_complete_layer_library


def validate_complete_result(result: dict[str, Any]) -> None:
    """Require the complete tested layer, not merely another longer prefix."""
    if not result.get("source", {}).get("production_source_unchanged"):
        raise implementation.ComposedMoeError("production source changed")
    policy = result.get("source", {}).get("layer_residual_policy")
    if policy is None:
        raise implementation.ComposedMoeError("final residual policy metadata missing")
    layers = result.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise implementation.ComposedMoeError("complete-layer result has no layers")
    for layer, value in layers.items():
        decision = value.get("decision", {})
        if decision.get("classification") != "single_token_sparse_layer_is_bfloat16_exact":
            raise implementation.ComposedMoeError(
                f"layer {layer} did not become BF16-exact: {decision}"
            )
        if decision.get("first_nonexact_stage") is not None:
            raise implementation.ComposedMoeError(
                f"layer {layer} still names a non-exact stage"
            )
        layer_out = value.get("stages", {}).get("layer_out", {})
        if layer_out.get("bfloat16_exact_fraction") != 1.0:
            raise implementation.ComposedMoeError(
                f"layer {layer} layer_out is not BF16-exact"
            )
        if layer_out.get("raw_exact_fraction") != 1.0:
            raise implementation.ComposedMoeError(
                f"layer {layer} layer_out is not raw-exact"
            )


def _out_path(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", required=True)
    args, _ = parser.parse_known_args(argv)
    return Path(args.out)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    rc = composed_runner.main(arguments)
    if rc:
        return rc
    out = _out_path(arguments)
    result = json.loads(out.read_text(encoding="utf-8"))
    validate_complete_result(result)
    result["format"] = "inkling-portable-bfloat16-complete-layer-probe"
    result["version"] = 1
    result["backend"] = torch_fingerprint()
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
