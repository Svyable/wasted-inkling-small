#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the bounded official reference with deterministic activation routing.

Synthetic fixtures retain the raw official router path. An official
``thinkingmachines/Inkling-Small`` fixture delegates to the source-bound
canonical route runner so cutoff-tie nondeterminism cannot request an expert
outside the bounded evidence fixture. Raw official routing remains independently
exercised by router discovery and cutoff-tie audit workflows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import inkling_fixture_reference as implementation
from inkling_release_config import (
    build_transformers_text_config,
    canonicalize_router_pairs,
)

_original_run_layer_reference = implementation.run_layer_reference


def _run_layer_reference(*args, **kwargs):
    values = _original_run_layer_reference(*args, **kwargs)
    canonicalize_router_pairs(values)
    return values


def _argument_value(name: str) -> str | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else None


def _is_official_fixture(path: str | None) -> bool:
    if not path:
        return False
    try:
        manifest = json.loads(
            (Path(path) / "fixture.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("model_id") == "thinkingmachines/Inkling-Small"


implementation._text_config = build_transformers_text_config
implementation.run_layer_reference = _run_layer_reference


if __name__ == "__main__":
    fixture = _argument_value("--fixture")
    if _is_official_fixture(fixture):
        from run_inkling_fixture_reference_canonical import main as canonical_main

        if "--route-override" not in sys.argv:
            sys.argv.extend(
                ["--route-override", "docs/OFFICIAL-ROUTER-SELECTION.json"]
            )
        raise SystemExit(canonical_main())
    raise SystemExit(implementation.main())
