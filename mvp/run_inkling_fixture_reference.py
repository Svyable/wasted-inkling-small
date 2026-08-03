#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the bounded official reference with release-schema normalization."""
from __future__ import annotations

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


implementation._text_config = build_transformers_text_config
implementation.run_layer_reference = _run_layer_reference


if __name__ == "__main__":
    raise SystemExit(implementation.main())
