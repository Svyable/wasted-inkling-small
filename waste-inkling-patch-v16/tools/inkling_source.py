#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Small dependency-free helpers shared by Inkling conversion probes."""

from __future__ import annotations

from typing import Any

INKLING_ARCHES = {
    "InklingForConditionalGeneration",
    "InklingForCausalLM",
}
INKLING_MODEL_TYPES = {"inkling_mm_model", "inkling_text"}


def config_identifies_inkling(config: dict[str, Any]) -> bool:
    """Return true only when the source config explicitly names Inkling.

    Do not infer architecture from a nested ``text_config``: WASTE's existing
    converter uses that shape for Kimi K3, while Inkling uses it as well.
    """
    archs = config.get("architectures")
    if isinstance(archs, list) and any(
        isinstance(value, str) and value in INKLING_ARCHES for value in archs
    ):
        return True
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type in INKLING_MODEL_TYPES:
        return True
    text = config.get("text_config")
    return bool(
        isinstance(text, dict)
        and isinstance(text.get("model_type"), str)
        and text["model_type"] in INKLING_MODEL_TYPES
    )
