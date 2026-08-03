#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bridge the released Inkling config schema to Transformers' runtime schema.

The released checkpoint uses:

* ``dense_intermediate_size`` for the dense MLP width;
* ``intermediate_size`` for the routed-expert width.

Transformers 5.14.x uses ``intermediate_size`` for dense layers and
``moe_intermediate_size`` for routed experts. Passing the release JSON through
unchanged silently leaves the routed width at the library default. This module
performs the translation explicitly and refuses conflicting or incomplete
geometry.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch


class ReleaseConfigError(RuntimeError):
    """The release schema cannot be translated without guessing."""


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ReleaseConfigError(f"release text config needs positive integer {key!r}")
    return value


def normalize_release_text_config(config_json: dict[str, Any]) -> dict[str, Any]:
    """Return an ``InklingTextConfig``-shaped copy of release JSON.

    The input may be the full multimodal ``config.json`` or its ``text_config``
    object. The source mapping is not mutated.
    """
    if not isinstance(config_json, dict):
        raise ReleaseConfigError("model config is not a JSON object")
    nested = config_json.get("text_config")
    text = nested if isinstance(nested, dict) else config_json

    dense = _positive_int(text, "dense_intermediate_size")
    routed = _positive_int(text, "intermediate_size")
    explicit_moe = text.get("moe_intermediate_size")
    if explicit_moe is not None:
        if not isinstance(explicit_moe, int) or isinstance(explicit_moe, bool):
            raise ReleaseConfigError("moe_intermediate_size must be an integer when present")
        if explicit_moe != routed:
            raise ReleaseConfigError(
                "conflicting routed widths: release intermediate_size="
                f"{routed}, moe_intermediate_size={explicit_moe}"
            )

    normalized = dict(text)
    normalized.pop("dense_intermediate_size", None)
    normalized["intermediate_size"] = dense
    normalized["moe_intermediate_size"] = routed
    return normalized


def build_transformers_text_config(config_json: dict[str, Any]) -> Any:
    """Construct and verify the official Transformers text config."""
    try:
        from transformers import InklingTextConfig
    except (ImportError, AttributeError) as exc:
        raise ReleaseConfigError(
            "Transformers with InklingTextConfig support is required"
        ) from exc

    normalized = normalize_release_text_config(config_json)
    config = InklingTextConfig.from_dict(normalized)
    dense = normalized["intermediate_size"]
    routed = normalized["moe_intermediate_size"]
    if int(config.intermediate_size) != dense:
        raise ReleaseConfigError(
            "Transformers changed dense intermediate during construction: "
            f"expected {dense}, got {config.intermediate_size}"
        )
    if int(config.moe_intermediate_size) != routed:
        raise ReleaseConfigError(
            "Transformers changed routed intermediate during construction: "
            f"expected {routed}, got {config.moe_intermediate_size}"
        )
    config._attn_implementation = "eager"
    return config


def canonicalize_router_pairs(values: MutableMapping[str, torch.Tensor]) -> list[str]:
    """Sort routed ``(expert_id, weight)`` pairs by expert ID in-place.

    The official router uses ``topk(sorted=False)`` while the C runtime emits a
    different slot order. Sorting IDs and gathering their weights gives both
    archives one deterministic representation without separating the pair.
    """
    canonicalized: list[str] = []
    suffix = ".routed_index"
    for index_name in sorted(name for name in values if name.endswith(suffix)):
        base = index_name[: -len(suffix)]
        weight_name = base + ".routed_weight"
        if weight_name not in values:
            raise ReleaseConfigError(f"router indices have no paired weights: {base}")
        index = values[index_name]
        weight = values[weight_name]
        if not isinstance(index, torch.Tensor) or not isinstance(weight, torch.Tensor):
            raise ReleaseConfigError(f"router pair is not tensor-valued: {base}")
        if index.shape != weight.shape or index.ndim < 1:
            raise ReleaseConfigError(
                f"router pair shape mismatch for {base}: {tuple(index.shape)} vs "
                f"{tuple(weight.shape)}"
            )
        if index.is_floating_point() or not weight.is_floating_point():
            raise ReleaseConfigError(f"router pair dtypes are invalid for {base}")

        ordered, order = torch.sort(index.to(torch.int64), dim=-1, stable=True)
        if ordered.shape[-1] > 1 and bool((ordered[..., 1:] == ordered[..., :-1]).any()):
            raise ReleaseConfigError(f"router selected a duplicate expert for {base}")
        values[index_name] = torch.gather(index, -1, order).contiguous()
        values[weight_name] = torch.gather(weight, -1, order).contiguous()
        canonicalized.append(base)
    return canonicalized
