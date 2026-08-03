#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Translate the released Inkling text config to the C layer-harness schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class CConfigError(RuntimeError):
    """The C layer config cannot be derived without guessing."""


def _text(config_json: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config_json, dict):
        raise CConfigError("model config is not a JSON object")
    nested = config_json.get("text_config")
    value = nested if isinstance(nested, dict) else config_json
    return dict(value)


def _integer(config: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CConfigError(f"text config needs integer {key!r} >= {minimum}")
    return value


def _number(config: dict[str, Any], key: str, *, positive: bool = False) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CConfigError(f"text config needs numeric {key!r}")
    out = float(value)
    if positive and out <= 0:
        raise CConfigError(f"text config needs positive {key!r}")
    return out


def normalized_c_layer_config(config_json: dict[str, Any]) -> dict[str, Any]:
    """Return exactly the JSON shape consumed by ``inkling_layer_parity.py``."""
    text = _text(config_json)
    layers = _integer(text, "num_hidden_layers")
    local_ids = text.get("local_layer_ids")
    if (
        not isinstance(local_ids, list)
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value >= layers
            for value in local_ids
        )
        or len(set(local_ids)) != len(local_ids)
    ):
        raise CConfigError("text config has invalid local_layer_ids")

    config = {
        "n_layers": layers,
        "hidden": _integer(text, "hidden_size"),
        "vocab": _integer(text, "vocab_size"),
        "unpadded_vocab": _integer(text, "unpadded_vocab_size"),
        "max_context": _integer(text, "model_max_length"),
        "global_heads": _integer(text, "num_attention_heads"),
        "global_kv_heads": _integer(text, "num_key_value_heads"),
        "global_head_dim": _integer(text, "head_dim"),
        "local_heads": _integer(text, "swa_num_attention_heads"),
        "local_kv_heads": _integer(text, "swa_num_key_value_heads"),
        "local_head_dim": _integer(text, "swa_head_dim"),
        "sliding_window": _integer(text, "sliding_window_size"),
        "d_rel": _integer(text, "d_rel"),
        "rel_extent": _integer(text, "rel_extent"),
        "conv_kernel": _integer(text, "sconv_kernel_size"),
        "dense_layers": _integer(text, "dense_mlp_idx", minimum=0),
        "dense_intermediate": _integer(text, "dense_intermediate_size"),
        # Release schema: intermediate_size is routed width.
        "moe_intermediate": _integer(text, "intermediate_size"),
        "n_routed_experts": _integer(text, "n_routed_experts"),
        "top_k": _integer(text, "num_experts_per_tok"),
        "n_shared_experts": _integer(text, "n_shared_experts"),
        "rms_eps": _number(text, "rms_norm_eps", positive=True),
        "route_scale": _number(text, "route_scale", positive=True),
        "logits_width_multiplier": _number(
            text, "logits_mup_width_multiplier", positive=True
        ),
        "log_scaling_n_floor": _integer(text, "log_scaling_n_floor"),
        "log_scaling_alpha": _number(text, "log_scaling_alpha"),
        "local_layer_ids": list(local_ids),
    }

    if config["unpadded_vocab"] > config["vocab"]:
        raise CConfigError("unpadded_vocab_size exceeds vocab_size")
    if config["dense_layers"] > layers:
        raise CConfigError("dense_mlp_idx exceeds num_hidden_layers")
    if config["top_k"] > config["n_routed_experts"]:
        raise CConfigError("num_experts_per_tok exceeds n_routed_experts")
    return config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        value = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
        config = normalized_c_layer_config(value)
        Path(args.out).write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, CConfigError) as exc:
        ap.error(str(exc))
        return 2
    print(f"wrote normalized C layer config to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
