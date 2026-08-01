#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Identify and validate the official Thinking Machines Inkling-Small release.

The model was released after the earlier WASTE Inkling audit.  This module is
the single source of truth for the exact released profile.  It deliberately
separates:

* architecture/profile identity (config fields),
* checkpoint packaging identity (index size and shard count), and
* local conversion readiness (required files and text tensor names).

A future Inkling-family model may still use the generic planner, but it must not
be labelled Inkling-Small unless every profile field below matches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ReleaseError(RuntimeError):
    pass


MODEL_ID = "thinkingmachines/Inkling-Small"
RELEASE_UPLOAD_COMMIT = "21152b5"
CHECKPOINT_TOTAL_SIZE = 531_912_898_740
CHECKPOINT_SHARDS = 32

# Exact fields serialized in the official 2026-08-01 config.  Values that are
# semantically relevant to text parity are pinned here; cosmetic/library
# metadata can change without invalidating the profile.
TEXT_PROFILE: dict[str, Any] = {
    "model_max_length": 1_048_576,
    "torch_dtype": "bfloat16",
    "hidden_size": 4096,
    "num_hidden_layers": 42,
    "vocab_size": 201_024,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "d_rel": 16,
    "rel_extent": 1024,
    "q_bias": False,
    "o_bias": False,
    "log_scaling_n_floor": 128_000,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-6,
    "use_embed_norm": True,
    "local_layer_ids": [
        0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16,
        18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32,
        33, 34, 36, 37, 38, 39, 40,
    ],
    "dense_mlp_idx": 2,
    "use_sconv": True,
    "sconv_kernel_size": 4,
    "unpadded_vocab_size": 200_058,
    "logits_mup_width_multiplier": 16.0,
    "final_logit_softcapping": None,
    "swa_head_dim": 128,
    "swa_num_attention_heads": 32,
    "swa_num_key_value_heads": 8,
    "sliding_window_size": 512,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "n_shared_experts": 2,
    "shared_expert_sink": True,
    "dense_intermediate_size": 16_384,
    "intermediate_size": 2048,
    "route_scale": 8.0,
    "use_gate_bias": True,
    "gate_activation": "sigmoid",
    "norm_after_topk": True,
    "use_global_scale": True,
}

AUDIO_PROFILE: dict[str, Any] = {
    "decoder_dmodel": 4096,
    "n_mel_bins": 80,
    "mel_vocab_size": 16,
    "bias": False,
    "dmel_min_value": -7.0,
    "dmel_max_value": 2.0,
    "use_audio_norm": True,
    "audio_mode": "dmel",
}

VISION_PROFILE: dict[str, Any] = {
    "vision_encoder_type": "hmlp",
    "decoder_dmodel": 4096,
    "patch_size": 40,
    "temporal_patch_size": 2,
    "n_channels": 3,
    "n_layers": 4,
    "use_vision_norm": True,
}

REQUIRED_SIDECARS = (
    "chat_template.jinja",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tiktoken/tokenizer.model",
)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseError(f"missing required file: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as exc:
        raise ReleaseError(f"cannot hash {path}: {exc}") from exc
    return h.hexdigest()


def _mismatches(actual: dict[str, Any], expected: dict[str, Any], prefix: str) -> list[str]:
    out: list[str] = []
    for key, want in expected.items():
        got = actual.get(key)
        if got != want:
            out.append(f"{prefix}.{key}: expected {want!r}, got {got!r}")
    return out


def required_text_tensor_names(config: dict[str, Any]) -> set[str]:
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ReleaseError("config.json text_config is missing")
    layers = text.get("num_hidden_layers")
    dense = text.get("dense_mlp_idx")
    if not isinstance(layers, int) or not isinstance(dense, int):
        raise ReleaseError("config lacks layer/dense counts")

    names = {
        "model.llm.embed.weight",
        "model.llm.embed_norm.weight",
        "model.llm.norm.weight",
        "model.llm.unembed.weight",
    }
    for layer in range(layers):
        p = f"model.llm.layers.{layer}"
        names.update(
            {
                f"{p}.attn_norm.weight",
                f"{p}.mlp_norm.weight",
                f"{p}.attn.wq_du.weight",
                f"{p}.attn.wk_dv.weight",
                f"{p}.attn.wv_dv.weight",
                f"{p}.attn.wr_du.weight",
                f"{p}.attn.wo_ud.weight",
                f"{p}.attn.q_norm.weight",
                f"{p}.attn.k_norm.weight",
                f"{p}.attn.rel_logits_proj.proj",
                f"{p}.attn.k_sconv.weight",
                f"{p}.attn.v_sconv.weight",
                f"{p}.attn_sconv.weight",
                f"{p}.mlp_sconv.weight",
            }
        )
        if layer < dense:
            names.update(
                {
                    f"{p}.mlp.w13_dn.weight",
                    f"{p}.mlp.w2_md.weight",
                    f"{p}.mlp.global_scale",
                }
            )
        else:
            names.update(
                {
                    f"{p}.mlp.experts.w13_weight",
                    f"{p}.mlp.experts.w2_weight",
                    f"{p}.mlp.gate.weight",
                    f"{p}.mlp.gate.bias",
                    f"{p}.mlp.gate.global_scale",
                    f"{p}.mlp.shared_experts.shared_w13_weight",
                    f"{p}.mlp.shared_experts.shared_w2_weight",
                }
            )
    return names


def inspect_release(src: Path | str) -> dict[str, Any]:
    root = Path(src)
    cfg = _load(root / "config.json")
    index = _load(root / "model.safetensors.index.json")
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    audio = cfg.get("audio_config") if isinstance(cfg.get("audio_config"), dict) else {}
    vision = cfg.get("vision_config") if isinstance(cfg.get("vision_config"), dict) else {}

    profile_errors: list[str] = []
    if cfg.get("architectures") != ["InklingForConditionalGeneration"]:
        profile_errors.append("config.architectures is not the official Inkling-Small value")
    if cfg.get("model_type") != "inkling_mm_model":
        profile_errors.append("config.model_type is not inkling_mm_model")
    if cfg.get("eos_token_id") != 200006:
        profile_errors.append("config.eos_token_id is not 200006")
    profile_errors.extend(_mismatches(text, TEXT_PROFILE, "text_config"))
    profile_errors.extend(_mismatches(audio, AUDIO_PROFILE, "audio_config"))
    profile_errors.extend(_mismatches(vision, VISION_PROFILE, "vision_config"))

    wm = index.get("weight_map")
    if not isinstance(wm, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in wm.items()):
        raise ReleaseError("invalid model.safetensors.index.json weight_map")
    metadata = index.get("metadata")
    total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
    shards = sorted(set(wm.values()))
    expected_names = required_text_tensor_names(cfg)
    missing_text = sorted(expected_names - set(wm))
    unexpected_text_dialect = sorted(
        name for name in wm if name.startswith("model.language_model.")
    )
    missing_assets = [name for name in REQUIRED_SIDECARS if not (root / name).is_file()]

    package_errors: list[str] = []
    if total_size != CHECKPOINT_TOTAL_SIZE:
        package_errors.append(
            f"index metadata.total_size: expected {CHECKPOINT_TOTAL_SIZE}, got {total_size!r}"
        )
    if len(shards) != CHECKPOINT_SHARDS:
        package_errors.append(
            f"index shard count: expected {CHECKPOINT_SHARDS}, got {len(shards)}"
        )
    if missing_text:
        package_errors.append(f"{len(missing_text)} required text tensor names are missing")
    if unexpected_text_dialect:
        package_errors.append("checkpoint mixes provider-raw and normalized text tensor names")

    return {
        "schema": "waste.inkling-release-inspection.v1",
        "model_id": MODEL_ID,
        "release_upload_commit": RELEASE_UPLOAD_COMMIT,
        "source": str(root),
        "config_sha256": sha256(root / "config.json"),
        "index_sha256": sha256(root / "model.safetensors.index.json"),
        "profile": {
            "match": not profile_errors,
            "errors": profile_errors,
        },
        "package": {
            "match": not package_errors,
            "errors": package_errors,
            "total_size": total_size,
            "shards": shards,
            "shard_count": len(shards),
            "weight_map_count": len(wm),
            "required_text_tensor_count": len(expected_names),
            "missing_text_tensors": missing_text,
        },
        "assets": {
            "required": list(REQUIRED_SIDECARS),
            "missing": missing_assets,
            "complete": not missing_assets,
        },
        "official_small": not profile_errors and not package_errors and not missing_assets,
    }


def require_official_small(src: Path | str, *, require_assets: bool = True) -> dict[str, Any]:
    report = inspect_release(src)
    errors = list(report["profile"]["errors"]) + list(report["package"]["errors"])
    if require_assets and report["assets"]["missing"]:
        errors.append(f"missing official sidecars: {', '.join(report['assets']['missing'])}")
    if errors:
        raise ReleaseError("not the official Inkling-Small release: " + "; ".join(errors))
    return report
