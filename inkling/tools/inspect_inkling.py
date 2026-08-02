#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Inspect an official Inkling checkpoint without loading tensor payloads.

This is deliberately a source-checkpoint probe, not a converter.  It reads
config/processor/tokenizer sidecars, the safetensors index, and (optionally)
small safetensors headers.  Every reported architecture field carries its
provenance, and missing data is emitted as an explicit blocker or deferred
multimodal item.

Usage:
    python3 tools/inspect_inkling.py --src /path/to/checkpoint
    python3 tools/inspect_inkling.py --src /path/to/checkpoint --json report.json
    python3 tools/inspect_inkling.py --src /path/to/checkpoint --strict text

Exit status is non-zero only for malformed input or when --strict requests a
capability whose source assets are incomplete.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from inkling_source import config_identifies_inkling
from inkling_release import ReleaseError, inspect_release

PROV_CONFIG = "confirmed from config"
PROV_WEIGHTS = "inferred from weights"
PROV_CODE = "inferred from code"
PROV_UNKNOWN = "unknown"

class InspectError(RuntimeError):
    pass


def load_json(path: Path, required: bool = False) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
    except FileNotFoundError:
        if required:
            raise InspectError(f"missing required file: {path}")
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise InspectError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InspectError(f"expected a JSON object in {path}")
    return value


def field(value: Any, provenance: str, source: str, note: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "value": value,
        "provenance": provenance,
        "source": source,
    }
    if note:
        out["note"] = note
    return out


def unknown(source: str, note: str) -> dict[str, Any]:
    return field(None, PROV_UNKNOWN, source, note)


def first(d: dict[str, Any], *names: str) -> tuple[Any, str | None]:
    for name in names:
        if name in d and d[name] is not None:
            return d[name], name
    return None, None


def list_ints(value: Any) -> list[int] | None:
    if not isinstance(value, list) or any(not isinstance(v, int) for v in value):
        return None
    return value


def tokenizer_added_tokens(*configs: dict[str, Any] | None) -> dict[str, int]:
    """Return exact added-token string -> id mappings from official sidecars.

    Token IDs are intentionally resolved from serialized tokenizer metadata,
    never from token spelling conventions or known Inkling defaults.
    """
    out: dict[str, int] = {}
    for config in configs:
        if not isinstance(config, dict):
            continue
        decoder = config.get("added_tokens_decoder")
        if isinstance(decoder, dict):
            for raw_id, item in decoder.items():
                try:
                    token_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if isinstance(item, dict) and isinstance(item.get("content"), str):
                    out[item["content"]] = token_id
        added = config.get("added_tokens")
        if isinstance(added, list):
            for item in added:
                if not isinstance(item, dict) or not isinstance(item.get("content"), str):
                    continue
                token_id = item.get("id")
                if isinstance(token_id, int):
                    out[item["content"]] = token_id
    return out


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read only a safetensors header.  Tensor bytes are never touched."""
    try:
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise InspectError(f"short safetensors header length: {path}")
            (header_len,) = struct.unpack("<Q", raw)
            # A corrupt length must not turn a metadata probe into an OOM.
            if header_len < 2 or header_len > (256 << 20):
                raise InspectError(f"implausible safetensors header length {header_len}: {path}")
            payload = f.read(header_len)
            if len(payload) != header_len:
                raise InspectError(f"truncated safetensors header: {path}")
    except OSError as exc:
        raise InspectError(f"cannot read safetensors header {path}: {exc}") from exc
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise InspectError(f"invalid safetensors header JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InspectError(f"safetensors header is not an object: {path}")
    return value


@dataclass
class TensorMeta:
    name: str
    shard: str
    dtype: str | None = None
    shape: list[int] | None = None


class TensorCatalog:
    """Index plus lazy shard-header metadata."""

    def __init__(self, root: Path):
        self.root = root
        self.weight_map: dict[str, str] = {}
        self._headers: dict[str, dict[str, Any]] = {}
        self.index_path = root / "model.safetensors.index.json"
        index = load_json(self.index_path)
        if index is not None:
            wm = index.get("weight_map")
            if not isinstance(wm, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in wm.items()):
                raise InspectError(f"invalid weight_map in {self.index_path}")
            self.weight_map = dict(wm)
        else:
            # Single-file and small synthetic checkpoints remain inspectable.
            shards = sorted(root.glob("*.safetensors"))
            for shard in shards:
                hdr = read_safetensors_header(shard)
                self._headers[shard.name] = hdr
                for name in hdr:
                    if name != "__metadata__":
                        self.weight_map[name] = shard.name

    def names(self) -> Iterable[str]:
        return self.weight_map.keys()

    def have(self, name: str) -> bool:
        return name in self.weight_map

    def matching(self, pattern: str) -> list[str]:
        rx = re.compile(pattern)
        return sorted(name for name in self.weight_map if rx.search(name))

    def meta(self, name: str) -> TensorMeta | None:
        shard = self.weight_map.get(name)
        if shard is None:
            return None
        if shard not in self._headers:
            path = self.root / shard
            if not path.is_file():
                return TensorMeta(name=name, shard=shard)
            self._headers[shard] = read_safetensors_header(path)
        item = self._headers[shard].get(name)
        if not isinstance(item, dict):
            return TensorMeta(name=name, shard=shard)
        shape = item.get("shape")
        return TensorMeta(
            name=name,
            shard=shard,
            dtype=item.get("dtype") if isinstance(item.get("dtype"), str) else None,
            shape=shape if isinstance(shape, list) and all(isinstance(x, int) for x in shape) else None,
        )

    def payload_status(self, required_names: Iterable[str]) -> dict[str, Any]:
        names = sorted(set(required_names))
        missing_index_entries = [name for name in names if name not in self.weight_map]
        shards = sorted({self.weight_map[name] for name in names if name in self.weight_map})
        missing_shards = [name for name in shards if not (self.root / name).is_file()]
        missing_headers: list[str] = []
        dtype_counts: dict[str, int] = {}
        for name in names:
            meta = self.meta(name)
            if meta is None:
                continue
            if meta.shape is None or meta.dtype is None:
                missing_headers.append(name)
                continue
            dtype_counts[meta.dtype] = dtype_counts.get(meta.dtype, 0) + 1
        return {
            "required_tensor_count": len(names),
            "required_shards": shards,
            "missing_index_entries": missing_index_entries,
            "missing_shards": missing_shards,
            "missing_header_metadata": missing_headers,
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "complete": not missing_index_entries and not missing_shards and not missing_headers,
        }


class Report:
    def __init__(self, src: Path):
        self.data: dict[str, Any] = {
            "schema": "waste.inkling-source-inspection.v1",
            "source": str(src),
            "architecture": {},
            "fields": {},
            "tensor_layout": {},
            "assets": {},
            "checks": [],
            "blocking": {"text": [], "vision": [], "audio": []},
            "deferred": [],
        }

    def set_field(self, name: str, value: dict[str, Any]) -> None:
        self.data["fields"][name] = value

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.data["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    def block(self, capability: str, message: str) -> None:
        bucket = self.data["blocking"][capability]
        if message not in bucket:
            bucket.append(message)

    def defer(self, message: str) -> None:
        if message not in self.data["deferred"]:
            self.data["deferred"].append(message)


def detect_dialect(names: set[str]) -> str:
    if "model.llm.embed.weight" in names or any(n.startswith("model.llm.layers.") for n in names):
        return "provider_raw"
    if "model.language_model.embed_tokens.weight" in names or any(
        n.startswith("model.language_model.layers.") for n in names
    ):
        return "transformers_normalized"
    return "unknown"


def representative_name(dialect: str, raw: str, hf: str) -> str:
    return raw if dialect == "provider_raw" else hf


def layer_prefix(dialect: str, layer: int) -> str:
    if dialect == "provider_raw":
        return f"model.llm.layers.{layer}"
    return f"model.language_model.layers.{layer}"


def add_cfg_field(report: Report, text: dict[str, Any], output: str, *keys: str, note: str | None = None) -> Any:
    value, key = first(text, *keys)
    if key is None:
        report.set_field(output, unknown("config.json:text_config", f"none of {', '.join(keys)} is present"))
        return None
    report.set_field(output, field(value, PROV_CONFIG, f"config.json:text_config.{key}", note))
    return value


def inspect(src: Path) -> Report:
    report = Report(src)
    cfg = load_json(src / "config.json", required=True)
    assert cfg is not None
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    if not isinstance(text, dict):
        raise InspectError("config.json text_config is not an object")

    archs = cfg.get("architectures")
    architectures = [x for x in archs if isinstance(x, str)] if isinstance(archs, list) else []
    model_type = cfg.get("model_type") if isinstance(cfg.get("model_type"), str) else text.get("model_type")
    is_inkling = config_identifies_inkling(cfg)
    report.data["architecture"] = {
        "is_inkling": is_inkling,
        "architectures": field(architectures, PROV_CONFIG, "config.json.architectures"),
        "model_type": field(model_type, PROV_CONFIG, "config.json.model_type"),
    }
    try:
        release = inspect_release(src)
    except ReleaseError as exc:
        release = {
            "schema": "waste.inkling-release-inspection.v1",
            "official_small": False,
            "error": str(exc),
        }
    report.data["architecture"]["release"] = release
    report.data["architecture"]["profile"] = (
        "official-inkling-small"
        if release.get("profile", {}).get("match")
        else "generic-inkling"
    )
    if not is_inkling:
        report.block("text", "config does not identify an Inkling architecture")

    n_layers = add_cfg_field(report, text, "num_hidden_layers", "num_hidden_layers")
    hidden = add_cfg_field(report, text, "hidden_size", "hidden_size")
    add_cfg_field(report, text, "model_max_length", "model_max_length", "max_position_embeddings")
    add_cfg_field(report, text, "vocab_size", "vocab_size")
    add_cfg_field(report, text, "unpadded_vocab_size", "unpadded_vocab_size")
    add_cfg_field(report, text, "rms_norm_eps", "rms_norm_eps")
    add_cfg_field(report, text, "dense_intermediate_size", "dense_intermediate_size")
    # Provider config calls routed width `intermediate_size`; normalized HF config
    # calls it `moe_intermediate_size`.
    routed_inter, routed_key = first(text, "moe_intermediate_size", "intermediate_size")
    if routed_key:
        report.set_field(
            "routed_intermediate_size",
            field(routed_inter, PROV_CONFIG, f"config.json:text_config.{routed_key}"),
        )
    else:
        report.set_field("routed_intermediate_size", unknown("config.json:text_config", "missing routed width"))
    n_experts = add_cfg_field(report, text, "num_routed_experts", "n_routed_experts", "num_experts")
    top_k = add_cfg_field(
        report, text, "num_experts_per_token", "num_experts_per_tok", "num_experts_per_token"
    )
    n_shared = add_cfg_field(report, text, "num_shared_experts", "n_shared_experts", "num_shared_experts")
    dense_idx = add_cfg_field(report, text, "dense_mlp_idx", "dense_mlp_idx", "first_k_dense_replace")
    route_scale = add_cfg_field(report, text, "route_scale", "route_scale", "routed_scaling_factor")
    add_cfg_field(report, text, "gate_activation", "gate_activation")
    add_cfg_field(report, text, "norm_after_topk", "norm_after_topk", "moe_renormalize")
    add_cfg_field(report, text, "shared_expert_sink", "shared_expert_sink")
    add_cfg_field(report, text, "use_global_scale", "use_global_scale")
    add_cfg_field(report, text, "logits_mup_width_multiplier", "logits_mup_width_multiplier")

    local_ids = list_ints(text.get("local_layer_ids"))
    if local_ids is None:
        report.set_field("local_layer_ids", unknown("config.json:text_config.local_layer_ids", "missing or malformed"))
        report.set_field("global_layer_ids", unknown("derived from local_layer_ids", "cannot derive cadence"))
    else:
        report.set_field("local_layer_ids", field(local_ids, PROV_CONFIG, "config.json:text_config.local_layer_ids"))
        if isinstance(n_layers, int) and n_layers >= 0:
            local_set = set(local_ids)
            globals_ = [i for i in range(n_layers) if i not in local_set]
            report.set_field(
                "global_layer_ids",
                field(globals_, PROV_CONFIG, "derived as complement of config local_layer_ids"),
            )
        else:
            report.set_field("global_layer_ids", unknown("derived from local_layer_ids", "num_hidden_layers missing"))
    add_cfg_field(report, text, "sliding_window_size", "sliding_window_size", "sliding_window")
    add_cfg_field(report, text, "global_num_attention_heads", "num_attention_heads")
    add_cfg_field(report, text, "global_num_key_value_heads", "num_key_value_heads")
    add_cfg_field(report, text, "global_head_dim", "head_dim")
    add_cfg_field(report, text, "local_num_attention_heads", "swa_num_attention_heads")
    add_cfg_field(report, text, "local_num_key_value_heads", "swa_num_key_value_heads")
    add_cfg_field(report, text, "local_head_dim", "swa_head_dim")
    d_rel = add_cfg_field(report, text, "relative_state_dim", "d_rel")
    rel_extent = add_cfg_field(report, text, "relative_extent", "rel_extent")
    add_cfg_field(report, text, "short_conv_kernel_size", "sconv_kernel_size", "conv_kernel_size")
    add_cfg_field(report, text, "log_scaling_n_floor", "log_scaling_n_floor")
    add_cfg_field(report, text, "log_scaling_alpha", "log_scaling_alpha")

    catalog = TensorCatalog(src)
    names = set(catalog.names())
    dialect = detect_dialect(names)
    report.data["tensor_layout"]["dialect"] = field(
        dialect,
        PROV_WEIGHTS if dialect != "unknown" else PROV_UNKNOWN,
        "model.safetensors.index.json:weight_map",
    )
    report.data["tensor_layout"]["tensor_count"] = len(names)
    report.data["assets"]["safetensors_index"] = {
        "path": str(catalog.index_path.relative_to(src)) if catalog.index_path.exists() else None,
        "present": catalog.index_path.exists(),
        "tensor_count": len(names),
    }
    if not names:
        report.block("text", "no safetensors index or tensor headers were found")
    if dialect == "unknown":
        report.block("text", "unknown Inkling tensor naming dialect")

    core_pairs = [
        ("embed", "model.llm.embed.weight", "model.language_model.embed_tokens.weight"),
        ("embed_norm", "model.llm.embed_norm.weight", "model.language_model.embed_norm.weight"),
        ("final_norm", "model.llm.norm.weight", "model.language_model.norm.weight"),
        ("unembed", "model.llm.unembed.weight", "lm_head.weight"),
    ]
    core: dict[str, Any] = {}
    required_text_names: set[str] = set()
    if dialect != "unknown":
        for label, raw, hf in core_pairs:
            name = representative_name(dialect, raw, hf)
            required_text_names.add(name)
            meta = catalog.meta(name)
            core[label] = {
                "name": name,
                "present": meta is not None,
                "shape": meta.shape if meta else None,
                "dtype": meta.dtype if meta else None,
                "shard": meta.shard if meta else None,
            }
            report.check(f"tensor:{label}", meta is not None, name)
            if meta is None:
                report.block("text", f"missing required tensor {name}")
    report.data["tensor_layout"]["core"] = core

    # Verify the exact layer grammar from names.  The converter must not infer
    # this solely from an architecture label because provider and HF-normalized
    # checkpoints intentionally use different layouts.
    layer_summary: dict[str, Any] = {"checked": 0, "missing": []}
    if dialect != "unknown" and isinstance(n_layers, int) and 0 <= n_layers <= 512:
        dense_count = dense_idx if isinstance(dense_idx, int) and dense_idx >= 0 else 0
        for layer in range(n_layers):
            p = layer_prefix(dialect, layer)
            if dialect == "provider_raw":
                common = [
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
                ]
                if layer < dense_count:
                    mlp = [
                        f"{p}.mlp.w13_dn.weight",
                        f"{p}.mlp.w2_md.weight",
                        f"{p}.mlp.global_scale",
                    ]
                else:
                    mlp = [
                        f"{p}.mlp.experts.w13_weight",
                        f"{p}.mlp.experts.w2_weight",
                        f"{p}.mlp.gate.weight",
                        f"{p}.mlp.gate.bias",
                        f"{p}.mlp.gate.global_scale",
                        f"{p}.mlp.shared_experts.shared_w13_weight",
                        f"{p}.mlp.shared_experts.shared_w2_weight",
                    ]
            else:
                common = [
                    f"{p}.input_layernorm.weight",
                    f"{p}.post_attention_layernorm.weight",
                    f"{p}.self_attn.q_proj.weight",
                    f"{p}.self_attn.k_proj.weight",
                    f"{p}.self_attn.v_proj.weight",
                    f"{p}.self_attn.r_proj.weight",
                    f"{p}.self_attn.o_proj.weight",
                    f"{p}.self_attn.q_norm.weight",
                    f"{p}.self_attn.k_norm.weight",
                    f"{p}.self_attn.rel_logits_proj.proj",
                    f"{p}.self_attn.k_sconv.conv1d.weight",
                    f"{p}.self_attn.v_sconv.conv1d.weight",
                    f"{p}.attn_sconv.conv1d.weight",
                    f"{p}.mlp_sconv.conv1d.weight",
                ]
                if layer < dense_count:
                    mlp = [
                        f"{p}.mlp.gate_proj.weight",
                        f"{p}.mlp.up_proj.weight",
                        f"{p}.mlp.down_proj.weight",
                        f"{p}.mlp.global_scale",
                    ]
                else:
                    mlp = [
                        f"{p}.mlp.experts.gate_up_proj",
                        f"{p}.mlp.experts.down_proj",
                        f"{p}.mlp.gate.weight",
                        f"{p}.mlp.gate.e_score_correction_bias",
                        f"{p}.mlp.gate.global_scale",
                        f"{p}.mlp.shared_experts.gate_proj",
                        f"{p}.mlp.shared_experts.up_proj",
                        f"{p}.mlp.shared_experts.down_proj",
                    ]
            required_text_names.update(common + mlp)
            missing = [name for name in common + mlp if name not in names]
            layer_summary["checked"] += 1
            if missing:
                layer_summary["missing"].append({"layer": layer, "tensors": missing})
        if layer_summary["missing"]:
            report.block("text", f"{len(layer_summary['missing'])} layers have missing required tensor names")
    report.data["tensor_layout"]["layers"] = layer_summary

    # Inventory completeness and payload availability are different gates.
    # The official index is enough to audit names before a 500+ GB download,
    # but conversion must not begin until every referenced base-text shard and
    # every required tensor header is present locally.
    metadata_text_ready = not report.data["blocking"]["text"]
    payload = catalog.payload_status(required_text_names)
    report.data["assets"]["text_payloads"] = payload
    if metadata_text_ready and payload["missing_shards"]:
        report.block(
            "text",
            f"{len(payload['missing_shards'])} required base-text safetensors shards are not present locally",
        )
    elif metadata_text_ready and payload["missing_header_metadata"]:
        report.block(
            "text",
            f"{len(payload['missing_header_metadata'])} required base-text tensors lack readable dtype/shape metadata",
        )

    mtp_names = sorted(name for name in names if name.startswith("model.mtp."))
    report.data["tensor_layout"]["mtp"] = {
        "present": bool(mtp_names),
        "tensor_count": len(mtp_names),
        "action": "exclude from baseline conversion",
    }
    if mtp_names:
        report.defer("MTP tensors are present: exclude them from text-parity conversion until base-model logits match")

    quant_companions = sorted(
        name for name in names
        if name.endswith((".original_shape", ".scale", ".scale2", ".input_amax"))
    )
    quant_cfg = load_json(src / "quantization_config.json") or load_json(src / "hf_quant_config.json")
    report.data["assets"]["source_quantization"] = {
        "config_present": quant_cfg is not None,
        "companion_tensor_count": len(quant_companions),
        "kind": "packed/quantized" if quant_cfg is not None or quant_companions else "native-or-unknown",
    }
    if quant_cfg is not None or quant_companions:
        report.defer("quantized source detected: establish BF16 parity first; add NVFP4/MXFP8 readers only as separate adapters")

    has_rel_weights = any(name.endswith("attn.rel_logits_proj.proj") or name.endswith("self_attn.rel_logits_proj.proj") for name in names)
    report.set_field(
        "positional_encoding",
        field(
            "hidden-state-conditioned relative logits" if has_rel_weights else None,
            PROV_WEIGHTS if has_rel_weights else PROV_UNKNOWN,
            "relative projection tensor names plus d_rel/rel_extent",
            "the exact scoring equation must remain sourced from the official implementation",
        ),
    )
    if d_rel is None or rel_extent is None or not has_rel_weights:
        report.block("text", "relative-position metadata or projection tensors are incomplete")

    # Sidecars: no prompt-format guessing.  Raw tokenization remains valid when
    # the official template is absent.
    tokenizer_candidates = [
        src / "tokenizer.model",
        src / "tiktoken.model",
        src / "tiktoken" / "tokenizer.model",
    ]
    tok = next((p for p in tokenizer_candidates if p.is_file()), None)
    template = src / "chat_template.jinja"
    tok_cfg = load_json(src / "tokenizer_config.json")
    tokenizer_json = load_json(src / "tokenizer.json")
    processor_cfg = load_json(src / "processor_config.json")
    token_ids = tokenizer_added_tokens(tok_cfg, tokenizer_json)
    report.data["assets"]["tokenizer"] = {
        "present": tok is not None,
        "path": str(tok.relative_to(src)) if tok else None,
    }
    report.data["assets"]["chat_template"] = {
        "present": template.is_file() or bool(tok_cfg and tok_cfg.get("chat_template")),
        "path": "chat_template.jinja" if template.is_file() else None,
        "source": "tokenizer_config.json.chat_template" if tok_cfg and tok_cfg.get("chat_template") else None,
    }
    if tok is None:
        report.block("text", "official tokenizer asset was not found (including tiktoken/tokenizer.model)")
    if not report.data["assets"]["chat_template"]["present"]:
        report.defer("official chat template absent: preserve raw-token/raw-text mode and do not guess a prompt format")

    eos = cfg.get("eos_token_id")
    report.set_field(
        "eos_token_id",
        field(eos, PROV_CONFIG, "config.json.eos_token_id") if isinstance(eos, int) else unknown("config.json.eos_token_id", "missing"),
    )
    token_fields = {
        "image_token_id": ("image_token_id", "image_token"),
        "audio_token_id": ("audio_token_id", "audio_token"),
        "image_bos_token_id": ("image_bos_token_id", "image_bos_token"),
        "audio_bos_token_id": ("audio_bos_token_id", "audio_bos_token"),
    }
    for out_name, (cfg_name, proc_name) in token_fields.items():
        if isinstance(cfg.get(cfg_name), int):
            report.set_field(out_name, field(cfg[cfg_name], PROV_CONFIG, f"config.json.{cfg_name}"))
        elif processor_cfg and isinstance(processor_cfg.get(proc_name), str):
            token = processor_cfg[proc_name]
            token_id = token_ids.get(token)
            if token_id is None:
                report.set_field(
                    out_name,
                    unknown(
                        f"processor_config.json.{proc_name} + tokenizer metadata",
                        f"official token {token!r} has no serialized ID mapping",
                    ),
                )
                capability = "vision" if out_name.startswith("image_") else "audio"
                report.block(capability, f"cannot resolve official {proc_name} to a tokenizer ID")
            else:
                report.set_field(
                    out_name,
                    field(
                        token_id,
                        PROV_CONFIG,
                        f"processor_config.json.{proc_name} + tokenizer_config.json.added_tokens_decoder",
                        f"token={token}",
                    ),
                )
        else:
            report.set_field(out_name, unknown("config/processor sidecars", "missing"))

    vision = cfg.get("vision_config") if isinstance(cfg.get("vision_config"), dict) else None
    report.data["assets"]["vision_config"] = bool(vision)
    if vision:
        for out_name, keys in {
            "vision_encoder_type": ("vision_encoder_type",),
            "vision_patch_size": ("patch_size",),
            "vision_temporal_patch_size": ("temporal_patch_size",),
            "vision_channels": ("n_channels", "num_channels"),
            "vision_layers": ("n_layers", "num_hidden_layers"),
            "vision_text_hidden_size": ("decoder_dmodel", "text_hidden_size"),
        }.items():
            value, key = first(vision, *keys)
            report.set_field(
                out_name,
                field(value, PROV_CONFIG, f"config.json:vision_config.{key}") if key else unknown("config.json:vision_config", "missing"),
            )
        visual_names = [n for n in names if n.startswith("model.visual.") or ".vision_tower." in n]
        vision_layers, _ = first(vision, "n_layers", "num_hidden_layers")
        expected_visual: list[str] = []
        if dialect == "provider_raw" and isinstance(vision_layers, int):
            expected_visual = ["model.visual.final_norm.weight"]
            expected_visual += [f"model.visual.layers.linear_{i}.weight" for i in range(vision_layers)]
            expected_visual += [f"model.visual.layers.norm_{i}.weight" for i in range(max(0, vision_layers - 1))]
        elif dialect == "transformers_normalized" and isinstance(vision_layers, int):
            expected_visual = ["model.vision_tower.final_norm.weight"]
            expected_visual += [f"model.vision_tower.encoder_layers.{i}.projection.weight" for i in range(vision_layers)]
            expected_visual += [f"model.vision_tower.encoder_layers.{i}.layer_norm.weight" for i in range(max(0, vision_layers - 1))]
        missing_visual = [name for name in expected_visual if name not in names]
        report.data["tensor_layout"]["vision"] = {
            "tensor_count": len(visual_names),
            "expected": expected_visual,
            "missing": missing_visual,
        }
        if not visual_names:
            report.block("vision", "vision config exists but no vision tower tensors were found")
        elif missing_visual:
            report.block("vision", f"vision tower is missing {len(missing_visual)} required tensor names")
        if not processor_cfg or not isinstance(processor_cfg.get("image_processor"), dict):
            report.block("vision", "processor_config.json image preprocessing metadata is missing")
    else:
        report.block("vision", "vision_config is absent")

    audio = cfg.get("audio_config") if isinstance(cfg.get("audio_config"), dict) else None
    report.data["assets"]["audio_config"] = bool(audio)
    if audio:
        for out_name, keys in {
            "audio_num_mel_bins": ("n_mel_bins",),
            "audio_mel_vocab_size": ("mel_vocab_size",),
            "audio_text_hidden_size": ("decoder_dmodel", "text_hidden_size"),
            "audio_dmel_min": ("dmel_min_value",),
            "audio_dmel_max": ("dmel_max_value",),
            "audio_mode": ("audio_mode",),
        }.items():
            value, key = first(audio, *keys)
            report.set_field(
                out_name,
                field(value, PROV_CONFIG, f"config.json:audio_config.{key}") if key else unknown("config.json:audio_config", "missing"),
            )
        audio_names = [n for n in names if n.startswith("model.audio.") or ".audio_tower." in n]
        expected_audio = (
            ["model.audio.encoder.weight", "model.audio.final_norm.weight"]
            if dialect == "provider_raw"
            else [
                "model.audio_tower.embed_audio_tokens.embed_audio_tokens.weight",
                "model.audio_tower.norm.weight",
            ]
            if dialect == "transformers_normalized"
            else []
        )
        missing_audio = [name for name in expected_audio if name not in names]
        report.data["tensor_layout"]["audio"] = {
            "tensor_count": len(audio_names),
            "expected": expected_audio,
            "missing": missing_audio,
        }
        if not audio_names:
            report.block("audio", "audio config exists but no audio embedding tensors were found")
        elif missing_audio:
            report.block("audio", f"audio tower is missing {len(missing_audio)} required tensor names")
        feature = processor_cfg.get("feature_extractor") if processor_cfg and isinstance(processor_cfg.get("feature_extractor"), dict) else None
        if not feature:
            report.block("audio", "processor_config.json audio feature-extractor metadata is missing")
    else:
        report.block("audio", "audio_config is absent")

    for name, value in {
        "num_hidden_layers": n_layers,
        "hidden_size": hidden,
        "num_routed_experts": n_experts,
        "num_experts_per_token": top_k,
        "num_shared_experts": n_shared,
        "dense_mlp_idx": dense_idx,
        "routed_intermediate_size": routed_inter,
        "route_scale": route_scale,
    }.items():
        if value is None:
            report.block("text", f"required text architecture field is unknown: {name}")

    report.data["ready"] = {
        "text_metadata": metadata_text_ready,
        "text_payloads": bool(metadata_text_ready and payload["complete"]),
        "text_source": not report.data["blocking"]["text"],
        "vision_source": not report.data["blocking"]["vision"],
        "audio_source": not report.data["blocking"]["audio"],
        # This tool intentionally cannot claim runtime parity.
        "waste_runtime": False,
    }
    return report


def human(report: Report) -> str:
    d = report.data
    lines = [
        f"source: {d['source']}",
        f"inkling: {'yes' if d['architecture']['is_inkling'] else 'no'}",
        f"tensor dialect: {d['tensor_layout'].get('dialect', {}).get('value')}",
        f"text metadata ready: {'yes' if d['ready']['text_metadata'] else 'no'}",
        f"text payloads ready: {'yes' if d['ready']['text_payloads'] else 'no'}",
        f"text source ready: {'yes' if d['ready']['text_source'] else 'no'}",
        f"vision source ready: {'yes' if d['ready']['vision_source'] else 'no'}",
        f"audio source ready: {'yes' if d['ready']['audio_source'] else 'no'}",
    ]
    for capability in ("text", "vision", "audio"):
        blockers = d["blocking"][capability]
        if blockers:
            lines.append(f"{capability} blockers:")
            lines.extend(f"  - {x}" for x in blockers)
    if d["deferred"]:
        lines.append("deferred:")
        lines.extend(f"  - {x}" for x in d["deferred"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="official checkpoint directory")
    ap.add_argument("--json", type=Path, help="write the complete machine-readable report")
    ap.add_argument(
        "--strict",
        choices=("none", "text", "vision", "audio", "all"),
        default="none",
        help="return 2 if the requested source capability has blockers",
    )
    args = ap.parse_args(argv)
    try:
        report = inspect(args.src)
    except InspectError as exc:
        print(f"inspect_inkling: {exc}", file=sys.stderr)
        return 1

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json.with_name(args.json.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(report.data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, args.json)
    print(human(report))

    requested = ("text", "vision", "audio") if args.strict == "all" else (args.strict,)
    if args.strict != "none" and any(report.data["blocking"][cap] for cap in requested):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
