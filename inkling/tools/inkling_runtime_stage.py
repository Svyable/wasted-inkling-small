#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Publish the converter-private Inkling runtime index.

The index binds completed trunk and BF16 expert staging outputs into one fixed,
little-endian metadata file that the dependency-free C parity runtime can open.
It is *not* a WASTE manifest and cannot be opened by the public runtime.

`runtime-stage.bin` is published last, after source hashes, tensor identities,
shapes, bank coverage, and optional payload CRCs have been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

from inkling_plan import PlanError, build_plan, write_json_atomic
from inkling_release import ReleaseError, inspect_release, require_official_small
from inkling_stage import StageError, verify_bank
from inkling_trunk import TrunkStageError, verify_tensor
from inkling_qtrunk import QTrunkError, verify_qtensor
from inkling_vq import VQError, VQSpec, verify_layer_outputs


class RuntimeStageError(RuntimeError):
    pass


MAGIC = 0x54524B49  # IKRT
VERSION = 1  # legacy BF16 private-stage index
VERSION_VQ = 2
VERSION_QTRUNK_VQ = 3
BANK_FORMAT_BF16 = 0
BANK_FORMAT_VQ = 1
HEADER_BYTES = 256
LAYER_ENTRY_BYTES = 32
TENSOR_ENTRY_BYTES = 256
BANK_ENTRY_BYTES = 128
FLAG_OFFICIAL_SMALL = 1 << 0
TENSOR_FLAG_ROW_BACKED = 1 << 0
TENSOR_FLAG_QUANTIZED = 1 << 1

DTYPE_CODE = {"BF16": 1, "F16": 2, "F32": 3, "Q8": 4, "Q4": 5}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeStageError(f"missing stage metadata: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeStageError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeStageError(f"expected JSON object: {path}")
    return value


def _hex32(value: str, name: str) -> bytes:
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise RuntimeStageError(f"invalid {name} SHA-256") from exc
    if len(raw) != 32:
        raise RuntimeStageError(f"invalid {name} SHA-256 length")
    return raw


def _fixed(text: str, size: int, name: str) -> bytes:
    raw = text.encode("utf-8")
    if not raw or len(raw) >= size or b"\0" in raw:
        raise RuntimeStageError(f"{name} does not fit fixed field: {text!r}")
    return raw + b"\0" * (size - len(raw))


def _safe_relative(path: str) -> str:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise RuntimeStageError(f"unsafe staged relative path: {path!r}")
    return p.as_posix()


def _config_values(plan: dict[str, Any]) -> dict[str, Any]:
    cfg = plan["manifest_preview"]["config"]
    text = cfg["text_config"]
    local = set(text["local_layer_ids"])
    layers = []
    for i in range(text["num_hidden_layers"]):
        is_local = i in local
        layers.append(
            {
                "layer": i,
                "is_local": is_local,
                "num_heads": text["swa_num_attention_heads"] if is_local else text["num_attention_heads"],
                "num_kv_heads": text["swa_num_key_value_heads"] if is_local else text["num_key_value_heads"],
                "head_dim": text["swa_head_dim"] if is_local else text["head_dim"],
                "relative_extent": text["sliding_window_size"] if is_local else text["rel_extent"],
                "sparse": i >= text["dense_mlp_idx"],
            }
        )
    return {
        "n_layers": text["num_hidden_layers"],
        "hidden": text["hidden_size"],
        "vocab": text["vocab_size"],
        "unpadded_vocab": text.get("unpadded_vocab_size", text["vocab_size"]),
        "max_context": text["model_max_length"],
        "global_heads": text["num_attention_heads"],
        "global_kv_heads": text["num_key_value_heads"],
        "global_head_dim": text["head_dim"],
        "local_heads": text["swa_num_attention_heads"],
        "local_kv_heads": text["swa_num_key_value_heads"],
        "local_head_dim": text["swa_head_dim"],
        "sliding_window": text["sliding_window_size"],
        "d_rel": text["d_rel"],
        "rel_extent": text["rel_extent"],
        "conv_kernel": text["sconv_kernel_size"],
        "dense_layers": text["dense_mlp_idx"],
        "dense_intermediate": text["dense_intermediate_size"],
        "moe_intermediate": text["intermediate_size"],
        "n_routed_experts": text["n_routed_experts"],
        "top_k": text["num_experts_per_tok"],
        "n_shared_experts": text["n_shared_experts"],
        "rms_eps": text["rms_norm_eps"],
        "route_scale": text["route_scale"],
        "logits_width_multiplier": text["logits_mup_width_multiplier"],
        "log_scaling_n_floor": text["log_scaling_n_floor"],
        "log_scaling_alpha": text["log_scaling_alpha"],
        "layers": layers,
    }


def _expected_targets(plan: dict[str, Any]) -> set[str]:
    targets: set[str] = set()
    for item in plan["trunk"]:
        role = item["role"]
        if role.endswith(".fused_gate_up"):
            base = role[: -len("fused_gate_up")]
            targets.add("inkling." + base + "gate")
            targets.add("inkling." + base + "up")
        else:
            targets.add("inkling." + role)
    return targets


def _validate_stages(
    src: Path,
    stage_dir: Path,
    *,
    verify: bool,
    require_official: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        plan = build_plan(src, require_payloads=False)
        release = require_official_small(src) if require_official else inspect_release(src)
    except (PlanError, ReleaseError) as exc:
        raise RuntimeStageError(str(exc)) from exc
    trunk = _load(stage_dir / "trunk-stage.json")
    experts = _load(stage_dir / "stage.json")

    source = trunk.get("source")
    esource = experts.get("source")
    if not isinstance(source, dict) or not isinstance(esource, dict):
        raise RuntimeStageError("stage metadata lacks source identity")
    for key in ("config_sha256", "index_sha256"):
        want = release[key]
        if source.get(key) != want or esource.get(key) != want:
            raise RuntimeStageError(f"stage source {key} does not match checkpoint")

    tensors = trunk.get("tensors")
    if not isinstance(tensors, list):
        raise RuntimeStageError("trunk-stage.json tensors is not an array")
    by_target: dict[str, dict[str, Any]] = {}
    for item in tensors:
        if not isinstance(item, dict) or not isinstance(item.get("target"), str):
            raise RuntimeStageError("malformed trunk-stage tensor entry")
        target = item["target"]
        if target in by_target:
            raise RuntimeStageError(f"duplicate staged tensor target: {target}")
        file_name = _safe_relative(str(item.get("file", "")))
        path = stage_dir / "trunk-stage" / file_name
        if verify:
            try:
                verify_tensor(path, item, verify_crc=True)
            except (OSError, TrunkStageError) as exc:
                raise RuntimeStageError(str(exc)) from exc
        elif not path.is_file():
            raise RuntimeStageError(f"missing staged tensor: {path}")
        by_target[target] = item
    expected = _expected_targets(plan)
    missing = sorted(expected - set(by_target))
    extra = sorted(set(by_target) - expected)
    if missing or extra:
        raise RuntimeStageError(
            f"staged trunk target set differs from plan: {len(missing)} missing, {len(extra)} extra"
        )

    banks = experts.get("banks")
    if not isinstance(banks, list):
        raise RuntimeStageError("stage.json banks is not an array")
    by_layer: dict[int, dict[str, Any]] = {}
    for item in banks:
        if not isinstance(item, dict) or not isinstance(item.get("layer"), int):
            raise RuntimeStageError("malformed expert bank entry")
        layer = item["layer"]
        if layer in by_layer:
            raise RuntimeStageError(f"duplicate expert bank for layer {layer}")
        if not item.get("complete"):
            raise RuntimeStageError(f"expert bank layer {layer} is not complete")
        file_name = _safe_relative(str(item.get("file", "")))
        path = stage_dir / file_name
        if verify:
            try:
                verify_bank(path, item, verify_crc=True)
            except (OSError, StageError) as exc:
                raise RuntimeStageError(str(exc)) from exc
        elif not path.is_file():
            raise RuntimeStageError(f"missing expert bank: {path}")
        by_layer[layer] = item
    sparse = set(plan["model"]["sparse_mlp_layers"])
    if set(by_layer) != sparse:
        raise RuntimeStageError(
            f"expert bank layer set differs from plan: expected {sorted(sparse)}, got {sorted(by_layer)}"
        )
    return plan, release, trunk, experts



def _validate_vq_stages(
    src: Path,
    stage_dir: Path,
    *,
    verify: bool,
    require_official: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate canonical trunk artifacts plus final WEXP/VQ layer banks."""
    try:
        plan = build_plan(src, require_payloads=False)
        release = require_official_small(src) if require_official else inspect_release(src)
    except (PlanError, ReleaseError) as exc:
        raise RuntimeStageError(str(exc)) from exc
    trunk = _load(stage_dir / "trunk-stage.json")
    experts = _load(stage_dir / "vq-stage.json")
    source = trunk.get("source")
    if not isinstance(source, dict):
        raise RuntimeStageError("trunk stage metadata lacks source identity")
    for key in ("config_sha256", "index_sha256"):
        if source.get(key) != release[key] or experts.get(f"source_{key}") != release[key]:
            raise RuntimeStageError(f"stage source {key} does not match checkpoint")

    tensors = trunk.get("tensors")
    if not isinstance(tensors, list):
        raise RuntimeStageError("trunk-stage.json tensors is not an array")
    by_target: dict[str, dict[str, Any]] = {}
    for item in tensors:
        if not isinstance(item, dict) or not isinstance(item.get("target"), str):
            raise RuntimeStageError("malformed trunk-stage tensor entry")
        target = item["target"]
        if target in by_target:
            raise RuntimeStageError(f"duplicate staged tensor target: {target}")
        file_name = _safe_relative(str(item.get("file", "")))
        path = stage_dir / "trunk-stage" / file_name
        if verify:
            try:
                verify_tensor(path, item, verify_crc=True)
            except (OSError, TrunkStageError) as exc:
                raise RuntimeStageError(str(exc)) from exc
        elif not path.is_file():
            raise RuntimeStageError(f"missing staged tensor: {path}")
        by_target[target] = item
    expected = _expected_targets(plan)
    if set(by_target) != expected:
        missing = sorted(expected - set(by_target))
        extra = sorted(set(by_target) - expected)
        raise RuntimeStageError(
            f"staged trunk target set differs from plan: {len(missing)} missing, {len(extra)} extra"
        )

    layers = experts.get("layers")
    if experts.get("schema") != "waste.inkling-vq-stage.v1" or not isinstance(layers, list):
        raise RuntimeStageError("vq-stage.json has an unsupported schema or layer list")
    by_layer: dict[int, dict[str, Any]] = {}
    quant_identity: dict[str, Any] | None = None
    for item in layers:
        if not isinstance(item, dict) or not isinstance(item.get("layer"), int):
            raise RuntimeStageError("malformed VQ layer entry")
        layer = item["layer"]
        if layer in by_layer or not item.get("complete"):
            raise RuntimeStageError(f"duplicate or incomplete VQ bank for layer {layer}")
        quant = item.get("quant")
        if not isinstance(quant, dict):
            raise RuntimeStageError(f"VQ layer {layer} lacks quantization metadata")
        identity = {k: int(quant.get(k, 0)) for k in ("stages", "vec_dim", "entries", "index_block")}
        try:
            spec = VQSpec(**identity)
            spec.validate()
        except (TypeError, VQError) as exc:
            raise RuntimeStageError(f"invalid VQ metadata for layer {layer}: {exc}") from exc
        if quant_identity is None:
            quant_identity = identity
        elif identity != quant_identity:
            raise RuntimeStageError("all Inkling VQ layer banks must share one geometry")
        bank_name = _safe_relative(str(item.get("file", "")))
        codebook_name = _safe_relative(str(item.get("codebooks_file", "")))
        expected_codebook = f"codebooks-L{layer}.bin"
        if codebook_name != expected_codebook:
            raise RuntimeStageError(
                f"VQ codebook path for layer {layer} must be {expected_codebook!r}"
            )
        bank = stage_dir / bank_name
        codebooks = stage_dir / codebook_name
        if verify:
            try:
                checked = verify_layer_outputs(
                    bank, codebooks, layer=layer, experts=int(item["experts"]),
                    hidden=int(item["hidden_size"]),
                    intermediate=int(item["intermediate_size"]),
                    spec=spec, codebook_base=int(item["codebook_base"]), verify_crc=True,
                )
            except (OSError, KeyError, TypeError, ValueError, VQError) as exc:
                raise RuntimeStageError(str(exc)) from exc
            if checked["record_bytes"] != int(item["record_bytes"]) or checked["bytes"] != int(item["bytes"]):
                raise RuntimeStageError(f"VQ metadata differs from bank layer {layer}")
        elif not bank.is_file() or not codebooks.is_file():
            raise RuntimeStageError(f"missing VQ bank or codebooks for layer {layer}")
        by_layer[layer] = item
    sparse = set(plan["model"]["sparse_mlp_layers"])
    if set(by_layer) != sparse or not experts.get("complete"):
        raise RuntimeStageError(
            f"VQ bank layer set differs from plan: expected {sorted(sparse)}, got {sorted(by_layer)}"
        )
    if quant_identity is None:
        raise RuntimeStageError("VQ stage has no layer geometry")
    return plan, release, trunk, experts, quant_identity

def _header_bytes(
    config: dict[str, Any], release: dict[str, Any], ntensors: int, nbanks: int,
    *, expert_format: str = "bf16", quant: dict[str, Any] | None = None,
    quantized_trunk: bool = False,
) -> bytes:
    if expert_format not in ("bf16", "vq"):
        raise RuntimeStageError(f"unsupported private expert format: {expert_format}")
    version = (VERSION_QTRUNK_VQ if quantized_trunk else
               VERSION if expert_format == "bf16" else VERSION_VQ)
    b = bytearray(HEADER_BYTES)
    struct.pack_into("<IHHI", b, 0, MAGIC, version, HEADER_BYTES,
                     FLAG_OFFICIAL_SMALL if release.get("official_small") else 0)
    struct.pack_into("<III", b, 12, LAYER_ENTRY_BYTES, TENSOR_ENTRY_BYTES, BANK_ENTRY_BYTES)
    struct.pack_into("<III", b, 24, config["n_layers"], ntensors, nbanks)
    ints = [
        config["hidden"], config["vocab"], config["unpadded_vocab"], config["max_context"],
        config["global_heads"], config["global_kv_heads"], config["global_head_dim"],
        config["local_heads"], config["local_kv_heads"], config["local_head_dim"],
        config["sliding_window"], config["d_rel"], config["rel_extent"], config["conv_kernel"],
        config["dense_layers"], config["dense_intermediate"], config["moe_intermediate"],
        config["n_routed_experts"], config["top_k"], config["n_shared_experts"],
    ]
    struct.pack_into("<20I", b, 36, *ints)
    struct.pack_into("<3f", b, 116, config["rms_eps"], config["route_scale"], config["logits_width_multiplier"])
    struct.pack_into("<If", b, 128, config["log_scaling_n_floor"], config["log_scaling_alpha"])
    total_size = release.get("package", {}).get("total_size")
    struct.pack_into("<Q", b, 136, total_size if isinstance(total_size, int) else 0)
    b[144:176] = _hex32(release["config_sha256"], "config")
    b[176:208] = _hex32(release["index_sha256"], "index")
    b[208:240] = hashlib.sha256(str(release.get("model_id", "generic-inkling")).encode()).digest()
    commit = str(release.get("release_upload_commit", ""))[:8].encode("ascii", "strict")
    b[240:248] = commit.ljust(8, b"\0")
    if expert_format == "vq":
        if not isinstance(quant, dict):
            raise RuntimeStageError("VQ runtime index requires quantization metadata")
        stages = int(quant.get("stages", 0))
        vec_dim = int(quant.get("vec_dim", 0))
        entries = int(quant.get("entries", 0))
        index_block = int(quant.get("index_block", 0))
        spec = VQSpec(stages=stages, vec_dim=vec_dim, entries=entries, index_block=index_block)
        try:
            spec.validate()
        except VQError as exc:
            raise RuntimeStageError(str(exc)) from exc
        if index_block > 255:
            raise RuntimeStageError("private VQ runtime index supports index_block <= 255")
        struct.pack_into("<BBBBHH", b, 248, BANK_FORMAT_VQ, stages, vec_dim, index_block, entries, 0)
    return bytes(b)


def _layer_bytes(item: dict[str, Any]) -> bytes:
    return struct.pack(
        "<8I",
        item["layer"], int(item["is_local"]), item["num_heads"],
        item["num_kv_heads"], item["head_dim"], item["relative_extent"],
        int(item["sparse"]), 0,
    )


def _tensor_bytes(stage_dir: Path, item: dict[str, Any]) -> bytes:
    b = bytearray(TENSOR_ENTRY_BYTES)
    target = item["target"]
    path = _safe_relative(f"trunk-stage/{item['file']}")
    b[0:128] = _fixed(target, 128, "canonical tensor name")
    b[128:208] = _fixed(path, 80, "staged tensor path")
    dtype = DTYPE_CODE.get(item.get("dtype"))
    shape = item.get("shape")
    if dtype is None or not isinstance(shape, list) or not 1 <= len(shape) <= 4 or any(
        not isinstance(x, int) or x <= 0 for x in shape
    ):
        raise RuntimeStageError(f"invalid staged tensor dtype/shape: {target}")
    flags = TENSOR_FLAG_ROW_BACKED if target in ("inkling.embed", "inkling.unembed") else 0
    struct.pack_into("<BBH", b, 208, dtype, len(shape), flags)
    padded = shape + [0] * (4 - len(shape))
    struct.pack_into("<4I", b, 212, *padded)
    struct.pack_into("<QQI", b, 228, int(item["payload_bytes"]), int(item["stored_bytes"]), int(item["crc32"]))
    return bytes(b)


def _qtensor_bytes(item: dict[str, Any]) -> bytes:
    b = bytearray(TENSOR_ENTRY_BYTES)
    target = item["target"]
    path = _safe_relative(f"qtrunk-stage/{item['file']}")
    b[0:128] = _fixed(target, 128, "canonical tensor name")
    b[128:208] = _fixed(path, 80, "quantized tensor path")
    bits = int(item.get("bits", 0))
    dtype = DTYPE_CODE.get("Q8" if bits == 8 else "Q4" if bits == 4 else "")
    shape = item.get("shape")
    if dtype is None or not isinstance(shape, list) or not 2 <= len(shape) <= 4 or any(
        not isinstance(x, int) or x <= 0 for x in shape
    ):
        raise RuntimeStageError(f"invalid quantized tensor dtype/shape: {target}")
    flags = TENSOR_FLAG_QUANTIZED
    if target in ("inkling.embed", "inkling.unembed"):
        flags |= TENSOR_FLAG_ROW_BACKED
    struct.pack_into("<BBH", b, 208, dtype, len(shape), flags)
    padded = shape + [0] * (4 - len(shape))
    struct.pack_into("<4I", b, 212, *padded)
    struct.pack_into("<QQI", b, 228, int(item["qbytes"]),
                     int(item["stored_bytes"]), int(item["q_crc32"]))
    return bytes(b)


def _bank_bytes(item: dict[str, Any], *, final_vq: bool = False) -> bytes:
    b = bytearray(BANK_ENTRY_BYTES)
    path = _safe_relative(str(item["file"]))
    b[:80] = _fixed(path, 80, "expert bank path")
    codebook_base = int(item.get("codebook_base", 0)) if final_vq else 0
    struct.pack_into(
        "<4IQQI",
        b,
        80,
        int(item["layer"]), int(item["experts"]), int(item["hidden_size"]),
        int(item["intermediate_size"]), int(item["record_bytes"]), int(item["bytes"]),
        codebook_base,
    )
    return bytes(b)


def publish_runtime_stage(
    src: Path,
    stage_dir: Path,
    *,
    verify: bool = True,
    require_official: bool = True,
) -> dict[str, Any]:
    plan, release, trunk, experts = _validate_stages(
        src, stage_dir, verify=verify, require_official=require_official
    )
    config = _config_values(plan)
    tensors = sorted(trunk["tensors"], key=lambda item: item["target"])
    banks = sorted(experts["banks"], key=lambda item: item["layer"])
    payload = bytearray()
    payload += _header_bytes(config, release, len(tensors), len(banks))
    for item in config["layers"]:
        payload += _layer_bytes(item)
    for item in tensors:
        payload += _tensor_bytes(stage_dir, item)
    for item in banks:
        payload += _bank_bytes(item)

    tmp = stage_dir / "runtime-stage.bin.tmp"
    final = stage_dir / "runtime-stage.bin"
    try:
        with tmp.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise

    meta = {
        "schema": "waste.inkling-private-runtime-stage.v1",
        "arch": "inkling-small" if release.get("official_small") else "inkling",
        "model_id": release.get("model_id"),
        "release_upload_commit": release.get("release_upload_commit"),
        "source": {
            "config_sha256": release["config_sha256"],
            "index_sha256": release["index_sha256"],
            "checkpoint_total_size": release.get("package", {}).get("total_size"),
        },
        "file": final.name,
        "bytes": final.stat().st_size,
        "layout": {
            "header_bytes": HEADER_BYTES,
            "layer_entry_bytes": LAYER_ENTRY_BYTES,
            "tensor_entry_bytes": TENSOR_ENTRY_BYTES,
            "bank_entry_bytes": BANK_ENTRY_BYTES,
        },
        "counts": {
            "layers": len(config["layers"]),
            "tensors": len(tensors),
            "banks": len(banks),
        },
        "status": {
            "private_runtime_stage_complete": True,
            "official_small": bool(release.get("official_small")),
            "waste_manifest_written": False,
            "public_runtime_supported": False,
            "official_weight_logits_parity": False,
        },
        "publication_rule": "runtime-stage.bin is converter-private; manifest.json remains absent",
    }
    write_json_atomic(stage_dir / "runtime-stage.json", meta)
    return meta



def publish_runtime_vq_stage(
    src: Path,
    stage_dir: Path,
    *,
    verify: bool = True,
    require_official: bool = True,
) -> dict[str, Any]:
    plan, release, trunk, experts, quant = _validate_vq_stages(
        src, stage_dir, verify=verify, require_official=require_official
    )
    config = _config_values(plan)
    tensors = sorted(trunk["tensors"], key=lambda item: item["target"])
    banks = sorted(experts["layers"], key=lambda item: item["layer"])
    payload = bytearray(_header_bytes(
        config, release, len(tensors), len(banks), expert_format="vq", quant=quant
    ))
    for item in config["layers"]:
        payload += _layer_bytes(item)
    for item in tensors:
        payload += _tensor_bytes(stage_dir, item)
    for item in banks:
        payload += _bank_bytes(item, final_vq=True)

    tmp = stage_dir / "runtime-stage.bin.tmp"
    final = stage_dir / "runtime-stage.bin"
    try:
        with tmp.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    meta = {
        "schema": "waste.inkling-private-runtime-stage.v2",
        "arch": "inkling-small" if release.get("official_small") else "inkling",
        "model_id": release.get("model_id"),
        "release_upload_commit": release.get("release_upload_commit"),
        "source": {
            "config_sha256": release["config_sha256"],
            "index_sha256": release["index_sha256"],
            "checkpoint_total_size": release.get("package", {}).get("total_size"),
        },
        "file": final.name,
        "bytes": final.stat().st_size,
        "expert_format": "WEXP/VQ",
        "expert_quant": quant,
        "layout": {
            "header_bytes": HEADER_BYTES, "layer_entry_bytes": LAYER_ENTRY_BYTES,
            "tensor_entry_bytes": TENSOR_ENTRY_BYTES, "bank_entry_bytes": BANK_ENTRY_BYTES,
        },
        "counts": {"layers": len(config["layers"]), "tensors": len(tensors), "banks": len(banks)},
        "status": {
            "private_runtime_stage_complete": True,
            "official_small": bool(release.get("official_small")),
            "final_expert_banks": True,
            "bf16_expert_stage_required": False,
            "waste_manifest_written": False,
            "public_runtime_supported": False,
            "official_weight_logits_parity": False,
        },
        "publication_rule": "runtime-stage.bin is converter-private; manifest.json remains absent",
    }
    write_json_atomic(stage_dir / "runtime-stage.json", meta)
    return meta

def _validate_qtrunk_vq_stages(
    src: Path, stage_dir: Path, *, verify: bool, require_official: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    plan, release, trunk, experts, quant = _validate_vq_stages(
        src, stage_dir, verify=verify, require_official=require_official
    )
    qmeta = _load(stage_dir / "qtrunk-stage.json")
    if qmeta.get("format") != "inkling-qtrunk-stage-v1":
        raise RuntimeStageError("qtrunk-stage.json has an unsupported format")
    h = hashlib.sha256((stage_dir / "trunk-stage.json").read_bytes()).hexdigest()
    if qmeta.get("source_manifest_sha256") != h:
        raise RuntimeStageError("quantized trunk was not built from this trunk-stage.json")
    qitems = qmeta.get("tensors")
    if not isinstance(qitems, list):
        raise RuntimeStageError("qtrunk-stage.json tensors is not an array")
    qby: dict[str, dict[str, Any]] = {}
    for item in qitems:
        if not isinstance(item, dict) or not isinstance(item.get("target"), str):
            raise RuntimeStageError("malformed quantized trunk entry")
        target = item["target"]
        if target in qby:
            raise RuntimeStageError(f"duplicate quantized trunk target: {target}")
        rel = _safe_relative(str(item.get("file", "")))
        path = stage_dir / "qtrunk-stage" / rel
        try:
            if verify:
                verify_qtensor(path, item)
            elif not path.is_file():
                raise RuntimeStageError(f"missing quantized tensor: {path}")
        except (OSError, QTrunkError) as exc:
            raise RuntimeStageError(str(exc)) from exc
        qby[target] = item
    canonical = {item["target"]: item for item in trunk["tensors"]}
    expected = _expected_targets(plan)
    merged: list[dict[str, Any]] = []
    for target in sorted(expected):
        if target in qby:
            merged.append(qby[target] | {"quantized": True})
        elif target in canonical:
            merged.append(canonical[target] | {"quantized": False})
        else:
            raise RuntimeStageError(f"missing canonical or quantized tensor: {target}")
    if set(qby) - expected:
        raise RuntimeStageError("quantized trunk contains unexpected tensor targets")
    supported_roles = {
        "q", "k", "v", "r", "o", "mlp.gate", "mlp.up", "mlp.down",
        "router.weight", "shared.gate", "shared.up", "shared.down",
    }
    for target in qby:
        if target in ("inkling.embed", "inkling.unembed"):
            continue
        parts = target.split(".", 3)
        if len(parts) != 4 or parts[0] != "inkling" or parts[1] != "layer" or parts[3] not in supported_roles:
            raise RuntimeStageError(f"quantized tensor has no runtime matrix backend: {target}")
    # Every matrix except the tiny convolution and relative-bias matrices may
    # be quantized. qtrunk policy intentionally leaves vectors/scalars in the
    # canonical stage; requiring at least embedding/unembedding/router proves
    # this is not a mislabeled v2 index.
    required_q = {"inkling.embed", "inkling.unembed"}
    required_q.update(t for t in expected if t.endswith("router.weight"))
    missing_q = sorted(required_q - set(qby))
    if missing_q:
        raise RuntimeStageError(f"quantized trunk is incomplete: {missing_q[:3]}")
    return plan, release, merged, experts, quant


def publish_runtime_qtrunk_vq_stage(
    src: Path, stage_dir: Path, *, verify: bool = True,
    require_official: bool = True,
) -> dict[str, Any]:
    plan, release, tensors, experts, quant = _validate_qtrunk_vq_stages(
        src, stage_dir, verify=verify, require_official=require_official
    )
    config = _config_values(plan)
    banks = sorted(experts["layers"], key=lambda item: item["layer"])
    payload = bytearray(_header_bytes(
        config, release, len(tensors), len(banks), expert_format="vq",
        quant=quant, quantized_trunk=True,
    ))
    for item in config["layers"]:
        payload += _layer_bytes(item)
    for item in tensors:
        payload += (_qtensor_bytes(item) if item.get("quantized")
                    else _tensor_bytes(stage_dir, item))
    for item in banks:
        payload += _bank_bytes(item, final_vq=True)
    tmp = stage_dir / "runtime-stage.bin.tmp"
    final = stage_dir / "runtime-stage.bin"
    try:
        with tmp.open("wb") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    qcount = sum(bool(x.get("quantized")) for x in tensors)
    meta = {
        "schema": "waste.inkling-private-runtime-stage.v3",
        "arch": "inkling-small" if release.get("official_small") else "inkling",
        "model_id": release.get("model_id"),
        "file": final.name, "bytes": final.stat().st_size,
        "expert_format": "WEXP/VQ", "expert_quant": quant,
        "trunk_format": "Q8/Q4 + canonical vectors",
        "counts": {"layers": len(config["layers"]), "tensors": len(tensors),
                   "quantized_tensors": qcount, "banks": len(banks)},
        "status": {"private_runtime_stage_complete": True,
                   "quantized_trunk": True, "final_expert_banks": True,
                   "waste_manifest_written": False,
                   "public_runtime_supported": False,
                   "official_weight_logits_parity": False},
        "publication_rule": "runtime-stage.bin is converter-private; manifest.json remains absent",
    }
    write_json_atomic(stage_dir / "runtime-stage.json", meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--stage", required=True, type=Path)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--allow-generic-inkling", action="store_true")
    ap.add_argument("--final-vq", action="store_true", help="publish from vq-stage.json instead of BF16 stage.json")
    ap.add_argument("--quantized-trunk", action="store_true", help="publish v3 index using qtrunk-stage.json plus final VQ banks")
    args = ap.parse_args(argv)
    try:
        if args.quantized_trunk:
            publisher = publish_runtime_qtrunk_vq_stage
        else:
            publisher = publish_runtime_vq_stage if args.final_vq else publish_runtime_stage
        meta = publisher(
            args.src, args.stage, verify=not args.no_verify,
            require_official=not args.allow_generic_inkling,
        )
    except RuntimeStageError as exc:
        print(f"inkling_runtime_stage: {exc}", file=sys.stderr)
        return 2
    print(
        f"runtime stage: {meta['counts']['layers']} layers, "
        f"{meta['counts']['tensors']} tensors, {meta['counts']['banks']} banks"
    )
    print("manifest written: no; public runtime supported: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
