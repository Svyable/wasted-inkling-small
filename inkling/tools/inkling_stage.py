#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Stage Inkling routed experts as aligned BF16 records.

This is deliberately not the WASTE container writer.  It consumes the exact
source adapter and produces a converter-internal staging directory whose commit
record is ``stage.json``.  No ``manifest.json`` is written, so an existing
WASTE runtime cannot mistake the output for a runnable model.

Each routed expert is one 4 KiB-aligned record containing gate, up, and down
BF16 matrices.  The record is self-describing enough for round-trip validation
and one positional read, while remaining independent of WASTE format v0's
VQ-only expert header.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from inkling_plan import PlanError, build_plan, write_json_atomic
from inkling_weights import ProviderRawInklingWeights, WeightError


class StageError(RuntimeError):
    pass


ALIGN = 4096
MAGIC = 0x46424B49  # "IKBF" little-endian: Inkling BF16 stage record
VERSION = 1
DTYPE_BF16 = 1
HEADER = struct.Struct("<IHHHBBIIQQQQI8x")
assert HEADER.size == 64


@dataclass(frozen=True)
class StageHeader:
    layer: int
    expert: int
    hidden: int
    intermediate: int
    gate_off: int
    up_off: int
    down_off: int
    payload_bytes: int
    crc32: int


def _align(n: int, alignment: int = ALIGN) -> int:
    return (n + alignment - 1) // alignment * alignment


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype != torch.bfloat16:
        raise StageError(f"stage payload must be BF16, got {tensor.dtype}")
    raw = tensor.detach().cpu().contiguous().view(torch.uint8)
    return raw.numpy().tobytes()


def record_geometry(hidden: int, intermediate: int) -> dict[str, int]:
    if hidden <= 0 or intermediate <= 0:
        raise StageError("hidden and intermediate sizes must be positive")
    gate_bytes = intermediate * hidden * 2
    up_bytes = gate_bytes
    down_bytes = hidden * intermediate * 2
    gate_off = HEADER.size
    up_off = gate_off + gate_bytes
    down_off = up_off + up_bytes
    payload_bytes = gate_bytes + up_bytes + down_bytes
    return {
        "gate_off": gate_off,
        "up_off": up_off,
        "down_off": down_off,
        "payload_bytes": payload_bytes,
        "record_bytes": _align(HEADER.size + payload_bytes),
    }


def parse_header(data: bytes) -> StageHeader:
    if len(data) < HEADER.size:
        raise StageError("short Inkling stage header")
    (
        magic,
        version,
        layer,
        expert,
        dtype,
        flags,
        hidden,
        intermediate,
        gate_off,
        up_off,
        down_off,
        payload_bytes,
        crc32,
    ) = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or dtype != DTYPE_BF16 or flags != 0:
        raise StageError("invalid Inkling stage header identity")
    geom = record_geometry(hidden, intermediate)
    for key, actual in (
        ("gate_off", gate_off),
        ("up_off", up_off),
        ("down_off", down_off),
        ("payload_bytes", payload_bytes),
    ):
        if actual != geom[key]:
            raise StageError(f"invalid {key} in Inkling stage header: {actual} != {geom[key]}")
    return StageHeader(
        layer=layer,
        expert=expert,
        hidden=hidden,
        intermediate=intermediate,
        gate_off=gate_off,
        up_off=up_off,
        down_off=down_off,
        payload_bytes=payload_bytes,
        crc32=crc32,
    )


def _source_dtype_ok(source: ProviderRawInklingWeights, layer: int) -> None:
    prefix = source._layer(layer)  # adapter owns the provider-raw grammar
    for name in (
        f"{prefix}.mlp.experts.w13_weight",
        f"{prefix}.mlp.experts.w2_weight",
    ):
        loc = source.reader.location(name)
        if loc.dtype != "BF16":
            raise StageError(
                f"BF16 parity staging requires BF16 source tensors; {name} is {loc.dtype}"
            )


def write_bank(
    source: ProviderRawInklingWeights,
    out_path: Path,
    layer: int,
    *,
    expert_limit: int | None = None,
) -> dict[str, Any]:
    if layer < source.dense_count or layer >= source.n_layers:
        raise StageError(f"layer {layer} is not a sparse Inkling layer")
    _source_dtype_ok(source, layer)
    count = source.n_experts if expert_limit is None else min(expert_limit, source.n_experts)
    if count <= 0:
        raise StageError("expert limit must select at least one expert")
    geom = record_geometry(source.hidden, source.intermediate)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp.open("wb") as f:
            for expert in range(count):
                weights = source.routed_expert(layer, expert, float32=False)
                gate = _tensor_bytes(weights.gate)
                up = _tensor_bytes(weights.up)
                down = _tensor_bytes(weights.down)
                payload = gate + up + down
                if len(payload) != geom["payload_bytes"]:
                    raise StageError(
                        f"layer {layer} expert {expert} payload is {len(payload)} bytes; "
                        f"expected {geom['payload_bytes']}"
                    )
                crc = zlib.crc32(payload) & 0xFFFFFFFF
                header = HEADER.pack(
                    MAGIC,
                    VERSION,
                    layer,
                    expert,
                    DTYPE_BF16,
                    0,
                    source.hidden,
                    source.intermediate,
                    geom["gate_off"],
                    geom["up_off"],
                    geom["down_off"],
                    geom["payload_bytes"],
                    crc,
                )
                f.write(header)
                f.write(payload)
                f.write(b"\0" * (geom["record_bytes"] - HEADER.size - len(payload)))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "layer": layer,
        "file": out_path.name,
        "dtype": "BF16",
        "experts": count,
        "model_experts": source.n_experts,
        "complete": count == source.n_experts,
        "hidden_size": source.hidden,
        "intermediate_size": source.intermediate,
        "header_bytes": HEADER.size,
        "payload_bytes": geom["payload_bytes"],
        "record_bytes": geom["record_bytes"],
        "bytes": out_path.stat().st_size,
        "alignment": ALIGN,
        "crc": "zlib crc32 over gate|up|down payload",
    }


def verify_bank(path: Path, meta: dict[str, Any], *, verify_crc: bool = True) -> None:
    record_bytes = int(meta["record_bytes"])
    count = int(meta["experts"])
    expected_size = record_bytes * count
    if path.stat().st_size != expected_size:
        raise StageError(f"{path} size differs from stage metadata")
    with path.open("rb") as f:
        for expert in range(count):
            rec = f.read(record_bytes)
            if len(rec) != record_bytes:
                raise StageError(f"short record {expert} in {path}")
            header = parse_header(rec)
            if header.layer != int(meta["layer"]) or header.expert != expert:
                raise StageError(f"record identity mismatch in {path}: {header}")
            if verify_crc:
                payload = rec[HEADER.size : HEADER.size + header.payload_bytes]
                if zlib.crc32(payload) & 0xFFFFFFFF != header.crc32:
                    raise StageError(f"record CRC mismatch: layer {header.layer} expert {expert}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _bank_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def _load_bank_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _resume_matches(
    meta: dict[str, Any],
    *,
    layer: int,
    file_name: str,
    count: int,
    source: ProviderRawInklingWeights,
    config_sha256: str,
    index_sha256: str,
) -> bool:
    expected = {
        "schema": "waste.inkling-bf16-bank.v1",
        "layer": layer,
        "file": file_name,
        "dtype": "BF16",
        "experts": count,
        "model_experts": source.n_experts,
        "hidden_size": source.hidden,
        "intermediate_size": source.intermediate,
        "source_config_sha256": config_sha256,
        "source_index_sha256": index_sha256,
    }
    return all(meta.get(key) == value for key, value in expected.items())


def stage_expert_banks(
    src: Path,
    out: Path,
    *,
    layers: Iterable[int] | None = None,
    expert_limit: int | None = None,
    verify: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    try:
        plan = build_plan(src, require_payloads=True)
        if plan["source_dialect"] != "provider_raw":
            raise StageError("BF16 staging currently accepts the official provider-raw checkpoint only")
        source = ProviderRawInklingWeights(src)
    except (PlanError, WeightError) as exc:
        raise StageError(str(exc)) from exc

    sparse = list(plan["model"]["sparse_mlp_layers"])
    selected = sparse if layers is None else list(layers)
    if len(set(selected)) != len(selected):
        raise StageError("duplicate layer in staging request")
    invalid = [layer for layer in selected if layer not in sparse]
    if invalid:
        raise StageError(f"requested non-sparse or out-of-range layers: {invalid}")

    out.mkdir(parents=True, exist_ok=True)
    config_sha256 = _sha256(src / "config.json")
    index_sha256 = _sha256(src / "model.safetensors.index.json")
    count = source.n_experts if expert_limit is None else min(expert_limit, source.n_experts)
    banks = []
    for layer in selected:
        bank_path = out / f"experts-L{layer}.bf16.stage"
        sidecar_path = _bank_sidecar(bank_path)
        meta = _load_bank_sidecar(sidecar_path) if resume else None
        reused = False
        if meta is not None and _resume_matches(
            meta,
            layer=layer,
            file_name=bank_path.name,
            count=count,
            source=source,
            config_sha256=config_sha256,
            index_sha256=index_sha256,
        ):
            try:
                verify_bank(bank_path, meta, verify_crc=verify)
                reused = True
            except (FileNotFoundError, OSError, StageError):
                meta = None
        if not reused:
            meta = write_bank(source, bank_path, layer, expert_limit=expert_limit)
            meta.update(
                {
                    "schema": "waste.inkling-bf16-bank.v1",
                    "source_config_sha256": config_sha256,
                    "source_index_sha256": index_sha256,
                }
            )
            if verify:
                verify_bank(bank_path, meta, verify_crc=True)
            write_json_atomic(sidecar_path, meta)
        assert meta is not None
        published = dict(meta)
        published["reused"] = reused
        published["sidecar"] = sidecar_path.name
        banks.append(published)

    stage = {
        "schema": "waste.inkling-bf16-expert-stage.v1",
        "arch": "inkling-small",
        "scope": "routed-experts-only",
        "source": {
            "path": str(src),
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "dialect": plan["source_dialect"],
        },
        "model": {
            "hidden_size": source.hidden,
            "intermediate_size": source.intermediate,
            "routed_experts": source.n_experts,
            "top_k": plan["model"]["top_k"],
            "shared_experts": plan["model"]["shared_experts"],
        },
        "banks": banks,
        "status": {
            "stage_complete_for_requested_layers": all(bank["complete"] for bank in banks),
            "waste_manifest_written": False,
            "waste_runtime_supported": False,
            "reference_parity": False,
        },
        "publication_rule": "stage.json is converter-internal; manifest.json remains absent",
    }
    write_json_atomic(out / "stage.json", stage)
    return stage
