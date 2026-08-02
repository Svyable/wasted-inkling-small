#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Resumable, bounded staging of Inkling's resident text trunk.

This is a converter-internal format, not a WASTE trunk.  Every canonical tensor
is stored in its own atomically-published file so large embeddings can be copied
in bounded chunks and fused gate/up tensors can be deinterleaved without ever
materializing them.  ``trunk-stage.json`` is published last.  ``manifest.json``
and ``trunk.bin`` remain absent until the integrated C loader and text forward
path are complete.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from inkling_plan import PlanError, build_plan, write_json_atomic
from inkling_weights import DTYPES, SafeTensorReader, TensorLocation, WeightError


class TrunkStageError(RuntimeError):
    pass


MAGIC = b"IKTN"
VERSION = 1
ALIGN = 4096
DTYPE_CODE = {"BF16": 1, "F16": 2, "F32": 3}
CODE_DTYPE = {value: key for key, value in DTYPE_CODE.items()}
# magic, version, dtype, ndim, flags, reserved, shape[4], payload bytes,
# payload crc32, canonical-name crc32, reserved.
HEADER = struct.Struct("<4sHBBHH4IQII20s")
assert HEADER.size == 64


@dataclass(frozen=True)
class TrunkJob:
    target: str
    source: str
    transform: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class TensorStageHeader:
    dtype: str
    shape: tuple[int, ...]
    payload_bytes: int
    crc32: int
    name_crc32: int


def _align(value: int, alignment: int = ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _shape_bytes(shape: Iterable[int], dtype: str) -> int:
    if dtype not in DTYPES:
        raise TrunkStageError(f"unsupported trunk dtype: {dtype}")
    return math.prod(shape) * DTYPES[dtype][1]


def _header_bytes(job: TrunkJob, crc32: int) -> bytes:
    if not 1 <= len(job.shape) <= 4:
        raise TrunkStageError(f"unsupported rank {len(job.shape)} for {job.target}")
    if any(dim < 0 or dim > 0xFFFFFFFF for dim in job.shape):
        raise TrunkStageError(f"invalid shape for {job.target}: {list(job.shape)}")
    padded_shape = list(job.shape) + [0] * (4 - len(job.shape))
    payload_bytes = _shape_bytes(job.shape, job.dtype)
    return HEADER.pack(
        MAGIC,
        VERSION,
        DTYPE_CODE[job.dtype],
        len(job.shape),
        0,
        0,
        *padded_shape,
        payload_bytes,
        crc32,
        zlib.crc32(job.target.encode("utf-8")) & 0xFFFFFFFF,
        b"\0" * 20,
    )


def parse_tensor_header(data: bytes) -> TensorStageHeader:
    if len(data) < HEADER.size:
        raise TrunkStageError("short Inkling trunk-stage header")
    (
        magic,
        version,
        dtype_code,
        ndim,
        flags,
        reserved,
        s0,
        s1,
        s2,
        s3,
        payload_bytes,
        crc32,
        name_crc32,
        reserved_bytes,
    ) = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or flags != 0 or reserved != 0:
        raise TrunkStageError("invalid Inkling trunk-stage header identity")
    if reserved_bytes != b"\0" * 20:
        raise TrunkStageError("nonzero Inkling trunk-stage reserved bytes")
    dtype = CODE_DTYPE.get(dtype_code)
    if dtype is None or ndim < 1 or ndim > 4:
        raise TrunkStageError("invalid Inkling trunk-stage dtype or rank")
    shape = (s0, s1, s2, s3)[:ndim]
    if _shape_bytes(shape, dtype) != payload_bytes:
        raise TrunkStageError("Inkling trunk-stage payload geometry mismatch")
    return TensorStageHeader(
        dtype=dtype,
        shape=shape,
        payload_bytes=payload_bytes,
        crc32=crc32,
        name_crc32=name_crc32,
    )


def _canonical_target(role: str) -> str:
    return f"inkling.{role}"


def build_trunk_jobs(plan: dict[str, Any], reader: SafeTensorReader) -> list[TrunkJob]:
    if plan.get("source_dialect") != "provider_raw":
        raise TrunkStageError("trunk staging currently accepts provider-raw Inkling checkpoints only")
    jobs: list[TrunkJob] = []
    targets: set[str] = set()
    for item in plan.get("trunk", []):
        if not isinstance(item, dict):
            raise TrunkStageError("malformed trunk entry in conversion plan")
        role = item.get("role")
        source = item.get("source")
        if not isinstance(role, str) or not isinstance(source, str):
            raise TrunkStageError("trunk plan entry lacks role/source")
        loc = reader.location(source)
        if role.endswith(".fused_gate_up"):
            if len(loc.shape) not in (2, 3):
                raise TrunkStageError(f"fused gate/up rank is unsupported for {source}: {loc.shape}")
            split_dim = 0 if len(loc.shape) == 2 else 1
            if loc.shape[split_dim] % 2:
                raise TrunkStageError(f"fused gate/up dimension is odd for {source}: {loc.shape}")
            out_shape = list(loc.shape)
            out_shape[split_dim] //= 2
            base = role[: -len("fused_gate_up")]
            for suffix, transform in (("gate", "deinterleave-even-rows"), ("up", "deinterleave-odd-rows")):
                target = _canonical_target(base + suffix)
                jobs.append(TrunkJob(target, source, transform, tuple(out_shape), loc.dtype))
        else:
            target = _canonical_target(role)
            if role.endswith(("k_sconv", "v_sconv", "attn_sconv", "mlp_sconv")):
                if len(loc.shape) != 3 or loc.shape[1] != 1:
                    raise TrunkStageError(
                        f"short-convolution tensor must be [channels,1,kernel]: {source} {loc.shape}"
                    )
                # The C kernel consumes [channels][kernel]. Removing the singleton
                # Conv1d group axis changes metadata only; payload bytes are already
                # contiguous in exactly that order.
                jobs.append(TrunkJob(target, source, "squeeze-conv-group-axis",
                                     (loc.shape[0], loc.shape[2]), loc.dtype))
            else:
                jobs.append(TrunkJob(target, source, "identity", loc.shape, loc.dtype))
    for job in jobs:
        if job.target in targets:
            raise TrunkStageError(f"duplicate canonical trunk target: {job.target}")
        targets.add(job.target)
    return sorted(jobs, key=lambda job: job.target)


def _artifact_paths(root: Path, ordinal: int) -> tuple[Path, Path]:
    path = root / f"{ordinal:05d}.tensor.stage"
    return path, path.with_name(path.name + ".json")


def _write_payload_file(path: Path, job: TrunkJob, chunks: Iterable[bytes]) -> dict[str, Any]:
    tmp = path.with_name(path.name + ".tmp")
    payload_bytes = _shape_bytes(job.shape, job.dtype)
    written = 0
    crc = 0
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp.open("w+b") as f:
            f.write(b"\0" * HEADER.size)
            for chunk in chunks:
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                crc = zlib.crc32(chunk, crc)
            if written != payload_bytes:
                raise TrunkStageError(
                    f"{job.target} wrote {written} payload bytes; expected {payload_bytes}"
                )
            stored_bytes = _align(HEADER.size + payload_bytes)
            f.write(b"\0" * (stored_bytes - HEADER.size - payload_bytes))
            f.seek(0)
            f.write(_header_bytes(job, crc & 0xFFFFFFFF))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "target": job.target,
        "source": job.source,
        "transform": job.transform,
        "dtype": job.dtype,
        "shape": list(job.shape),
        "header_bytes": HEADER.size,
        "payload_bytes": payload_bytes,
        "stored_bytes": path.stat().st_size,
        "alignment": ALIGN,
        "crc32": crc & 0xFFFFFFFF,
    }


def _identity_chunks(reader: SafeTensorReader, source: str, chunk_bytes: int) -> Iterable[bytes]:
    return reader.iter_bytes(source, chunk_bytes=chunk_bytes)


def _split_chunks(
    reader: SafeTensorReader,
    source: str,
    *,
    parity: int,
    chunk_bytes: int,
) -> Iterable[bytes]:
    loc = reader.location(source)
    if len(loc.shape) not in (2, 3):
        raise TrunkStageError(f"cannot split fused tensor rank {len(loc.shape)}: {source}")
    fused_rows = loc.shape[0] if len(loc.shape) == 2 else loc.shape[1]
    if fused_rows % 2:
        raise TrunkStageError(f"cannot split odd fused row count {fused_rows}: {source}")
    row_bytes = loc.shape[-1] * DTYPES[loc.dtype][1]
    for start, count, data in reader.iter_row_chunks(source, chunk_bytes=chunk_bytes):
        view = memoryview(data)
        out = bytearray()
        for local_row in range(count):
            flat_row = start + local_row
            row_within_group = flat_row % fused_rows
            if row_within_group % 2 == parity:
                begin = local_row * row_bytes
                out.extend(view[begin : begin + row_bytes])
        if out:
            yield bytes(out)



def _write_split_pair(
    reader: SafeTensorReader,
    source: str,
    even_path: Path,
    even_job: TrunkJob,
    odd_path: Path,
    odd_job: TrunkJob,
    *,
    chunk_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if even_job.dtype != odd_job.dtype or even_job.shape != odd_job.shape:
        raise TrunkStageError(f"split outputs disagree for {source}")
    loc = reader.location(source)
    if len(loc.shape) not in (2, 3):
        raise TrunkStageError(f"cannot split fused tensor rank {len(loc.shape)}: {source}")
    fused_rows = loc.shape[0] if len(loc.shape) == 2 else loc.shape[1]
    if fused_rows % 2:
        raise TrunkStageError(f"cannot split odd fused row count {fused_rows}: {source}")
    row_bytes = loc.shape[-1] * DTYPES[loc.dtype][1]
    payload_bytes = _shape_bytes(even_job.shape, even_job.dtype)
    even_tmp = even_path.with_name(even_path.name + ".tmp")
    odd_tmp = odd_path.with_name(odd_path.name + ".tmp")
    even_path.parent.mkdir(parents=True, exist_ok=True)
    written = [0, 0]
    crc = [0, 0]
    try:
        with even_tmp.open("w+b") as fe, odd_tmp.open("w+b") as fo:
            files = [fe, fo]
            for f in files:
                f.write(b"\0" * HEADER.size)
            for start, count, data in reader.iter_row_chunks(source, chunk_bytes=chunk_bytes):
                view = memoryview(data)
                buffers = [bytearray(), bytearray()]
                for local_row in range(count):
                    flat_row = start + local_row
                    parity = (flat_row % fused_rows) & 1
                    begin = local_row * row_bytes
                    buffers[parity].extend(view[begin : begin + row_bytes])
                for parity, buf in enumerate(buffers):
                    if not buf:
                        continue
                    files[parity].write(buf)
                    written[parity] += len(buf)
                    crc[parity] = zlib.crc32(buf, crc[parity])
            stored_bytes = _align(HEADER.size + payload_bytes)
            for parity, (f, job) in enumerate(zip(files, (even_job, odd_job))):
                if written[parity] != payload_bytes:
                    raise TrunkStageError(
                        f"{job.target} wrote {written[parity]} payload bytes; expected {payload_bytes}"
                    )
                f.write(b"\0" * (stored_bytes - HEADER.size - payload_bytes))
                f.seek(0)
                f.write(_header_bytes(job, crc[parity] & 0xFFFFFFFF))
                f.flush()
                os.fsync(f.fileno())
        os.replace(even_tmp, even_path)
        os.replace(odd_tmp, odd_path)
    except BaseException:
        for tmp in (even_tmp, odd_tmp):
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        raise

    def meta(path: Path, job: TrunkJob, parity: int) -> dict[str, Any]:
        return {
            "target": job.target,
            "source": job.source,
            "transform": job.transform,
            "dtype": job.dtype,
            "shape": list(job.shape),
            "header_bytes": HEADER.size,
            "payload_bytes": payload_bytes,
            "stored_bytes": path.stat().st_size,
            "alignment": ALIGN,
            "crc32": crc[parity] & 0xFFFFFFFF,
        }

    return meta(even_path, even_job, 0), meta(odd_path, odd_job, 1)

def verify_tensor(path: Path, meta: dict[str, Any], *, verify_crc: bool = True) -> None:
    expected_size = int(meta["stored_bytes"])
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise TrunkStageError(f"cannot stat {path}: {exc}") from exc
    if actual_size != expected_size or actual_size % ALIGN:
        raise TrunkStageError(f"invalid staged tensor size for {path}")
    with path.open("rb") as f:
        header_bytes = f.read(HEADER.size)
        header = parse_tensor_header(header_bytes)
        if header.dtype != meta["dtype"] or list(header.shape) != list(meta["shape"]):
            raise TrunkStageError(f"staged tensor header metadata mismatch: {path}")
        if header.payload_bytes != int(meta["payload_bytes"]) or header.crc32 != int(meta["crc32"]):
            raise TrunkStageError(f"staged tensor payload metadata mismatch: {path}")
        expected_name_crc = zlib.crc32(str(meta["target"]).encode("utf-8")) & 0xFFFFFFFF
        if header.name_crc32 != expected_name_crc:
            raise TrunkStageError(f"staged tensor target identity mismatch: {path}")
        if verify_crc:
            remaining = header.payload_bytes
            crc = 0
            while remaining:
                chunk = f.read(min(8 << 20, remaining))
                if not chunk:
                    raise TrunkStageError(f"short staged tensor payload: {path}")
                remaining -= len(chunk)
                crc = zlib.crc32(chunk, crc)
            if crc & 0xFFFFFFFF != header.crc32:
                raise TrunkStageError(f"staged tensor CRC mismatch: {path}")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _resume_matches(
    meta: dict[str, Any],
    job: TrunkJob,
    *,
    file_name: str,
    config_sha256: str,
    index_sha256: str,
) -> bool:
    expected = {
        "schema": "waste.inkling-trunk-tensor.v1",
        "file": file_name,
        "target": job.target,
        "source": job.source,
        "transform": job.transform,
        "dtype": job.dtype,
        "shape": list(job.shape),
        "source_config_sha256": config_sha256,
        "source_index_sha256": index_sha256,
    }
    return all(meta.get(key) == value for key, value in expected.items())


def stage_trunk(
    src: Path,
    out: Path,
    *,
    verify: bool = True,
    resume: bool = True,
    chunk_bytes: int = 8 << 20,
) -> dict[str, Any]:
    if chunk_bytes <= 0:
        raise TrunkStageError("chunk_bytes must be positive")
    try:
        plan = build_plan(src, require_payloads=True)
        reader = SafeTensorReader(src)
        jobs = build_trunk_jobs(plan, reader)
    except (PlanError, WeightError) as exc:
        raise TrunkStageError(str(exc)) from exc

    config_sha256 = _sha256(src / "config.json")
    index_sha256 = _sha256(src / "model.safetensors.index.json")
    root = out / "trunk-stage"
    root.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, list[tuple[int, TrunkJob]]] = {}
    for ordinal, job in enumerate(jobs):
        by_source.setdefault(job.source, []).append((ordinal, job))

    published: list[dict[str, Any]] = []
    completed: dict[str, dict[str, Any]] = {}
    for source_name, group in by_source.items():
        reusable = True
        group_meta: list[tuple[int, TrunkJob, Path, Path, dict[str, Any] | None]] = []
        for ordinal, job in group:
            path, sidecar = _artifact_paths(root, ordinal)
            meta = _load_json(sidecar) if resume else None
            if meta is None or not _resume_matches(
                meta,
                job,
                file_name=path.name,
                config_sha256=config_sha256,
                index_sha256=index_sha256,
            ):
                reusable = False
            else:
                try:
                    verify_tensor(path, meta, verify_crc=verify)
                except (OSError, TrunkStageError):
                    reusable = False
            group_meta.append((ordinal, job, path, sidecar, meta))

        if reusable:
            for ordinal, job, path, sidecar, meta in group_meta:
                assert meta is not None
                item = dict(meta)
                item.update({"ordinal": ordinal, "reused": True, "sidecar": sidecar.name})
                completed[job.target] = item
            continue

        transforms = {job.transform for _, job, _, _, _ in group_meta}
        if transforms in ({"identity"}, {"squeeze-conv-group-axis"}) and len(group_meta) == 1:
            ordinal, job, path, sidecar, _ = group_meta[0]
            meta = _write_payload_file(
                path,
                job,
                _identity_chunks(reader, source_name, chunk_bytes),
            )
            meta.update(
                {
                    "schema": "waste.inkling-trunk-tensor.v1",
                    "file": path.name,
                    "source_config_sha256": config_sha256,
                    "source_index_sha256": index_sha256,
                }
            )
            if verify:
                verify_tensor(path, meta, verify_crc=True)
            write_json_atomic(sidecar, meta)
            item = dict(meta)
            item.update({"ordinal": ordinal, "reused": False, "sidecar": sidecar.name})
            completed[job.target] = item
            continue

        if transforms != {"deinterleave-even-rows", "deinterleave-odd-rows"} or len(group_meta) != 2:
            raise TrunkStageError(f"unsupported trunk transform group for {source_name}: {sorted(transforms)}")
        ordered = sorted(
            group_meta,
            key=lambda item: 0 if item[1].transform == "deinterleave-even-rows" else 1,
        )
        even = ordered[0]
        odd = ordered[1]
        metas = _write_split_pair(
            reader, source_name, even[2], even[1], odd[2], odd[1], chunk_bytes=chunk_bytes
        )
        for (ordinal, job, path, sidecar, _), meta in zip((even, odd), metas):
            meta.update(
                {
                    "schema": "waste.inkling-trunk-tensor.v1",
                    "file": path.name,
                    "source_config_sha256": config_sha256,
                    "source_index_sha256": index_sha256,
                }
            )
            if verify:
                verify_tensor(path, meta, verify_crc=True)
            write_json_atomic(sidecar, meta)
            item = dict(meta)
            item.update({"ordinal": ordinal, "reused": False, "sidecar": sidecar.name})
            completed[job.target] = item

    for job in jobs:
        published.append(completed[job.target])

    stage = {
        "schema": "waste.inkling-resident-trunk-stage.v1",
        "arch": "inkling",
        "scope": "resident-text-trunk-only",
        "source": {
            "path": str(src),
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "dialect": plan["source_dialect"],
        },
        "tensors": published,
        "totals": {
            "tensors": len(published),
            "payload_bytes": sum(int(item["payload_bytes"]) for item in published),
            "stored_bytes": sum(int(item["stored_bytes"]) for item in published),
            "source_bytes_read": reader.bytes_read,
        },
        "status": {
            "trunk_stage_complete": True,
            "waste_trunk_written": False,
            "waste_manifest_written": False,
            "waste_runtime_supported": False,
            "reference_parity": False,
        },
        "publication_rule": "trunk-stage.json is converter-internal; manifest.json and trunk.bin remain absent",
    }
    write_json_atomic(out / "trunk-stage.json", stage)
    return stage
