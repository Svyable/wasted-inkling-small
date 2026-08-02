#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded Q8/Q4 conversion for canonical Inkling trunk-stage tensors.

This writes converter-private quantized tensor artifacts. It does not publish a
WASTE manifest and is intentionally independent from the public loader until
quantized full-model parity is established.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import torch

from inkling_trunk import HEADER as STAGE_HEADER, parse_tensor_header

MAGIC = 0x54514B49  # IKQT
VERSION = 1
ALIGN = 4096
FMT_Q8G = 2
FMT_Q4G = 3
HEADER = struct.Struct("<IHBBI4IQQQQIII24s")
assert HEADER.size == 96


class QTrunkError(RuntimeError):
    pass


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_identity(stage_root: Path) -> str:
    p = stage_root / "trunk-stage.json"
    if not p.is_file():
        raise QTrunkError("trunk-stage.json is missing; run --stage-trunk first")
    return _sha256(p)


def _policy(target: str, bits: int) -> int | None:
    # Vectors/scalars remain in the canonical BF16/F32 stage for now.
    # Vocabulary tables and router logits are sensitivity-first Q8. The rest
    # may use Q4 when explicitly requested.
    if target.endswith(("norm", "bias", "global_scale", "rel_proj", "sconv")):
        return None
    if target in ("inkling.embed", "inkling.unembed") or target.endswith("router.weight"):
        return FMT_Q8G
    return FMT_Q8G if bits == 8 else FMT_Q4G


def _read_stage_meta(root: Path) -> dict[str, Any]:
    try:
        meta = json.loads((root / "trunk-stage.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QTrunkError(f"cannot read trunk-stage.json: {exc}") from exc
    if not isinstance(meta.get("tensors"), list):
        raise QTrunkError("malformed trunk-stage.json")
    return meta


def _read_rows(path: Path, payload_offset: int, rows: int, cols: int,
               dtype: str, row0: int, count: int) -> torch.Tensor:
    es = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype)
    td = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}.get(dtype)
    if not es or td is None:
        raise QTrunkError(f"unsupported source dtype {dtype}")
    nbytes = count * cols * es
    with path.open("rb") as f:
        f.seek(payload_offset + row0 * cols * es)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise QTrunkError(f"short staged tensor read: {path}")
    return torch.frombuffer(bytearray(raw), dtype=td).reshape(count, cols).float()


def _quantize_rows(x: torch.Tensor, fmt: int, group: int) -> tuple[bytes, bytes]:
    rows, cols = x.shape
    ng = (cols + group - 1) // group
    padded = ng * group
    if padded != cols:
        x = torch.nn.functional.pad(x, (0, padded - cols))
    xg = x.reshape(rows, ng, group)
    if fmt == FMT_Q8G:
        scales = xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
        q = torch.clamp(torch.round(xg / scales), -127, 127).to(torch.int8)
        qb = q.contiguous().numpy().tobytes()
    elif fmt == FMT_Q4G:
        scales = xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0
        q = torch.clamp(torch.round(xg / scales), -8, 7).to(torch.int16).reshape(rows, padded)
        nib = (q + 8).to(torch.uint8)
        packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
        qb = packed.contiguous().numpy().tobytes()
    else:
        raise QTrunkError("unsupported quantized trunk format")
    sb = scales.squeeze(-1).to(torch.float16).contiguous().numpy().tobytes()
    return qb, sb


def _header(target: str, fmt: int, group: int, shape: list[int], rows: int,
            cols: int, qbytes: int, sbytes: int, qcrc: int, scrc: int) -> bytes:
    padded = shape + [0] * (4 - len(shape))
    return HEADER.pack(MAGIC, VERSION, fmt, len(shape), group, *padded,
                       rows, cols, qbytes, sbytes, qcrc, scrc,
                       zlib.crc32(target.encode()) & 0xFFFFFFFF, b"\0" * 24)


def verify_qtensor(path: Path, meta: dict[str, Any]) -> None:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise QTrunkError(f"short quantized trunk artifact: {path}")
    values = HEADER.unpack_from(raw)
    magic, version, fmt, ndim, group = values[:5]
    shape = list(values[5:9])[:ndim]
    rows, cols, qbytes, sbytes, qcrc, scrc, namecrc = values[9:16]
    if magic != MAGIC or version != VERSION or fmt not in (FMT_Q8G, FMT_Q4G) or not 1 <= ndim <= 4:
        raise QTrunkError(f"invalid quantized trunk header: {path}")
    if group != meta["group"] or shape != meta["shape"] or rows != meta["rows"] or cols != meta["cols"]:
        raise QTrunkError(f"quantized trunk geometry mismatch: {path}")
    if namecrc != zlib.crc32(meta["target"].encode()) & 0xFFFFFFFF:
        raise QTrunkError(f"quantized trunk name mismatch: {path}")
    qoff = HEADER.size; soff = qoff + qbytes
    if soff + sbytes > len(raw):
        raise QTrunkError(f"truncated quantized trunk artifact: {path}")
    if zlib.crc32(raw[qoff:soff]) & 0xFFFFFFFF != qcrc:
        raise QTrunkError(f"quantized payload CRC mismatch: {path}")
    if zlib.crc32(raw[soff:soff+sbytes]) & 0xFFFFFFFF != scrc:
        raise QTrunkError(f"quantized scale CRC mismatch: {path}")
    if len(raw) != _align(HEADER.size + qbytes + sbytes):
        raise QTrunkError(f"quantized trunk alignment mismatch: {path}")


def quantize_trunk(stage_root: Path, *, bits: int = 8, group: int = 128,
                   chunk_rows: int = 64, verify: bool = True,
                   resume: bool = True) -> dict[str, Any]:
    if bits not in (4, 8) or group < 2 or group % 2 or chunk_rows < 1:
        raise QTrunkError("bits must be 4/8, group must be positive even, and chunk_rows positive")
    stage = _read_stage_meta(stage_root)
    source_id = _source_identity(stage_root)
    out_dir = stage_root / "qtrunk-stage"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []
    qtotal = stotal = source_total = 0
    for ordinal, item in enumerate(stage["tensors"]):
        target = item["target"]
        shape = [int(x) for x in item["shape"]]
        fmt = _policy(target, bits)
        if fmt is None:
            continue
        if len(shape) < 2:
            continue
        rows = math.prod(shape[:-1]); cols = shape[-1]
        ng = (cols + group - 1) // group
        rowbytes = ng * group * (1 if fmt == FMT_Q8G else 1/2)
        if int(rowbytes) != rowbytes:
            raise QTrunkError(f"invalid Q4 row geometry for {target}")
        qbytes = rows * int(rowbytes)
        sbytes = rows * ng * 2
        filename = f"{ordinal:05d}.qtensor"
        path = out_dir / filename
        side = path.with_suffix(path.suffix + ".json")
        identity = {"source_manifest_sha256": source_id, "target": target,
                    "shape": shape, "format": fmt, "group": group,
                    "rows": rows, "cols": cols, "qbytes": qbytes,
                    "scale_bytes": sbytes}
        reused = False
        if resume and path.is_file() and side.is_file():
            try:
                old = json.loads(side.read_text(encoding="utf-8"))
                if all(old.get(k) == v for k, v in identity.items()):
                    verify_qtensor(path, old)
                    result.append(old | {"reused": True})
                    reused = True
            except (OSError, json.JSONDecodeError, QTrunkError):
                reused = False
        if reused:
            qtotal += qbytes; stotal += sbytes
            continue
        src = stage_root / "trunk-stage" / item["file"]
        raw_head = src.read_bytes()[:STAGE_HEADER.size]
        hdr = parse_tensor_header(raw_head)
        if list(hdr.shape) != shape or hdr.dtype != item["dtype"]:
            raise QTrunkError(f"source stage metadata mismatch for {target}")
        qtmp = path.with_name(path.name + ".qtmp")
        stmp = path.with_name(path.name + ".stmp")
        final_tmp = path.with_name(path.name + ".tmp")
        qcrc = scrc = 0; qw = sw = 0
        try:
            with qtmp.open("wb") as qf, stmp.open("wb") as sf:
                for row0 in range(0, rows, chunk_rows):
                    count = min(chunk_rows, rows - row0)
                    x = _read_rows(src, STAGE_HEADER.size, rows, cols, hdr.dtype, row0, count)
                    qb, sb = _quantize_rows(x, fmt, group)
                    qf.write(qb); sf.write(sb)
                    qcrc = zlib.crc32(qb, qcrc); scrc = zlib.crc32(sb, scrc)
                    qw += len(qb); sw += len(sb)
                    source_total += count * cols * {"BF16":2,"F16":2,"F32":4}[hdr.dtype]
            if qw != qbytes or sw != sbytes:
                raise QTrunkError(f"quantized byte count mismatch for {target}")
            with final_tmp.open("wb") as out, qtmp.open("rb") as qf, stmp.open("rb") as sf:
                out.write(_header(target, fmt, group, shape, rows, cols, qbytes, sbytes,
                                  qcrc & 0xFFFFFFFF, scrc & 0xFFFFFFFF))
                for f in (qf, sf):
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        out.write(chunk)
                out.write(b"\0" * (_align(HEADER.size + qbytes + sbytes) - HEADER.size - qbytes - sbytes))
                out.flush(); os.fsync(out.fileno())
            os.replace(final_tmp, path)
        finally:
            for tmp in (qtmp, stmp, final_tmp):
                try: tmp.unlink()
                except FileNotFoundError: pass
        meta = identity | {"file": filename, "bits": 8 if fmt == FMT_Q8G else 4,
                           "rowbytes": int(rowbytes), "scale_groups_per_row": ng,
                           "stored_bytes": path.stat().st_size,
                           "q_crc32": qcrc & 0xFFFFFFFF,
                           "scale_crc32": scrc & 0xFFFFFFFF,
                           "source_dtype": hdr.dtype, "reused": False}
        if verify: verify_qtensor(path, meta)
        _atomic_json(side, meta)
        result.append(meta); qtotal += qbytes; stotal += sbytes
    result.sort(key=lambda x: x["target"])
    output = {"format": "inkling-qtrunk-stage-v1", "source_manifest_sha256": source_id,
              "default_bits": bits, "group": group, "tensors": result,
              "totals": {"tensors": len(result), "quantized_bytes": qtotal,
                         "scale_bytes": stotal, "stored_bytes": sum(x["stored_bytes"] for x in result),
                         "source_bytes_read": source_total}}
    _atomic_json(stage_root / "qtrunk-stage.json", output)
    return output
