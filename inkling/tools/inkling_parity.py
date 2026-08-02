#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Bounded official-weight parity fixtures and activation archives for Inkling.

This module does not claim parity by itself.  It creates a small, deterministic
fixture from an official checkpoint so the Python reference and dependency-free
C runtime can compare the same tensors and intermediate activations without
copying the complete checkpoint.

The fixture is intentionally source-oriented: tensors retain their published
names and dtypes, while routed experts are stored as individual axis-0 slices.
Canonical conversion is exercised separately by the existing trunk/expert
adapters.  Every artifact is checksummed and the manifest is published last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from inkling_release import inspect_release
from inkling_weights import DTYPES, SafeTensorReader, WeightError


class ParityError(RuntimeError):
    pass


ARCHIVE_MAGIC = b"IKPA"
ARCHIVE_VERSION = 1
ARCHIVE_HEADER = struct.Struct("<4sHBBI4IQII20s")  # 64 bytes
assert ARCHIVE_HEADER.size == 64
DTYPE_CODE = {"F32": 1, "I32": 2}
CODE_DTYPE = {1: (torch.float32, 4), 2: (torch.int32, 4)}


@dataclass(frozen=True)
class ArchiveEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    payload_bytes: int
    crc32: int
    path: str


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _safe_name(name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")[:80] or "entry"
    return f"{stem}-{digest}.bin"


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    return tensor.view(torch.uint8).numpy().tobytes()


def write_archive_entry(root: Path, name: str, tensor: torch.Tensor) -> ArchiveEntry:
    if tensor.dtype == torch.float32:
        dtype = "F32"
    elif tensor.dtype in (torch.int32, torch.int64):
        dtype = "I32"
        tensor = tensor.to(torch.int32)
    else:
        raise ParityError(f"activation {name} must be F32 or integer, got {tensor.dtype}")
    if tensor.ndim < 1 or tensor.ndim > 4 or any(int(x) <= 0 for x in tensor.shape):
        raise ParityError(f"activation {name} has unsupported shape {list(tensor.shape)}")
    payload = _tensor_bytes(tensor)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    shape = tuple(int(x) for x in tensor.shape)
    padded = list(shape) + [0] * (4 - len(shape))
    path = root / _safe_name(name)
    tmp = path.with_name(path.name + ".tmp")
    header = ARCHIVE_HEADER.pack(
        ARCHIVE_MAGIC,
        ARCHIVE_VERSION,
        DTYPE_CODE[dtype],
        len(shape),
        0,
        *padded,
        len(payload),
        crc,
        zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF,
        b"\0" * 20,
    )
    try:
        with tmp.open("wb") as f:
            f.write(header)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return ArchiveEntry(name, dtype, shape, len(payload), crc, path.name)


def read_archive_entry(root: Path, entry: dict[str, Any]) -> torch.Tensor:
    name = entry.get("name")
    rel = entry.get("path")
    if not isinstance(name, str) or not isinstance(rel, str) or Path(rel).name != rel:
        raise ParityError("malformed activation archive entry")
    path = root / rel
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ParityError(f"cannot read activation {name}: {exc}") from exc
    if len(raw) < ARCHIVE_HEADER.size:
        raise ParityError(f"short activation file: {name}")
    values = ARCHIVE_HEADER.unpack_from(raw)
    magic, version, dtype_code, ndim, flags = values[:5]
    dims = values[5:9]
    payload_bytes, crc, name_crc, reserved = values[9:13]
    if magic != ARCHIVE_MAGIC or version != ARCHIVE_VERSION or flags != 0 or reserved != b"\0" * 20:
        raise ParityError(f"invalid activation header: {name}")
    if dtype_code not in CODE_DTYPE or ndim < 1 or ndim > 4:
        raise ParityError(f"invalid activation dtype/rank: {name}")
    shape = tuple(int(x) for x in dims[:ndim])
    torch_dtype, item_size = CODE_DTYPE[dtype_code]
    if any(x <= 0 for x in shape) or math.prod(shape) * item_size != payload_bytes:
        raise ParityError(f"activation geometry mismatch: {name}")
    payload = raw[ARCHIVE_HEADER.size:]
    if len(payload) != payload_bytes or zlib.crc32(payload) & 0xFFFFFFFF != crc:
        raise ParityError(f"activation payload corruption: {name}")
    if zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF != name_crc:
        raise ParityError(f"activation name mismatch: {name}")
    return torch.frombuffer(bytearray(payload), dtype=torch_dtype).reshape(shape).clone()


def write_activation_archive(out: Path | str, values: dict[str, torch.Tensor], *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    entries = [write_archive_entry(root, name, values[name]) for name in sorted(values)]
    manifest = {
        "format": "inkling-parity-activations",
        "version": ARCHIVE_VERSION,
        "metadata": metadata or {},
        "entries": [entry.__dict__ | {"shape": list(entry.shape)} for entry in entries],
    }
    _atomic_json(root / "activations.json", manifest)
    return manifest


def read_activation_archive(root: Path | str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    root = Path(root)
    try:
        manifest = json.loads((root / "activations.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParityError(f"cannot read activation archive: {exc}") from exc
    if manifest.get("format") != "inkling-parity-activations" or manifest.get("version") != ARCHIVE_VERSION:
        raise ParityError("unsupported activation archive")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ParityError("activation archive entries are missing")
    out: dict[str, torch.Tensor] = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("name") in out:
            raise ParityError("malformed or duplicate activation entry")
        out[entry["name"]] = read_archive_entry(root, entry)
    meta = manifest.get("metadata")
    return out, meta if isinstance(meta, dict) else {}


def _layer_tensor_names(reader: SafeTensorReader, layers: set[int]) -> list[str]:
    out = []
    for name in reader.names():
        match = re.match(r"model\.llm\.layers\.(\d+)\.", name)
        if match and int(match.group(1)) in layers and ".mlp.experts.w" not in name:
            out.append(name)
    return sorted(out)


def _global_tensor_names(reader: SafeTensorReader) -> list[str]:
    wanted = {
        "model.llm.embed.weight",
        "model.llm.embed_norm.weight",
        "model.llm.norm.weight",
        "model.llm.unembed.weight",
    }
    return sorted(wanted.intersection(reader.names()))


def parse_expert_selection(text: str) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    if not text:
        return result
    for group in text.split(";"):
        try:
            layer_text, experts_text = group.split(":", 1)
            layer = int(layer_text)
            experts = sorted({int(x) for x in experts_text.split(",") if x != ""})
        except ValueError as exc:
            raise ParityError(f"invalid expert selection {group!r}; use L:e,e;L:e") from exc
        if layer < 0 or not experts or any(e < 0 for e in experts):
            raise ParityError(f"invalid expert selection {group!r}")
        result[layer] = experts
    return result


def _copy_tensor(reader: SafeTensorReader, name: str, out: Path, *, max_bytes: int) -> dict[str, Any]:
    loc = reader.location(name)
    if loc.nbytes > max_bytes:
        raise ParityError(f"tensor {name} is {loc.nbytes} bytes, above per-tensor limit {max_bytes}")
    path = out / _safe_name(name)
    tmp = path.with_name(path.name + ".tmp")
    crc = 0
    written = 0
    try:
        with tmp.open("wb") as f:
            for chunk in reader.iter_bytes(name):
                f.write(chunk)
                crc = zlib.crc32(chunk, crc)
                written += len(chunk)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"name": name, "kind": "tensor", "dtype": loc.dtype, "shape": list(loc.shape),
            "bytes": written, "crc32": crc & 0xFFFFFFFF, "path": path.name}


def _copy_expert(reader: SafeTensorReader, name: str, expert: int, out: Path, *, max_bytes: int) -> dict[str, Any]:
    loc = reader.location(name)
    if not loc.shape or expert >= loc.shape[0]:
        raise ParityError(f"expert {expert} outside {name} shape {list(loc.shape)}")
    _, item_size = DTYPES[loc.dtype]
    nbytes = math.prod(loc.shape[1:]) * item_size
    if nbytes > max_bytes:
        raise ParityError(f"expert slice {name}[{expert}] is {nbytes} bytes, above limit {max_bytes}")
    tensor = reader.slice0(name, expert)
    payload = _tensor_bytes(tensor)
    path = out / _safe_name(f"{name}[{expert}]")
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"name": name, "kind": "axis0-slice", "axis0": expert, "dtype": loc.dtype,
            "shape": list(loc.shape[1:]), "bytes": len(payload),
            "crc32": zlib.crc32(payload) & 0xFFFFFFFF, "path": path.name}


def extract_parity_fixture(src: Path | str, out: Path | str, *, layers: Iterable[int],
                           experts: dict[int, list[int]] | None = None,
                           max_total_bytes: int = 8 << 30,
                           max_tensor_bytes: int = 2 << 30,
                           require_official: bool = True) -> dict[str, Any]:
    src = Path(src); out = Path(out)
    layer_set = set(int(x) for x in layers)
    if not layer_set or any(x < 0 for x in layer_set):
        raise ParityError("at least one nonnegative layer is required")
    release = inspect_release(src) if require_official else None
    if require_official and (not release.get("official_profile") or not release.get("text_payloads_ready")):
        raise ParityError("source is not a complete official Inkling-Small BF16 package")
    reader = SafeTensorReader(src)
    out.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    total = 0

    def add(entry: dict[str, Any]) -> None:
        nonlocal total
        total += int(entry["bytes"])
        if total > max_total_bytes:
            raise ParityError(f"fixture exceeds total byte limit {max_total_bytes}")
        entries.append(entry)

    for name in _global_tensor_names(reader) + _layer_tensor_names(reader, layer_set):
        # Vocabulary tables are intentionally not copied whole.  The caller can
        # add selected rows as activation inputs; copying either table would make
        # a supposedly bounded fixture several GiB.
        if name in {"model.llm.embed.weight", "model.llm.unembed.weight"}:
            continue
        add(_copy_tensor(reader, name, out, max_bytes=max_tensor_bytes))

    experts = experts or {}
    for layer, ids in sorted(experts.items()):
        if layer not in layer_set:
            raise ParityError(f"expert selection names unselected layer {layer}")
        for suffix in ("w13_weight", "w2_weight"):
            name = f"model.llm.layers.{layer}.mlp.experts.{suffix}"
            for expert in ids:
                add(_copy_expert(reader, name, expert, out, max_bytes=max_tensor_bytes))

    config_sha = hashlib.sha256((src / "config.json").read_bytes()).hexdigest()
    index_sha = hashlib.sha256((src / "model.safetensors.index.json").read_bytes()).hexdigest()
    manifest = {
        "format": "inkling-parity-fixture",
        "version": 1,
        "model_id": "thinkingmachines/Inkling-Small" if require_official else "synthetic",
        "layers": sorted(layer_set),
        "experts": {str(k): v for k, v in sorted(experts.items())},
        "source": {"config_sha256": config_sha, "index_sha256": index_sha},
        "total_payload_bytes": total,
        "reader_payload_bytes": reader.bytes_read,
        "entries": entries,
        "notes": [
            "embedding and unembedding tables are not copied whole",
            "routed expert entries are bounded axis-0 slices",
            "activations.json is a separate runtime/reference interchange format",
        ],
    }
    _atomic_json(out / "fixture.json", manifest)
    return manifest



def compare_activation_archives(reference: Path | str, candidate: Path | str, *,
                                atol: float = 1e-5, rtol: float = 1e-5) -> dict[str, Any]:
    if atol < 0 or rtol < 0:
        raise ParityError("tolerances must be nonnegative")
    ref, ref_meta = read_activation_archive(reference)
    got, got_meta = read_activation_archive(candidate)
    names = sorted(set(ref) | set(got))
    results = []
    passed = True
    for name in names:
        if name not in ref or name not in got:
            results.append({"name": name, "status": "missing",
                            "reference": name in ref, "candidate": name in got})
            passed = False
            continue
        a, b = ref[name], got[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            results.append({"name": name, "status": "shape-or-dtype",
                            "reference_shape": list(a.shape), "candidate_shape": list(b.shape),
                            "reference_dtype": str(a.dtype), "candidate_dtype": str(b.dtype)})
            passed = False
            continue
        if a.is_floating_point():
            delta = (a - b).abs()
            max_abs = float(delta.max()) if delta.numel() else 0.0
            denom = a.abs().clamp_min(1e-30)
            max_rel = float((delta / denom).max()) if delta.numel() else 0.0
            ok = bool(torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))
            results.append({"name": name, "status": "pass" if ok else "mismatch",
                            "max_abs": max_abs, "max_rel": max_rel,
                            "atol": atol, "rtol": rtol})
        else:
            mismatches = int((a != b).sum())
            ok = mismatches == 0
            results.append({"name": name, "status": "pass" if ok else "mismatch",
                            "mismatches": mismatches, "elements": a.numel()})
        passed = passed and ok
    return {"format": "inkling-parity-report", "version": 1, "passed": passed,
            "reference_metadata": ref_meta, "candidate_metadata": got_meta,
            "results": results}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-reference")
    ap.add_argument("--compare-candidate")
    ap.add_argument("--report")
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--layers", required=True, help="comma-separated decoder layer ids")
    ap.add_argument("--experts", default="", help="selected expert slices as L:e,e;L:e")
    ap.add_argument("--max-total-gib", type=float, default=8.0)
    ap.add_argument("--max-tensor-gib", type=float, default=2.0)
    args = ap.parse_args()
    if args.compare_reference or args.compare_candidate:
        if not args.compare_reference or not args.compare_candidate:
            ap.error("--compare-reference and --compare-candidate are required together")
        try:
            report = compare_activation_archives(args.compare_reference, args.compare_candidate,
                                                 atol=args.atol, rtol=args.rtol)
        except ParityError as exc:
            ap.error(str(exc))
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report:
            Path(args.report).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0 if report["passed"] else 2
    if not args.src or not args.out:
        ap.error("--src and --out are required for fixture extraction")
    try:
        layers = [int(x) for x in args.layers.split(",") if x != ""]
        manifest = extract_parity_fixture(
            args.src, args.out, layers=layers, experts=parse_expert_selection(args.experts),
            max_total_bytes=int(args.max_total_gib * (1 << 30)),
            max_tensor_bytes=int(args.max_tensor_gib * (1 << 30)),
        )
    except (ParityError, WeightError, ValueError) as exc:
        ap.error(str(exc))
    print(f"wrote bounded parity fixture with {len(manifest['entries'])} entries and "
          f"{manifest['total_payload_bytes']} payload bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
