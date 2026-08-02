#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Direct Inkling routed-expert conversion to WASTE VQ records.

This module is the first converter path that writes the final WASTE expert
record and codebook layouts.  It deliberately writes one layer at a time and
publishes only ``vq-stage.json``; ``manifest.json`` remains gated on public
runtime integration and official-weight parity.

Memory is bounded to:

* one source expert (gate/up/down);
* a fixed reservoir of normalized 8-value training vectors per matrix kind;
* one quantization distance chunk; and
* the codebooks for one layer.

The output is compatible with ``waste_expert_hdr`` and
``waste_codebook_hdr`` from ``src/waste_format.h``.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from inspect_inkling import InspectError
from inkling_plan import PlanError, build_plan, write_json_atomic
from inkling_weights import ExpertWeights, ProviderRawInklingWeights, WeightError


class VQError(RuntimeError):
    pass


MAGIC_EXPERT = 0x50584557      # WEXP
MAGIC_CODEBOOK = 0x4B424357    # WCBK
ALIGN = 4096
FMT_VQ3R = 4
FMT_VQ2R = 5
EXPERT_HEADER = struct.Struct("<IHHBBHHHIIIIIIII")
CODEBOOK_HEADER = struct.Struct("<IHBBII")
assert EXPERT_HEADER.size == 48
assert CODEBOOK_HEADER.size == 16

KIND_NAMES = ("gate", "up", "down")


@dataclass(frozen=True)
class VQSpec:
    stages: int = 3
    vec_dim: int = 8
    entries: int = 256
    index_block: int = 64

    def validate(self) -> None:
        if self.stages not in (2, 3):
            raise VQError("WASTE expert VQ stages must be 2 or 3")
        if self.vec_dim <= 0 or self.vec_dim > 64:
            raise VQError("invalid VQ vector dimension")
        if self.entries <= 1 or self.entries > 256:
            raise VQError("VQ entries must be in 2..256")
        if self.index_block <= 0 or self.index_block > 4096:
            raise VQError("invalid VQ index block")

    @property
    def fmt(self) -> int:
        return FMT_VQ3R if self.stages == 3 else FMT_VQ2R


@dataclass
class QuantizedMatrix:
    indices: torch.Tensor  # uint8 [stages, rows*cols/vec_dim]
    scales: torch.Tensor   # float16 [rows]
    shape: tuple[int, int]


@dataclass
class ParsedRecord:
    layer: int
    expert: int
    fmt: int
    codebook_base: int
    record_bytes: int
    offsets: tuple[int, int, int]
    correction_off: int
    crc32: int


def _align(value: int, alignment: int = ALIGN) -> int:
    if value < 0 or alignment <= 0:
        raise VQError("invalid alignment request")
    return (value + alignment - 1) // alignment * alignment


def _raw(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _source_hashes(src: Path) -> tuple[str, str]:
    return _sha256(src / "config.json"), _sha256(src / "model.safetensors.index.json")


def _assign_nearest(x: torch.Tensor, centroids: torch.Tensor, chunk: int) -> torch.Tensor:
    if x.ndim != 2 or centroids.ndim != 2 or x.shape[1] != centroids.shape[1]:
        raise VQError("invalid nearest-centroid shapes")
    if chunk <= 0:
        raise VQError("assignment chunk must be positive")
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    cn = (centroids * centroids).sum(1)
    for start in range(0, x.shape[0], chunk):
        part = x[start : start + chunk]
        distances = (part * part).sum(1, keepdim=True) + cn.unsqueeze(0)
        distances.addmm_(part, centroids.T, beta=1.0, alpha=-2.0)
        out[start : start + part.shape[0]] = distances.argmin(1)
    return out


def train_codebooks(
    vectors: torch.Tensor,
    spec: VQSpec,
    *,
    device: str | torch.device = "cpu",
    iterations: int = 10,
    assign_chunk: int = 32768,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Train residual codebooks on normalized vectors."""
    spec.validate()
    if vectors.ndim != 2 or vectors.shape[1] != spec.vec_dim:
        raise VQError(f"training vectors must be [n,{spec.vec_dim}]")
    if vectors.shape[0] < spec.entries:
        raise VQError(
            f"need at least {spec.entries} training vectors, received {vectors.shape[0]}"
        )
    if iterations < 1:
        raise VQError("k-means iterations must be positive")
    dev = torch.device(device)
    residual = vectors.to(device=dev, dtype=torch.float32).contiguous()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    books: list[torch.Tensor] = []
    for stage in range(spec.stages):
        perm = torch.randperm(residual.shape[0], generator=generator)[: spec.entries]
        centroids = residual[perm.to(residual.device)].clone()
        for _ in range(iterations):
            assignment = _assign_nearest(residual, centroids, assign_chunk)
            sums = torch.zeros_like(centroids)
            sums.index_add_(0, assignment, residual)
            counts = torch.bincount(assignment, minlength=spec.entries).to(residual.dtype)
            live = counts > 0
            centroids[live] = sums[live] / counts[live, None]
        books.append(centroids.detach().cpu().contiguous())
        assignment = _assign_nearest(residual, centroids, assign_chunk)
        residual = residual - centroids[assignment]
    return books


def _normalized_vectors(matrix: torch.Tensor, vec_dim: int) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[1] % vec_dim:
        raise VQError(
            f"matrix shape {list(matrix.shape)} is incompatible with vector dimension {vec_dim}"
        )
    matrix = matrix.float().contiguous()
    scale = matrix.abs().amax(1, keepdim=True).clamp_min(1e-8)
    return (matrix / scale).reshape(-1, vec_dim)


def collect_training_vectors(
    source: ProviderRawInklingWeights,
    layer: int,
    expert_ids: Sequence[int],
    spec: VQSpec,
    *,
    train_vectors: int,
    seed: int = 1234,
) -> dict[str, torch.Tensor]:
    """Read every sampled expert once and build bounded per-kind reservoirs."""
    spec.validate()
    if not expert_ids:
        raise VQError("at least one codebook-sample expert is required")
    if train_vectors < spec.entries:
        raise VQError("train_vectors must be at least the codebook entry count")
    per = max(1, math.ceil(train_vectors / len(expert_ids)))
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in KIND_NAMES}
    for expert in expert_ids:
        weights = source.routed_expert(layer, expert, float32=True)
        for kind, matrix in zip(KIND_NAMES, (weights.gate, weights.up, weights.down)):
            vectors = _normalized_vectors(matrix, spec.vec_dim)
            count = min(per, vectors.shape[0])
            generator = torch.Generator(device="cpu").manual_seed(
                seed + layer * 100003 + expert * 97 + KIND_NAMES.index(kind) * 13
            )
            selected = torch.randperm(vectors.shape[0], generator=generator)[:count]
            chunks[kind].append(vectors[selected].cpu())
    output: dict[str, torch.Tensor] = {}
    for kind in KIND_NAMES:
        values = torch.cat(chunks[kind], dim=0)
        if values.shape[0] < spec.entries:
            raise VQError(
                f"{kind} reservoir has {values.shape[0]} vectors; need {spec.entries}"
            )
        output[kind] = values[:train_vectors].contiguous()
    return output


_NATIVE_VQ: tuple[Any, Any] | bool | None = None


def _native_vq() -> tuple[Any, Any] | None:
    global _NATIVE_VQ
    if _NATIVE_VQ is False:
        return None
    if _NATIVE_VQ is not None:
        return _NATIVE_VQ
    root = Path(__file__).resolve().parents[1]
    for name in ("libwastevq.dll", "libwastevq.so", "libwastevq.dylib"):
        path = root / name
        if not path.exists():
            continue
        lib = ctypes.CDLL(str(path))
        fp = ctypes.POINTER(ctypes.c_float)
        lib.waste_vq_encode.argtypes = [
            fp,
            ctypes.c_int,
            fp,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
        ]
        lib.waste_vq_encode.restype = None
        _NATIVE_VQ = (lib, ctypes)
        return _NATIVE_VQ
    _NATIVE_VQ = False
    return None


def quantize_matrix(
    matrix: torch.Tensor,
    books: Sequence[torch.Tensor],
    spec: VQSpec,
    *,
    device: str | torch.device = "cpu",
    assign_chunk: int = 32768,
    native: bool = True,
) -> QuantizedMatrix:
    spec.validate()
    if len(books) != spec.stages:
        raise VQError("codebook stage count mismatch")
    if matrix.ndim != 2 or matrix.shape[1] % spec.vec_dim:
        raise VQError("expert matrix is not vector-dimension aligned")
    rows, cols = map(int, matrix.shape)
    work = matrix.float().contiguous()
    scales = work.abs().amax(1, keepdim=True).clamp_min(1e-8)
    vectors = (work / scales).reshape(-1, spec.vec_dim).contiguous()

    native_lib = _native_vq() if native and torch.device(device).type == "cpu" else None
    if native_lib is not None:
        lib, cmod = native_lib
        stacked = torch.stack([book.float().cpu() for book in books]).contiguous()
        output = torch.empty(vectors.shape[0] * spec.stages, dtype=torch.uint8)
        fp = cmod.POINTER(cmod.c_float)
        lib.waste_vq_encode(
            cmod.cast(vectors.data_ptr(), fp),
            vectors.shape[0],
            cmod.cast(stacked.data_ptr(), fp),
            spec.stages,
            spec.entries,
            spec.vec_dim,
            cmod.cast(output.data_ptr(), cmod.POINTER(cmod.c_uint8)),
            0,
        )
        indices = output.view(vectors.shape[0], spec.stages).T.contiguous()
    else:
        dev = torch.device(device)
        residual = vectors.to(dev)
        encoded: list[torch.Tensor] = []
        for book in books:
            centroids = book.to(device=dev, dtype=torch.float32)
            idx = _assign_nearest(residual, centroids, assign_chunk)
            encoded.append(idx.to(torch.uint8).cpu())
            residual = residual - centroids[idx]
        indices = torch.stack(encoded).contiguous()
    return QuantizedMatrix(indices, scales.flatten().half().cpu(), (rows, cols))


def block_indices(indices: torch.Tensor, shape: tuple[int, int], spec: VQSpec) -> torch.Tensor:
    rows, cols = shape
    if indices.dtype != torch.uint8 or list(indices.shape) != [spec.stages, rows * cols // spec.vec_dim]:
        raise VQError("invalid VQ index tensor")
    vectors_per_row = cols // spec.vec_dim
    tensor = indices.view(spec.stages, rows, vectors_per_row).permute(1, 2, 0)
    pad = (-rows) % spec.index_block
    if pad:
        tensor = torch.cat(
            [
                tensor,
                torch.zeros(
                    pad,
                    vectors_per_row,
                    spec.stages,
                    dtype=torch.uint8,
                ),
            ],
            dim=0,
        )
    blocks = tensor.shape[0] // spec.index_block
    return tensor.view(blocks, spec.index_block, vectors_per_row, spec.stages).permute(
        0, 2, 1, 3
    ).contiguous()


def unblock_indices(data: bytes, shape: tuple[int, int], spec: VQSpec) -> torch.Tensor:
    rows, cols = shape
    vectors_per_row = cols // spec.vec_dim
    padded_rows = _align(rows, spec.index_block)
    expected = padded_rows * vectors_per_row * spec.stages
    if len(data) != expected:
        raise VQError(f"blocked index payload is {len(data)} bytes; expected {expected}")
    blocks = padded_rows // spec.index_block
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(
        blocks, vectors_per_row, spec.index_block, spec.stages
    )
    return (
        raw.permute(0, 2, 1, 3)
        .reshape(padded_rows, vectors_per_row, spec.stages)[:rows]
        .permute(2, 0, 1)
        .reshape(spec.stages, rows * vectors_per_row)
        .contiguous()
    )


def write_codebooks(
    path: Path,
    base: int,
    books: dict[str, Sequence[torch.Tensor]],
    spec: VQSpec,
) -> dict[str, Any]:
    spec.validate()
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as f:
            for kind_index, kind in enumerate(KIND_NAMES):
                values = books.get(kind)
                if values is None or len(values) != spec.stages:
                    raise VQError(f"missing {kind} codebooks")
                for stage, book in enumerate(values):
                    if list(book.shape) != [spec.entries, spec.vec_dim]:
                        raise VQError(f"invalid {kind} codebook shape {list(book.shape)}")
                    codebook_id = base + kind_index * spec.stages + stage
                    if codebook_id > 0xFFFF:
                        raise VQError("codebook id exceeds WASTE u16 field")
                    f.write(
                        CODEBOOK_HEADER.pack(
                            MAGIC_CODEBOOK,
                            codebook_id,
                            spec.fmt,
                            spec.vec_dim,
                            spec.entries,
                            0,
                        )
                    )
                    f.write(_raw(book.half()))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {
        "file": path.name,
        "base": base,
        "records": len(KIND_NAMES) * spec.stages,
        "bytes": path.stat().st_size,
    }


def _record_body(
    matrices: Sequence[QuantizedMatrix], spec: VQSpec
) -> tuple[bytes, tuple[int, int, int], int]:
    if len(matrices) != 3:
        raise VQError("expert record requires gate, up, and down")
    offset = EXPERT_HEADER.size
    offsets: list[int] = []
    body = bytearray()
    for matrix in matrices:
        offsets.append(offset)
        payload = _raw(block_indices(matrix.indices, matrix.shape, spec))
        body.extend(payload)
        offset += len(payload)
    correction_off = offset
    for matrix in matrices:
        payload = _raw(matrix.scales)
        body.extend(payload)
        offset += len(payload)
    return bytes(body), (offsets[0], offsets[1], offsets[2]), correction_off


def write_expert_record(
    f: Any,
    layer: int,
    expert: int,
    codebook_base: int,
    matrices: Sequence[QuantizedMatrix],
    spec: VQSpec,
) -> int:
    body, offsets, correction_off = _record_body(matrices, spec)
    total = EXPERT_HEADER.size + len(body)
    record_bytes = _align(total)
    blocks = record_bytes // ALIGN
    header = EXPERT_HEADER.pack(
        MAGIC_EXPERT,
        layer,
        expert,
        spec.fmt,
        0,
        codebook_base,
        0,
        0,
        blocks,
        offsets[0],
        offsets[1],
        offsets[2],
        correction_off,
        zlib.crc32(body) & 0xFFFFFFFF,
        0,
        0,
    )
    f.write(header)
    f.write(body)
    f.write(b"\0" * (record_bytes - total))
    return record_bytes


def parse_record_header(data: bytes) -> ParsedRecord:
    if len(data) < EXPERT_HEADER.size:
        raise VQError("short WEXP header")
    (
        magic,
        layer,
        expert,
        fmt,
        flags,
        codebook_base,
        lowrank,
        reserved0,
        blocks,
        gate_off,
        up_off,
        down_off,
        correction_off,
        crc32,
        reserved1,
        reserved2,
    ) = EXPERT_HEADER.unpack_from(data)
    if magic != MAGIC_EXPERT or fmt not in (FMT_VQ2R, FMT_VQ3R) or flags != 0:
        raise VQError("invalid WEXP record identity")
    if lowrank or reserved0 or reserved1 or reserved2 or blocks <= 0:
        raise VQError("unsupported or nonzero WEXP reserved fields")
    record_bytes = blocks * ALIGN
    if not (
        EXPERT_HEADER.size <= gate_off < up_off < down_off < correction_off <= record_bytes
    ):
        raise VQError("invalid WEXP payload offsets")
    return ParsedRecord(
        layer,
        expert,
        fmt,
        codebook_base,
        record_bytes,
        (gate_off, up_off, down_off),
        correction_off,
        crc32,
    )


def read_codebooks(path: Path, spec: VQSpec) -> dict[int, torch.Tensor]:
    spec.validate()
    record_bytes = CODEBOOK_HEADER.size + spec.entries * spec.vec_dim * 2
    raw = path.read_bytes()
    if len(raw) % record_bytes:
        raise VQError("codebook file ends inside a record")
    output: dict[int, torch.Tensor] = {}
    for offset in range(0, len(raw), record_bytes):
        magic, codebook_id, fmt, vec_dim, entries, reserved = CODEBOOK_HEADER.unpack_from(raw, offset)
        if (
            magic != MAGIC_CODEBOOK
            or fmt != spec.fmt
            or vec_dim != spec.vec_dim
            or entries != spec.entries
            or reserved != 0
            or codebook_id in output
        ):
            raise VQError("invalid WASTE codebook record")
        start = offset + CODEBOOK_HEADER.size
        payload = raw[start : start + spec.entries * spec.vec_dim * 2]
        output[codebook_id] = torch.frombuffer(
            bytearray(payload), dtype=torch.float16
        ).reshape(spec.entries, spec.vec_dim).float()
    return output


def _index_payload_bytes(shape: tuple[int, int], spec: VQSpec) -> int:
    rows, cols = shape
    if rows <= 0 or cols <= 0 or cols % spec.vec_dim:
        raise VQError("invalid matrix shape for WEXP geometry")
    return _align(rows, spec.index_block) * (cols // spec.vec_dim) * spec.stages


def record_geometry(hidden: int, intermediate: int, spec: VQSpec) -> dict[str, int]:
    """Exact final WEXP geometry for one Inkling routed expert."""
    spec.validate()
    shapes = ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))
    index_bytes = [_index_payload_bytes(shape, spec) for shape in shapes]
    scale_bytes = (2 * intermediate + hidden) * 2
    body_bytes = sum(index_bytes) + scale_bytes
    return {
        "gate_index_bytes": index_bytes[0],
        "up_index_bytes": index_bytes[1],
        "down_index_bytes": index_bytes[2],
        "scale_bytes": scale_bytes,
        "body_bytes": body_bytes,
        "record_bytes": _align(EXPERT_HEADER.size + body_bytes),
    }


def dequantize_record(
    record: bytes,
    books: dict[int, torch.Tensor],
    shapes: Sequence[tuple[int, int]],
    spec: VQSpec,
    *,
    expected_layer: int | None = None,
    expected_expert: int | None = None,
    verify_crc: bool = True,
) -> ExpertWeights:
    header = parse_record_header(record)
    if len(record) != header.record_bytes:
        raise VQError("WEXP record byte length differs from header")
    if expected_layer is not None and header.layer != expected_layer:
        raise VQError("WEXP layer identity mismatch")
    if expected_expert is not None and header.expert != expected_expert:
        raise VQError("WEXP expert identity mismatch")
    if header.fmt != spec.fmt:
        raise VQError("WEXP format differs from requested VQ spec")
    scale_end = header.correction_off + sum(shape[0] for shape in shapes) * 2
    if scale_end > len(record):
        raise VQError("WEXP correction payload exceeds record")
    if verify_crc and zlib.crc32(record[EXPERT_HEADER.size:scale_end]) & 0xFFFFFFFF != header.crc32:
        raise VQError("WEXP payload CRC mismatch")

    results: list[torch.Tensor] = []
    scale_cursor = header.correction_off
    for kind, shape in enumerate(shapes):
        begin = header.offsets[kind]
        end = header.offsets[kind + 1] if kind < 2 else header.correction_off
        expected_indices = _index_payload_bytes(shape, spec)
        if end - begin != expected_indices:
            raise VQError("WEXP index payload size differs from shape")
        indices = unblock_indices(record[begin:end], shape, spec).view(
            spec.stages, shape[0], shape[1] // spec.vec_dim
        )
        rows = shape[0]
        scales = torch.frombuffer(
            bytearray(record[scale_cursor : scale_cursor + rows * 2]), dtype=torch.float16
        ).float()
        scale_cursor += rows * 2
        reconstructed = torch.zeros(
            shape[0], shape[1] // spec.vec_dim, spec.vec_dim, dtype=torch.float32
        )
        for stage in range(spec.stages):
            book_id = header.codebook_base + kind * spec.stages + stage
            book = books.get(book_id)
            if book is None or list(book.shape) != [spec.entries, spec.vec_dim]:
                raise VQError(f"missing codebook {book_id}")
            reconstructed += book[indices[stage].long()]
        results.append(reconstructed.reshape(shape) * scales[:, None])
    return ExpertWeights(gate=results[0], up=results[1], down=results[2])


def verify_layer_outputs(
    bank_path: Path,
    codebook_path: Path,
    *,
    layer: int,
    experts: int,
    hidden: int,
    intermediate: int,
    spec: VQSpec,
    codebook_base: int,
    verify_crc: bool = True,
) -> dict[str, Any]:
    books = read_codebooks(codebook_path, spec)
    expected_ids = set(range(codebook_base, codebook_base + 3 * spec.stages))
    if set(books) != expected_ids:
        raise VQError("layer codebook ids are not dense at their declared base")
    raw_size = bank_path.stat().st_size
    with bank_path.open("rb") as f:
        first = f.read(EXPERT_HEADER.size)
        header = parse_record_header(first)
        if header.layer != layer or header.expert != 0 or header.codebook_base != codebook_base:
            raise VQError("first WEXP record identity mismatch")
        if raw_size != header.record_bytes * experts:
            raise VQError("expert bank size differs from record count")
        shapes = ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))
        for expert in range(experts):
            f.seek(expert * header.record_bytes)
            record = f.read(header.record_bytes)
            dequantize_record(
                record,
                books,
                shapes,
                spec,
                expected_layer=layer,
                expected_expert=expert,
                verify_crc=verify_crc,
            )
    return {
        "record_bytes": header.record_bytes,
        "bytes": raw_size,
        "experts": experts,
        "codebooks": len(books),
    }


def _sample_ids(n_experts: int, count: int) -> list[int]:
    if count <= 0:
        raise VQError("codebook sample count must be positive")
    count = min(count, n_experts)
    if count == 1:
        return [0]
    return sorted({round(i * (n_experts - 1) / (count - 1)) for i in range(count)})


def quantize_layer(
    source: ProviderRawInklingWeights,
    out: Path,
    layer: int,
    *,
    spec: VQSpec = VQSpec(),
    expert_limit: int | None = None,
    codebook_sample: int = 12,
    train_vectors: int = 300000,
    kmeans_iterations: int = 10,
    device: str = "cpu",
    assign_chunk: int = 32768,
    verify: bool = True,
) -> dict[str, Any]:
    spec.validate()
    if layer < source.dense_count or layer >= source.n_layers:
        raise VQError(f"layer {layer} is not a sparse Inkling layer")
    count = source.n_experts if expert_limit is None else min(expert_limit, source.n_experts)
    if count <= 0:
        raise VQError("expert limit must select at least one expert")
    sample_pool = count if expert_limit is not None else source.n_experts
    sample_ids = _sample_ids(sample_pool, codebook_sample)
    vectors = collect_training_vectors(
        source, layer, sample_ids, spec, train_vectors=train_vectors
    )
    books = {
        kind: train_codebooks(
            vectors[kind],
            spec,
            device=device,
            iterations=kmeans_iterations,
            assign_chunk=assign_chunk,
            seed=layer * 17 + kind_index,
        )
        for kind_index, kind in enumerate(KIND_NAMES)
    }

    codebook_base = (layer - source.dense_count) * 3 * spec.stages
    bank_path = out / f"experts-L{layer}.bin"
    codebook_path = out / f"codebooks-L{layer}.bin"
    out.mkdir(parents=True, exist_ok=True)
    codebook_meta = write_codebooks(codebook_path, codebook_base, books, spec)
    tmp = bank_path.with_name(bank_path.name + ".tmp")
    record_bytes = 0
    try:
        with tmp.open("wb") as f:
            for expert in range(count):
                weights = source.routed_expert(layer, expert, float32=True)
                matrices = [
                    quantize_matrix(
                        matrix,
                        books[kind],
                        spec,
                        device=device,
                        assign_chunk=assign_chunk,
                    )
                    for kind, matrix in zip(
                        KIND_NAMES, (weights.gate, weights.up, weights.down)
                    )
                ]
                this_record = write_expert_record(
                    f, layer, expert, codebook_base, matrices, spec
                )
                if record_bytes and this_record != record_bytes:
                    raise VQError("expert record size changed within a layer")
                record_bytes = this_record
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, bank_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        bank_path.unlink(missing_ok=True)
        codebook_path.unlink(missing_ok=True)
        raise

    if verify:
        try:
            verify_layer_outputs(
                bank_path,
                codebook_path,
                layer=layer,
                experts=count,
                hidden=source.hidden,
                intermediate=source.intermediate,
                spec=spec,
                codebook_base=codebook_base,
            )
        except BaseException:
            bank_path.unlink(missing_ok=True)
            codebook_path.unlink(missing_ok=True)
            raise
    return {
        "schema": "waste.inkling-vq-layer.v1",
        "layer": layer,
        "file": bank_path.name,
        "codebooks_file": codebook_path.name,
        "experts": count,
        "model_experts": source.n_experts,
        "complete": count == source.n_experts,
        "hidden_size": source.hidden,
        "intermediate_size": source.intermediate,
        "record_bytes": record_bytes,
        "geometry": record_geometry(source.hidden, source.intermediate, spec),
        "bytes": bank_path.stat().st_size,
        "codebook_base": codebook_base,
        "codebooks": codebook_meta,
        "quant": {
            "fmt": "VQ3R" if spec.stages == 3 else "VQ2R",
            "stages": spec.stages,
            "vec_dim": spec.vec_dim,
            "entries": spec.entries,
            "index_block": spec.index_block,
            "bits_per_weight": float(spec.stages),
        },
        "training": {
            "algorithm": "residual-lloyd-v1",
            "codebook_sample": codebook_sample,
            "requested_train_vectors": train_vectors,
            "sample_experts": sample_ids,
            "vectors_per_kind": {kind: int(vectors[kind].shape[0]) for kind in KIND_NAMES},
            "iterations": kmeans_iterations,
            "device": str(device),
        },
    }


def _sidecar_matches(
    meta: dict[str, Any],
    *,
    layer: int,
    source: ProviderRawInklingWeights,
    spec: VQSpec,
    count: int,
    config_sha: str,
    index_sha: str,
    codebook_sample: int,
    train_vectors: int,
    kmeans_iterations: int,
    device: str,
) -> bool:
    quant = meta.get("quant") if isinstance(meta.get("quant"), dict) else {}
    training = meta.get("training") if isinstance(meta.get("training"), dict) else {}
    return (
        meta.get("schema") == "waste.inkling-vq-layer.v1"
        and meta.get("layer") == layer
        and meta.get("experts") == count
        and meta.get("model_experts") == source.n_experts
        and meta.get("source_config_sha256") == config_sha
        and meta.get("source_index_sha256") == index_sha
        and quant.get("stages") == spec.stages
        and quant.get("vec_dim") == spec.vec_dim
        and quant.get("entries") == spec.entries
        and quant.get("index_block") == spec.index_block
        and training.get("algorithm") == "residual-lloyd-v1"
        and training.get("codebook_sample") == codebook_sample
        and training.get("requested_train_vectors") == train_vectors
        and training.get("iterations") == kmeans_iterations
        and training.get("device") == str(device)
    )


def quantize_expert_banks(
    src: Path,
    out: Path,
    *,
    layers: Iterable[int] | None = None,
    spec: VQSpec = VQSpec(),
    expert_limit: int | None = None,
    codebook_sample: int = 12,
    train_vectors: int = 300000,
    kmeans_iterations: int = 10,
    device: str = "cpu",
    assign_chunk: int = 32768,
    verify: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    try:
        plan = build_plan(src, require_payloads=True)
        source = ProviderRawInklingWeights(src)
    except (InspectError, PlanError, WeightError) as exc:
        raise VQError(str(exc)) from exc
    if plan.get("source_dialect") != "provider_raw":
        raise VQError("direct VQ conversion accepts the official provider-raw checkpoint only")
    selected = list(layers) if layers is not None else list(range(source.dense_count, source.n_layers))
    if not selected or len(set(selected)) != len(selected):
        raise VQError("layers must be a nonempty unique list")
    for layer in selected:
        if layer < source.dense_count or layer >= source.n_layers:
            raise VQError(f"layer {layer} is not sparse")
    count = source.n_experts if expert_limit is None else min(expert_limit, source.n_experts)
    config_sha, index_sha = _source_hashes(src)
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for layer in selected:
        sidecar = out / f"experts-L{layer}.bin.json"
        bank = out / f"experts-L{layer}.bin"
        codebooks = out / f"codebooks-L{layer}.bin"
        reused = False
        if resume:
            try:
                old = json.loads(sidecar.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                old = None
            if isinstance(old, dict) and _sidecar_matches(
                old,
                layer=layer,
                source=source,
                spec=spec,
                count=count,
                config_sha=config_sha,
                index_sha=index_sha,
                codebook_sample=codebook_sample,
                train_vectors=train_vectors,
                kmeans_iterations=kmeans_iterations,
                device=device,
            ):
                try:
                    verify_layer_outputs(
                        bank,
                        codebooks,
                        layer=layer,
                        experts=count,
                        hidden=source.hidden,
                        intermediate=source.intermediate,
                        spec=spec,
                        codebook_base=int(old["codebook_base"]),
                        verify_crc=verify,
                    )
                    meta = old
                    reused = True
                except (OSError, KeyError, TypeError, ValueError, VQError):
                    reused = False
        if not reused:
            try:
                meta = quantize_layer(
                    source,
                    out,
                    layer,
                    spec=spec,
                    expert_limit=expert_limit,
                    codebook_sample=codebook_sample,
                    train_vectors=train_vectors,
                    kmeans_iterations=kmeans_iterations,
                    device=device,
                    assign_chunk=assign_chunk,
                    verify=verify,
                )
            except WeightError as exc:
                raise VQError(str(exc)) from exc
            meta["source_config_sha256"] = config_sha
            meta["source_index_sha256"] = index_sha
            write_json_atomic(sidecar, meta)
        result = dict(meta)
        result["reused"] = reused
        results.append(result)
    stage = {
        "schema": "waste.inkling-vq-stage.v1",
        "architecture": "inkling",
        "official_small": bool(plan.get("official_small")),
        "source_config_sha256": config_sha,
        "source_index_sha256": index_sha,
        "layers": results,
        "totals": {
            "bank_bytes": sum(int(item.get("bytes", 0)) for item in results),
            "codebook_bytes": sum(int(item.get("codebooks", {}).get("bytes", 0)) for item in results),
        },
        "complete": all(item.get("complete") for item in results),
        "manifest_published": False,
    }
    write_json_atomic(out / "vq-stage.json", stage)
    return stage
