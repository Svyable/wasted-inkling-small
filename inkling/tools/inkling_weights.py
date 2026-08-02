#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Bounded safetensors access and official Inkling weight transforms.

The reader understands native BF16/F16/F32 safetensors and can read a contiguous
slice along tensor axis 0 without materializing the whole tensor.  Inkling's
routed expert tensors are expert-major, so this is the primitive the converter
needs to keep memory bounded to one expert.

The interleave operation is a direct, independently implemented equivalent of
the official Apache-2.0 Transformers checkpoint conversion operation.  It is
kept here rather than importing Transformers so the conversion tool remains
small and deterministic.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch


class WeightError(RuntimeError):
    pass


DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
}


@dataclass(frozen=True)
class TensorLocation:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    file_offset: int
    nbytes: int


class SafeTensorReader:
    """Lazy sharded safetensors reader with bounded axis-0 slices."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        index_path = self.root / "model.safetensors.index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise WeightError(f"missing {index_path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise WeightError(f"cannot read {index_path}: {exc}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or any(
            not isinstance(name, str) or not isinstance(shard, str)
            for name, shard in weight_map.items()
        ):
            raise WeightError(f"invalid weight_map in {index_path}")
        self.weight_map: dict[str, str] = dict(weight_map)
        self._headers: dict[str, tuple[dict[str, Any], int]] = {}
        self.bytes_read = 0

    def names(self) -> Iterator[str]:
        return iter(self.weight_map)

    def _header(self, shard: str) -> tuple[dict[str, Any], int]:
        if shard in self._headers:
            return self._headers[shard]
        path = self.root / shard
        try:
            with path.open("rb") as f:
                raw = f.read(8)
                if len(raw) != 8:
                    raise WeightError(f"short safetensors prefix: {path}")
                (header_len,) = struct.unpack("<Q", raw)
                if header_len < 2 or header_len > (256 << 20):
                    raise WeightError(f"implausible safetensors header length {header_len}: {path}")
                payload = f.read(header_len)
                if len(payload) != header_len:
                    raise WeightError(f"truncated safetensors header: {path}")
        except OSError as exc:
            raise WeightError(f"cannot read {path}: {exc}") from exc
        try:
            header = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise WeightError(f"invalid safetensors header JSON in {path}: {exc}") from exc
        if not isinstance(header, dict):
            raise WeightError(f"safetensors header is not an object: {path}")
        value = (header, 8 + header_len)
        self._headers[shard] = value
        return value

    def location(self, name: str) -> TensorLocation:
        shard = self.weight_map.get(name)
        if shard is None:
            raise WeightError(f"tensor not present in index: {name}")
        header, base = self._header(shard)
        item = header.get(name)
        if not isinstance(item, dict):
            raise WeightError(f"tensor not present in shard header: {name} ({shard})")
        dtype = item.get("dtype")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if dtype not in DTYPES:
            raise WeightError(f"unsupported source dtype {dtype!r} for {name}; BF16 parity source required")
        if not isinstance(shape, list) or any(not isinstance(x, int) or x < 0 for x in shape):
            raise WeightError(f"invalid shape for {name}: {shape!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(x, int) or x < 0 for x in offsets)
            or offsets[1] < offsets[0]
        ):
            raise WeightError(f"invalid data_offsets for {name}: {offsets!r}")
        _, item_size = DTYPES[dtype]
        expected = math.prod(shape) * item_size
        actual = offsets[1] - offsets[0]
        if actual != expected:
            raise WeightError(
                f"payload size mismatch for {name}: shape/dtype require {expected} bytes, header says {actual}"
            )
        return TensorLocation(
            name=name,
            shard=shard,
            dtype=dtype,
            shape=tuple(shape),
            file_offset=base + offsets[0],
            nbytes=actual,
        )

    def _read(self, loc: TensorLocation, offset: int, nbytes: int) -> bytes:
        path = self.root / loc.shard
        try:
            with path.open("rb") as f:
                f.seek(offset)
                data = f.read(nbytes)
        except OSError as exc:
            raise WeightError(f"cannot read {loc.name} from {path}: {exc}") from exc
        if len(data) != nbytes:
            raise WeightError(
                f"short read for {loc.name}: wanted {nbytes} bytes at {offset}, received {len(data)}"
            )
        self.bytes_read += nbytes
        return data

    @staticmethod
    def _decode(data: bytes, dtype_name: str, shape: tuple[int, ...]) -> torch.Tensor:
        dtype, _ = DTYPES[dtype_name]
        # bytearray gives torch owned, writable storage and avoids the warning
        # emitted for an immutable bytes buffer.
        return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)

    def iter_bytes(self, name: str, *, chunk_bytes: int = 8 << 20) -> Iterator[bytes]:
        """Yield one tensor's raw contiguous payload without materializing it."""
        if chunk_bytes <= 0:
            raise WeightError("chunk_bytes must be positive")
        loc = self.location(name)
        path = self.root / loc.shard
        remaining = loc.nbytes
        try:
            with path.open("rb") as f:
                f.seek(loc.file_offset)
                while remaining:
                    data = f.read(min(chunk_bytes, remaining))
                    if not data:
                        raise WeightError(
                            f"short read for {loc.name}: {remaining} payload bytes remain"
                        )
                    remaining -= len(data)
                    self.bytes_read += len(data)
                    yield data
        except OSError as exc:
            raise WeightError(f"cannot stream {loc.name} from {path}: {exc}") from exc

    def iter_row_chunks(
        self, name: str, *, chunk_bytes: int = 8 << 20
    ) -> Iterator[tuple[int, int, bytes]]:
        """Yield contiguous chunks aligned to rows of the tensor's last axis."""
        loc = self.location(name)
        if not loc.shape:
            raise WeightError(f"cannot row-stream scalar tensor {name}")
        _, item_size = DTYPES[loc.dtype]
        row_bytes = loc.shape[-1] * item_size
        if row_bytes <= 0:
            raise WeightError(f"cannot row-stream zero-width tensor {name}")
        rows = math.prod(loc.shape[:-1])
        rows_per_chunk = max(1, chunk_bytes // row_bytes)
        path = self.root / loc.shard
        try:
            with path.open("rb") as f:
                f.seek(loc.file_offset)
                start = 0
                while start < rows:
                    count = min(rows_per_chunk, rows - start)
                    need = count * row_bytes
                    data = f.read(need)
                    if len(data) != need:
                        raise WeightError(
                            f"short row read for {loc.name}: wanted {need}, received {len(data)}"
                        )
                    self.bytes_read += need
                    yield start, count, data
                    start += count
        except OSError as exc:
            raise WeightError(f"cannot row-stream {loc.name} from {path}: {exc}") from exc

    def tensor(self, name: str, *, float32: bool = False) -> torch.Tensor:
        loc = self.location(name)
        out = self._decode(self._read(loc, loc.file_offset, loc.nbytes), loc.dtype, loc.shape)
        return out.float() if float32 else out

    def slice0(
        self,
        name: str,
        start: int,
        count: int = 1,
        *,
        float32: bool = False,
    ) -> torch.Tensor:
        loc = self.location(name)
        if not loc.shape:
            raise WeightError(f"cannot axis-0 slice scalar tensor {name}")
        if start < 0 or count < 1 or start + count > loc.shape[0]:
            raise WeightError(
                f"axis-0 slice [{start}:{start + count}] outside {name} shape {list(loc.shape)}"
            )
        _, item_size = DTYPES[loc.dtype]
        stride_elems = math.prod(loc.shape[1:])
        stride_bytes = stride_elems * item_size
        shape = (count, *loc.shape[1:])
        data = self._read(loc, loc.file_offset + start * stride_bytes, count * stride_bytes)
        out = self._decode(data, loc.dtype, shape)
        if count == 1:
            out = out[0]
        return out.float() if float32 else out


def interleave(tensor: torch.Tensor, dim: int, *, inverse: bool = False) -> torch.Tensor:
    """Apply the official Inkling checkpoint Interleave operation exactly."""
    if dim < 0:
        dim += tensor.ndim
    if dim < 0 or dim >= tensor.ndim:
        raise WeightError(f"interleave dimension {dim} outside rank {tensor.ndim}")
    size = tensor.shape[dim]
    if size % 2:
        raise WeightError(f"interleave dimension must be even, got {size} in shape {list(tensor.shape)}")
    shape = list(tensor.shape)
    if inverse:
        shape[dim : dim + 1] = [2, size // 2]
    else:
        shape[dim : dim + 1] = [size // 2, 2]
    return tensor.reshape(shape).transpose(dim, dim + 1).reshape(tensor.shape).contiguous()


def split_fused_gate_up(tensor: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = interleave(tensor, dim)
    gate, up = torch.chunk(normalized, 2, dim=dim)
    return gate.contiguous(), up.contiguous()


@dataclass
class ExpertWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


class ProviderRawInklingWeights:
    """Official provider-raw Inkling text tensor adapter."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        try:
            config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WeightError(f"cannot read Inkling config: {exc}") from exc
        text = config.get("text_config")
        if not isinstance(text, dict):
            raise WeightError("Inkling config has no text_config object")
        self.config = config
        self.text = text
        self.reader = SafeTensorReader(self.root)
        self.n_layers = self._int("num_hidden_layers")
        self.n_experts = self._int("n_routed_experts")
        self.n_shared = self._int("n_shared_experts")
        self.hidden = self._int("hidden_size")
        self.intermediate = self._int("intermediate_size")
        self.dense_intermediate = self._int("dense_intermediate_size")
        self.dense_count = self._int("dense_mlp_idx", minimum=0)

    def _int(self, key: str, *, minimum: int = 1) -> int:
        value = self.text.get(key)
        if not isinstance(value, int) or value < minimum:
            raise WeightError(f"invalid or missing text_config.{key}: {value!r}")
        return value

    @staticmethod
    def _layer(layer: int) -> str:
        return f"model.llm.layers.{layer}"

    def dense(self, layer: int, *, float32: bool = True) -> ExpertWeights:
        if layer < 0 or layer >= self.dense_count:
            raise WeightError(f"layer {layer} is not a dense Inkling MLP layer")
        p = self._layer(layer)
        fused = self.reader.tensor(f"{p}.mlp.w13_dn.weight", float32=float32)
        down = self.reader.tensor(f"{p}.mlp.w2_md.weight", float32=float32)
        expected_fused = [2 * self.dense_intermediate, self.hidden]
        expected_down = [self.hidden, self.dense_intermediate]
        if list(fused.shape) != expected_fused or list(down.shape) != expected_down:
            raise WeightError(
                f"dense layer {layer} shapes differ from config: fused {list(fused.shape)} vs {expected_fused}, "
                f"down {list(down.shape)} vs {expected_down}"
            )
        gate, up = split_fused_gate_up(fused, 0)
        return ExpertWeights(gate=gate, up=up, down=down.contiguous())

    def routed_expert(self, layer: int, expert: int, *, float32: bool = True) -> ExpertWeights:
        if layer < self.dense_count or layer >= self.n_layers:
            raise WeightError(f"layer {layer} is not a sparse Inkling MLP layer")
        if expert < 0 or expert >= self.n_experts:
            raise WeightError(f"expert {expert} outside 0..{self.n_experts - 1}")
        p = self._layer(layer)
        # Full source tensors are [expert, fused/down rows, input].  Reading one
        # axis-0 slice bounds memory to one expert.  Interleave axis 1 on the
        # full tensor is equivalent to axis 0 after taking that slice.
        fused = self.reader.slice0(
            f"{p}.mlp.experts.w13_weight", expert, float32=float32
        )
        down = self.reader.slice0(
            f"{p}.mlp.experts.w2_weight", expert, float32=float32
        )
        expected_fused = [2 * self.intermediate, self.hidden]
        expected_down = [self.hidden, self.intermediate]
        if list(fused.shape) != expected_fused or list(down.shape) != expected_down:
            raise WeightError(
                f"expert {expert} layer {layer} shapes differ from config: "
                f"fused {list(fused.shape)} vs {expected_fused}, down {list(down.shape)} vs {expected_down}"
            )
        gate, up = split_fused_gate_up(fused, 0)
        return ExpertWeights(gate=gate, up=up, down=down.contiguous())

    def shared(self, layer: int, *, float32: bool = True) -> ExpertWeights:
        if layer < self.dense_count or layer >= self.n_layers:
            raise WeightError(f"layer {layer} is not a sparse Inkling MLP layer")
        p = self._layer(layer)
        fused = self.reader.tensor(
            f"{p}.mlp.shared_experts.shared_w13_weight", float32=float32
        )
        down = self.reader.tensor(
            f"{p}.mlp.shared_experts.shared_w2_weight", float32=float32
        )
        expected_fused = [self.n_shared, 2 * self.intermediate, self.hidden]
        expected_down = [self.n_shared, self.hidden, self.intermediate]
        if list(fused.shape) != expected_fused or list(down.shape) != expected_down:
            raise WeightError(
                f"shared layer {layer} shapes differ from config: fused {list(fused.shape)} vs {expected_fused}, "
                f"down {list(down.shape)} vs {expected_down}"
            )
        gate, up = split_fused_gate_up(fused, 1)
        return ExpertWeights(gate=gate, up=up, down=down.contiguous())
