#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build bounded Inkling parity fixtures with strict HTTP range reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHARD_RE = re.compile(r"^[A-Za-z0-9_.-]+\.safetensors$")


class RemoteFixtureError(RuntimeError):
    pass


def require(ok: Any, message: str) -> None:
    if not ok:
        raise RemoteFixtureError(message)


def parse_experts(text: str) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    if not text:
        return out
    for group in text.split(";"):
        try:
            layer_text, ids_text = group.split(":", 1)
            layer = int(layer_text)
            ids = sorted({int(value) for value in ids_text.split(",") if value})
        except ValueError as exc:
            raise RemoteFixtureError(f"invalid expert selection {group!r}") from exc
        require(layer >= 0 and ids and min(ids) >= 0,
                f"invalid expert selection {group!r}")
        out[layer] = ids
    return out


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            new.remove_header("Authorization")
            new.unredirected_hdrs.pop("Authorization", None)
        return new


class HttpRanges:
    def __init__(self, base: str, revision: str, *, token: str | None = None,
                 timeout: float = 60, retries: int = 3) -> None:
        parsed = urllib.parse.urlparse(base)
        require(parsed.scheme in ("http", "https") and parsed.netloc,
                "base URL must use HTTP(S)")
        require(COMMIT_RE.fullmatch(revision) is not None,
                "revision must be an immutable lowercase hex commit")
        require(timeout > 0 and retries > 0, "timeout/retries must be positive")
        self.base, self.revision = base.rstrip("/"), revision
        self.token, self.timeout, self.retries = token, timeout, retries
        self.bytes_read = self.requests = 0
        self.opener = urllib.request.build_opener(SafeRedirect())

    def url(self, name: str) -> str:
        require(Path(name).name == name and name not in (".", ".."),
                f"unsafe remote path {name!r}")
        return f"{self.base}/resolve/{urllib.parse.quote(self.revision, safe='')}/{urllib.parse.quote(name, safe='')}"

    def open(self, name: str, byte_range: tuple[int, int] | None = None):
        headers = {"Accept-Encoding": "identity", "User-Agent": "wasted-inkling-range-fixture/1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if byte_range:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        request = urllib.request.Request(self.url(name), headers=headers)
        for attempt in range(self.retries):
            self.requests += 1
            try:
                return self.opener.open(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                retry = exc.code in (429, 500, 502, 503, 504)
                if not retry or attempt + 1 == self.retries:
                    raise RemoteFixtureError(f"HTTP {exc.code} reading {name}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt + 1 == self.retries:
                    raise RemoteFixtureError(f"cannot read {name}: {exc}") from exc
            time.sleep(0.25 * 2**attempt)
        raise AssertionError("unreachable")

    def file(self, name: str, limit: int = 16 << 20) -> bytes:
        with self.open(name) as response:
            length = response.headers.get("Content-Length")
            require(length is None or int(length) <= limit,
                    f"metadata {name} is above {limit} bytes")
            data = response.read(limit + 1)
        require(len(data) <= limit, f"metadata {name} is above {limit} bytes")
        self.bytes_read += len(data)
        return data

    def range(self, name: str, start: int, size: int,
              chunk: int = 8 << 20) -> Iterator[bytes]:
        require(start >= 0 and size > 0 and chunk > 0, "invalid byte range")
        end = start + size - 1
        with self.open(name, (start, end)) as response:
            status = getattr(response, "status", response.getcode())
            require(status == 206,
                    f"server ignored Range for {name}: expected 206, got {status}")
            require(response.headers.get("Content-Encoding") in (None, "identity"),
                    f"encoded range response for {name}")
            length = response.headers.get("Content-Length")
            require(length is None or int(length) == size,
                    f"wrong range Content-Length for {name}")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)",
                                 response.headers.get("Content-Range", ""))
            require(match is not None and (int(match[1]), int(match[2])) == (start, end),
                    f"wrong Content-Range for {name}")
            remaining = size
            while remaining:
                data = response.read(min(chunk, remaining))
                require(data and len(data) <= remaining,
                        f"short or oversized range response for {name}")
                remaining -= len(data)
                self.bytes_read += len(data)
                yield data
            require(not response.read(1), f"oversized range response for {name}")

    def read_range(self, name: str, start: int, size: int) -> bytes:
        return b"".join(self.range(name, start, size))


@dataclass(frozen=True)
class Location:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    size: int


class RemoteSafetensors:
    def __init__(self, source: HttpRanges, index_raw: bytes) -> None:
        try:
            index = json.loads(index_raw)
        except json.JSONDecodeError as exc:
            raise RemoteFixtureError(f"invalid index JSON: {exc}") from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        require(isinstance(weight_map, dict) and weight_map,
                "index has no weight_map")
        require(all(isinstance(k, str) and isinstance(v, str) and SHARD_RE.fullmatch(v)
                    for k, v in weight_map.items()),
                "index contains an unsafe shard entry")
        self.source, self.weight_map = source, dict(weight_map)
        self.headers: dict[str, tuple[dict[str, Any], int]] = {}

    def header(self, shard: str) -> tuple[dict[str, Any], int]:
        if shard not in self.headers:
            (length,) = struct.unpack("<Q", self.source.read_range(shard, 0, 8))
            require(2 <= length <= 256 << 20,
                    f"implausible safetensors header in {shard}")
            try:
                header = json.loads(self.source.read_range(shard, 8, length))
            except json.JSONDecodeError as exc:
                raise RemoteFixtureError(f"invalid safetensors header in {shard}") from exc
            require(isinstance(header, dict), f"invalid safetensors header in {shard}")
            self.headers[shard] = header, 8 + length
        return self.headers[shard]

    def location(self, name: str) -> Location:
        shard = self.weight_map.get(name)
        require(shard is not None, f"tensor absent from index: {name}")
        header, base = self.header(shard)
        item = header.get(name)
        require(isinstance(item, dict), f"tensor absent from shard header: {name}")
        dtype, shape, offsets = item.get("dtype"), item.get("shape"), item.get("data_offsets")
        require(dtype in DTYPE_BYTES, f"unsupported dtype {dtype!r} for {name}")
        require(isinstance(shape, list) and 1 <= len(shape) <= 4
                and all(isinstance(x, int) and x > 0 for x in shape),
                f"invalid shape for {name}")
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(isinstance(x, int) and x >= 0 for x in offsets)
                and offsets[1] >= offsets[0], f"invalid offsets for {name}")
        size = offsets[1] - offsets[0]
        require(size == math.prod(shape) * DTYPE_BYTES[dtype],
                f"payload geometry mismatch for {name}")
        return Location(name, shard, dtype, tuple(shape), base + offsets[0], size)


@dataclass(frozen=True)
class Entry:
    name: str
    kind: str
    axis0: int | None
    dtype: str
    shape: tuple[int, ...]
    size: int
    shard: str
    offset: int

    @property
    def label(self) -> str:
        return self.name if self.axis0 is None else f"{self.name}[{self.axis0}]"

    @property
    def nbytes(self) -> int:
        return self.size


COMMON_SUFFIXES = (
    "attn_norm.weight", "mlp_norm.weight", "attn.wq_du.weight",
    "attn.wk_dv.weight", "attn.wv_dv.weight", "attn.wr_du.weight",
    "attn.wo_ud.weight", "attn.q_norm.weight", "attn.k_norm.weight",
    "attn.rel_logits_proj.proj", "attn.k_sconv.weight",
    "attn.v_sconv.weight", "attn_sconv.weight", "mlp_sconv.weight",
)
DENSE_SUFFIXES = ("mlp.w13_dn.weight", "mlp.w2_md.weight", "mlp.global_scale")
SPARSE_SUFFIXES = (
    "mlp.gate.weight", "mlp.gate.bias", "mlp.gate.global_scale",
    "mlp.shared_experts.shared_w13_weight",
    "mlp.shared_experts.shared_w2_weight",
)


def required_layer(layer: int, sparse: bool) -> set[str]:
    prefix = f"model.llm.layers.{layer}."
    suffixes = COMMON_SUFFIXES + (SPARSE_SUFFIXES if sparse else DENSE_SUFFIXES)
    return {prefix + suffix for suffix in suffixes}


class Plan:
    def __init__(self, model: str, source: HttpRanges, config: bytes, index: bytes,
                 layers: set[int], experts: dict[int, list[int]], entries: list[Entry]) -> None:
        self.model, self.revision = model, source.revision
        self.config, self.index = config, index
        self.layers, self.experts, self.entries = tuple(sorted(layers)), experts, entries
        self.metadata_bytes, self.requests = source.bytes_read, source.requests

    @property
    def metadata_bytes_read(self) -> int:
        return self.metadata_bytes

    @property
    def payload_bytes(self) -> int:
        return sum(entry.size for entry in self.entries)

    def summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model, "revision": self.revision,
            "layers": list(self.layers),
            "experts": {str(k): v for k, v in sorted(self.experts.items())},
            "entries": len(self.entries), "payload_bytes": self.payload_bytes,
            "metadata_bytes_read": self.metadata_bytes, "requests": self.requests,
            "shards": sorted({entry.shard for entry in self.entries}),
            "config_sha256": hashlib.sha256(self.config).hexdigest(),
            "index_sha256": hashlib.sha256(self.index).hexdigest(),
        }


def build_plan(source: HttpRanges, model: str, layers: Iterable[int],
               experts: dict[int, list[int]] | None = None,
               max_total: int = 8 << 30, max_entry: int = 2 << 30) -> Plan:
    selected = set(map(int, layers))
    require(selected and min(selected) >= 0, "at least one nonnegative layer is required")
    config_raw = source.file("config.json")
    index_raw = source.file("model.safetensors.index.json")
    try:
        config = json.loads(config_raw)
    except json.JSONDecodeError as exc:
        raise RemoteFixtureError("invalid config JSON") from exc
    text = config.get("text_config") if isinstance(config, dict) else None
    require(isinstance(text, dict), "config has no text_config")
    n_layers, n_experts, dense = (text.get(k) for k in
                                  ("num_hidden_layers", "n_routed_experts", "dense_mlp_idx"))
    require(isinstance(n_layers, int) and n_layers > max(selected), "selected layer is out of range")
    require(isinstance(n_experts, int) and n_experts > 0, "invalid n_routed_experts")
    require(isinstance(dense, int) and 0 <= dense <= n_layers, "invalid dense_mlp_idx")
    experts = experts or {}
    for layer in selected:
        require((layer < dense and layer not in experts) or (layer >= dense and experts.get(layer)),
                f"layer {layer} has invalid routed expert selection")
    for layer, ids in experts.items():
        require(layer in selected and ids and all(0 <= i < n_experts for i in ids),
                f"invalid expert selection for layer {layer}")

    reader = RemoteSafetensors(source, index_raw)
    available = set(reader.weight_map)
    required = {"model.llm.embed_norm.weight", "model.llm.norm.weight"}
    for layer in selected:
        required |= required_layer(layer, layer >= dense)
    missing = sorted(required - available)
    require(not missing, "release index is missing required tensors: " + ", ".join(missing))

    entries: list[Entry] = []
    total = 0

    def add(entry: Entry) -> None:
        nonlocal total
        require(entry.size <= max_entry, f"{entry.label} exceeds per-entry limit")
        total += entry.size
        require(total <= max_total, "fixture exceeds total byte limit")
        entries.append(entry)

    for name in sorted(required):
        loc = reader.location(name)
        add(Entry(name, "tensor", None, loc.dtype, loc.shape, loc.size, loc.shard, loc.offset))
    for layer, ids in sorted(experts.items()):
        for suffix in ("w13_weight", "w2_weight"):
            name = f"model.llm.layers.{layer}.mlp.experts.{suffix}"
            loc = reader.location(name)
            stride = math.prod(loc.shape[1:]) * DTYPE_BYTES[loc.dtype]
            for expert in sorted(set(ids)):
                add(Entry(name, "axis0-slice", expert, loc.dtype, loc.shape[1:],
                          stride, loc.shard, loc.offset + expert * stride))
    require(len({(entry.name, entry.axis0) for entry in entries}) == len(entries),
            "duplicate fixture entries")
    return Plan(model, source, config_raw, index_raw, selected, experts, entries)


def safe_name(label: str) -> str:
    digest = hashlib.sha256(label.encode()).hexdigest()[:16]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")[:80] or "entry"
    return f"{stem}-{digest}.bin"


def extract(source: HttpRanges, plan: Plan, out: Path | str) -> dict[str, Any]:
    require(source.revision == plan.revision, "source revision differs from plan")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / "fixture.json").exists(), "fixture manifest already exists")
    metadata_start = source.bytes_read
    manifest_entries = []
    for entry in plan.entries:
        path = root / safe_name(entry.label)
        tmp = path.with_name(path.name + ".tmp")
        crc = written = 0
        try:
            with tmp.open("wb") as handle:
                for block in source.range(entry.shard, entry.offset, entry.size):
                    handle.write(block)
                    crc = zlib.crc32(block, crc)
                    written += len(block)
                handle.flush()
                os.fsync(handle.fileno())
            require(written == entry.size, f"short payload for {entry.label}")
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        item = {"name": entry.name, "kind": entry.kind, "dtype": entry.dtype,
                "shape": list(entry.shape), "bytes": written,
                "crc32": crc & 0xFFFFFFFF, "path": path.name}
        if entry.axis0 is not None:
            item["axis0"] = entry.axis0
        manifest_entries.append(item)
    manifest = {
        "format": "inkling-parity-fixture", "version": 1,
        "model_id": plan.model, "layers": list(plan.layers),
        "experts": {str(k): v for k, v in sorted(plan.experts.items())},
        "source": {"repository": plan.model, "revision": plan.revision,
                   "config_sha256": hashlib.sha256(plan.config).hexdigest(),
                   "index_sha256": hashlib.sha256(plan.index).hexdigest()},
        "total_payload_bytes": sum(entry["bytes"] for entry in manifest_entries),
        "reader_payload_bytes": source.bytes_read - metadata_start,
        "remote_bytes_read": source.bytes_read,
        "remote_requests": source.requests,
        "entries": manifest_entries,
        "notes": ["strict HTTP ranges from an immutable revision",
                  "vocabulary tables omitted", "routed experts are axis-0 slices"],
    }
    tmp = root / "fixture.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, root / "fixture.json")
    return manifest


# Compatibility names used by the MVP tests and handoff documentation.
HttpRangeSource = HttpRanges
_required_layer_names = required_layer


def build_remote_plan(source: HttpRanges, *, model_id: str, layers: Iterable[int],
                      experts: dict[int, list[int]] | None = None,
                      max_total_bytes: int = 8 << 30,
                      max_tensor_bytes: int = 2 << 30) -> Plan:
    return build_plan(source, model_id, layers, experts, max_total_bytes, max_tensor_bytes)


def extract_remote_fixture(source: HttpRanges, plan: Plan, out: Path | str,
                           *, chunk_bytes: int = 8 << 20) -> dict[str, Any]:
    del chunk_bytes
    return extract(source, plan, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="thinkingmachines/Inkling-Small")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--experts", default="")
    ap.add_argument("--out")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--max-total-gib", type=float, default=8)
    ap.add_argument("--max-entry-gib", type=float, default=2)
    ap.add_argument("--base-url", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        require(MODEL_RE.fullmatch(args.repo), f"invalid model id {args.repo!r}")
        base = args.base_url or f"https://huggingface.co/{args.repo}"
        source = HttpRanges(base, args.revision, token=os.environ.get("HF_TOKEN"))
        plan = build_plan(source, args.repo,
                          [int(value) for value in args.layers.split(",") if value],
                          parse_experts(args.experts),
                          int(args.max_total_gib * (1 << 30)),
                          int(args.max_entry_gib * (1 << 30)))
        if args.plan_only:
            result = plan.summary()
        else:
            require(args.out, "--out is required unless --plan-only is used")
            result = extract(source, plan, args.out)
    except (RemoteFixtureError, ValueError, OSError) as exc:
        ap.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
