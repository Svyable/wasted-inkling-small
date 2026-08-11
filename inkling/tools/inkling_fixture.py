#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Read and validate a bounded Inkling parity fixture.

`inkling_parity.py` extracts a fixture — selected decoder layers, selected
axis-0 routed-expert slices, hash-bound to the source `config.json` and
safetensors index — so parity work does not require the complete 532 GB
checkpoint.  Nothing consumed those fixtures, which is why the official side of
the trace protocol still loaded the whole model.  This module is that consumer.

It is deliberately dependency-free: validating a fixture is a bytes-and-CRC
problem, and a laptop should be able to check one without importing torch.
Callers that need tensors convert at their own boundary.

Every accessor is fail-closed.  A fixture that is missing a tensor, has a
corrupt payload, or does not cover a requested layer raises rather than
returning something plausible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

FIXTURE_FORMAT = "inkling-parity-fixture"
FIXTURE_VERSION = 1

# Source dtypes a parity fixture may contain. Kept independent of
# inkling_weights.DTYPES on purpose: that table maps to torch dtypes and this
# module must import nothing.
ITEM_SIZE = {"BF16": 2, "F16": 2, "F32": 4}

KIND_TENSOR = "tensor"
KIND_SLICE = "axis0-slice"

# The release name of the unembedding table. Bounded final-head evidence takes
# axis-0 rows of it; the whole table never enters a fixture.
UNEMBED_NAME = "model.llm.unembed.weight"


class FixtureError(RuntimeError):
    pass


def _bf16_to_f32(raw: bytes) -> array:
    """Widen little-endian BF16 to F32 by placing it in the high half."""
    wide = bytearray(len(raw) * 2)
    wide[2::4] = raw[0::2]
    wide[3::4] = raw[1::2]
    out = array("f")
    out.frombytes(bytes(wide))
    if sys.byteorder != "little":
        out.byteswap()
    return out


def _f16_to_f32(raw: bytes) -> array:
    halves = array("H")
    halves.frombytes(raw)
    if sys.byteorder != "little":
        halves.byteswap()
    out = array("f", [0.0]) * len(halves)
    for i, h in enumerate(halves):
        sign = -1.0 if h >> 15 else 1.0
        exponent = (h >> 10) & 0x1F
        mantissa = h & 0x3FF
        if exponent == 0:
            value = math.ldexp(mantissa, -24)
        elif exponent == 0x1F:
            value = math.inf if mantissa == 0 else math.nan
        else:
            value = math.ldexp(mantissa + 0x400, exponent - 25)
        out[i] = sign * value
    return out


def decode_f32(raw: bytes, dtype: str) -> array:
    """Decode a source-dtype payload to an array('f') of float32 values."""
    item = ITEM_SIZE.get(dtype)
    if item is None:
        raise FixtureError(f"unsupported dtype {dtype!r}")
    if len(raw) % item:
        raise FixtureError(f"{dtype} payload of {len(raw)} bytes is not a whole number of values")
    if dtype == "F32":
        out = array("f")
        out.frombytes(raw)
        if sys.byteorder != "little":
            out.byteswap()
        return out
    return _bf16_to_f32(raw) if dtype == "BF16" else _f16_to_f32(raw)


@dataclass(frozen=True)
class FixtureEntry:
    """One stored tensor or expert slice. `shape` is the stored shape: for an
    axis-0 slice it is the source shape with the expert axis already removed."""

    name: str
    kind: str
    axis0: int | None
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    crc32: int
    path: str

    @property
    def key(self) -> tuple[str, int | None]:
        return (self.name, self.axis0)

    @property
    def label(self) -> str:
        return self.name if self.axis0 is None else f"{self.name}[{self.axis0}]"


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def _entry_from_json(value: Any, root: Path) -> FixtureEntry:
    _require(isinstance(value, dict), "fixture entry is not an object")
    name = value.get("name")
    kind = value.get("kind")
    dtype = value.get("dtype")
    shape = value.get("shape")
    nbytes = value.get("bytes")
    crc = value.get("crc32")
    rel = value.get("path")

    _require(isinstance(name, str) and name, "fixture entry has no name")
    _require(kind in (KIND_TENSOR, KIND_SLICE), f"unsupported fixture kind for {name}: {kind!r}")
    _require(dtype in ITEM_SIZE, f"unsupported fixture dtype for {name}: {dtype!r}")
    _require(
        isinstance(shape, list) and 1 <= len(shape) <= 4
        and all(isinstance(x, int) and x > 0 for x in shape),
        f"invalid fixture shape for {name}: {shape!r}",
    )
    _require(isinstance(nbytes, int) and nbytes >= 0, f"invalid byte count for {name}")
    _require(isinstance(crc, int) and 0 <= crc <= 0xFFFFFFFF, f"invalid crc32 for {name}")
    _require(isinstance(rel, str) and rel and Path(rel).name == rel,
             f"fixture path for {name} must be a plain file name")

    axis0 = value.get("axis0")
    if kind == KIND_SLICE:
        _require(isinstance(axis0, int) and axis0 >= 0,
                 f"axis-0 slice {name} has no valid axis0 index")
    else:
        _require(axis0 is None, f"tensor entry {name} must not carry an axis0 index")

    expected = math.prod(shape) * ITEM_SIZE[dtype]
    _require(nbytes == expected,
             f"fixture entry {name} declares {nbytes} bytes, geometry implies {expected}")

    path = root / rel
    try:
        actual = path.stat().st_size
    except OSError as exc:
        raise FixtureError(f"fixture payload missing for {name}: {exc}") from exc
    _require(actual == nbytes,
             f"fixture payload for {name} is {actual} bytes, manifest says {nbytes}")

    return FixtureEntry(name, kind, axis0 if kind == KIND_SLICE else None,
                        dtype, tuple(int(x) for x in shape), nbytes, crc, rel)


class Fixture:
    """A validated, bounded slice of an official checkpoint."""

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root
        self.model_id = manifest.get("model_id")
        self.source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}

        layers = manifest.get("layers")
        _require(isinstance(layers, list)
                 and all(isinstance(x, int) and x >= 0 for x in layers),
                 "fixture manifest has no valid layer list")
        _require(len(set(layers)) == len(layers), "fixture layer list has duplicates")
        self.layers: tuple[int, ...] = tuple(sorted(layers))

        rows_raw = manifest.get("vocab_rows") or []
        _require(isinstance(rows_raw, list), "fixture vocabulary rows are not a list")
        _require(all(isinstance(x, int) and x >= 0 for x in rows_raw),
                 "fixture vocabulary rows are not nonnegative integers")
        _require(len(set(rows_raw)) == len(rows_raw),
                 "fixture vocabulary row list has duplicates")
        self.vocab_rows: tuple[int, ...] = tuple(sorted(rows_raw))
        # A fixture must cover something: layers, vocabulary rows, or both. An
        # empty layer list is legitimate for a head-only fixture.
        _require(self.layers or self.vocab_rows,
                 "fixture covers neither layers nor vocabulary rows")

        experts_raw = manifest.get("experts") or {}
        _require(isinstance(experts_raw, dict), "fixture expert selection is not an object")
        experts: dict[int, tuple[int, ...]] = {}
        for key, ids in experts_raw.items():
            try:
                layer = int(key)
            except (TypeError, ValueError) as exc:
                raise FixtureError(f"invalid expert layer key {key!r}") from exc
            _require(layer in self.layers, f"expert selection names unselected layer {layer}")
            _require(isinstance(ids, list) and ids
                     and all(isinstance(x, int) and x >= 0 for x in ids),
                     f"invalid expert id list for layer {layer}")
            _require(len(set(ids)) == len(ids), f"duplicate expert ids for layer {layer}")
            experts[layer] = tuple(sorted(ids))
        self.experts = experts

        entries = manifest.get("entries")
        _require(isinstance(entries, list) and entries, "fixture manifest has no entries")
        self._entries: dict[tuple[str, int | None], FixtureEntry] = {}
        for value in entries:
            entry = _entry_from_json(value, root)
            _require(entry.key not in self._entries,
                     f"duplicate fixture entry {entry.label}")
            self._entries[entry.key] = entry

        total = manifest.get("total_payload_bytes")
        if isinstance(total, int):
            actual = sum(e.nbytes for e in self._entries.values())
            _require(total == actual,
                     f"fixture declares {total} payload bytes, entries hold {actual}")

    # -- inspection ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def payload_bytes(self) -> int:
        return sum(e.nbytes for e in self._entries.values())

    def names(self) -> list[str]:
        return sorted({e.name for e in self._entries.values()})

    def entries(self) -> list[FixtureEntry]:
        return [self._entries[k] for k in sorted(self._entries, key=lambda k: (k[0], k[1] or -1))]

    def has(self, name: str, axis0: int | None = None) -> bool:
        return (name, axis0) in self._entries

    def entry(self, name: str, axis0: int | None = None) -> FixtureEntry:
        try:
            return self._entries[(name, axis0)]
        except KeyError:
            label = name if axis0 is None else f"{name}[{axis0}]"
            raise FixtureError(f"fixture does not contain {label}") from None

    def layer_entries(self, layer: int) -> list[FixtureEntry]:
        prefix = f"model.llm.layers.{layer}."
        return [e for e in self.entries() if e.name.startswith(prefix)]

    # -- payloads --------------------------------------------------------

    def raw(self, name: str, axis0: int | None = None, *, verify: bool = True) -> bytes:
        """Return one entry's stored payload in its source dtype."""
        entry = self.entry(name, axis0)
        try:
            payload = (self.root / entry.path).read_bytes()
        except OSError as exc:
            raise FixtureError(f"cannot read {entry.label}: {exc}") from exc
        if len(payload) != entry.nbytes:
            raise FixtureError(f"{entry.label} is {len(payload)} bytes, expected {entry.nbytes}")
        if verify and zlib.crc32(payload) & 0xFFFFFFFF != entry.crc32:
            raise FixtureError(f"{entry.label} failed CRC verification")
        return payload

    def values(self, name: str, axis0: int | None = None, *,
               verify: bool = True) -> array:
        """Decode one entry to float32, without torch.

        BF16 and F16 are widened here so a ctypes harness can feed the C
        runtime directly. The widening is exact — every BF16 and F16 value is
        representable in F32 — so this introduces no tolerance of its own.
        """
        entry = self.entry(name, axis0)
        payload = self.raw(name, axis0, verify=verify)
        return decode_f32(payload, entry.dtype)

    def verify(self) -> int:
        """CRC-check every payload. Returns the number of bytes verified."""
        total = 0
        for entry in self.entries():
            total += len(self.raw(entry.name, entry.axis0, verify=True))
        return total

    # -- reference-side assembly -----------------------------------------

    def layer_state_dict_entries(self, layer: int) -> dict[str, FixtureEntry]:
        """Map one layer's fixture entries to official module-relative keys.

        `model.llm.layers.7.attn.q_proj.weight` becomes `attn.q_proj.weight`,
        which is what an official decoder-layer module expects. Routed expert
        slices are excluded: they are stored per expert and cannot be loaded
        into a module that expects the full `[n_routed, …]` tensor. Use
        `expert_slices()` for those and check coverage first.
        """
        self.require_layers([layer])
        prefix = f"model.llm.layers.{layer}."
        out: dict[str, FixtureEntry] = {}
        for entry in self.entries():
            if entry.kind != KIND_TENSOR or not entry.name.startswith(prefix):
                continue
            key = entry.name[len(prefix):]
            _require(key not in out, f"duplicate state-dict key {key} for layer {layer}")
            out[key] = entry
        _require(out, f"fixture holds no dense tensors for layer {layer}")
        return out

    def expert_slices(self, layer: int, name_suffix: str) -> dict[int, FixtureEntry]:
        """Return `{expert_id: entry}` for one routed-expert tensor."""
        self.require_layers([layer])
        name = f"model.llm.layers.{layer}.mlp.experts.{name_suffix}"
        out = {e.axis0: e for e in self.entries()
               if e.kind == KIND_SLICE and e.name == name and e.axis0 is not None}
        _require(out, f"fixture holds no {name_suffix} slices for layer {layer}")
        return dict(sorted(out.items()))

    def vocab_row_slices(self) -> dict[int, FixtureEntry]:
        """Return `{vocabulary row: entry}` for the unembedding table.

        The full table is gigabytes, so final-head evidence selects rows the
        same way sparse-layer evidence selects experts: as axis-0 slices.
        """
        out = {e.axis0: e for e in self.entries()
               if e.kind == KIND_SLICE and e.name == UNEMBED_NAME
               and e.axis0 is not None}
        _require(out, "fixture holds no unembedding row slices")
        _require(tuple(sorted(out)) == self.vocab_rows,
                 "unembedding slices disagree with the declared vocabulary rows")
        return dict(sorted(out.items()))

    # -- fail-closed coverage checks -------------------------------------

    def require_layers(self, layers: Iterable[int]) -> None:
        missing = sorted(set(int(x) for x in layers) - set(self.layers))
        if missing:
            raise FixtureError(
                f"fixture covers layers {list(self.layers)}; requested {missing} are absent")

    def require_vocab_rows(self, rows: Iterable[int]) -> None:
        missing = sorted(set(int(x) for x in rows) - set(self.vocab_rows))
        if missing:
            raise FixtureError(
                f"fixture carries vocabulary rows {list(self.vocab_rows)}; "
                f"requested {missing} are absent")

    def require_experts(self, layer: int, ids: Iterable[int]) -> None:
        self.require_layers([layer])
        have = set(self.experts.get(layer, ()))
        missing = sorted(set(int(x) for x in ids) - have)
        if missing:
            raise FixtureError(
                f"fixture layer {layer} carries experts {sorted(have)}; "
                f"requested {missing} are absent")

    def bound_to(self, config_sha256: str, index_sha256: str) -> bool:
        """True when this fixture was extracted from the named source files."""
        return (self.source.get("config_sha256") == config_sha256
                and self.source.get("index_sha256") == index_sha256)


def load_fixture(root: Path | str) -> Fixture:
    root = Path(root)
    try:
        manifest = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError(f"cannot read fixture manifest: {exc}") from exc
    _require(isinstance(manifest, dict), "fixture manifest is not an object")
    _require(manifest.get("format") == FIXTURE_FORMAT,
             f"not an Inkling parity fixture: {manifest.get('format')!r}")
    _require(manifest.get("version") == FIXTURE_VERSION,
             f"unsupported fixture version {manifest.get('version')!r}")
    return Fixture(root, manifest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="inspect a bounded Inkling parity fixture")
    ap.add_argument("--fixture", required=True, help="fixture directory")
    ap.add_argument("--verify", action="store_true", help="CRC-check every payload")
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = ap.parse_args(argv)

    try:
        fixture = load_fixture(args.fixture)
        verified = fixture.verify() if args.verify else 0
    except FixtureError as exc:
        ap.error(str(exc))
        return 2  # unreachable; argparse exits

    summary = {
        "model_id": fixture.model_id,
        "layers": list(fixture.layers),
        "experts": {str(k): list(v) for k, v in sorted(fixture.experts.items())},
        "vocab_rows": list(fixture.vocab_rows),
        "entries": len(fixture),
        "payload_bytes": fixture.payload_bytes,
        "verified_bytes": verified,
        "source": fixture.source,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"fixture {args.fixture}")
        print(f"  model      {fixture.model_id}")
        print(f"  layers     {list(fixture.layers)}")
        for layer, ids in sorted(fixture.experts.items()):
            print(f"  experts L{layer} {list(ids)}")
        if fixture.vocab_rows:
            print(f"  vocab rows {list(fixture.vocab_rows)}")
        print(f"  entries    {len(fixture)}")
        print(f"  payload    {fixture.payload_bytes} bytes")
        if args.verify:
            print(f"  verified   {verified} bytes, all CRCs match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
