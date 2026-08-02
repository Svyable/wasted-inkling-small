#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the private Inkling C runtime with named activation tracing.

The shared library must include the private Inkling sources.  This tool is a
parity/debug surface, not the public WASTE API.  It copies callback values
immediately and writes the CRC-protected archive format from inkling_parity.py.
"""
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
from typing import Any

import torch

from inkling_parity import write_activation_archive

FP = ctypes.POINTER(ctypes.c_float)
TraceFloat = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, FP, ctypes.c_size_t
)
TraceInt = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
)


class Trace(ctypes.Structure):
    _fields_ = [("emit_float", TraceFloat), ("emit_int", TraceInt), ("ctx", ctypes.c_void_p)]


class PrivateOptions(ctypes.Structure):
    _fields_ = [("context_capacity", ctypes.c_int),
                ("verify_crc", ctypes.c_int),
                ("require_official_small", ctypes.c_int)]


class TraceCollector:
    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}
        self.position = 0
        self._float_cb = TraceFloat(self._emit_float)
        self._int_cb = TraceInt(self._emit_int)
        self.c_trace = Trace(self._float_cb, self._int_cb, None)

    def _name(self, layer: int, point: bytes) -> str:
        text = point.decode("ascii", "strict")
        scope = "model" if layer < 0 else f"layer.{layer}"
        return f"token.{self.position}.{scope}.{text}"

    def _emit_float(self, _ctx: Any, layer: int, point: bytes, data: FP, count: int) -> int:
        self.values[self._name(layer, point)] = torch.tensor(
            [data[i] for i in range(count)], dtype=torch.float32
        )
        return 0

    def _emit_int(self, _ctx: Any, layer: int, point: bytes,
                  data: ctypes.POINTER(ctypes.c_int), count: int) -> int:
        self.values[self._name(layer, point)] = torch.tensor(
            [data[i] for i in range(count)], dtype=torch.int32
        )
        return 0


def configure_library(lib: ctypes.CDLL) -> None:
    lib.waste_inkling_private_options_init.argtypes = [ctypes.POINTER(PrivateOptions)]
    lib.waste_inkling_private_open.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(PrivateOptions),
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t,
    ]
    lib.waste_inkling_private_open.restype = ctypes.c_int
    lib.waste_inkling_private_step_trace.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, FP, ctypes.c_size_t,
        ctypes.POINTER(Trace),
    ]
    lib.waste_inkling_private_step_trace.restype = ctypes.c_int
    lib.waste_inkling_private_close.argtypes = [ctypes.c_void_p]


def trace_private_runtime(library: Path | str, stage: Path | str, tokens: list[int],
                          out: Path | str, *, logits_count: int = 200058,
                          context_capacity: int = 4096,
                          verify_crc: bool = True,
                          require_official_small: bool = True) -> dict[str, Any]:
    if not tokens or any(t < 0 for t in tokens):
        raise ValueError("at least one nonnegative token id is required")
    lib = ctypes.CDLL(str(library))
    configure_library(lib)
    options = PrivateOptions()
    lib.waste_inkling_private_options_init(ctypes.byref(options))
    options.context_capacity = context_capacity
    options.verify_crc = int(verify_crc)
    options.require_official_small = int(require_official_small)
    runtime = ctypes.c_void_p()
    detail = ctypes.create_string_buffer(512)
    rc = lib.waste_inkling_private_open(
        str(stage).encode(), ctypes.byref(options), ctypes.byref(runtime),
        detail, len(detail),
    )
    if rc:
        raise RuntimeError(detail.value.decode(errors="replace") or f"private open failed: {rc}")
    collector = TraceCollector()
    logits = (ctypes.c_float * logits_count)()
    try:
        for position, token in enumerate(tokens):
            collector.position = position
            rc = lib.waste_inkling_private_step_trace(
                runtime, token, position, logits, logits_count,
                ctypes.byref(collector.c_trace),
            )
            if rc:
                raise RuntimeError(f"private trace step failed at position {position}: {rc}")
    finally:
        lib.waste_inkling_private_close(runtime)
    return write_activation_archive(
        out, collector.values,
        metadata={"runtime": "waste-private-c", "tokens": tokens,
                  "context_capacity": context_capacity},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--tokens", required=True, help="comma-separated token ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--logits-count", type=int, default=200058)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--no-crc", action="store_true")
    ap.add_argument("--allow-nonofficial", action="store_true")
    args = ap.parse_args()
    tokens = [int(x) for x in args.tokens.split(",") if x]
    trace_private_runtime(
        args.library, args.stage, tokens, args.out,
        logits_count=args.logits_count, context_capacity=args.context,
        verify_crc=not args.no_crc,
        require_official_small=not args.allow_nonofficial,
    )
    print(f"wrote C activation trace to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
