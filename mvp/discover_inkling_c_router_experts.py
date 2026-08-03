#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record C Inkling router choices without loading routed expert payloads.

The C layer emits post-attention normalization, router logits, indices, and
weights before calling ``expert_get``. This probe rejects that lookup
deliberately, then continues with the next externally supplied hidden state.
Attention and attention-branch convolution state have already advanced; MLP
state cannot influence a later router because every position receives its own
layer input.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "inkling" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from discover_inkling_router_experts import input_sha256
from inkling_compact_layer_parity import (
    ExpertGet,
    ExpertWeights,
    bind_sparse_layer_compact,
)
from inkling_fixture import FixtureError, load_fixture
from inkling_layer_parity import (
    LayerParityError,
    LayerScratch,
    LayerState,
    MatrixBackend,
    TraceCollector,
    _build_config,
    build_library,
    configure_library,
)


class CRouterDiscoveryError(RuntimeError):
    """C routing could not be observed at the expected boundary."""


class _RejectExperts:
    def __init__(self, layer: int) -> None:
        self.layer = layer
        self.requests: list[int] = []
        self.callback = ExpertGet(self._get)

    def _get(self, _ctx, layer: int, expert: int, _out) -> int:
        if layer != self.layer:
            return -1
        self.requests.append(int(expert))
        return -1


def discover_c_layer_router(
    lib,
    fixture,
    cfg: dict[str, Any],
    layer: int,
    inputs: list[list[float]],
) -> dict[str, Any]:
    """Capture every C router row while refusing expert computation."""
    if layer < int(cfg["dense_layers"]):
        raise CRouterDiscoveryError(f"layer {layer} is not sparse")
    compact = bind_sparse_layer_compact(fixture, cfg, layer)
    config = _build_config(lib, cfg)
    hidden = int(cfg["hidden"])
    is_local = layer in set(cfg.get("local_layer_ids", []))
    kv_heads = int(cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"])
    head_dim = int(cfg["local_head_dim"] if is_local else cfg["global_head_dim"])
    conv_k = int(cfg["conv_kernel"])
    cap = (
        min(max(len(inputs), 1), int(cfg["sliding_window"]))
        if is_local
        else max(len(inputs), 1)
    )

    nfloat = lib.waste_inkling_layer_scratch_floats(
        ctypes.byref(config), layer, cap
    )
    nint = lib.waste_inkling_layer_scratch_ints(ctypes.byref(config))
    if not nfloat:
        raise CRouterDiscoveryError("scratch sizing failed")
    fbuf = (ctypes.c_float * nfloat)()
    ibuf = (ctypes.c_int * max(nint, 1))()
    scratch = LayerScratch()
    if lib.waste_inkling_layer_scratch_init(
        ctypes.byref(scratch),
        ctypes.byref(config),
        layer,
        cap,
        fbuf,
        ctypes.c_size_t(nfloat),
        ibuf,
        ctypes.c_size_t(nint),
    ):
        raise CRouterDiscoveryError("scratch init failed")

    kv = (ctypes.c_float * (cap * kv_heads * head_dim))()
    vv = (ctypes.c_float * (cap * kv_heads * head_dim))()
    kconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    vconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    aconv = (ctypes.c_float * (hidden * conv_k))()
    mconv = (ctypes.c_float * (hidden * conv_k))()
    state = LayerState()
    if lib.waste_inkling_layer_state_init(
        ctypes.byref(state),
        ctypes.byref(config),
        layer,
        cap,
        kv,
        vv,
        kconv,
        vconv,
        aconv,
        mconv,
    ):
        raise CRouterDiscoveryError("state init failed")

    collector = TraceCollector()
    backend = MatrixBackend()
    reject = _RejectExperts(layer)
    reject_ptr = ctypes.cast(reject.callback, ctypes.c_void_p)
    router_inputs: list[list[float]] = []
    topk_indices: list[list[int]] = []
    topk_weights: list[list[float]] = []
    router_logits: list[list[float]] = []

    for position, row in enumerate(inputs):
        if len(row) != hidden:
            raise CRouterDiscoveryError(
                f"input {position} has {len(row)} values, hidden is {hidden}"
            )
        collector.position = position
        x = (ctypes.c_float * hidden)(*row)
        before = len(reject.requests)
        rc = lib.waste_inkling_layer_step_backend_trace(
            ctypes.byref(config),
            layer,
            ctypes.byref(compact.weights),
            ctypes.byref(backend),
            ctypes.byref(state),
            x,
            position,
            ctypes.byref(scratch),
            reject_ptr,
            None,
            ctypes.byref(collector.c_trace),
        )
        if rc == 0:
            raise CRouterDiscoveryError(
                f"layer {layer} position {position} completed despite rejected expert lookup"
            )
        if len(reject.requests) != before + 1:
            raise CRouterDiscoveryError(
                f"layer {layer} position {position} made {len(reject.requests) - before} "
                "expert requests; expected one rejected lookup"
            )
        prefix = f"token.{position}.layer.{layer}."
        input_name = prefix + "post_attention_norm"
        index_name = prefix + "routed_index"
        weight_name = prefix + "routed_weight"
        logits_name = prefix + "router_logits"
        for name in (input_name, index_name, weight_name, logits_name):
            if name not in collector.values:
                raise CRouterDiscoveryError(
                    f"layer {layer} position {position} did not emit {name}"
                )
        router_inputs.append(
            [float(value) for value in collector.values[input_name]]
        )
        topk_indices.append(
            [int(value) for value in collector.values[index_name]]
        )
        topk_weights.append(
            [float(value) for value in collector.values[weight_name]]
        )
        router_logits.append(
            [float(value) for value in collector.values[logits_name]]
        )

    return {
        "seed_fixture_experts": list(fixture.experts.get(layer, ())),
        "selected_experts": sorted(
            {value for row in topk_indices for value in row}
        ),
        "first_rejected_expert": list(reject.requests),
        "router_inputs": router_inputs,
        "topk_indices": topk_indices,
        "topk_weights": topk_weights,
        "router_logits": router_logits,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--library")
    ap.add_argument("--src", default=str(REPO / "inkling" / "src"))
    ap.add_argument("--input-dtype", choices=("bfloat16",), default="bfloat16")
    args = ap.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(inputs, list) or not inputs:
            raise CRouterDiscoveryError("--inputs must contain a nonempty JSON list")
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise CRouterDiscoveryError("--layers must be nonempty")
        library = Path(args.library) if args.library else build_library(
            Path(args.src), Path(args.out).with_suffix(".so")
        )
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        tensor = torch.tensor(inputs, dtype=torch.float32)
        result = {
            "format": "inkling-c-router-discovery",
            "version": 2,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": len(inputs),
            "input_dtype": args.input_dtype,
            "inputs_sha256": input_sha256(tensor, torch.bfloat16),
            "layers": {
                str(layer): discover_c_layer_router(
                    lib, fixture, cfg, layer, inputs
                )
                for layer in layers
            },
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        CRouterDiscoveryError,
        FixtureError,
        LayerParityError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        ap.error(str(exc))
        return 2
    summary = {
        "format": result["format"],
        "version": result["version"],
        "inputs_sha256": result["inputs_sha256"],
        "layers": {
            layer: {
                "selected_experts": value["selected_experts"],
                "first_rejected_expert": value["first_rejected_expert"],
                "positions": len(value["topk_indices"]),
            }
            for layer, value in result["layers"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
