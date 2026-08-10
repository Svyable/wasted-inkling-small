#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run checked-in BF16 sparse execution across the eight source-bound tokens.

This is the stateful promotion gate after position-zero layers 2 and 5 became
BF16-exact without source rewriting.  One checked-in C layer state is retained
across all eight inputs so K/V caches and all four short-convolution states are
part of the claim.  Canonical route IDs and #51 exact weights remain an
explicit downstream counterfactual adapter; raw candidate routes are preserved.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

import torch

import diagnose_inkling_bf16_stateful_mlp_boundary as boundary
import diagnose_inkling_portable_bf16_composed_moe as composed
import diagnose_inkling_portable_bf16_stateful_sparse_layer as stateful
from checked_bf16_profile_collector import (
    CheckedPostReductionWeightHelper,
    build_checked_weight_helper,
)
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import CanonicalRouteError, RouteRow, load_canonical_routes
from inkling_compact_layer_parity import bind_sparse_layer_compact
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import DTYPES, FixtureReferenceError, verify_config_binding
from inkling_layer_parity import LayerScratch, LayerState, LayerParityError, _build_config
from inkling_release_config import build_transformers_text_config
from run_inkling_checked_bf16_sparse_layer import (
    BF16_REFERENCE,
    build_checked_library,
    configure_profile_library,
)
from run_inkling_fixture_reference_canonical import CanonicalOfficialRouteError


class CheckedStatefulError(RuntimeError):
    pass


def run_c_stateful_profile(
    lib: Any,
    fixture: Any,
    cfg: dict[str, Any],
    layer: int,
    inputs: list[list[float]],
    rows: list[RouteRow],
    provider: composed.CapturingMoeProvider,
    collector: stateful.StatefulExactWeightCollector,
) -> dict[str, Any]:
    compact = bind_sparse_layer_compact(fixture, cfg, layer)
    if compact.provider is None:
        raise CheckedStatefulError("sparse fixture has no expert provider")
    required = {expert for row in rows for expert in row.indices}
    missing = sorted(required.difference(compact.provider.experts))
    if missing:
        raise CheckedStatefulError(f"layer {layer} fixture omits experts {missing}")
    for field in (
        "wq", "wk", "wv", "wr", "wo", "router_weight",
        "shared_gate", "shared_up", "shared_down",
    ):
        setattr(compact.weights, field, ctypes.POINTER(ctypes.c_float)())

    config = _build_config(lib, cfg)
    hidden = int(cfg["hidden"])
    is_local = layer in set(cfg.get("local_layer_ids", []))
    kv_heads = int(cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"])
    head_dim = int(cfg["local_head_dim"] if is_local else cfg["global_head_dim"])
    conv_k = int(cfg["conv_kernel"])
    capacity = (
        min(max(len(inputs), 1), int(cfg["sliding_window"]))
        if is_local else max(len(inputs), 1)
    )
    nfloat = lib.waste_inkling_layer_scratch_floats(ctypes.byref(config), layer, capacity)
    nint = lib.waste_inkling_layer_scratch_ints(ctypes.byref(config))
    fbuf = (ctypes.c_float * nfloat)()
    ibuf = (ctypes.c_int * max(nint, 1))()
    scratch = LayerScratch()
    if lib.waste_inkling_layer_scratch_init(
        ctypes.byref(scratch), ctypes.byref(config), layer, capacity,
        fbuf, ctypes.c_size_t(nfloat), ibuf, ctypes.c_size_t(nint),
    ):
        raise CheckedStatefulError("scratch init failed")

    kv = (ctypes.c_float * (capacity * kv_heads * head_dim))()
    vv = (ctypes.c_float * (capacity * kv_heads * head_dim))()
    kconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    vconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    aconv = (ctypes.c_float * (hidden * conv_k))()
    mconv = (ctypes.c_float * (hidden * conv_k))()
    layer_state = LayerState()
    if lib.waste_inkling_layer_state_init(
        ctypes.byref(layer_state), ctypes.byref(config), layer, capacity,
        kv, vv, kconv, vconv, aconv, mconv,
    ):
        raise CheckedStatefulError("state init failed")

    outputs: list[torch.Tensor] = []
    preexpert: list[torch.Tensor] = []
    for position, input_row in enumerate(inputs):
        if len(input_row) != hidden:
            raise CheckedStatefulError(
                f"input {position} has {len(input_row)} values; expected {hidden}"
            )
        collector.position = position
        provider.router_logits = None
        x = (ctypes.c_float * hidden)(*input_row)
        rc = lib.waste_inkling_layer_step_backend_trace_profile(
            ctypes.byref(config), layer, ctypes.byref(compact.weights),
            ctypes.byref(provider.backend), ctypes.byref(layer_state), x, position,
            ctypes.byref(scratch), ctypes.c_void_p(), None,
            ctypes.byref(collector.c_trace), BF16_REFERENCE,
        )
        if provider.error:
            raise CheckedStatefulError(provider.error)
        if collector.error:
            raise CheckedStatefulError(collector.error)
        if rc:
            raise CheckedStatefulError(
                f"checked C layer failed at position {position} with status {rc}"
            )
        outputs.append(torch.tensor([float(x[i]) for i in range(hidden)], dtype=torch.float32))
        key = f"token.{position}.layer.{layer}.post_attention_norm"
        if key not in collector.values:
            raise CheckedStatefulError(f"candidate trace omitted {key}")
        preexpert.append(torch.tensor(collector.values[key], dtype=torch.float32))

    expected_positions = set(range(len(inputs)))
    if collector.applied_positions != expected_positions:
        raise CheckedStatefulError(
            f"canonical IDs applied at {sorted(collector.applied_positions)}; "
            f"expected {sorted(expected_positions)}"
        )
    return {
        "outputs": outputs,
        "post_attention_norm": preexpert,
        "backend_calls": provider.calls,
        "candidate_routes": [
            collector.values[f"token.{position}.layer.{layer}.candidate_routed_index"]
            for position in range(len(inputs))
        ],
        "canonical_routes": [
            collector.values[f"token.{position}.layer.{layer}.routed_index"]
            for position in range(len(inputs))
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--c-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--layers", default="2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        cfg = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(input_values, list) or len(input_values) != 8:
            raise CheckedStatefulError("--inputs must contain eight source-bound rows")
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers or any(layer < int(cfg["dense_layers"]) for layer in layers):
            raise CheckedStatefulError("checked stateful profile is sparse-layer only")
        routes = load_canonical_routes(args.selection, fixture, input_values, layers)
        for layer in layers:
            required = {expert for row in routes[layer] for expert in row.indices}
            if set(fixture.experts.get(layer, ())) != required:
                raise CheckedStatefulError(f"layer {layer} fixture expert union changed")
        dtype = DTYPES["bfloat16"]
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        if input_sha256(inputs, dtype) != selection["inputs_sha256"]:
            raise CheckedStatefulError("input hash differs from route archive")

        library, source = build_checked_library(Path(args.out).with_suffix(".runtime.so"))
        lib = ctypes.CDLL(str(library))
        configure_profile_library(lib)
        helper = CheckedPostReductionWeightHelper(
            build_checked_weight_helper(Path(args.out).with_suffix(".weights.so"))
        )
        original_runner = stateful.run_c_stateful
        stateful.run_c_stateful = run_c_stateful_profile
        try:
            analyses = {
                str(layer): boundary.analyze_layer(
                    lib, helper, fixture, config, cfg, layer, inputs,
                    input_values, routes[layer], device=torch.device("cpu"), dtype=dtype,
                )
                for layer in layers
            }
        finally:
            stateful.run_c_stateful = original_runner

        nonexact = sorted(
            int(layer) for layer, value in analyses.items()
            if value["decision"]["classification"] != "stateful_sparse_mlp_ladder_exact"
        )
        result = {
            "format": "inkling-checked-in-bfloat16-stateful-sparse-profile",
            "version": 1,
            "positions": len(input_values),
            "inputs_sha256": input_sha256(inputs, dtype),
            "source": source,
            "execution_profile": composed.collect_execution_profile(),
            "evidence_outcome": {
                "classification": (
                    "checked_in_stateful_sparse_profile_exact"
                    if not nonexact else "checked_in_stateful_sparse_profile_mismatch"
                ),
                "tested_layers": layers,
                "nonexact_layers": nonexact,
            },
            "layers": analyses,
        }
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (
        CanonicalOfficialRouteError, CanonicalRouteError, CheckedStatefulError,
        FixtureError, FixtureReferenceError, LayerParityError, RuntimeError,
        OSError, ValueError, KeyError,
    ) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())