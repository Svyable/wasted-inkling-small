#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the checked-in Inkling BF16 sparse-layer profile on official bytes.

This is the promotion gate between the retained evidence transforms and normal
checked-in C.  It compiles ``inkling/src`` unchanged, calls
``waste_inkling_layer_step_backend_trace_profile(..., BF16_REFERENCE)``, and
compares the same named activations used by the temporary evidence stack.

Routing remains two explicit contracts.  The raw deterministic WASTE route and
weights are preserved in the report.  For downstream arithmetic only, the
source-bound canonical official IDs and exact reference-profile weights are
injected through an explicit #51 adapter.  That keeps a platform-specific
top-k tie from changing the expert bank while still making any raw router
mismatch visible instead of laundering it into a parity claim.

All kernel-sensitive matrices, including routed experts, are supplied only by
the native BF16 matrix backend.  The C call deliberately receives no
``expert_get`` callback, so a green result proves the promoted profile does not
silently depend on a second resident/streaming expert storage path.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import diagnose_inkling_portable_bf16_composed_moe as composed
from checked_bf16_profile_collector import (
    CheckedExactWeightCollector,
    CheckedPostReductionWeightHelper,
    build_checked_weight_helper,
)
from diagnose_inkling_pre_router import _capture_official_pre_router
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import (
    CanonicalRouteError,
    RouteRow,
    load_canonical_routes,
)
from inkling_compact_layer_parity import bind_sparse_layer_compact
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    build_layer_from_fixture,
    verify_config_binding,
)
from inkling_layer_parity import (
    LayerParityError,
    LayerScratch,
    LayerState,
    _build_config,
    build_library,
    configure_library,
)
from inkling_release_config import build_transformers_text_config
from run_inkling_fixture_reference_canonical import CanonicalOfficialRouteError


BF16_REFERENCE = 1
ALL_POINTS = tuple(composed.PRE_ROUTER_POINTS) + tuple(composed.STAGES)


class CheckedBfloat16ProfileError(RuntimeError):
    """The checked-in BF16 promotion profile cannot proceed without guessing."""


def build_checked_library(out: Path) -> tuple[Path, dict[str, Any]]:
    source_root = Path(__file__).resolve().parents[2] / "inkling" / "src"
    layer = (source_root / "inkling_layer.c").read_bytes()
    attention = (source_root / "inkling_attention.c").read_bytes()
    library = build_library(source_root, out)
    return library, {
        "production_source_unchanged": True,
        "compiled_source_root": str(source_root),
        "layer_sha256": hashlib.sha256(layer).hexdigest(),
        "attention_sha256": hashlib.sha256(attention).hexdigest(),
        "numeric_profile": "WASTE_INKLING_NUMERIC_BF16_REFERENCE",
        "numeric_profile_value": BF16_REFERENCE,
        "source_rewriting": False,
        "routed_expert_storage": "matrix-backend-only",
    }


def configure_profile_library(lib: ctypes.CDLL) -> None:
    configure_library(lib)
    fn = lib.waste_inkling_layer_step_backend_trace_profile
    fn.restype = ctypes.c_int
    fn.argtypes = list(lib.waste_inkling_layer_step_backend_trace.argtypes) + [
        ctypes.c_int
    ]


def run_c_layer_profile(
    lib: Any,
    fixture: Any,
    cfg: dict[str, Any],
    layer: int,
    input_row: list[float],
    row: RouteRow,
    provider: composed.CapturingMoeProvider,
    collector: CheckedExactWeightCollector,
) -> dict[str, Any]:
    compact = bind_sparse_layer_compact(fixture, cfg, layer)
    if compact.provider is None:
        raise CheckedBfloat16ProfileError("sparse fixture has no expert provider")
    missing = sorted(set(row.indices).difference(compact.provider.experts))
    if missing:
        raise CheckedBfloat16ProfileError(
            f"layer {layer} fixture omits experts {missing}"
        )

    # Force every kernel-sensitive matrix through the native BF16 backend.  The
    # checked-in profile itself also refuses the scalar resident fallback.
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
    capacity = 1

    nfloat = lib.waste_inkling_layer_scratch_floats(
        ctypes.byref(config), layer, capacity
    )
    nint = lib.waste_inkling_layer_scratch_ints(ctypes.byref(config))
    fbuf = (ctypes.c_float * nfloat)()
    ibuf = (ctypes.c_int * max(nint, 1))()
    scratch = LayerScratch()
    if lib.waste_inkling_layer_scratch_init(
        ctypes.byref(scratch), ctypes.byref(config), layer, capacity,
        fbuf, ctypes.c_size_t(nfloat), ibuf, ctypes.c_size_t(nint),
    ):
        raise CheckedBfloat16ProfileError("scratch init failed")

    kv = (ctypes.c_float * (capacity * kv_heads * head_dim))()
    vv = (ctypes.c_float * (capacity * kv_heads * head_dim))()
    kconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    vconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    aconv = (ctypes.c_float * (hidden * conv_k))()
    mconv = (ctypes.c_float * (hidden * conv_k))()
    state = LayerState()
    if lib.waste_inkling_layer_state_init(
        ctypes.byref(state), ctypes.byref(config), layer, capacity,
        kv, vv, kconv, vconv, aconv, mconv,
    ):
        raise CheckedBfloat16ProfileError("state init failed")

    collector.position = 0
    x = (ctypes.c_float * hidden)(*input_row)
    rc = lib.waste_inkling_layer_step_backend_trace_profile(
        ctypes.byref(config), layer, ctypes.byref(compact.weights),
        ctypes.byref(provider.backend), ctypes.byref(state), x, 0,
        ctypes.byref(scratch), ctypes.c_void_p(), None,
        ctypes.byref(collector.c_trace), BF16_REFERENCE,
    )
    if provider.error:
        raise CheckedBfloat16ProfileError(provider.error)
    if collector.error:
        raise CheckedBfloat16ProfileError(collector.error)
    if rc:
        raise CheckedBfloat16ProfileError(
            f"checked-in BF16 layer failed with status {rc}"
        )
    if collector.applied_positions != {0}:
        raise CheckedBfloat16ProfileError("canonical IDs were not injected")

    stages: dict[str, torch.Tensor] = {}
    for point in ALL_POINTS:
        key = f"token.0.layer.{layer}.{point}"
        if key not in collector.values:
            raise CheckedBfloat16ProfileError(f"C trace missed {key}")
        stages[point] = torch.tensor(collector.values[key], dtype=torch.float32)

    return {
        "stages": stages,
        "candidate_routed_index": collector.values[
            f"token.0.layer.{layer}.candidate_routed_index"
        ],
        "candidate_routed_weight": collector.values[
            f"token.0.layer.{layer}.candidate_routed_weight"
        ],
        "candidate_shared_weight": collector.values[
            f"token.0.layer.{layer}.candidate_shared_weight"
        ],
        "canonical_routed_index": collector.values[
            f"token.0.layer.{layer}.routed_index"
        ],
        "exact_routed_weight": collector.values[
            f"token.0.layer.{layer}.routed_weight"
        ],
        "exact_shared_weight": collector.values[
            f"token.0.layer.{layer}.shared_weight"
        ],
        "backend_calls": provider.calls,
        "expert_get_supplied": False,
    }


def first_nonexact(metrics: dict[str, dict[str, Any]]) -> str | None:
    for point in ALL_POINTS:
        if metrics[point]["bfloat16_exact_fraction"] != 1.0:
            return point
    return None


def analyze_layer(
    lib: Any,
    helper: CheckedPostReductionWeightHelper,
    fixture: Any,
    config: Any,
    cfg: dict[str, Any],
    layer: int,
    inputs: torch.Tensor,
    input_values: list[list[float]],
    row: RouteRow,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    pre, _, _, _ = _capture_official_pre_router(
        fixture, config, layer, inputs, device=device, dtype=dtype
    )
    module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    official_moe, official_route = composed.official_stages(
        module,
        pre["post_attention_norm"][0],
        pre["post_attention_residual"][0],
        row,
    )
    official = {
        **{
            point: pre[point][0].detach().float().reshape(-1)
            for point in composed.PRE_ROUTER_POINTS
        },
        **official_moe,
    }

    provider = composed.CapturingMoeProvider(layer, module)
    collector = CheckedExactWeightCollector(
        row,
        provider,
        helper,
        int(module.mlp.gate.num_experts),
        float(module.mlp.gate.route_scale),
        float(module.mlp.gate.global_scale.detach().float().cpu().reshape(-1)[0]),
    )
    candidate = run_c_layer_profile(
        lib, fixture, cfg, layer, input_values[0], row, provider, collector
    )
    comparisons = {
        point: composed.metrics(official[point], candidate["stages"][point])
        for point in ALL_POINTS
    }
    first = first_nonexact(comparisons)
    raw_route = candidate["candidate_routed_index"]
    canonical_route = candidate["canonical_routed_index"]
    route_matches = raw_route == canonical_route
    routed_weights_match = (
        route_matches
        and candidate["candidate_routed_weight"] == candidate["exact_routed_weight"]
    )
    shared_weights_match = (
        candidate["candidate_shared_weight"] == candidate["exact_shared_weight"]
    )
    return {
        "decision": {
            "classification": (
                "checked_in_bf16_sparse_layer_exact"
                if first is None
                else "checked_in_bf16_sparse_layer_mismatch"
            ),
            "first_nonexact_stage": first,
            "exact_bfloat16_prefix": (
                list(ALL_POINTS)
                if first is None
                else list(ALL_POINTS[: ALL_POINTS.index(first)])
            ),
        },
        "stages": comparisons,
        "official_route": official_route,
        "candidate_routed_index": raw_route,
        "canonical_routed_index": canonical_route,
        "candidate_routed_weight": candidate["candidate_routed_weight"],
        "candidate_shared_weight": candidate["candidate_shared_weight"],
        "exact_routed_weight": candidate["exact_routed_weight"],
        "exact_shared_weight": candidate["exact_shared_weight"],
        "router_contract": {
            "raw_route_matches_canonical": route_matches,
            "raw_routed_weights_match_exact": routed_weights_match,
            "raw_shared_weights_match_exact": shared_weights_match,
            "downstream_uses_canonical_ids_and_exact_weights": True,
        },
        "backend_contract": {
            "routed_experts_use_matrix_backend_only": True,
            "expert_get_supplied": candidate["expert_get_supplied"],
        },
        "backend_calls": candidate["backend_calls"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--c-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    args = parser.parse_args(argv)

    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        cfg = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(input_values, list) or len(input_values) != 8:
            raise CheckedBfloat16ProfileError(
                "--inputs must contain the eight source-bound rows"
            )
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers or any(layer < int(cfg["dense_layers"]) for layer in layers):
            raise CheckedBfloat16ProfileError(
                "the checked-in sparse BF16 gate requires sparse layers"
            )
        routes = load_canonical_routes(
            args.selection, fixture, input_values, layers
        )
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise CheckedBfloat16ProfileError(
                "the checked-in BF16 promotion probe requires CPU"
            )
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        if input_sha256(inputs, dtype) != selection["inputs_sha256"]:
            raise CheckedBfloat16ProfileError("input hash differs from route archive")

        library, source = build_checked_library(Path(args.out).with_suffix(".runtime.so"))
        lib = ctypes.CDLL(str(library))
        configure_profile_library(lib)
        helper = CheckedPostReductionWeightHelper(
            build_checked_weight_helper(Path(args.out).with_suffix(".weights.so"))
        )
        analyses = {
            str(layer): analyze_layer(
                lib,
                helper,
                fixture,
                config,
                cfg,
                layer,
                inputs,
                input_values,
                routes[layer][0],
                device=device,
                dtype=dtype,
            )
            for layer in layers
        }
        nonexact = sorted(
            int(layer)
            for layer, value in analyses.items()
            if value["decision"]["classification"]
            != "checked_in_bf16_sparse_layer_exact"
        )
        result = {
            "format": "inkling-checked-in-bfloat16-sparse-profile",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "executed_position": 0,
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "execution_profile": composed.collect_execution_profile(),
            "source": source,
            "evidence_outcome": {
                "classification": (
                    "checked_in_bf16_sparse_profile_exact"
                    if not nonexact
                    else "checked_in_bf16_sparse_profile_mismatch"
                ),
                "tested_layers": layers,
                "nonexact_layers": nonexact,
            },
            "layers": analyses,
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        CanonicalOfficialRouteError,
        CanonicalRouteError,
        CheckedBfloat16ProfileError,
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())