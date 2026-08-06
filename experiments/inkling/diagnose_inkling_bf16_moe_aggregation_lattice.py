#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Map Inkling BF16 sparse-MoE weighting and accumulation semantics.

The complete expert-local path is already bit-exact.  This evidence-only probe
therefore holds every gate/up/activation/down tensor fixed and varies only:

* routed contribution completion and accumulation order/dtype;
* shared gamma placement before or after the down projection;
* shared summation dtype and completion;
* the final routed-plus-shared addition.

Both official and actual temporary-C aggregate reconstructions must be exact
before any counterfactual is interpreted.  Production sources remain unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from diagnose_inkling_portable_bf16_single_token_moe import (
    MoeMatrixProvider,
    SingleTokenMoeError,
)
from diagnose_inkling_portable_bf16_single_token_moe_activation import (
    MoeActivationError,
    build_activation_library,
)
from diagnose_inkling_pre_router import (
    PreRouterDiagnosisError,
    _capture_official_pre_router,
)
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import (
    CanonicalRouteCollector,
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
    configure_library,
)
from inkling_release_config import build_transformers_text_config
from run_inkling_fixture_reference_canonical import (
    CanonicalOfficialGate,
    CanonicalOfficialRouteError,
)


class MoeAggregationError(RuntimeError):
    """The MoE aggregation lattice cannot be trusted without exact anchors."""


def bf16(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(torch.bfloat16).cpu().contiguous()


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu().contiguous()
    candidate = candidate.detach().float().cpu().contiguous()
    if tuple(reference.shape) != tuple(candidate.shape):
        raise MoeAggregationError(
            f"shape mismatch {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    if not reference.numel():
        raise MoeAggregationError("cannot compare empty tensors")
    delta = (reference - candidate).abs()
    reference_bf16 = reference.to(torch.bfloat16).float()
    candidate_bf16 = candidate.to(torch.bfloat16).float()
    bf16_delta = (reference_bf16 - candidate_bf16).abs()
    return {
        "count": int(reference.numel()),
        "raw_exact_fraction": float(reference.eq(candidate).float().mean()),
        "raw_max_abs": float(delta.max()),
        "raw_mean_abs": float(delta.mean()),
        "bfloat16_exact_fraction": float(
            reference_bf16.eq(candidate_bf16).float().mean()
        ),
        "bfloat16_max_abs": float(bf16_delta.max()),
        "bfloat16_mean_abs": float(bf16_delta.mean()),
    }


def _first_exact(
    variants: dict[str, dict[str, Any]], preferred: tuple[str, ...]
) -> str | None:
    for name in preferred:
        value = variants[name]
        if value["raw_exact_fraction"] == 1.0:
            return name
    return None


def classify_lattice(
    anchors: dict[str, dict[str, Any]],
    routed: dict[str, dict[str, Any]],
    shared: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
    shared_weights: dict[str, Any],
) -> dict[str, Any]:
    failed = [
        name
        for name, value in anchors.items()
        if value["raw_exact_fraction"] != 1.0
    ]
    if failed:
        return {
            "classification": "aggregation_anchor_failed",
            "failed_anchors": failed,
            "routed_policy": None,
            "shared_policy": None,
            "final_add_policy": None,
        }
    routed_policy = _first_exact(
        routed,
        (
            "route_order_bfloat16_sequential",
            "expert_order_bfloat16_sequential",
            "expert_order_native_index_add",
        ),
    )
    shared_policy = _first_exact(
        shared,
        (
            "official_gamma_after_down_bfloat16_contribution_float32_sum_bfloat16",
            "official_gamma_before_down_float32_sum_bfloat16",
            "official_gamma_before_down_bfloat16_sequential",
        ),
    )
    final_policy = _first_exact(
        final,
        (
            "float32_add_result_bfloat16",
            "native_bfloat16_add",
        ),
    )
    if not routed_policy or not shared_policy or not final_policy:
        return {
            "classification": "aggregation_policy_remains_incomplete",
            "failed_anchors": [],
            "routed_policy": routed_policy,
            "shared_policy": shared_policy,
            "final_add_policy": final_policy,
            "candidate_shared_weights_bfloat16_exact": (
                shared_weights["bfloat16_exact_fraction"] == 1.0
            ),
        }
    return {
        "classification": "moe_weighting_and_accumulation_policy_identified",
        "failed_anchors": [],
        "routed_policy": routed_policy,
        "shared_policy": shared_policy,
        "final_add_policy": final_policy,
        "candidate_shared_weights_bfloat16_exact": (
            shared_weights["bfloat16_exact_fraction"] == 1.0
        ),
    }


def run_c_aggregate(
    lib: Any,
    fixture: Any,
    cfg: dict[str, Any],
    layer: int,
    input_row: list[float],
    route: RouteRow,
    provider: MoeMatrixProvider,
) -> dict[str, Any]:
    compact = bind_sparse_layer_compact(fixture, cfg, layer)
    if compact.provider is None:
        raise MoeAggregationError("sparse fixture has no expert provider")
    available = set(compact.provider.experts)
    missing = sorted(set(route.indices).difference(available))
    if missing:
        raise MoeAggregationError(
            f"layer {layer} fixture omits canonical experts {missing}"
        )
    for field in (
        "wq",
        "wk",
        "wv",
        "wr",
        "wo",
        "router_weight",
        "shared_gate",
        "shared_up",
        "shared_down",
    ):
        setattr(compact.weights, field, ctypes.POINTER(ctypes.c_float)())

    config = _build_config(lib, cfg)
    hidden = int(cfg["hidden"])
    is_local = layer in set(cfg.get("local_layer_ids", []))
    kv_heads = int(
        cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"]
    )
    head_dim = int(
        cfg["local_head_dim"] if is_local else cfg["global_head_dim"]
    )
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
        ctypes.byref(scratch),
        ctypes.byref(config),
        layer,
        capacity,
        fbuf,
        ctypes.c_size_t(nfloat),
        ibuf,
        ctypes.c_size_t(nint),
    ):
        raise MoeAggregationError("scratch init failed")
    kv = (ctypes.c_float * (capacity * kv_heads * head_dim))()
    vv = (ctypes.c_float * (capacity * kv_heads * head_dim))()
    kconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    vconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    aconv = (ctypes.c_float * (hidden * conv_k))()
    mconv = (ctypes.c_float * (hidden * conv_k))()
    state = LayerState()
    if lib.waste_inkling_layer_state_init(
        ctypes.byref(state),
        ctypes.byref(config),
        layer,
        capacity,
        kv,
        vv,
        kconv,
        vconv,
        aconv,
        mconv,
    ):
        raise MoeAggregationError("state init failed")

    collector = CanonicalRouteCollector([route])
    collector.position = 0
    x = (ctypes.c_float * hidden)(*input_row)
    expert_get = ctypes.cast(compact.provider.callback, ctypes.c_void_p)
    rc = lib.waste_inkling_layer_step_backend_trace(
        ctypes.byref(config),
        layer,
        ctypes.byref(compact.weights),
        ctypes.byref(provider.backend),
        ctypes.byref(state),
        x,
        0,
        ctypes.byref(scratch),
        expert_get,
        None,
        ctypes.byref(collector.c_trace),
    )
    if provider.error:
        raise MoeAggregationError(provider.error)
    if collector.error:
        raise MoeAggregationError(collector.error)
    if rc:
        raise MoeAggregationError(f"C layer failed with status {rc}")
    if collector.applied_positions != {0}:
        raise MoeAggregationError("canonical route was not applied")

    def values(point: str) -> list[float]:
        name = f"token.0.layer.{layer}.{point}"
        if name not in collector.values:
            raise MoeAggregationError(f"C trace missed {name}")
        return [float(item) for item in collector.values[name]]

    return {
        "moe_out": torch.tensor(values("moe_out"), dtype=torch.float32),
        "candidate_shared_weights": torch.tensor(
            values("shared_weight"), dtype=torch.float32
        ),
        "canonical_routed_weights": torch.tensor(
            values("routed_weight"), dtype=torch.float32
        ),
        "candidate_routed_weights": torch.tensor(
            values("candidate_routed_weight"), dtype=torch.float32
        ),
        "backend_calls": provider.calls,
    }


def routed_components(
    module: Any,
    normalized: torch.Tensor,
    routed_ids: torch.Tensor,
    routed_weights: torch.Tensor,
) -> tuple[list[int], list[torch.Tensor], torch.Tensor]:
    ids = [int(value) for value in routed_ids[0].detach().cpu().tolist()]
    weights = routed_weights[0].detach().to(torch.bfloat16).cpu()
    downs: list[torch.Tensor] = []
    for expert in ids:
        fused = module.mlp.experts.gate_up[str(expert)]
        gate, up = F.linear(normalized, fused).chunk(2, dim=-1)
        gated = F.silu(gate) * up
        downs.append(
            F.linear(gated, module.mlp.experts.down[str(expert)])[0]
            .detach()
            .to(torch.bfloat16)
            .cpu()
        )
    return ids, downs, weights


def routed_variants(
    ids: list[int], downs: list[torch.Tensor], weights: torch.Tensor
) -> dict[str, torch.Tensor]:
    hidden = int(downs[0].numel())
    route_positions = list(range(len(ids)))
    expert_positions = sorted(route_positions, key=lambda pos: ids[pos])

    def float32_sum(order: list[int], complete_contribution: bool) -> torch.Tensor:
        acc = torch.zeros(hidden, dtype=torch.float32)
        for pos in order:
            term = downs[pos] * weights[pos]
            if complete_contribution:
                term = term.to(torch.bfloat16)
            else:
                term = downs[pos].float() * weights[pos].float()
            acc = acc + term.float()
        return acc

    def bfloat16_sum(order: list[int]) -> torch.Tensor:
        acc = torch.zeros(hidden, dtype=torch.bfloat16)
        for pos in order:
            contribution = (downs[pos] * weights[pos]).to(torch.bfloat16)
            acc = (acc + contribution).to(torch.bfloat16)
        return acc

    native = torch.zeros((1, hidden), dtype=torch.bfloat16)
    index = torch.tensor([0], dtype=torch.int64)
    for pos in expert_positions:
        contribution = (downs[pos] * weights[pos]).reshape(1, -1)
        native.index_add_(0, index, contribution.to(torch.bfloat16))

    route_complete_f32 = float32_sum(route_positions, True)
    return {
        "route_order_float32": float32_sum(route_positions, False),
        "route_order_bfloat16_contribution_float32_sum": route_complete_f32,
        "route_order_bfloat16_contribution_float32_sum_bfloat16": (
            route_complete_f32.to(torch.bfloat16)
        ),
        "route_order_bfloat16_sequential": bfloat16_sum(route_positions),
        "expert_order_bfloat16_sequential": bfloat16_sum(expert_positions),
        "expert_order_native_index_add": native[0],
    }


def shared_components(
    module: Any,
    normalized: torch.Tensor,
    official_gammas: torch.Tensor,
    candidate_gammas: torch.Tensor,
) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    official_gammas = official_gammas[0].detach().to(torch.bfloat16).cpu()
    candidate_gammas = candidate_gammas.detach().float().cpu()
    products: list[torch.Tensor] = []
    unweighted_downs: list[torch.Tensor] = []
    official_before_downs: list[torch.Tensor] = []
    candidate_before_downs: list[torch.Tensor] = []
    n_shared = int(module.mlp.shared_experts.n_shared_experts)
    if official_gammas.numel() != n_shared or candidate_gammas.numel() != n_shared:
        raise MoeAggregationError("shared gamma geometry changed")
    for expert in range(n_shared):
        gate = F.linear(normalized, module.mlp.shared_experts.gate_proj[expert])
        up = F.linear(normalized, module.mlp.shared_experts.up_proj[expert])
        product = (F.silu(gate) * up).to(torch.bfloat16)
        down_weight = module.mlp.shared_experts.down_proj[expert]
        unweighted = F.linear(product, down_weight)[0].to(torch.bfloat16)
        official_weighted = (
            product * official_gammas[expert].to(product.device)
        ).to(torch.bfloat16)
        candidate_weighted = (
            product
            * candidate_gammas[expert]
            .to(torch.bfloat16)
            .to(product.device)
        ).to(torch.bfloat16)
        products.append(product[0].detach().cpu())
        unweighted_downs.append(unweighted.detach().cpu())
        official_before_downs.append(
            F.linear(official_weighted, down_weight)[0]
            .detach()
            .to(torch.bfloat16)
            .cpu()
        )
        candidate_before_downs.append(
            F.linear(candidate_weighted, down_weight)[0]
            .detach()
            .to(torch.bfloat16)
            .cpu()
        )
    return {
        "official_gammas": official_gammas,
        "candidate_gammas": candidate_gammas,
        "products": products,
        "unweighted_downs": unweighted_downs,
        "official_before_downs": official_before_downs,
        "candidate_before_downs": candidate_before_downs,
    }


def shared_variants(components: dict[str, Any]) -> dict[str, torch.Tensor]:
    official_gammas = components["official_gammas"]
    candidate_gammas = components["candidate_gammas"]
    unweighted = components["unweighted_downs"]
    official_before = components["official_before_downs"]
    candidate_before = components["candidate_before_downs"]
    hidden = int(unweighted[0].numel())

    def after_down(gammas: torch.Tensor, complete: bool) -> torch.Tensor:
        acc = torch.zeros(hidden, dtype=torch.float32)
        for expert, down in enumerate(unweighted):
            if complete:
                term = (
                    down * gammas[expert].to(torch.bfloat16)
                ).to(torch.bfloat16)
            else:
                term = down.float() * gammas[expert].float()
            acc = acc + term.float()
        return acc

    def sum_before(values: list[torch.Tensor], bfloat16_sequential: bool) -> torch.Tensor:
        if bfloat16_sequential:
            acc = torch.zeros(hidden, dtype=torch.bfloat16)
            for value in values:
                acc = (acc + value.to(torch.bfloat16)).to(torch.bfloat16)
            return acc
        return torch.stack([value.float() for value in values]).sum(0).to(
            torch.bfloat16
        )

    official_after_complete = after_down(official_gammas, True)
    return {
        "candidate_gamma_after_down_float32_sum": after_down(
            candidate_gammas, False
        ),
        "official_gamma_after_down_float32_sum": after_down(
            official_gammas, False
        ),
        "official_gamma_after_down_bfloat16_contribution_float32_sum_bfloat16": (
            official_after_complete.to(torch.bfloat16)
        ),
        "candidate_gamma_before_down_float32_sum_bfloat16": sum_before(
            candidate_before, False
        ),
        "official_gamma_before_down_float32_sum_bfloat16": sum_before(
            official_before, False
        ),
        "official_gamma_before_down_bfloat16_sequential": sum_before(
            official_before, True
        ),
    }


def c_combined_reconstruction(
    ids: list[int],
    routed_downs: list[torch.Tensor],
    routed_weights: torch.Tensor,
    shared_downs: list[torch.Tensor],
    candidate_shared_weights: torch.Tensor,
) -> torch.Tensor:
    del ids  # C follows canonical route slot order, already represented by lists.
    hidden = int(routed_downs[0].numel())
    acc = torch.zeros(hidden, dtype=torch.float32)
    for down, weight in zip(routed_downs, routed_weights):
        acc = acc + down.float() * weight.float()
    for down, weight in zip(shared_downs, candidate_shared_weights):
        acc = acc + down.float() * weight.float()
    return acc


def analyze_layer(
    lib: Any,
    fixture: Any,
    config: Any,
    cfg: dict[str, Any],
    layer: int,
    inputs: torch.Tensor,
    input_values: list[list[float]],
    route: RouteRow,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    pre_router, _, _, _ = _capture_official_pre_router(
        fixture,
        config,
        layer,
        inputs,
        device=device,
        dtype=dtype,
    )
    module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    normalized = pre_router["post_attention_norm"][0].to(
        device=device, dtype=torch.bfloat16
    ).reshape(1, -1)
    gate = CanonicalOfficialGate(module.mlp.gate, [route])
    _router_logits, routed_weights, routed_ids, shared_gammas = gate(normalized)
    if not gate.applied:
        raise MoeAggregationError("canonical official gate was not applied")

    provider = MoeMatrixProvider(layer, module)
    candidate = run_c_aggregate(
        lib, fixture, cfg, layer, input_values[0], route, provider
    )

    official_routed = module.mlp.experts(
        normalized, routed_ids, routed_weights
    )[0].detach().cpu()
    official_shared = module.mlp.shared_experts(
        normalized, gammas=shared_gammas
    )[0].detach().cpu()
    official_moe = (official_routed + official_shared).detach().cpu()

    ids, routed_downs, official_routed_weights = routed_components(
        module, normalized, routed_ids, routed_weights
    )
    routed_values = routed_variants(
        ids, routed_downs, official_routed_weights
    )
    routed_metrics = {
        name: metrics(official_routed, value)
        for name, value in routed_values.items()
    }

    shared_parts = shared_components(
        module,
        normalized,
        shared_gammas,
        candidate["candidate_shared_weights"],
    )
    shared_values = shared_variants(shared_parts)
    shared_metrics = {
        name: metrics(official_shared, value)
        for name, value in shared_values.items()
    }

    manual_official_routed = routed_values["expert_order_native_index_add"]
    manual_official_shared = shared_values[
        "official_gamma_before_down_float32_sum_bfloat16"
    ]
    final_values = {
        "float32_add": manual_official_routed.float()
        + manual_official_shared.float(),
        "float32_add_result_bfloat16": (
            manual_official_routed.float() + manual_official_shared.float()
        ).to(torch.bfloat16),
        "native_bfloat16_add": (
            manual_official_routed.to(torch.bfloat16)
            + manual_official_shared.to(torch.bfloat16)
        ),
    }
    final_metrics = {
        name: metrics(official_moe, value)
        for name, value in final_values.items()
    }

    c_reconstruction = c_combined_reconstruction(
        ids,
        routed_downs,
        torch.tensor(route.weights, dtype=torch.float32),
        shared_parts["unweighted_downs"],
        candidate["candidate_shared_weights"],
    )
    anchors = {
        "candidate_c_combined_reconstruction": metrics(
            candidate["moe_out"], c_reconstruction
        ),
        "official_routed_reconstruction": metrics(
            official_routed, manual_official_routed
        ),
        "official_shared_reconstruction": metrics(
            official_shared, manual_official_shared
        ),
        "official_final_reconstruction": metrics(
            official_moe, final_values["native_bfloat16_add"]
        ),
    }
    shared_weight_metrics = metrics(
        shared_gammas[0].detach().cpu(),
        candidate["candidate_shared_weights"],
    )
    decision = classify_lattice(
        anchors,
        routed_metrics,
        shared_metrics,
        final_metrics,
        shared_weight_metrics,
    )
    return {
        "decision": decision,
        "anchors": anchors,
        "shared_weight_metrics": shared_weight_metrics,
        "official_shared_gammas": shared_gammas[0]
        .detach()
        .float()
        .cpu()
        .tolist(),
        "candidate_shared_weights": candidate[
            "candidate_shared_weights"
        ].tolist(),
        "routed_variants": routed_metrics,
        "shared_variants": shared_metrics,
        "final_add_variants": final_metrics,
        "candidate_moe_out": metrics(official_moe, candidate["moe_out"]),
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
            raise MoeAggregationError(
                "--inputs must contain the eight source-bound deterministic rows"
            )
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise MoeAggregationError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection, fixture, input_values, layers
        )
        position_zero = {layer: routes[layer][0] for layer in layers}
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise MoeAggregationError("the aggregation lattice requires CPU")

        library, source = build_activation_library(
            Path(args.out).with_suffix(".so")
        )
        if not source["production_source_unchanged"]:
            raise MoeAggregationError("production source changed")
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        analyses = {
            str(layer): analyze_layer(
                lib,
                fixture,
                config,
                cfg,
                layer,
                inputs,
                input_values,
                position_zero[layer],
                device=device,
                dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-bfloat16-moe-aggregation-lattice",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "source_tokens": len(input_values),
            "executed_position": 0,
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "source": source,
            "layers": analyses,
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
        PreRouterDiagnosisError,
        CanonicalRouteError,
        CanonicalOfficialRouteError,
        SingleTokenMoeError,
        MoeActivationError,
        MoeAggregationError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
