#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Locate the first stateful sparse-MLP boundary after #45.

#45 proves that the full pre-expert path stays BF16-exact at all eight source-
bound positions. Layer 5 is fully raw-exact, while layer 2 first diverges at
final output position 3. This evidence-only probe compares the downstream ladder
per position without changing the composed candidate:

1. injected fixed-ID routed weights;
2. injected shared gammas;
3. aggregate ``moe_out``;
4. MLP short-convolution ``mlp_branch``;
5. final ``layer_out``.

The official side independently reconstructs the full-sequence sparse MLP and
short convolution, then requires its reconstructed final layer output to be
raw-exact to the canonical official decoder-layer output before interpreting
candidate differences. Production sources and public APIs remain unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

import torch

# Install #44's copied-source final-residual transform before building C.
import run_inkling_portable_bf16_complete_sparse_layer  # noqa: F401
import diagnose_inkling_portable_bf16_composed_moe as composed
import diagnose_inkling_portable_bf16_stateful_sparse_layer as stateful
from diagnose_inkling_pre_router import _capture_official_pre_router
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import CanonicalRouteError, RouteRow, load_canonical_routes
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    build_layer_from_fixture,
    verify_config_binding,
)
from inkling_layer_parity import LayerParityError, configure_library
from inkling_release_config import build_transformers_text_config
from run_inkling_fixture_reference_canonical import (
    CanonicalOfficialGate,
    CanonicalOfficialRouteError,
    run_layer_reference_canonical,
)


class StatefulMlpBoundaryError(RuntimeError):
    """The stateful MLP boundary cannot be classified without exact anchors."""


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu().contiguous().reshape(-1)
    candidate = candidate.detach().float().cpu().contiguous().reshape(-1)
    if tuple(reference.shape) != tuple(candidate.shape):
        raise StatefulMlpBoundaryError(
            f"shape mismatch {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    if not reference.numel():
        raise StatefulMlpBoundaryError("cannot compare empty tensors")
    delta = (reference - candidate).abs()
    rb = reference.to(torch.bfloat16).float()
    cb = candidate.to(torch.bfloat16).float()
    bd = (rb - cb).abs()
    return {
        "count": int(reference.numel()),
        "raw_exact_fraction": float(reference.eq(candidate).float().mean()),
        "raw_max_abs": float(delta.max()),
        "raw_mean_abs": float(delta.mean()),
        "bfloat16_exact_fraction": float(rb.eq(cb).float().mean()),
        "bfloat16_max_abs": float(bd.max()),
        "bfloat16_mean_abs": float(bd.mean()),
    }


def _first_nonexact(values: list[dict[str, Any]], field: str) -> int | None:
    return next((i for i, value in enumerate(values) if value[field] != 1.0), None)


def classify_ladder(stages: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    policies = (
        ("routed_weight", "raw_exact_fraction"),
        ("shared_weight", "raw_exact_fraction"),
        ("moe_out", "raw_exact_fraction"),
        # #38 proves hidden FP32 MLP-branch values need not match; BF16-visible
        # exactness is the contract consumed by the final residual boundary.
        ("mlp_branch", "bfloat16_exact_fraction"),
        ("layer_out", "raw_exact_fraction"),
    )
    first: dict[str, int | None] = {
        stage: _first_nonexact(stages[stage], field)
        for stage, field in policies
    }
    for stage, _field in policies:
        if first[stage] is not None:
            return {
                "classification": f"stateful_{stage}_mismatch",
                "first_boundary_stage": stage,
                "first_boundary_position": first[stage],
                "first_nonexact_positions": first,
            }
    return {
        "classification": "stateful_sparse_mlp_ladder_exact",
        "first_boundary_stage": None,
        "first_boundary_position": None,
        "first_nonexact_positions": first,
    }


def _trace_tensor(collector: Any, layer: int, position: int, point: str) -> torch.Tensor:
    key = f"token.{position}.layer.{layer}.{point}"
    values = collector.values.get(key)
    if values is None:
        raise StatefulMlpBoundaryError(f"candidate trace omitted {key}")
    return torch.tensor(values, dtype=torch.float32)


def official_mlp_sequence(
    fixture: Any,
    config: Any,
    layer: int,
    normalized: torch.Tensor,
    residual: torch.Tensor,
    rows: list[RouteRow],
    canonical_outputs: dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    normalized = normalized.detach().to(device=device, dtype=dtype)
    residual = residual.detach().to(device=device, dtype=dtype)
    gate = CanonicalOfficialGate(module.mlp.gate, rows)
    module.mlp.gate = gate

    with torch.no_grad():
        _logits, routed_weights, _ids, shared_gammas = gate(normalized)
        moe_out = module.mlp(normalized)
        mlp_branch = module.mlp_sconv(
            moe_out.unsqueeze(0),
            past_key_values=None,
            conv_mask=None,
        )[0]
        layer_out = residual + mlp_branch

    if not gate.applied:
        raise StatefulMlpBoundaryError(
            f"canonical gate was not applied for layer {layer}"
        )
    anchors: list[dict[str, Any]] = []
    for position in range(normalized.shape[0]):
        key = f"token.{position}.layer.{layer}.output"
        if key not in canonical_outputs:
            raise StatefulMlpBoundaryError(f"official trace omitted {key}")
        anchor = metrics(canonical_outputs[key], layer_out[position])
        if anchor["raw_exact_fraction"] != 1.0:
            raise StatefulMlpBoundaryError(
                f"layer {layer} official MLP reconstruction is not raw-exact "
                f"at position {position}"
            )
        anchors.append(anchor)
    return {
        "routed_weight": routed_weights.detach().float().cpu(),
        "shared_weight": shared_gammas.detach().float().cpu(),
        "moe_out": moe_out.detach().float().cpu(),
        "mlp_branch": mlp_branch.detach().float().cpu(),
        "layer_out": layer_out.detach().float().cpu(),
        "layer_out_anchor": anchors,
    }


def analyze_layer(
    lib: Any,
    helper: Any,
    fixture: Any,
    config: Any,
    cfg: dict[str, Any],
    layer: int,
    inputs: torch.Tensor,
    input_values: list[list[float]],
    rows: list[RouteRow],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    official_pre, _, _, _ = _capture_official_pre_router(
        fixture, config, layer, inputs, device=device, dtype=dtype
    )
    canonical_outputs = run_layer_reference_canonical(
        fixture,
        config,
        layer,
        input_values,
        rows,
        device=device,
        dtype=dtype,
    )
    official = official_mlp_sequence(
        fixture,
        config,
        layer,
        official_pre["post_attention_norm"],
        official_pre["post_attention_residual"],
        rows,
        canonical_outputs,
        device=device,
        dtype=dtype,
    )

    provider_module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    provider = composed.CapturingMoeProvider(layer, provider_module)
    collector = stateful.StatefulExactWeightCollector(
        rows,
        provider,
        helper,
        int(provider_module.mlp.gate.num_experts),
        float(provider_module.mlp.gate.route_scale),
        float(
            provider_module.mlp.gate.global_scale.detach()
            .float().cpu().reshape(-1)[0]
        ),
    )
    candidate = stateful.run_c_stateful(
        lib, fixture, cfg, layer, input_values, rows, provider, collector
    )

    stages: dict[str, list[dict[str, Any]]] = {
        "routed_weight": [],
        "shared_weight": [],
        "moe_out": [],
        "mlp_branch": [],
        "layer_out": [],
    }
    for position in range(len(input_values)):
        stages["routed_weight"].append(
            metrics(
                official["routed_weight"][position],
                _trace_tensor(collector, layer, position, "routed_weight"),
            )
        )
        stages["shared_weight"].append(
            metrics(
                official["shared_weight"][position],
                _trace_tensor(collector, layer, position, "shared_weight"),
            )
        )
        stages["moe_out"].append(
            metrics(
                official["moe_out"][position],
                _trace_tensor(collector, layer, position, "moe_out"),
            )
        )
        stages["mlp_branch"].append(
            metrics(
                official["mlp_branch"][position],
                _trace_tensor(collector, layer, position, "mlp_branch"),
            )
        )
        stages["layer_out"].append(
            metrics(
                official["layer_out"][position],
                candidate["outputs"][position],
            )
        )

    preexpert = [
        metrics(
            official_pre["post_attention_norm"][position],
            candidate["post_attention_norm"][position],
        )
        for position in range(len(input_values))
    ]
    if any(value["bfloat16_exact_fraction"] != 1.0 for value in preexpert):
        raise StatefulMlpBoundaryError(
            f"layer {layer} pre-expert regression invalidates MLP diagnosis"
        )
    return {
        "decision": classify_ladder(stages),
        "preexpert": preexpert,
        "stages": stages,
        "official_layer_out_anchor": official["layer_out_anchor"],
        "candidate_routes": candidate["candidate_routes"],
        "canonical_routes": candidate["canonical_routes"],
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
            raise StatefulMlpBoundaryError(
                "--inputs must contain the eight source-bound deterministic rows"
            )
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise StatefulMlpBoundaryError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection, fixture, input_values, layers
        )
        for layer in layers:
            required = {expert for row in routes[layer] for expert in row.indices}
            if set(fixture.experts.get(layer, ())) != required:
                raise StatefulMlpBoundaryError(
                    f"layer {layer} fixture expert union changed"
                )
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise StatefulMlpBoundaryError("the stateful MLP probe requires CPU")

        library, source = composed.build_composed_library(
            Path(args.out).with_suffix(".runtime.so")
        )
        if not source["production_source_unchanged"]:
            raise StatefulMlpBoundaryError("production source changed")
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        helper = composed.FixedWeightHelper(
            composed.build_helper(Path(args.out).with_suffix(".weights.so"))
        )
        analyses = {
            str(layer): analyze_layer(
                lib, helper, fixture, config, cfg, layer, inputs,
                input_values, routes[layer], device=device, dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-bfloat16-stateful-mlp-boundary-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "positions": len(input_values),
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
        CanonicalRouteError,
        CanonicalOfficialRouteError,
        StatefulMlpBoundaryError,
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
