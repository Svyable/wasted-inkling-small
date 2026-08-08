#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the composed BF16 sparse-layer policies statefully across all tokens.

The position-zero arithmetic is already exact in #44. This evidence-only probe
keeps one C layer state alive across the complete source-bound eight-token
sequence and compares two per-position anchors:

* ``post_attention_norm`` against the official full-sequence pre-router trace;
* final decoder-layer output against the official canonical full-sequence layer.

Canonical routed IDs remain an evidence adapter and raw candidate choices are
preserved separately. Production source and the public matrix-backend ABI are
unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

import torch

# Installs #44's copied-source final-residual transform on top of #36.
import run_inkling_portable_bf16_complete_sparse_layer  # noqa: F401
import diagnose_inkling_portable_bf16_composed_moe as composed
from diagnose_inkling_pre_router import _capture_official_pre_router
from diagnose_inkling_bf16_router_weight_lattice import selected_logits
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
    TraceCollector,
    _build_config,
    configure_library,
)
from inkling_release_config import build_transformers_text_config
from run_inkling_fixture_reference_canonical import (
    CanonicalOfficialRouteError,
    run_layer_reference_canonical,
)
from discover_inkling_router_experts import input_sha256


class StatefulSparseLayerError(RuntimeError):
    """Stateful parity cannot proceed without exact source-bound contracts."""


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu().contiguous().reshape(-1)
    candidate = candidate.detach().float().cpu().contiguous().reshape(-1)
    if tuple(reference.shape) != tuple(candidate.shape):
        raise StatefulSparseLayerError(
            f"shape mismatch {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
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


def classify_positions(
    preexpert: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    first_pre = next(
        (i for i, value in enumerate(preexpert)
         if value["bfloat16_exact_fraction"] != 1.0),
        None,
    )
    first_output_raw = next(
        (i for i, value in enumerate(outputs)
         if value["raw_exact_fraction"] != 1.0),
        None,
    )
    first_output_bf16 = next(
        (i for i, value in enumerate(outputs)
         if value["bfloat16_exact_fraction"] != 1.0),
        None,
    )
    if first_pre is not None:
        classification = "stateful_preexpert_mismatch"
    elif first_output_raw is None:
        classification = "stateful_sparse_layer_outputs_raw_exact"
    elif first_output_bf16 is None:
        classification = "stateful_sparse_layer_outputs_bfloat16_exact"
    else:
        classification = "stateful_sparse_layer_output_mismatch"
    return {
        "classification": classification,
        "first_preexpert_nonexact_position": first_pre,
        "first_output_raw_nonexact_position": first_output_raw,
        "first_output_bfloat16_nonexact_position": first_output_bf16,
    }


class StatefulExactWeightCollector(CanonicalRouteCollector):
    """Inject exact fixed-ID weights for the collector's current position."""

    def __init__(
        self,
        rows: list[RouteRow],
        provider: composed.CapturingMoeProvider,
        helper: Any,
        n_routed: int,
        route_scale: float,
        global_scale: float,
    ) -> None:
        super().__init__(rows)
        self.provider = provider
        self.helper = helper
        self.n_routed = int(n_routed)
        self.route_scale = float(route_scale)
        self.global_scale = float(global_scale)
        self.exact_by_position: dict[int, torch.Tensor] = {}

    def _compute_weights(self) -> torch.Tensor:
        cached = self.exact_by_position.get(self.position)
        if cached is not None:
            return cached
        if self.provider.router_logits is None:
            raise StatefulSparseLayerError(
                "router logits were not captured before route weights"
            )
        row = self.rows[self.position]
        ids = torch.tensor(row.indices, dtype=torch.int64)
        selected = selected_logits(
            self.provider.router_logits,
            ids,
            self.n_routed,
        )
        weights = self.helper.evaluate(
            selected,
            self.route_scale,
            self.global_scale,
            1,
            composed.ROUTER_POLICY_FLAGS,
        )
        self.exact_by_position[self.position] = weights
        return weights

    def _store_candidate(
        self,
        layer: int,
        point: bytes,
        data: Any,
        count: int,
    ) -> None:
        name = self._name(layer, point)
        self.values[name] = [float(data[index]) for index in range(count)]
        self.dtypes[name] = "F32"

    def _emit_float(self, ctx, layer, point, data, count) -> int:
        try:
            decoded = point.decode("ascii", "strict")
            count = int(count)
            if decoded == "routed_weight":
                row = self._row(count)
                if row is None:
                    return 1
                self._store_candidate(
                    layer, b"candidate_routed_weight", data, count
                )
                weights = self._compute_weights()
                if weights.numel() < count:
                    raise StatefulSparseLayerError(
                        "exact routed-weight vector is too short"
                    )
                for index in range(count):
                    data[index] = float(weights[index])
            elif decoded == "shared_weight":
                row = self.rows[self.position]
                self._store_candidate(
                    layer, b"candidate_shared_weight", data, count
                )
                weights = self._compute_weights()
                start = len(row.indices)
                if weights.numel() != start + count:
                    raise StatefulSparseLayerError(
                        "exact shared-weight width changed"
                    )
                for index in range(count):
                    data[index] = float(weights[start + index])
            return TraceCollector._emit_float(
                self, ctx, layer, point, data, count
            )
        except Exception as exc:  # ctypes callbacks cannot propagate safely.
            self.error = str(exc)
            return 1


def run_c_stateful(
    lib: Any,
    fixture: Any,
    cfg: dict[str, Any],
    layer: int,
    inputs: list[list[float]],
    rows: list[RouteRow],
    provider: composed.CapturingMoeProvider,
    collector: StatefulExactWeightCollector,
) -> dict[str, Any]:
    compact = bind_sparse_layer_compact(fixture, cfg, layer)
    if compact.provider is None:
        raise StatefulSparseLayerError("sparse fixture has no expert provider")
    required = {expert for row in rows for expert in row.indices}
    missing = sorted(required.difference(compact.provider.experts))
    if missing:
        raise StatefulSparseLayerError(
            f"layer {layer} fixture omits experts {missing}"
        )
    for field in (
        "wq", "wk", "wv", "wr", "wo", "router_weight",
        "shared_gate", "shared_up", "shared_down",
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
    capacity = (
        min(max(len(inputs), 1), int(cfg["sliding_window"]))
        if is_local else max(len(inputs), 1)
    )
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
        raise StatefulSparseLayerError("scratch init failed")
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
        raise StatefulSparseLayerError("state init failed")

    outputs: list[torch.Tensor] = []
    preexpert: list[torch.Tensor] = []
    expert_get = ctypes.cast(compact.provider.callback, ctypes.c_void_p)
    for position, input_row in enumerate(inputs):
        if len(input_row) != hidden:
            raise StatefulSparseLayerError(
                f"input {position} has {len(input_row)} values; expected {hidden}"
            )
        collector.position = position
        provider.router_logits = None
        x = (ctypes.c_float * hidden)(*input_row)
        rc = lib.waste_inkling_layer_step_backend_trace(
            ctypes.byref(config), layer, ctypes.byref(compact.weights),
            ctypes.byref(provider.backend), ctypes.byref(state), x, position,
            ctypes.byref(scratch), expert_get, None,
            ctypes.byref(collector.c_trace),
        )
        if provider.error:
            raise StatefulSparseLayerError(provider.error)
        if collector.error:
            raise StatefulSparseLayerError(collector.error)
        if rc:
            raise StatefulSparseLayerError(
                f"C layer failed at position {position} with status {rc}"
            )
        outputs.append(
            torch.tensor([float(x[i]) for i in range(hidden)], dtype=torch.float32)
        )
        key = f"token.{position}.layer.{layer}.post_attention_norm"
        if key not in collector.values:
            raise StatefulSparseLayerError(f"candidate trace omitted {key}")
        preexpert.append(torch.tensor(collector.values[key], dtype=torch.float32))

    expected_positions = set(range(len(inputs)))
    if collector.applied_positions != expected_positions:
        raise StatefulSparseLayerError(
            f"canonical IDs applied at {sorted(collector.applied_positions)}; "
            f"expected {sorted(expected_positions)}"
        )
    return {
        "outputs": outputs,
        "post_attention_norm": preexpert,
        "backend_calls": provider.calls,
        "candidate_routes": [
            collector.values[
                f"token.{position}.layer.{layer}.candidate_routed_index"
            ]
            for position in range(len(inputs))
        ],
        "canonical_routes": [
            collector.values[f"token.{position}.layer.{layer}.routed_index"]
            for position in range(len(inputs))
        ],
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
    official = run_layer_reference_canonical(
        fixture,
        config,
        layer,
        input_values,
        rows,
        device=device,
        dtype=dtype,
    )
    module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    provider = composed.CapturingMoeProvider(layer, module)
    collector = StatefulExactWeightCollector(
        rows,
        provider,
        helper,
        int(module.mlp.gate.num_experts),
        float(module.mlp.gate.route_scale),
        float(module.mlp.gate.global_scale.detach().float().cpu().reshape(-1)[0]),
    )
    candidate = run_c_stateful(
        lib, fixture, cfg, layer, input_values, rows, provider, collector
    )
    pre_metrics = []
    output_metrics = []
    for position in range(len(input_values)):
        pre_metrics.append(
            metrics(
                official_pre["post_attention_norm"][position],
                candidate["post_attention_norm"][position],
            )
        )
        key = f"token.{position}.layer.{layer}.output"
        if key not in official:
            raise StatefulSparseLayerError(f"official trace omitted {key}")
        output_metrics.append(
            metrics(official[key], candidate["outputs"][position])
        )
    return {
        "decision": classify_positions(pre_metrics, output_metrics),
        "post_attention_norm": pre_metrics,
        "outputs": output_metrics,
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
            raise StatefulSparseLayerError(
                "--inputs must contain the eight source-bound deterministic rows"
            )
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise StatefulSparseLayerError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection, fixture, input_values, layers
        )
        for layer in layers:
            required = {expert for row in routes[layer] for expert in row.indices}
            if set(fixture.experts.get(layer, ())) != required:
                raise StatefulSparseLayerError(
                    f"layer {layer} fixture expert union changed"
                )
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise StatefulSparseLayerError("the stateful C probe requires CPU")

        library, source = composed.build_composed_library(
            Path(args.out).with_suffix(".runtime.so")
        )
        if not source["production_source_unchanged"]:
            raise StatefulSparseLayerError("production source changed")
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
            "format": "inkling-portable-bfloat16-stateful-sparse-layer-probe",
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
        StatefulSparseLayerError,
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
