#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Locate and test Inkling precision drift before routing."""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Any, Callable, Collection

import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "inkling" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from analyze_inkling_router_mismatch import (
    _route,
    _tensor_metrics,
    _variant_summary,
)
from discover_inkling_c_router_experts import _RejectExperts
from discover_inkling_router_experts import (
    _CaptureExperts,
    _RouterCaptured,
    _causal_mask,
    input_sha256,
)
from inkling_compact_layer_parity import bind_sparse_layer_compact
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    _import_transformers,
    build_layer_from_fixture,
    verify_config_binding,
)
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
from inkling_release_config import build_transformers_text_config


class PreRouterDiagnosisError(RuntimeError):
    """The pre-router trace cannot be aligned without guessing."""


POINTS = (
    "input_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "relative_proj_input",
    "k_sconv",
    "v_sconv",
    "attention_out",
    "attention_branch",
    "post_attention_residual",
    "post_attention_norm",
)


def _sequence_tensor(value: Any, tokens: int) -> torch.Tensor:
    tensor = value
    if isinstance(tensor, (tuple, list)):
        tensor = next((item for item in tensor if isinstance(item, torch.Tensor)), None)
    if not isinstance(tensor, torch.Tensor):
        raise PreRouterDiagnosisError("official hook output has no tensor")
    tensor = tensor.detach()
    if tensor.ndim >= 1 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.shape[0] != tokens:
        raise PreRouterDiagnosisError(
            f"official hook has {tensor.shape[0]} positions; expected {tokens}"
        )
    return tensor.reshape(tokens, -1).contiguous()


def _capture_official_pre_router(
    fixture,
    config,
    layer: int,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    dict[str, torch.Tensor],
    Any,
    torch.Tensor,
    torch.Tensor,
]:
    """Capture official stages plus exact routed IDs and weights."""
    _, _, classes, _ = _import_transformers()
    _, DynamicCache = classes
    module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    tokens = int(inputs.shape[0])
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def output_hook(point: str) -> Callable:
        def hook(_module, _args, output):
            captured[point] = _sequence_tensor(output, tokens)

        return hook

    def input_hook(point: str) -> Callable:
        def hook(_module, args):
            if not args:
                raise PreRouterDiagnosisError(f"{point} prehook received no input")
            captured[point] = _sequence_tensor(args[0], tokens)

        return hook

    attn = module.self_attn
    handles.append(
        module.input_layernorm.register_forward_hook(output_hook("input_norm"))
    )
    handles.append(attn.q_proj.register_forward_hook(output_hook("q_proj")))
    handles.append(attn.k_proj.register_forward_hook(output_hook("k_proj")))
    handles.append(attn.v_proj.register_forward_hook(output_hook("v_proj")))
    handles.append(
        attn.r_proj.register_forward_hook(output_hook("relative_proj_input"))
    )
    handles.append(attn.k_sconv.register_forward_hook(output_hook("k_sconv")))
    handles.append(attn.v_sconv.register_forward_hook(output_hook("v_sconv")))
    handles.append(
        attn.o_proj.register_forward_pre_hook(input_hook("attention_out"))
    )
    handles.append(
        module.attn_sconv.register_forward_hook(output_hook("attention_branch"))
    )
    handles.append(
        module.post_attention_layernorm.register_forward_pre_hook(
            input_hook("post_attention_residual")
        )
    )
    handles.append(
        module.post_attention_layernorm.register_forward_hook(
            output_hook("post_attention_norm")
        )
    )
    module.mlp.experts = _CaptureExperts()
    prefix = inputs.to(device=device, dtype=dtype).unsqueeze(0)
    try:
        try:
            with torch.no_grad():
                module(
                    prefix,
                    attention_mask=_causal_mask(
                        config,
                        layer,
                        tokens,
                        device=device,
                        dtype=dtype,
                    ),
                    conv_mask=None,
                    past_key_values=DynamicCache(config=config),
                )
        except _RouterCaptured as result:
            indices = result.indices
            weights = result.weights
        else:
            raise PreRouterDiagnosisError(
                "official layer completed without reaching routed experts"
            )
    finally:
        for handle in handles:
            handle.remove()
    missing = [point for point in POINTS if point not in captured]
    if missing:
        raise PreRouterDiagnosisError(
            "official trace missed points: " + ", ".join(missing)
        )
    return captured, module, indices, weights


def capture_official_pre_router(
    fixture,
    config,
    layer: int,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Capture full-sequence official tensors corresponding to C trace points."""
    captured, _, _, _ = _capture_official_pre_router(
        fixture,
        config,
        layer,
        inputs,
        device=device,
        dtype=dtype,
    )
    return captured


def _round_float32_pointer_to_bfloat16(
    data: ctypes.POINTER(ctypes.c_float),
    count: int,
) -> None:
    """Round writable float32 storage to BF16, retaining it as float32."""
    raw = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint32))
    for index in range(int(count)):
        bits = int(raw[index])
        # Preserve infinities and NaN payloads. Finite values use round-to-nearest
        # ties-to-even, matching the standard float32 -> BF16 conversion.
        if bits & 0x7F800000 != 0x7F800000:
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
            raw[index] = bits & 0xFFFF0000


class _QuantizingTraceCollector(TraceCollector):
    """Diagnostic collector that requantizes selected writable C buffers."""

    def __init__(self, quantize_points: Collection[str]) -> None:
        self.quantize_points = frozenset(quantize_points)
        super().__init__()

    def _emit_float(self, ctx, layer, point, data, count) -> int:
        decoded = point.decode("ascii", "strict")
        if decoded in self.quantize_points:
            _round_float32_pointer_to_bfloat16(data, int(count))
        return super()._emit_float(ctx, layer, point, data, count)


class _InjectingTraceCollector(TraceCollector):
    """Diagnostic collector that replaces one or more C stages with an oracle."""

    def __init__(self, replacements: dict[str, torch.Tensor]) -> None:
        self.replacements = {
            point: tensor.detach().float().cpu().reshape(tensor.shape[0], -1).tolist()
            for point, tensor in replacements.items()
        }
        self.error: str | None = None
        super().__init__()

    def _emit_float(self, ctx, layer, point, data, count) -> int:
        decoded = point.decode("ascii", "strict")
        rows = self.replacements.get(decoded)
        if rows is not None:
            if self.position >= len(rows):
                self.error = (
                    f"oracle point {decoded} has no position {self.position}"
                )
                return 1
            row = rows[self.position]
            if len(row) != int(count):
                self.error = (
                    f"oracle point {decoded} position {self.position} has "
                    f"{len(row)} values; C emitted {int(count)}"
                )
                return 1
            for index, value in enumerate(row):
                data[index] = float(value)
        return super()._emit_float(ctx, layer, point, data, count)


def _capture_c_pre_router(
    lib,
    fixture,
    cfg: dict[str, Any],
    layer: int,
    inputs: list[list[float]],
    *,
    quantize_points: Collection[str] = (),
    replace_stages: dict[str, torch.Tensor] | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Capture C stages and route, optionally requantizing traced buffers."""
    replacement_points = set((replace_stages or {}).keys())
    unknown = sorted(
        set(quantize_points).union(replacement_points).difference(POINTS)
    )
    if unknown:
        raise PreRouterDiagnosisError(
            "unknown mutation trace points: " + ", ".join(unknown)
        )
    if quantize_points and replace_stages:
        raise PreRouterDiagnosisError(
            "quantization and oracle replacement are separate counterfactuals"
        )
    compact = bind_sparse_layer_compact(fixture, cfg, layer)
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
        raise PreRouterDiagnosisError("scratch sizing failed")
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
        raise PreRouterDiagnosisError("scratch init failed")
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
        raise PreRouterDiagnosisError("state init failed")

    if replace_stages:
        collector = _InjectingTraceCollector(replace_stages)
    elif quantize_points:
        collector = _QuantizingTraceCollector(quantize_points)
    else:
        collector = TraceCollector()
    backend = MatrixBackend()
    reject = _RejectExperts(layer)
    reject_ptr = ctypes.cast(reject.callback, ctypes.c_void_p)
    for position, row in enumerate(inputs):
        if len(row) != hidden:
            raise PreRouterDiagnosisError(
                f"input {position} has {len(row)} values; hidden is {hidden}"
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
        collector_error = getattr(collector, "error", None)
        if collector_error:
            raise PreRouterDiagnosisError(collector_error)
        if rc == 0 or len(reject.requests) != before + 1:
            raise PreRouterDiagnosisError(
                f"layer {layer} position {position} did not stop at one expert lookup"
            )

    out: dict[str, torch.Tensor] = {}
    route_indices = []
    route_weights = []
    for position in range(len(inputs)):
        for point, rows in (
            ("routed_index", route_indices),
            ("routed_weight", route_weights),
        ):
            name = f"token.{position}.layer.{layer}.{point}"
            if name not in collector.values:
                raise PreRouterDiagnosisError(f"C trace missed {name}")
            rows.append(collector.values[name])
    for point in POINTS:
        rows = []
        for position in range(len(inputs)):
            name = f"token.{position}.layer.{layer}.{point}"
            if name not in collector.values:
                raise PreRouterDiagnosisError(f"C trace missed {name}")
            rows.append(collector.values[name])
        out[point] = torch.tensor(rows, dtype=torch.float32)
    return (
        out,
        torch.tensor(route_indices, dtype=torch.int64),
        torch.tensor(route_weights, dtype=torch.float32),
    )


def capture_c_pre_router(
    lib,
    fixture,
    cfg: dict[str, Any],
    layer: int,
    inputs: list[list[float]],
) -> dict[str, torch.Tensor]:
    """Capture all C trace points before deliberately rejected expert lookup."""
    captured, _, _ = _capture_c_pre_router(
        lib, fixture, cfg, layer, inputs
    )
    return captured


def compare_pre_router(
    official: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    compare_dtype: torch.dtype,
) -> dict[str, Any]:
    stages = {}
    first_nonexact = None
    for point in POINTS:
        metrics = _tensor_metrics(
            official[point], candidate[point], compare_dtype=compare_dtype
        )
        stages[point] = metrics
        if (
            first_nonexact is None
            and metrics["quantized_exact_fraction"] < 1.0
        ):
            first_nonexact = point
    return {
        "first_nonexact_bfloat16_stage": first_nonexact,
        "all_stages_bfloat16_exact": first_nonexact is None,
        "stages": stages,
    }


def _official_route_on_state(
    gate: Any,
    state: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the official gate and router on a supplied pre-router state."""
    with torch.no_grad():
        logits = F.linear(
            state.to(device=gate.weight.device, dtype=dtype),
            gate.weight.detach().to(dtype),
        )
        indices, weights, _ = _route(
            logits,
            correction_bias=gate.e_score_correction_bias.detach().to(dtype),
            global_scale=gate.global_scale.detach().to(dtype),
            route_scale=float(gate.route_scale),
            top_k=int(gate.top_k),
            n_shared=int(gate.n_shared_experts),
        )
    return indices.cpu(), weights.cpu()


def _route_summary(
    name: str,
    indices: torch.Tensor,
    weights: torch.Tensor,
    official_indices: torch.Tensor,
    official_weights: torch.Tensor,
) -> dict[str, Any]:
    return _variant_summary(
        name,
        indices.cpu(),
        weights.cpu(),
        official_indices.cpu(),
        official_weights.cpu(),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--c-config", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    args = ap.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        c_config = json.loads(
            Path(args.c_config).read_text(encoding="utf-8")
        )
        input_values = json.loads(
            Path(args.inputs).read_text(encoding="utf-8")
        )
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [
            int(value) for value in args.layers.split(",") if value
        ]
        if not layers:
            raise PreRouterDiagnosisError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        library = build_library(
            REPO / "inkling" / "src", Path(args.out).with_suffix(".so")
        )
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        analyses = {}
        for layer in layers:
            official, module, official_indices, official_weights = (
                _capture_official_pre_router(
                    fixture,
                    config,
                    layer,
                    inputs,
                    device=device,
                    dtype=dtype,
                )
            )
            candidate, c_indices, c_weights = _capture_c_pre_router(
                lib, fixture, c_config, layer, input_values
            )
            requantized, q_indices, q_weights = _capture_c_pre_router(
                lib,
                fixture,
                c_config,
                layer,
                input_values,
                quantize_points=POINTS,
            )
            analysis = compare_pre_router(
                official, candidate, compare_dtype=dtype
            )
            analysis["counterfactuals"] = {
                "requantize_all_traced_stages": compare_pre_router(
                    official,
                    requantized,
                    compare_dtype=dtype,
                )
            }
            gate = module.mlp.gate
            baseline_official_indices, baseline_official_weights = (
                _official_route_on_state(
                    gate,
                    candidate["post_attention_norm"],
                    dtype=dtype,
                )
            )
            requantized_official_indices, requantized_official_weights = (
                _official_route_on_state(
                    gate,
                    requantized["post_attention_norm"],
                    dtype=dtype,
                )
            )
            analysis["routing"] = {
                "c_emitted": _route_summary(
                    "c_emitted",
                    c_indices,
                    c_weights,
                    official_indices,
                    official_weights,
                ),
                "c_emitted_after_trace_requantization": _route_summary(
                    "c_emitted_after_trace_requantization",
                    q_indices,
                    q_weights,
                    official_indices,
                    official_weights,
                ),
                "official_bfloat16_router_on_c_state": _route_summary(
                    "official_bfloat16_router_on_c_state",
                    baseline_official_indices,
                    baseline_official_weights,
                    official_indices,
                    official_weights,
                ),
                "official_bfloat16_router_on_requantized_c_state": (
                    _route_summary(
                        "official_bfloat16_router_on_requantized_c_state",
                        requantized_official_indices,
                        requantized_official_weights,
                        official_indices,
                        official_weights,
                    )
                ),
            }
            analysis["counterfactual_limitations"] = [
                "only existing writable C trace boundaries are requantized",
                "q/k head normalization is not an exposed trace boundary",
                "attention score, softmax, and value-matmul internals are not exposed",
                "C router logits are traced only after expert selection",
            ]
            analyses[str(layer)] = analysis
        result = {
            "format": "inkling-pre-router-diagnosis",
            "version": 2,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
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
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        ap.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
