#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Execute one source-bound sparse MoE token with native BF16 matrices.

The complete pre-expert path is already BF16-exact.  This evidence-only probe
therefore executes only deterministic position zero through the real C layer
while the official side evaluates the same post-attention-normalized row through
the fixture-backed sparse MLP.

A temporary C source adds private matrix kinds for routed expert gate/up/down
operations.  The checked-in public header is not modified.  All Q/K/V/R/O,
router, shared-expert, and routed-expert matrices are served by the pinned native
PyTorch BF16 linear kernel.  Canonical route pairs are injected only after the
candidate route is preserved, and expert_get must still prove every selected
expert exists in the bounded fixture.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from diagnose_inkling_native_bf16_backend import (
    FP,
    MAT_K,
    MAT_O,
    MAT_Q,
    MAT_R,
    MAT_ROUTER,
    MAT_V,
    MatvecCallback,
    NativeBfloat16BackendError,
    native_bfloat16_linear,
    transform_rms_source,
)
from diagnose_inkling_portable_bf16_attention import (
    PortableBfloat16AttentionError,
    transform_attention_source,
)
from diagnose_inkling_portable_bf16_preexpert import (
    PortableBfloat16PreExpertError,
    transform_residual_source,
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
    LAYER_SOURCES,
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
from run_inkling_fixture_reference_canonical import (
    CanonicalOfficialGate,
    CanonicalOfficialRouteError,
)


class SingleTokenMoeError(RuntimeError):
    """The bounded single-token MoE probe cannot continue without guessing."""


MAT_SHARED_GATE = 9
MAT_SHARED_UP = 10
MAT_SHARED_DOWN = 11
MAT_ROUTED_GATE = 12
MAT_ROUTED_UP = 13
MAT_ROUTED_DOWN = 14

STAGES = (
    "routed_gate0",
    "routed_up0",
    "routed_gated0",
    "routed_down0",
    "shared_gate0",
    "shared_up0",
    "shared_gated0",
    "shared_down0",
    "moe_out",
)

_PRIVATE_KIND_ANCHOR = "static float silu(float x) { return x / (1.0f + expf(-x)); }\n"
_PRIVATE_KIND_INSERT = """#define WASTE_IK_MAT_ROUTED_GATE_PROBE ((waste_inkling_matrix_kind)12)
#define WASTE_IK_MAT_ROUTED_UP_PROBE ((waste_inkling_matrix_kind)13)
#define WASTE_IK_MAT_ROUTED_DOWN_PROBE ((waste_inkling_matrix_kind)14)

static float silu(float x) { return x / (1.0f + expf(-x)); }
"""
_ROUTED_OLD = """            waste_inkling_expert_weights ew;
            if (get_routed(cfg, w, layer, s->routed_index[k],
                           expert_get, expert_ctx, &ew) ||
                expert_eval(s->branch, s->gate, s->up, &ew,
                            s->norm, hidden, inter)) return -1;
            const float gamma = s->routed_weight[k];
            for (int i = 0; i < hidden; i++) s->ff[i] += gamma * s->branch[i];
"""
_ROUTED_NEW = """            waste_inkling_expert_weights ew;
            const int expert = s->routed_index[k];
            if (get_routed(cfg, w, layer, expert,
                           expert_get, expert_ctx, &ew)) return -1;
            if (backend && backend->matvec) {
                if (apply_matvec(backend, layer,
                                 WASTE_IK_MAT_ROUTED_GATE_PROBE, expert,
                                 s->gate, NULL, s->norm, inter, hidden) ||
                    apply_matvec(backend, layer,
                                 WASTE_IK_MAT_ROUTED_UP_PROBE, expert,
                                 s->up, NULL, s->norm, inter, hidden)) return -1;
                if (k == 0 &&
                    (trace_f(trace, layer, "routed_gate0", s->gate,
                             (size_t)inter) ||
                     trace_f(trace, layer, "routed_up0", s->up,
                             (size_t)inter))) return -1;
                for (int i = 0; i < inter; i++)
                    s->gate[i] = silu(s->gate[i]) * s->up[i];
                if (k == 0 && trace_f(trace, layer, "routed_gated0",
                                      s->gate, (size_t)inter)) return -1;
                if (apply_matvec(backend, layer,
                                 WASTE_IK_MAT_ROUTED_DOWN_PROBE, expert,
                                 s->branch, NULL, s->gate,
                                 hidden, inter)) return -1;
            } else if (expert_eval(s->branch, s->gate, s->up, &ew,
                                   s->norm, hidden, inter)) return -1;
            if (k == 0 && trace_f(trace, layer, "routed_down0",
                                  s->branch, (size_t)hidden)) return -1;
            const float gamma = s->routed_weight[k];
            for (int i = 0; i < hidden; i++) s->ff[i] += gamma * s->branch[i];
"""
_SHARED_OLD = """                if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_GATE, e,
                                 s->gate, NULL, s->norm, inter, hidden) ||
                    apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_UP, e,
                                 s->up, NULL, s->norm, inter, hidden)) return -1;
                for (int j = 0; j < inter; j++)
                    s->gate[j] = silu(s->gate[j]) * s->up[j];
                if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_DOWN, e,
                                 s->branch, NULL, s->gate, hidden, inter)) return -1;
"""
_SHARED_NEW = """                if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_GATE, e,
                                 s->gate, NULL, s->norm, inter, hidden) ||
                    apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_UP, e,
                                 s->up, NULL, s->norm, inter, hidden)) return -1;
                if (e == 0 &&
                    (trace_f(trace, layer, "shared_gate0", s->gate,
                             (size_t)inter) ||
                     trace_f(trace, layer, "shared_up0", s->up,
                             (size_t)inter))) return -1;
                for (int j = 0; j < inter; j++)
                    s->gate[j] = silu(s->gate[j]) * s->up[j];
                if (e == 0 && trace_f(trace, layer, "shared_gated0",
                                      s->gate, (size_t)inter)) return -1;
                if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_DOWN, e,
                                 s->branch, NULL, s->gate, hidden, inter)) return -1;
                if (e == 0 && trace_f(trace, layer, "shared_down0",
                                      s->branch, (size_t)hidden)) return -1;
"""


def transform_moe_source(source: str) -> str:
    """Add private routed matrix callbacks and focused stage traces once."""
    transformed = source
    for old, new, label in (
        (_PRIVATE_KIND_ANCHOR, _PRIVATE_KIND_INSERT, "private kind anchor"),
        (_ROUTED_OLD, _ROUTED_NEW, "routed expert block"),
        (_SHARED_OLD, _SHARED_NEW, "shared expert backend block"),
    ):
        count = transformed.count(old)
        if count != 1:
            raise SingleTokenMoeError(
                f"expected exactly one {label}; found {count}"
            )
        transformed = transformed.replace(old, new, 1)
    return transformed


def build_single_token_moe_library(out: Path) -> tuple[Path, dict[str, Any]]:
    source_root = Path(__file__).resolve().parents[2] / "inkling" / "src"
    temporary = Path(tempfile.mkdtemp(prefix="inkling-bf16-single-token-moe-"))
    probe_root = temporary / "src"
    shutil.copytree(source_root, probe_root)

    layer_path = probe_root / "inkling_layer.c"
    attention_path = probe_root / "inkling_attention.c"
    layer_original = layer_path.read_text(encoding="utf-8")
    attention_original = attention_path.read_text(encoding="utf-8")
    layer_transformed = transform_moe_source(
        transform_residual_source(transform_rms_source(layer_original))
    )
    attention_transformed = transform_attention_source(attention_original)
    layer_path.write_text(layer_transformed, encoding="utf-8")
    attention_path.write_text(attention_transformed, encoding="utf-8")
    library = build_library(probe_root, out)
    return library, {
        "production_source_unchanged": (
            (source_root / "inkling_layer.c").read_text(encoding="utf-8")
            == layer_original
            and (source_root / "inkling_attention.c").read_text(encoding="utf-8")
            == attention_original
        ),
        "production_layer_sha256": hashlib.sha256(
            layer_original.encode()
        ).hexdigest(),
        "probe_layer_sha256": hashlib.sha256(
            layer_transformed.encode()
        ).hexdigest(),
        "production_attention_sha256": hashlib.sha256(
            attention_original.encode()
        ).hexdigest(),
        "probe_attention_sha256": hashlib.sha256(
            attention_transformed.encode()
        ).hexdigest(),
        "compiled_sources": list(LAYER_SOURCES),
        "private_matrix_kinds": {
            "routed_gate": MAT_ROUTED_GATE,
            "routed_up": MAT_ROUTED_UP,
            "routed_down": MAT_ROUTED_DOWN,
        },
    }


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu().contiguous()
    candidate = candidate.detach().float().cpu().contiguous()
    if tuple(reference.shape) != tuple(candidate.shape):
        raise SingleTokenMoeError(
            f"shape mismatch {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    raw_delta = (reference - candidate).abs()
    reference_bf16 = reference.to(torch.bfloat16).float()
    candidate_bf16 = candidate.to(torch.bfloat16).float()
    bf16_delta = (reference_bf16 - candidate_bf16).abs()
    return {
        "count": int(reference.numel()),
        "raw_exact_fraction": float(reference.eq(candidate).float().mean()),
        "raw_max_abs": float(raw_delta.max()),
        "raw_mean_abs": float(raw_delta.mean()),
        "bfloat16_exact_fraction": float(
            reference_bf16.eq(candidate_bf16).float().mean()
        ),
        "bfloat16_max_abs": float(bf16_delta.max()),
        "bfloat16_mean_abs": float(bf16_delta.mean()),
    }


def classify_stages(stage_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prefix: list[str] = []
    for stage in STAGES:
        if stage_metrics[stage]["bfloat16_exact_fraction"] != 1.0:
            return {
                "classification": "single_token_moe_mismatch_isolated",
                "exact_bfloat16_prefix": prefix,
                "first_nonexact_stage": stage,
            }
        prefix.append(stage)
    return {
        "classification": "single_token_sparse_moe_is_bfloat16_exact",
        "exact_bfloat16_prefix": prefix,
        "first_nonexact_stage": None,
    }


class MoeMatrixProvider:
    """Serve attention, router, shared, and selected routed matrices in BF16."""

    def __init__(self, layer: int, module: Any) -> None:
        self.layer = int(layer)
        self.module = module
        self.calls: list[dict[str, Any]] = []
        self.error: str | None = None
        self.callback = MatvecCallback(self._call)
        self.backend = MatrixBackend(
            ctypes.cast(self.callback, ctypes.c_void_p),
            None,
        )

    @staticmethod
    def _vector(pointer: FP, cols: int) -> torch.Tensor:
        return torch.tensor(
            [float(pointer[index]) for index in range(cols)],
            dtype=torch.float32,
        )

    @staticmethod
    def _copy(pointer: FP, value: torch.Tensor, rows: int) -> None:
        value = value.detach().float().cpu().contiguous()
        if value.ndim != 1 or value.numel() != rows:
            raise SingleTokenMoeError(
                f"backend output {tuple(value.shape)} cannot fill {rows} rows"
            )
        for index, item in enumerate(value.tolist()):
            pointer[index] = float(item)

    def _weight(self, kind: int, index: int) -> tuple[str, torch.Tensor]:
        attn = self.module.self_attn
        if kind == MAT_Q:
            return "q_proj", attn.q_proj.weight
        if kind == MAT_K:
            return "k_proj", attn.k_proj.weight
        if kind == MAT_V:
            return "v_proj", attn.v_proj.weight
        if kind == MAT_R:
            return "relative_proj_input", attn.r_proj.weight
        if kind == MAT_O:
            return "o_proj", attn.o_proj.weight
        if kind == MAT_ROUTER:
            return "router", self.module.mlp.gate.weight
        if kind == MAT_SHARED_GATE:
            return "shared_gate", self.module.mlp.shared_experts.gate_proj[index]
        if kind == MAT_SHARED_UP:
            return "shared_up", self.module.mlp.shared_experts.up_proj[index]
        if kind == MAT_SHARED_DOWN:
            return "shared_down", self.module.mlp.shared_experts.down_proj[index]
        if kind in (MAT_ROUTED_GATE, MAT_ROUTED_UP):
            fused = self.module.mlp.experts.gate_up[str(index)]
            gate, up = fused.chunk(2, dim=0)
            return (
                "routed_gate" if kind == MAT_ROUTED_GATE else "routed_up",
                gate if kind == MAT_ROUTED_GATE else up,
            )
        if kind == MAT_ROUTED_DOWN:
            return "routed_down", self.module.mlp.experts.down[str(index)]
        raise SingleTokenMoeError(
            f"unexpected matrix request kind={kind} index={index}"
        )

    def _call(
        self,
        _ctx: int,
        layer: int,
        kind: int,
        index: int,
        x: FP,
        out: FP,
        rows: int,
        cols: int,
    ) -> int:
        try:
            if layer != self.layer:
                raise SingleTokenMoeError(
                    f"backend requested layer {layer}; expected {self.layer}"
                )
            name, weight = self._weight(kind, index)
            weight = weight.detach().cpu().contiguous()
            if tuple(weight.shape) != (rows, cols):
                raise SingleTokenMoeError(
                    f"{name}[{index}] shape {tuple(weight.shape)} != {(rows, cols)}"
                )
            value = native_bfloat16_linear(weight, self._vector(x, cols))
            self._copy(out, value, rows)
            self.calls.append(
                {
                    "kind": name,
                    "index": int(index),
                    "rows": int(rows),
                    "cols": int(cols),
                }
            )
            return 0
        except Exception as exc:  # ctypes callbacks cannot propagate safely.
            self.error = str(exc)
            return -1


def official_moe_stages(
    module: Any,
    normalized: torch.Tensor,
    row: RouteRow,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    normalized = normalized.detach().to(torch.bfloat16).reshape(1, -1)
    gate = CanonicalOfficialGate(module.mlp.gate, [row])
    router_logits, routed_weights, routed_ids, shared_gammas = gate(normalized)
    if not gate.applied:
        raise SingleTokenMoeError("canonical official gate was not applied")

    expert = int(routed_ids[0, 0])
    fused = module.mlp.experts.gate_up[str(expert)]
    routed_gate, routed_up = F.linear(normalized, fused).chunk(2, dim=-1)
    routed_gated = F.silu(routed_gate) * routed_up
    routed_down = F.linear(
        routed_gated, module.mlp.experts.down[str(expert)]
    )

    shared_gate = F.linear(
        normalized, module.mlp.shared_experts.gate_proj[0]
    )
    shared_up = F.linear(
        normalized, module.mlp.shared_experts.up_proj[0]
    )
    shared_gated = F.silu(shared_gate) * shared_up
    shared_down = F.linear(
        shared_gated, module.mlp.shared_experts.down_proj[0]
    )

    routed_out = module.mlp.experts(
        normalized, routed_ids, routed_weights
    )
    shared_out = module.mlp.shared_experts(
        normalized, gammas=shared_gammas
    )
    moe_out = routed_out + shared_out
    return {
        "routed_gate0": routed_gate[0],
        "routed_up0": routed_up[0],
        "routed_gated0": routed_gated[0],
        "routed_down0": routed_down[0],
        "shared_gate0": shared_gate[0],
        "shared_up0": shared_up[0],
        "shared_gated0": shared_gated[0],
        "shared_down0": shared_down[0],
        "moe_out": moe_out[0],
    }, {
        "router_logits": router_logits.detach().float().cpu().tolist(),
        "routed_ids": routed_ids.detach().cpu().tolist(),
        "routed_weights": routed_weights.detach().float().cpu().tolist(),
        "shared_gammas": shared_gammas.detach().float().cpu().tolist(),
        "routed_output": routed_out.detach().float().cpu().tolist(),
        "shared_output": shared_out.detach().float().cpu().tolist(),
    }


def run_c_single_token(
    lib: Any,
    fixture: Any,
    cfg: dict[str, Any],
    layer: int,
    input_row: list[float],
    route: RouteRow,
    provider: MoeMatrixProvider,
) -> dict[str, Any]:
    compact = bind_sparse_layer_compact(fixture, cfg, layer)
    available = set(compact.provider.experts if compact.provider else ())
    missing = sorted(set(route.indices).difference(available))
    if missing:
        raise SingleTokenMoeError(
            f"layer {layer} fixture omits canonical experts {missing}"
        )
    for field in (
        "wq", "wk", "wv", "wr", "wo", "router_weight",
        "shared_gate", "shared_up", "shared_down",
    ):
        setattr(compact.weights, field, FP())

    config = _build_config(lib, cfg)
    hidden = int(cfg["hidden"])
    if len(input_row) != hidden:
        raise SingleTokenMoeError(
            f"input row has {len(input_row)} values; expected {hidden}"
        )
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
        ctypes.byref(scratch), ctypes.byref(config), layer, capacity,
        fbuf, ctypes.c_size_t(nfloat), ibuf, ctypes.c_size_t(nint),
    ):
        raise SingleTokenMoeError("scratch init failed")
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
        raise SingleTokenMoeError("state init failed")

    collector = CanonicalRouteCollector([route])
    collector.position = 0
    x = (ctypes.c_float * hidden)(*input_row)
    expert_get = ctypes.cast(compact.provider.callback, ctypes.c_void_p)
    rc = lib.waste_inkling_layer_step_backend_trace(
        ctypes.byref(config), layer, ctypes.byref(compact.weights),
        ctypes.byref(provider.backend), ctypes.byref(state), x, 0,
        ctypes.byref(scratch), expert_get, None,
        ctypes.byref(collector.c_trace),
    )
    if provider.error:
        raise SingleTokenMoeError(provider.error)
    if collector.error:
        raise SingleTokenMoeError(collector.error)
    if rc:
        raise SingleTokenMoeError(f"C layer failed with status {rc}")
    if collector.applied_positions != {0}:
        raise SingleTokenMoeError("canonical route was not applied at position zero")

    stages: dict[str, torch.Tensor] = {}
    for stage in STAGES:
        name = f"token.0.layer.{layer}.{stage}"
        if name not in collector.values:
            raise SingleTokenMoeError(f"C trace missed {name}")
        stages[stage] = torch.tensor(
            collector.values[name], dtype=torch.float32
        )
    return {
        "stages": stages,
        "candidate_routed_index": collector.values[
            f"token.0.layer.{layer}.candidate_routed_index"
        ],
        "candidate_routed_weight": collector.values[
            f"token.0.layer.{layer}.candidate_routed_weight"
        ],
        "canonical_routed_index": collector.values[
            f"token.0.layer.{layer}.routed_index"
        ],
        "canonical_routed_weight": collector.values[
            f"token.0.layer.{layer}.routed_weight"
        ],
        "backend_calls": provider.calls,
    }


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
        fixture, config, layer, inputs,
        device=device, dtype=dtype,
    )
    module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    official, official_route = official_moe_stages(
        module, pre_router["post_attention_norm"][0], route
    )
    provider = MoeMatrixProvider(layer, module)
    candidate = run_c_single_token(
        lib, fixture, cfg, layer, input_values[0], route, provider
    )
    comparisons = {
        stage: metrics(official[stage], candidate["stages"][stage])
        for stage in STAGES
    }
    return {
        "decision": classify_stages(comparisons),
        "stages": comparisons,
        "official_route": official_route,
        "candidate_routed_index": candidate["candidate_routed_index"],
        "candidate_routed_weight": candidate["candidate_routed_weight"],
        "canonical_routed_index": candidate["canonical_routed_index"],
        "canonical_routed_weight": candidate["canonical_routed_weight"],
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
            raise SingleTokenMoeError(
                "--inputs must contain the eight source-bound deterministic rows"
            )
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise SingleTokenMoeError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection, fixture, input_values, layers
        )
        position_zero = {layer: routes[layer][0] for layer in layers}
        for layer, row in position_zero.items():
            declared = set(fixture.experts.get(layer, ()))
            if declared != set(row.indices):
                raise SingleTokenMoeError(
                    f"layer {layer} fixture experts {sorted(declared)} differ from "
                    f"position-zero route {sorted(row.indices)}"
                )
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise SingleTokenMoeError("the C probe requires --device cpu")
        library, source = build_single_token_moe_library(
            Path(args.out).with_suffix(".so")
        )
        if not source["production_source_unchanged"]:
            raise SingleTokenMoeError("production source changed")
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        analyses = {
            str(layer): analyze_layer(
                lib, fixture, config, cfg, layer, inputs, input_values,
                position_zero[layer], device=device, dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-portable-bfloat16-single-token-moe-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "source_tokens": len(input_values),
            "executed_position": 0,
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "fixture_experts": {
                str(layer): list(fixture.experts.get(layer, ()))
                for layer in layers
            },
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
        NativeBfloat16BackendError,
        PortableBfloat16AttentionError,
        PortableBfloat16PreExpertError,
        CanonicalRouteError,
        CanonicalOfficialRouteError,
        SingleTokenMoeError,
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
