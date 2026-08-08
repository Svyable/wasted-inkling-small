#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compose the proven BF16 router and sparse-MoE aggregation policies.

This evidence-only probe keeps raw C routing observable, injects only the
source-bound canonical IDs, recomputes their routed/shared weights from the
captured exact BF16 router logits using the portable logsumexp policy from #35,
and applies the aggregation policy from #34 in a copied C source tree.

The run continues through the causal one-token MLP short convolution and final
layer residual so that the first boundary after an exact ``moe_out`` is named
without requiring the full eight-token expert union. Production sources and the
public matrix-backend ABI remain unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from diagnose_inkling_bf16_router_weight_lattice import (
    F_FINAL_BF16,
    F_GLOBAL_SCALE_BF16,
    F_LOGP_BF16,
    F_LSE_BF16,
    F_NORMALIZED_BF16,
    F_OUTPUT_DELTA_BF16,
    F_ROUTE_SCALE_BF16,
    FixedWeightHelper,
    build_helper,
    selected_logits,
)
from diagnose_inkling_native_bf16_backend import MAT_ROUTER, transform_rms_source
from diagnose_inkling_portable_bf16_attention import transform_attention_source
from diagnose_inkling_portable_bf16_preexpert import transform_residual_source
from diagnose_inkling_portable_bf16_single_token_moe import (
    MoeMatrixProvider,
    SingleTokenMoeError,
    transform_moe_source,
)
from diagnose_inkling_portable_bf16_single_token_moe_activation import (
    MoeActivationError,
    transform_activation_source,
)
from diagnose_inkling_pre_router import PreRouterDiagnosisError, _capture_official_pre_router
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


class ComposedMoeError(RuntimeError):
    """The composed BF16 MoE candidate cannot proceed without exact contracts."""


ROUTER_POLICY_FLAGS = (
    F_LOGP_BF16
    | F_LSE_BF16
    | F_OUTPUT_DELTA_BF16
    | F_NORMALIZED_BF16
    | F_ROUTE_SCALE_BF16
    | F_GLOBAL_SCALE_BF16
    | F_FINAL_BF16
)

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
    "mlp_branch",
    "layer_out",
)

_CPUINFO_FIELDS = {
    "vendor_id": "vendor_id",
    "cpu family": "family",
    "model": "model",
    "model name": "model_name",
    "stepping": "stepping",
    "microcode": "microcode",
}


def _first_cpuinfo(path: Path) -> dict[str, Any]:
    """Read the first Linux processor block without making it a run gate."""
    values: dict[str, Any] = {
        "available": False,
        **{field: None for field in _CPUINFO_FIELDS.values()},
    }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        if not line.strip():
            break
        key, separator, value = line.partition(":")
        if separator and key.strip() in _CPUINFO_FIELDS:
            values[_CPUINFO_FIELDS[key.strip()]] = value.strip()
    values["available"] = any(
        values[field] is not None for field in _CPUINFO_FIELDS.values()
    )
    return values


def collect_execution_profile(
    *,
    environ: Mapping[str, str] | None = None,
    cpuinfo_path: Path = Path("/proc/cpuinfo"),
    torch_module: Any = torch,
    logical_cpu_count: int | None = None,
) -> dict[str, Any]:
    """Bind a numerical result to the hosted runner and CPU class that made it."""
    env = os.environ if environ is None else environ
    if logical_cpu_count is None:
        logical_cpu_count = os.cpu_count()
    cpu = _first_cpuinfo(cpuinfo_path)
    cpu.update(
        {
            "logical_cpu_count": logical_cpu_count,
            "machine": platform.machine(),
        }
    )
    profile = {
        "schema_version": 2,
        "workflow": {
            "event_name": env.get("GITHUB_EVENT_NAME"),
            "job": env.get("GITHUB_JOB"),
            "ref": env.get("GITHUB_REF"),
            "run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
            "run_id": env.get("GITHUB_RUN_ID"),
            "sha": env.get("GITHUB_SHA"),
        },
        "runner": {
            "runner_id": env.get("RUNNER_NAME"),
            "os": env.get("RUNNER_OS"),
            "arch": env.get("RUNNER_ARCH"),
            "image_os": env.get("ImageOS"),
            "image_version": env.get("ImageVersion"),
        },
        "cpu": cpu,
        "torch": {
            "version": str(torch_module.__version__),
            "cpu_capability": torch_module.backends.cpu.get_cpu_capability(),
            "num_threads": int(torch_module.get_num_threads()),
            "num_interop_threads": int(torch_module.get_num_interop_threads()),
        },
        "thread_environment": {
            name: env.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "ATEN_CPU_CAPABILITY",
            )
        },
    }
    host_class = {
        "runner": {
            key: profile["runner"][key]
            for key in ("os", "arch", "image_os", "image_version")
        },
        "cpu": profile["cpu"],
    }
    profile["host_class_sha256"] = hashlib.sha256(
        json.dumps(host_class, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    reference_profile = {
        "host_class_sha256": profile["host_class_sha256"],
        "torch": profile["torch"],
        "thread_environment": profile["thread_environment"],
    }
    profile["reference_profile_sha256"] = hashlib.sha256(
        json.dumps(
            reference_profile,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return profile

_ROUTED_LOOP_OLD = """        for (int i = 0; i < hidden; i++) s->ff[i] = 0.0f;
        for (int k = 0; k < cfg->top_k; k++) {
"""
_ROUTED_LOOP_NEW = """        for (int i = 0; i < hidden; i++) s->ff[i] = bf16_round_probe(0.0f);
        for (int a = 0; a < cfg->top_k; a++) {
            for (int b = a + 1; b < cfg->top_k; b++) {
                if (s->routed_index[b] < s->routed_index[a]) {
                    const int tmp_index = s->routed_index[a];
                    const float tmp_weight = s->routed_weight[a];
                    s->routed_index[a] = s->routed_index[b];
                    s->routed_weight[a] = s->routed_weight[b];
                    s->routed_index[b] = tmp_index;
                    s->routed_weight[b] = tmp_weight;
                }
            }
        }
        for (int k = 0; k < cfg->top_k; k++) {
"""
_ROUTED_ACCUM_OLD = """            const float gamma = s->routed_weight[k];
            for (int i = 0; i < hidden; i++) s->ff[i] += gamma * s->branch[i];
"""
_ROUTED_ACCUM_NEW = """            const float gamma = bf16_round_probe(s->routed_weight[k]);
            for (int i = 0; i < hidden; i++) {
                const float contribution = bf16_round_probe(
                    bf16_round_probe(s->branch[i]) * gamma);
                s->ff[i] = bf16_round_probe(
                    bf16_round_probe(s->ff[i]) + contribution);
            }
"""
_SHARED_BEGIN_OLD = """        const size_t gh = (size_t)inter * hidden;
        const size_t dh = (size_t)hidden * inter;
"""
_SHARED_BEGIN_NEW = """        for (int i = 0; i < hidden; i++) s->attn[i] = 0.0f;
        const size_t gh = (size_t)inter * hidden;
        const size_t dh = (size_t)hidden * inter;
"""
_SHARED_DOWN_OLD = """                if (e == 0 && trace_f(trace, layer, "shared_gated0",
                                      s->gate, (size_t)inter)) return -1;
                if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_DOWN, e,
                                 s->branch, NULL, s->gate, hidden, inter)) return -1;
                if (e == 0 && trace_f(trace, layer, "shared_down0",
                                      s->branch, (size_t)hidden)) return -1;
"""
_SHARED_DOWN_NEW = """                if (e == 0 && trace_f(trace, layer, "shared_gated0",
                                      s->gate, (size_t)inter)) return -1;
                const float shared_gamma = bf16_round_probe(s->shared_weight[e]);
                for (int j = 0; j < inter; j++)
                    s->gate[j] = bf16_round_probe(
                        bf16_round_probe(s->gate[j]) * shared_gamma);
                if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_DOWN, e,
                                 s->branch, NULL, s->gate, hidden, inter)) return -1;
                if (e == 0 && trace_f(trace, layer, "shared_down0",
                                      s->branch, (size_t)hidden)) return -1;
"""
_SHARED_ACCUM_OLD = """            const float gamma = s->shared_weight[e];
            for (int i = 0; i < hidden; i++) s->ff[i] += gamma * s->branch[i];
        }
        if (trace_f(trace, layer, "moe_out", s->ff, (size_t)hidden)) return -1;
"""
_SHARED_ACCUM_NEW = """            for (int i = 0; i < hidden; i++) s->attn[i] += s->branch[i];
        }
        for (int i = 0; i < hidden; i++) {
            const float routed = bf16_round_probe(s->ff[i]);
            const float shared = bf16_round_probe(s->attn[i]);
            s->ff[i] = bf16_round_probe(routed + shared);
        }
        if (trace_f(trace, layer, "moe_out", s->ff, (size_t)hidden)) return -1;
"""


def transform_aggregation_source(source: str) -> str:
    transformed = source
    for old, new, label in (
        (_ROUTED_LOOP_OLD, _ROUTED_LOOP_NEW, "routed expert-order loop"),
        (_ROUTED_ACCUM_OLD, _ROUTED_ACCUM_NEW, "routed BF16 accumulation"),
        (_SHARED_BEGIN_OLD, _SHARED_BEGIN_NEW, "shared FP32 accumulator"),
        (_SHARED_DOWN_OLD, _SHARED_DOWN_NEW, "shared gamma-before-down block"),
        (_SHARED_ACCUM_OLD, _SHARED_ACCUM_NEW, "shared/final aggregation block"),
    ):
        count = transformed.count(old)
        if count != 1:
            raise ComposedMoeError(f"expected exactly one {label}; found {count}")
        transformed = transformed.replace(old, new, 1)
    return transformed


def build_composed_library(out: Path) -> tuple[Path, dict[str, Any]]:
    source_root = Path(__file__).resolve().parents[2] / "inkling" / "src"
    temporary = Path(tempfile.mkdtemp(prefix="inkling-bf16-composed-moe-"))
    probe_root = temporary / "src"
    shutil.copytree(source_root, probe_root)
    layer_path = probe_root / "inkling_layer.c"
    attention_path = probe_root / "inkling_attention.c"
    layer_original = layer_path.read_text(encoding="utf-8")
    attention_original = attention_path.read_text(encoding="utf-8")
    layer_transformed = transform_aggregation_source(
        transform_activation_source(
            transform_moe_source(
                transform_residual_source(transform_rms_source(layer_original))
            )
        )
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
        "production_layer_sha256": hashlib.sha256(layer_original.encode()).hexdigest(),
        "probe_layer_sha256": hashlib.sha256(layer_transformed.encode()).hexdigest(),
        "production_attention_sha256": hashlib.sha256(attention_original.encode()).hexdigest(),
        "probe_attention_sha256": hashlib.sha256(attention_transformed.encode()).hexdigest(),
        "compiled_sources": list(LAYER_SOURCES),
        "router_policy": {"family": "logsumexp", "flags": ROUTER_POLICY_FLAGS},
        "aggregation_policy": {
            "routed_order": "ascending_expert_id",
            "routed_accumulation": "bfloat16_sequential",
            "shared_gamma": "before_down",
            "shared_accumulation": "float32_sum_then_bfloat16",
            "final_add": "bfloat16_completed",
        },
    }


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu().contiguous()
    candidate = candidate.detach().float().cpu().contiguous()
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ComposedMoeError(
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


def tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    """Retain exact float32 and BF16 payload bits for paired-arm comparison."""
    raw = tensor.detach().float().cpu().contiguous()
    bfloat16 = raw.to(torch.bfloat16).contiguous()
    raw_bits = raw.view(torch.int32).reshape(-1)
    bfloat16_bits = bfloat16.view(torch.int16).reshape(-1)
    return {
        "shape": list(raw.shape),
        "byte_order": sys.byteorder,
        "float32_bits": raw_bits.tolist(),
        "float32_sha256": hashlib.sha256(
            raw_bits.numpy().tobytes(order="C")
        ).hexdigest(),
        "bfloat16_bits": bfloat16_bits.tolist(),
        "bfloat16_sha256": hashlib.sha256(
            bfloat16_bits.numpy().tobytes(order="C")
        ).hexdigest(),
    }


def classify_stages(stage_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prefix: list[str] = []
    for stage in STAGES:
        if stage_metrics[stage]["bfloat16_exact_fraction"] != 1.0:
            return {
                "classification": "composed_moe_mismatch_isolated",
                "exact_bfloat16_prefix": prefix,
                "first_nonexact_stage": stage,
            }
        prefix.append(stage)
    return {
        "classification": "single_token_sparse_layer_is_bfloat16_exact",
        "exact_bfloat16_prefix": prefix,
        "first_nonexact_stage": None,
    }


def complete_evidence_outcome(
    analyses: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    exact_layers = sorted(
        int(layer)
        for layer, value in analyses.items()
        if value["decision"]["classification"]
        == "single_token_sparse_layer_is_bfloat16_exact"
    )
    nonexact_layers = sorted(
        int(layer) for layer in analyses if int(layer) not in exact_layers
    )
    return {
        "classification": (
            "complete_sparse_layers_are_bfloat16_exact"
            if not nonexact_layers
            else "complete_sparse_layer_mismatch_observed"
        ),
        "exact_layers": exact_layers,
        "nonexact_layers": nonexact_layers,
    }


def classify_dispatch_pair(
    native: Mapping[str, Any],
    forced_avx2: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared native/forced-AVX2 interpretation table."""
    native_profile = native["execution_profile"]
    forced_profile = forced_avx2["execution_profile"]
    host_class = native_profile["host_class_sha256"]
    if forced_profile["host_class_sha256"] != host_class:
        raise ComposedMoeError("dispatch-pair arms ran on different host classes")
    layers = sorted(native["layers"], key=int)
    if layers != sorted(forced_avx2["layers"], key=int):
        raise ComposedMoeError("dispatch-pair layer sets differ")

    layer_out_bitwise: dict[str, Any] = {}
    for layer in layers:
        native_payloads = native["layers"][layer]["layer_out_payloads"]
        forced_payloads = forced_avx2["layers"][layer]["layer_out_payloads"]
        comparison: dict[str, bool] = {}
        for side in ("official", "candidate"):
            for representation in ("float32", "bfloat16"):
                key = f"{representation}_bits"
                comparison[f"{side}_{representation}_equal"] = (
                    native_payloads[side][key] == forced_payloads[side][key]
                )
        comparison["all_payloads_equal"] = all(comparison.values())
        layer_out_bitwise[layer] = comparison

    arms_equal = all(
        value["all_payloads_equal"] for value in layer_out_bitwise.values()
    )
    exact_classification = "complete_sparse_layers_are_bfloat16_exact"
    native_exact = (
        native["evidence_outcome"]["classification"] == exact_classification
    )
    forced_exact = (
        forced_avx2["evidence_outcome"]["classification"]
        == exact_classification
    )

    if native_exact and forced_exact:
        if arms_equal:
            classification = "both_dispatch_profiles_exact_and_bitwise_equal"
            reference_profile_bound = False
            next_action = "no_failure_reproduced"
        else:
            classification = (
                "both_dispatch_profiles_exact_but_layer_out_is_dispatch_variant"
            )
            reference_profile_bound = True
            next_action = "scope_official_exactness_to_named_reference_profiles"
        denominator_live = False
    elif not native_exact and forced_exact:
        classification = "forced_avx2_closes_native_mismatch"
        reference_profile_bound = True
        denominator_live = False
        next_action = "scope_official_exactness_to_named_reference_profiles"
    elif native_exact and not forced_exact:
        classification = "forced_avx2_introduces_mismatch"
        reference_profile_bound = True
        denominator_live = False
        next_action = "scope_official_exactness_to_named_reference_profiles"
    elif arms_equal:
        classification = "dispatch_invariant_mismatch_keeps_denominator_defect_live"
        reference_profile_bound = False
        denominator_live = True
        next_action = "investigate_logsumexp_denominator_reduction"
    else:
        classification = "dispatch_variant_mismatch_remains_profile_bound"
        reference_profile_bound = True
        denominator_live = False
        next_action = "scope_profiles_before_investigating_residual_mismatch"

    return {
        "format": "inkling-bfloat16-dispatch-pair-classification",
        "version": 1,
        "host_class_sha256": host_class,
        "native_reference_profile_sha256": native_profile[
            "reference_profile_sha256"
        ],
        "forced_avx2_reference_profile_sha256": forced_profile[
            "reference_profile_sha256"
        ],
        "native_exact": native_exact,
        "forced_avx2_exact": forced_exact,
        "arms_bitwise_equal": arms_equal,
        "layer_out_bitwise": layer_out_bitwise,
        "classification": classification,
        "reference_profile_bound": reference_profile_bound,
        "denominator_defect_branch_live": denominator_live,
        "next_action": next_action,
    }


class CapturingMoeProvider(MoeMatrixProvider):
    def __init__(self, layer: int, module: Any) -> None:
        super().__init__(layer, module)
        self.router_logits: torch.Tensor | None = None

    def _call(self, ctx, layer, kind, index, x, out, rows, cols):
        rc = super()._call(ctx, layer, kind, index, x, out, rows, cols)
        if rc == 0 and kind == MAT_ROUTER:
            self.router_logits = torch.tensor(
                [float(out[position]) for position in range(rows)],
                dtype=torch.float32,
            )
        return rc


class ExactWeightCollector(CanonicalRouteCollector):
    def __init__(
        self,
        row: RouteRow,
        provider: CapturingMoeProvider,
        helper: FixedWeightHelper,
        n_routed: int,
        route_scale: float,
        global_scale: float,
    ) -> None:
        super().__init__([row])
        self.provider = provider
        self.helper = helper
        self.n_routed = int(n_routed)
        self.route_scale = float(route_scale)
        self.global_scale = float(global_scale)
        self.exact_weights: torch.Tensor | None = None

    def _compute_weights(self) -> torch.Tensor:
        if self.exact_weights is not None:
            return self.exact_weights
        if self.provider.router_logits is None:
            raise ComposedMoeError("router logits were not captured before route weights")
        row = self.rows[0]
        ids = torch.tensor(row.indices, dtype=torch.int64)
        selected = selected_logits(self.provider.router_logits, ids, self.n_routed)
        self.exact_weights = self.helper.evaluate(
            selected,
            self.route_scale,
            self.global_scale,
            1,
            ROUTER_POLICY_FLAGS,
        )
        return self.exact_weights

    def _emit_float(self, ctx, layer, point, data, count) -> int:
        decoded = point.decode("ascii", "strict")
        if decoded == "routed_weight":
            if int(count) != len(self.rows[0].indices):
                self.error = "routed-weight width changed"
                return 1
            self._preserve("candidate_routed_weight", data, int(count), integer=False)
            weights = self._compute_weights()
            for index in range(int(count)):
                data[index] = float(weights[index])
        elif decoded == "shared_weight":
            self._preserve("candidate_shared_weight", data, int(count), integer=False)
            weights = self._compute_weights()
            start = len(self.rows[0].indices)
            if weights.numel() != start + int(count):
                self.error = "shared-weight width changed"
                return 1
            for index in range(int(count)):
                data[index] = float(weights[start + index])
        return TraceCollector._emit_float(self, ctx, layer, point, data, count)


def official_stages(
    module: Any,
    normalized: torch.Tensor,
    residual: torch.Tensor,
    row: RouteRow,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    normalized = normalized.detach().to(torch.bfloat16).reshape(1, -1)
    gate = CanonicalOfficialGate(module.mlp.gate, [row])
    _, routed_weights, routed_ids, shared_gammas = gate(normalized)
    if not gate.applied:
        raise ComposedMoeError("canonical official gate was not applied")
    ids = [int(value) for value in routed_ids[0].detach().cpu().tolist()]
    first_position = min(range(len(ids)), key=lambda position: ids[position])
    first_expert = ids[first_position]
    fused = module.mlp.experts.gate_up[str(first_expert)]
    routed_gate, routed_up = F.linear(normalized, fused).chunk(2, dim=-1)
    routed_gated = F.silu(routed_gate) * routed_up
    routed_down = F.linear(
        routed_gated, module.mlp.experts.down[str(first_expert)]
    )
    shared_gate = F.linear(normalized, module.mlp.shared_experts.gate_proj[0])
    shared_up = F.linear(normalized, module.mlp.shared_experts.up_proj[0])
    shared_gated = F.silu(shared_gate) * shared_up
    shared_weighted = shared_gated * shared_gammas[0, 0]
    shared_down = F.linear(
        shared_weighted, module.mlp.shared_experts.down_proj[0]
    )
    routed_out = module.mlp.experts(
        normalized, routed_ids, routed_weights
    )
    shared_out = module.mlp.shared_experts(
        normalized, gammas=shared_gammas
    )
    moe_out = routed_out + shared_out
    mlp_branch = module.mlp_sconv(
        moe_out.reshape(1, 1, -1),
        past_key_values=None,
        conv_mask=None,
    )
    layer_out = residual.detach().to(torch.bfloat16).reshape(1, 1, -1) + mlp_branch
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
        "mlp_branch": mlp_branch[0, 0],
        "layer_out": layer_out[0, 0],
    }, {
        "sorted_first_routed_expert": first_expert,
        "routed_ids": routed_ids.detach().cpu().tolist(),
        "routed_weights": routed_weights.detach().float().cpu().tolist(),
        "shared_gammas": shared_gammas.detach().float().cpu().tolist(),
    }


def run_c_layer(
    lib: Any,
    fixture: Any,
    cfg: dict[str, Any],
    layer: int,
    input_row: list[float],
    row: RouteRow,
    provider: CapturingMoeProvider,
    collector: ExactWeightCollector,
) -> dict[str, Any]:
    compact = bind_sparse_layer_compact(fixture, cfg, layer)
    if compact.provider is None:
        raise ComposedMoeError("sparse fixture has no expert provider")
    missing = sorted(set(row.indices).difference(compact.provider.experts))
    if missing:
        raise ComposedMoeError(f"layer {layer} fixture omits experts {missing}")
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
        raise ComposedMoeError("scratch init failed")
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
        raise ComposedMoeError("state init failed")
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
        raise ComposedMoeError(provider.error)
    if collector.error:
        raise ComposedMoeError(collector.error)
    if rc:
        raise ComposedMoeError(f"C layer failed with status {rc}")
    if collector.applied_positions != {0}:
        raise ComposedMoeError("canonical IDs were not injected")
    stages: dict[str, torch.Tensor] = {}
    for stage in STAGES:
        key = f"token.0.layer.{layer}.{stage}"
        if key not in collector.values:
            raise ComposedMoeError(f"C trace missed {key}")
        stages[stage] = torch.tensor(collector.values[key], dtype=torch.float32)
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
    }


def analyze_layer(
    lib: Any,
    helper: FixedWeightHelper,
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
    official, official_route = official_stages(
        module,
        pre["post_attention_norm"][0],
        pre["post_attention_residual"][0],
        row,
    )
    provider = CapturingMoeProvider(layer, module)
    collector = ExactWeightCollector(
        row,
        provider,
        helper,
        int(module.mlp.gate.num_experts),
        float(module.mlp.gate.route_scale),
        float(module.mlp.gate.global_scale.detach().float().cpu().reshape(-1)[0]),
    )
    candidate = run_c_layer(
        lib, fixture, cfg, layer, input_values[0], row, provider, collector
    )
    comparisons = {
        stage: metrics(official[stage], candidate["stages"][stage])
        for stage in STAGES
    }
    return {
        "decision": classify_stages(comparisons),
        "stages": comparisons,
        "layer_out_payloads": {
            "official": tensor_payload(official["layer_out"]),
            "candidate": tensor_payload(candidate["stages"]["layer_out"]),
        },
        "official_route": official_route,
        "candidate_routed_index": candidate["candidate_routed_index"],
        "candidate_routed_weight": candidate["candidate_routed_weight"],
        "candidate_shared_weight": candidate["candidate_shared_weight"],
        "canonical_routed_index": candidate["canonical_routed_index"],
        "exact_routed_weight": candidate["exact_routed_weight"],
        "exact_shared_weight": candidate["exact_shared_weight"],
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
        values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(values, list) or len(values) != 8:
            raise ComposedMoeError("--inputs must contain eight source-bound rows")
        inputs = torch.tensor(values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise ComposedMoeError("--layers must be nonempty")
        routes = load_canonical_routes(args.selection, fixture, values, layers)
        for layer in layers:
            if set(fixture.experts.get(layer, ())) != set(routes[layer][0].indices):
                raise ComposedMoeError(f"layer {layer} fixture expert set changed")
        device = torch.device(args.device)
        if device.type != "cpu":
            raise ComposedMoeError("the composed C probe requires CPU")
        dtype = DTYPES[args.dtype]
        helper = FixedWeightHelper(
            build_helper(Path(args.out).with_suffix(".weights.so"))
        )
        library, source = build_composed_library(
            Path(args.out).with_suffix(".runtime.so")
        )
        if not source["production_source_unchanged"]:
            raise ComposedMoeError("production source changed")
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        analyses = {
            str(layer): analyze_layer(
                lib, helper, fixture, config, cfg, layer, inputs, values,
                routes[layer][0], device=device, dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-portable-bfloat16-composed-moe-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "source_tokens": len(values),
            "executed_position": 0,
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "execution_profile": collect_execution_profile(),
            "evidence_outcome": complete_evidence_outcome(analyses),
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
        ComposedMoeError,
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
