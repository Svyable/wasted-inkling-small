#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Name the first unmodeled BF16 router-normalization stage.

#48 proves the layer-2 canonical selected router logits are raw-exact between
official prefill and the C backend at all eight positions, yet position 3 has no
exact policy in the existing 4,096 final-weight lattice. This evidence-only
probe instruments the logsumexp-family scalar helper and tracks prefix-exact
policies through the official normalization stages:

selected logits -> logsigmoid -> logsumexp -> normalized -> route scale ->
final global-scaled weights.

The first stage where every policy that was exact through the previous stage
fails is the next arithmetic primitive to investigate. Experts are never
executed; production source and public APIs remain unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from diagnose_inkling_bf16_router_weight_lattice import (
    F_DENOM_BF16,
    F_FINAL_BF16,
    F_GLOBAL_SCALE_BF16,
    F_LOGDENOM_BF16,
    F_LOGP_BF16,
    F_LSE_BF16,
    F_NORMALIZED_BF16,
    F_OUTPUT_DELTA_BF16,
    F_REDUCE_DELTA_BF16,
    F_REDUCE_EXP_BF16,
    F_ROUTE_SCALE_BF16,
    FLAG_COUNT,
    metrics,
    official_manual_weights,
    policy_description,
    selected_logits,
    semantic_penalty,
)
from diagnose_inkling_pre_router import PreRouterDiagnosisError, _capture_official_pre_router
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import CanonicalRouteError, load_canonical_routes
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import DTYPES, FixtureReferenceError, build_layer_from_fixture, verify_config_binding
from inkling_release_config import build_transformers_text_config
from run_inkling_fixture_reference_canonical import CanonicalOfficialGate, CanonicalOfficialRouteError


class RouterNormalizationStageError(RuntimeError):
    """The router stage lattice cannot continue without exact anchors."""


LEGACY_FLAGS = (
    F_LOGP_BF16
    | F_LSE_BF16
    | F_OUTPUT_DELTA_BF16
    | F_NORMALIZED_BF16
    | F_ROUTE_SCALE_BF16
    | F_GLOBAL_SCALE_BF16
    | F_FINAL_BF16
)

STAGES = (
    "logsigmoid",
    "logsumexp",
    "normalized",
    "after_route_scale",
    "final",
)

_C_SOURCE = r'''
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#define F_LOGP_BF16 (1u << 0)
#define F_REDUCE_DELTA_BF16 (1u << 1)
#define F_REDUCE_EXP_BF16 (1u << 2)
#define F_DENOM_BF16 (1u << 3)
#define F_LOGDENOM_BF16 (1u << 4)
#define F_LSE_BF16 (1u << 5)
#define F_OUTPUT_DELTA_BF16 (1u << 6)
#define F_NORMALIZED_BF16 (1u << 7)
#define F_ROUTE_SCALE_BF16 (1u << 8)
#define F_GLOBAL_SCALE_BF16 (1u << 9)
#define F_FINAL_BF16 (1u << 10)

static float bf16_round_probe(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float logsigmoid_probe(float x) {
    if (x >= 0.0f) return -log1pf(expf(-x));
    return x - log1pf(expf(x));
}

int inkling_probe_router_stages(
    const float *selected_logits, size_t n,
    float route_scale, float global_scale, unsigned flags,
    float *out_logp, float *out_lse, float *out_normalized,
    float *out_route, float *out_global, float *out_final) {
    if (!selected_logits || !out_logp || !out_lse || !out_normalized ||
        !out_route || !out_global || !out_final || n == 0 || n > 16)
        return -1;

    float logp[16], max_logp = -FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        float v = logsigmoid_probe(selected_logits[i]);
        if (flags & F_LOGP_BF16) v = bf16_round_probe(v);
        logp[i] = v;
        out_logp[i] = v;
        if (v > max_logp) max_logp = v;
    }

    float denom = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = logp[i] - max_logp;
        if (flags & F_REDUCE_DELTA_BF16) d = bf16_round_probe(d);
        float term = expf(d);
        if (flags & F_REDUCE_EXP_BF16) term = bf16_round_probe(term);
        denom += term;
        if (flags & F_DENOM_BF16) denom = bf16_round_probe(denom);
    }
    if (!(denom > 0.0f) || !isfinite(denom)) return -1;

    float ld = logf(denom);
    if (flags & F_LOGDENOM_BF16) ld = bf16_round_probe(ld);
    float lse = max_logp + ld;
    if (flags & F_LSE_BF16) lse = bf16_round_probe(lse);
    out_lse[0] = lse;

    for (size_t i = 0; i < n; i++) {
        float d = logp[i] - lse;
        if (flags & F_OUTPUT_DELTA_BF16) d = bf16_round_probe(d);
        float normalized = expf(d);
        if (flags & F_NORMALIZED_BF16)
            normalized = bf16_round_probe(normalized);
        out_normalized[i] = normalized;

        float route = normalized * route_scale;
        if (flags & F_ROUTE_SCALE_BF16)
            route = bf16_round_probe(route);
        out_route[i] = route;

        float global = route * global_scale;
        if (flags & F_GLOBAL_SCALE_BF16)
            global = bf16_round_probe(global);
        out_global[i] = global;

        float final = global;
        if (flags & F_FINAL_BF16)
            final = bf16_round_probe(final);
        out_final[i] = final;
    }
    return 0;
}
'''


def build_stage_helper(out: Path) -> Path:
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        raise RouterNormalizationStageError("no C compiler available")
    root = Path(tempfile.mkdtemp(prefix="inkling-router-stage-lattice-"))
    source = root / "probe.c"
    source.write_text(_C_SOURCE, encoding="utf-8")
    result = subprocess.run(
        [
            cc,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(out),
            "-lm",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RouterNormalizationStageError(
            f"stage helper build failed:\n{result.stderr}"
        )
    return out


class StageHelper:
    def __init__(self, library: Path) -> None:
        self.lib = ctypes.CDLL(str(library))
        self.fn = self.lib.inkling_probe_router_stages
        fp = ctypes.POINTER(ctypes.c_float)
        self.fn.argtypes = [
            fp,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint,
            fp,
            fp,
            fp,
            fp,
            fp,
            fp,
        ]
        self.fn.restype = ctypes.c_int

    def evaluate(
        self,
        logits: torch.Tensor,
        route_scale: float,
        global_scale: float,
        flags: int,
    ) -> dict[str, torch.Tensor]:
        values = logits.detach().float().cpu().contiguous().reshape(-1)
        n = int(values.numel())
        if not 0 < n <= 16:
            raise RouterNormalizationStageError(
                f"selected width {n} outside helper bounds"
            )
        ib = (ctypes.c_float * n)(*values.tolist())
        logp = (ctypes.c_float * n)()
        lse = (ctypes.c_float * 1)()
        normalized = (ctypes.c_float * n)()
        route = (ctypes.c_float * n)()
        global_out = (ctypes.c_float * n)()
        final = (ctypes.c_float * n)()
        if self.fn(
            ib,
            n,
            route_scale,
            global_scale,
            flags,
            logp,
            lse,
            normalized,
            route,
            global_out,
            final,
        ):
            raise RouterNormalizationStageError("router stage helper failed")
        return {
            "logsigmoid": torch.tensor(
                [float(logp[i]) for i in range(n)], dtype=torch.float32
            ),
            "logsumexp": torch.tensor([float(lse[0])], dtype=torch.float32),
            "normalized": torch.tensor(
                [float(normalized[i]) for i in range(n)], dtype=torch.float32
            ),
            "after_route_scale": torch.tensor(
                [float(route[i]) for i in range(n)], dtype=torch.float32
            ),
            "final": torch.tensor(
                [float(final[i]) for i in range(n)], dtype=torch.float32
            ),
        }


def _official_stages(manual: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "logsigmoid": manual["logsigmoid"].detach().float().reshape(-1),
        "logsumexp": manual["logsumexp"].detach().float().reshape(-1),
        "normalized": manual["normalized"].detach().float().reshape(-1),
        "after_route_scale": manual["after_route_scale"].detach().float().reshape(-1),
        "final": manual["final"].detach().float().reshape(-1),
    }


def _metric_rank(value: dict[str, Any], flags: int) -> tuple[Any, ...]:
    return (
        -value["raw_exact_fraction"],
        -value["bfloat16_exact_fraction"],
        value["raw_max_abs"],
        value["bfloat16_max_abs"],
        semantic_penalty(1, flags),
        int(flags).bit_count(),
        flags,
    )


def stage_prefix_lattice(
    helper: StageHelper,
    selected: torch.Tensor,
    official: dict[str, torch.Tensor],
    route_scale: float,
    global_scale: float,
) -> dict[str, Any]:
    evaluations = {
        flags: helper.evaluate(selected, route_scale, global_scale, flags)
        for flags in range(1 << FLAG_COUNT)
    }
    survivors = list(range(1 << FLAG_COUNT))
    summaries: dict[str, Any] = {}
    first_zero: str | None = None
    for stage in STAGES:
        candidates = []
        exact = []
        for flags in survivors:
            value = metrics(official[stage], evaluations[flags][stage])
            candidates.append((flags, value))
            if value["raw_exact_fraction"] == 1.0:
                exact.append(flags)
        best = min(
            candidates,
            key=lambda item: _metric_rank(item[1], item[0]),
        ) if candidates else None
        summaries[stage] = {
            "prefix_exact_policy_count": len(exact),
            "best_policy": (
                policy_description(1, best[0]) if best is not None else None
            ),
            "best_metrics": best[1] if best is not None else None,
        }
        survivors = exact
        if not survivors and first_zero is None:
            first_zero = stage
    final_survivors = sorted(
        survivors,
        key=lambda flags: (
            semantic_penalty(1, flags),
            int(flags).bit_count(),
            policy_description(1, flags)["key"],
        ),
    )
    legacy = helper.evaluate(
        selected, route_scale, global_scale, LEGACY_FLAGS
    )
    legacy_metrics = {
        stage: metrics(official[stage], legacy[stage])
        for stage in STAGES
    }
    return {
        "first_zero_prefix_stage": first_zero,
        "stages": summaries,
        "final_exact_policy_count": len(final_survivors),
        "preferred_final_policy": (
            policy_description(1, final_survivors[0])
            if final_survivors else None
        ),
        "legacy_stage_metrics": legacy_metrics,
    }


def classify_positions(positions: list[dict[str, Any]]) -> dict[str, Any]:
    first_final_failure = next(
        (
            item["position"]
            for item in positions
            if item["final_exact_policy_count"] == 0
        ),
        None,
    )
    if first_final_failure is None:
        return {
            "classification": "modeled_logsumexp_router_pipeline_generalizes",
            "first_unmodeled_position": None,
            "first_unmodeled_stage": None,
        }
    item = positions[first_final_failure]
    stage = item["first_zero_prefix_stage"]
    mapping = {
        "logsigmoid": "router_logsigmoid_primitive_mismatch",
        "logsumexp": "router_logsumexp_reduction_mismatch",
        "normalized": "router_normalized_exp_primitive_mismatch",
        "after_route_scale": "router_route_scale_mismatch",
        "final": "router_global_scale_or_final_completion_mismatch",
    }
    return {
        "classification": mapping.get(
            stage, "router_normalization_stage_unclassified"
        ),
        "first_unmodeled_position": first_final_failure,
        "first_unmodeled_stage": stage,
    }


def analyze_layer(
    helper: StageHelper,
    fixture: Any,
    config: Any,
    layer: int,
    inputs: torch.Tensor,
    rows: list[Any],
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
    normalized = pre["post_attention_norm"].to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        logits = F.linear(
            normalized,
            module.mlp.gate.weight.detach().to(device=device, dtype=torch.bfloat16),
        ).detach().float().cpu()
    gate = CanonicalOfficialGate(module.mlp.gate, rows)
    _, routed_weights, routed_ids, shared_gammas = gate(normalized)
    if not gate.applied:
        raise RouterNormalizationStageError(
            "canonical official gate was not applied"
        )

    n_routed = int(module.mlp.gate.num_experts)
    route_scale = float(module.mlp.gate.route_scale)
    global_tensor = module.mlp.gate.global_scale.detach().cpu()
    global_scale = float(global_tensor.float().reshape(-1)[0])

    positions: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if routed_ids[position].detach().cpu().tolist() != list(row.indices):
            raise RouterNormalizationStageError(
                f"layer {layer} position {position} canonical routed IDs changed"
            )
        ids = torch.tensor(row.indices, dtype=torch.int64)
        selected = selected_logits(logits[position], ids, n_routed)
        manual = official_manual_weights(
            selected, route_scale, global_tensor
        )
        official_final = torch.cat(
            (
                routed_weights[position].detach().cpu(),
                shared_gammas[position].detach().cpu(),
            )
        )
        anchor = metrics(official_final, manual["final"])
        if anchor["raw_exact_fraction"] != 1.0:
            raise RouterNormalizationStageError(
                f"layer {layer} position {position} official manual anchor failed"
            )
        lattice = stage_prefix_lattice(
            helper,
            selected,
            _official_stages(manual),
            route_scale,
            global_scale,
        )
        positions.append(
            {
                "position": position,
                "canonical_indices": list(row.indices),
                "official_manual_anchor": anchor,
                **lattice,
            }
        )
    return {
        "decision": classify_positions(positions),
        "positions": positions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
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
        values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(values, list) or len(values) != 8:
            raise RouterNormalizationStageError(
                "--inputs must contain eight source-bound deterministic rows"
            )
        inputs = torch.tensor(values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise RouterNormalizationStageError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection, fixture, values, layers
        )
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise RouterNormalizationStageError(
                "router normalization stage lattice requires CPU"
            )
        helper = StageHelper(
            build_stage_helper(Path(args.out).with_suffix(".stages.so"))
        )
        analyses = {
            str(layer): analyze_layer(
                helper,
                fixture,
                config,
                layer,
                inputs,
                routes[layer],
                device=device,
                dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-bfloat16-router-normalization-stage-lattice",
            "version": 2,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "source_tokens": len(values),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "legacy_policy_flags": LEGACY_FLAGS,
            "production_source_modified": False,
            "layers": analyses,
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        FixtureError,
        FixtureReferenceError,
        PreRouterDiagnosisError,
        CanonicalRouteError,
        CanonicalOfficialRouteError,
        RouterNormalizationStageError,
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
