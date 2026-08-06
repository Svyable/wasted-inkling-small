#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Classify Inkling BF16 router-weight normalization with fixed expert IDs.

The sparse-MoE aggregation policy is known, but the observed C shared weights
were computed from cutoff-tie alternative routed IDs while the official oracle
used source-bound canonical IDs.  This evidence-only probe separates those two
causes before proposing any arithmetic change.

For each layer it requires two exact anchors:

* the official gate output equals a manual execution of the published BF16
  ``logsigmoid -> logsumexp -> exp -> route_scale -> global_scale`` path;
* a standalone C helper reproduces the checked-in ``waste_inkling_route``
  normalization exactly when fed the actual C-selected IDs.

It then fixes the canonical routed IDs and enumerates portable C completion
policies across logsigmoid, logsumexp/ratio normalization, reduction, scaling,
and final BF16 boundaries.  Expert values are never evaluated.
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

from diagnose_inkling_pre_router import (
    PreRouterDiagnosisError,
    _capture_official_pre_router,
)
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import (
    CanonicalRouteError,
    RouteRow,
    load_canonical_routes,
)
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    build_layer_from_fixture,
    verify_config_binding,
)
from inkling_layer_parity import build_library
from inkling_release_config import build_transformers_text_config
from run_inkling_fixture_reference_canonical import (
    CanonicalOfficialGate,
    CanonicalOfficialRouteError,
)


class RouterWeightLatticeError(RuntimeError):
    """The router-weight lattice cannot be interpreted without exact anchors."""


F_LOGP_BF16 = 1 << 0
F_REDUCE_DELTA_BF16 = 1 << 1
F_REDUCE_EXP_BF16 = 1 << 2
F_DENOM_BF16 = 1 << 3
F_LOGDENOM_BF16 = 1 << 4
F_LSE_BF16 = 1 << 5
F_OUTPUT_DELTA_BF16 = 1 << 6
F_NORMALIZED_BF16 = 1 << 7
F_ROUTE_SCALE_BF16 = 1 << 8
F_GLOBAL_SCALE_BF16 = 1 << 9
F_FINAL_BF16 = 1 << 10
FLAG_COUNT = 11

FLAG_NAMES = (
    (F_LOGP_BF16, "logp_bf16"),
    (F_REDUCE_DELTA_BF16, "reduce_delta_bf16"),
    (F_REDUCE_EXP_BF16, "reduce_exp_bf16"),
    (F_DENOM_BF16, "denom_bf16"),
    (F_LOGDENOM_BF16, "logdenom_bf16"),
    (F_LSE_BF16, "lse_bf16"),
    (F_OUTPUT_DELTA_BF16, "output_delta_bf16"),
    (F_NORMALIZED_BF16, "normalized_bf16"),
    (F_ROUTE_SCALE_BF16, "route_scale_bf16"),
    (F_GLOBAL_SCALE_BF16, "global_scale_bf16"),
    (F_FINAL_BF16, "final_bf16"),
)

_C_SOURCE = r"""
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

static float bf16_round_probe(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float logsigmoid_probe(float x)
{
    if (x >= 0.0f) return -log1pf(expf(-x));
    return x - log1pf(expf(x));
}

int inkling_probe_fixed_router_weights(
    const float *selected_logits, size_t n,
    float route_scale, float global_scale,
    int family, unsigned flags, float *out)
{
    if (!selected_logits || !out || n == 0 || n > 16 ||
        (family != 0 && family != 1))
        return -1;

    float logp[16];
    float terms[16];
    float max_logp = -FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        float value = logsigmoid_probe(selected_logits[i]);
        if (flags & F_LOGP_BF16) value = bf16_round_probe(value);
        logp[i] = value;
        if (value > max_logp) max_logp = value;
    }

    float denom = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float delta = logp[i] - max_logp;
        if (flags & F_REDUCE_DELTA_BF16)
            delta = bf16_round_probe(delta);
        float term = expf(delta);
        if (flags & F_REDUCE_EXP_BF16)
            term = bf16_round_probe(term);
        terms[i] = term;
        denom += term;
        if (flags & F_DENOM_BF16)
            denom = bf16_round_probe(denom);
    }
    if (!(denom > 0.0f) || !isfinite(denom)) return -1;

    float lse = 0.0f;
    if (family == 1) {
        float logdenom = logf(denom);
        if (flags & F_LOGDENOM_BF16)
            logdenom = bf16_round_probe(logdenom);
        lse = max_logp + logdenom;
        if (flags & F_LSE_BF16) lse = bf16_round_probe(lse);
    }

    for (size_t i = 0; i < n; i++) {
        float value;
        if (family == 0) {
            value = terms[i] / denom;
        } else {
            float delta = logp[i] - lse;
            if (flags & F_OUTPUT_DELTA_BF16)
                delta = bf16_round_probe(delta);
            value = expf(delta);
        }
        if (flags & F_NORMALIZED_BF16)
            value = bf16_round_probe(value);
        value *= route_scale;
        if (flags & F_ROUTE_SCALE_BF16)
            value = bf16_round_probe(value);
        value *= global_scale;
        if (flags & F_GLOBAL_SCALE_BF16)
            value = bf16_round_probe(value);
        if (flags & F_FINAL_BF16)
            value = bf16_round_probe(value);
        out[i] = value;
    }
    return 0;
}
"""


def build_helper(out: Path) -> Path:
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        raise RouterWeightLatticeError("no C compiler available")
    root = Path(tempfile.mkdtemp(prefix="inkling-router-weight-lattice-"))
    source = root / "router_weight_probe.c"
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
        raise RouterWeightLatticeError(
            f"router-weight helper build failed:\n{result.stderr}"
        )
    return out


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu().contiguous()
    candidate = candidate.detach().float().cpu().contiguous()
    if tuple(reference.shape) != tuple(candidate.shape):
        raise RouterWeightLatticeError(
            f"shape mismatch {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    if not reference.numel():
        raise RouterWeightLatticeError("cannot compare empty router weights")
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


def flag_names(flags: int) -> list[str]:
    return [name for bit, name in FLAG_NAMES if flags & bit]


def policy_key(family: int, flags: int) -> str:
    return f"{'ratio' if family == 0 else 'logsumexp'}:{flags:03x}"


def policy_description(family: int, flags: int) -> dict[str, Any]:
    return {
        "key": policy_key(family, flags),
        "family": "ratio" if family == 0 else "logsumexp",
        "flags": int(flags),
        "boundaries": flag_names(flags),
    }


def semantic_penalty(family: int, flags: int) -> tuple[int, int, int]:
    expected = (
        F_LOGP_BF16
        | F_LSE_BF16
        | F_OUTPUT_DELTA_BF16
        | F_NORMALIZED_BF16
        | F_ROUTE_SCALE_BF16
        | F_GLOBAL_SCALE_BF16
        | F_FINAL_BF16
    )
    missing = (expected & ~flags).bit_count()
    extras = (flags & ~expected).bit_count()
    return (0 if family == 1 else 1, missing, extras)


class FixedWeightHelper:
    def __init__(self, library: Path) -> None:
        self.lib = ctypes.CDLL(str(library))
        self.fn = self.lib.inkling_probe_fixed_router_weights
        fp = ctypes.POINTER(ctypes.c_float)
        self.fn.argtypes = [
            fp,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_uint,
            fp,
        ]
        self.fn.restype = ctypes.c_int

    def evaluate(
        self,
        selected_logits: torch.Tensor,
        route_scale: float,
        global_scale: float,
        family: int,
        flags: int,
    ) -> torch.Tensor:
        values = (
            selected_logits.detach().float().cpu().contiguous().reshape(-1)
        )
        n = int(values.numel())
        if n <= 0 or n > 16:
            raise RouterWeightLatticeError(
                f"selected router width {n} is outside helper bounds"
            )
        input_buf = (ctypes.c_float * n)(*values.tolist())
        output_buf = (ctypes.c_float * n)()
        rc = self.fn(
            input_buf,
            n,
            float(route_scale),
            float(global_scale),
            int(family),
            int(flags),
            output_buf,
        )
        if rc:
            raise RouterWeightLatticeError(
                f"fixed router helper failed family={family} flags={flags}"
            )
        return torch.tensor(
            [float(output_buf[index]) for index in range(n)],
            dtype=torch.float32,
        )


class ActualCRouter:
    """Invoke the checked-in waste_inkling_route implementation directly."""

    def __init__(self, library: Path) -> None:
        self.lib = ctypes.CDLL(str(library))
        self.fn = self.lib.waste_inkling_route
        fp = ctypes.POINTER(ctypes.c_float)
        ip = ctypes.POINTER(ctypes.c_int)
        self.fn.argtypes = [
            fp,
            fp,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ip,
            fp,
            fp,
        ]
        self.fn.restype = ctypes.c_int

    def evaluate(
        self,
        logits: torch.Tensor,
        correction_bias: torch.Tensor,
        n_routed: int,
        n_shared: int,
        top_k: int,
        route_scale: float,
        global_scale: float,
    ) -> dict[str, torch.Tensor]:
        logits = logits.detach().float().cpu().contiguous().reshape(-1)
        correction_bias = (
            correction_bias.detach().float().cpu().contiguous().reshape(-1)
        )
        if logits.numel() != n_routed + n_shared:
            raise RouterWeightLatticeError("router logits geometry changed")
        if correction_bias.numel() != n_routed:
            raise RouterWeightLatticeError("router correction-bias geometry changed")
        logits_buf = (ctypes.c_float * logits.numel())(*logits.tolist())
        bias_buf = (ctypes.c_float * n_routed)(*correction_bias.tolist())
        ids = (ctypes.c_int * top_k)()
        routed = (ctypes.c_float * top_k)()
        shared = (ctypes.c_float * max(n_shared, 1))()
        rc = self.fn(
            logits_buf,
            bias_buf,
            n_routed,
            n_shared,
            top_k,
            float(route_scale),
            float(global_scale),
            ids,
            routed,
            shared,
        )
        if rc:
            raise RouterWeightLatticeError("checked-in C router failed")
        return {
            "indices": torch.tensor(
                [int(ids[index]) for index in range(top_k)],
                dtype=torch.int64,
            ),
            "routed": torch.tensor(
                [float(routed[index]) for index in range(top_k)],
                dtype=torch.float32,
            ),
            "shared": torch.tensor(
                [float(shared[index]) for index in range(n_shared)],
                dtype=torch.float32,
            ),
        }


def selected_logits(
    logits: torch.Tensor, ids: torch.Tensor, n_routed: int
) -> torch.Tensor:
    routed = logits[:n_routed].index_select(0, ids.to(torch.int64))
    shared = logits[n_routed:]
    return torch.cat((routed, shared), dim=0).contiguous()


def official_manual_weights(
    selected: torch.Tensor,
    route_scale: float,
    global_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    selected = selected.detach().to(torch.bfloat16)
    logp = F.logsigmoid(selected)
    lse = torch.logsumexp(logp, dim=-1, keepdim=True)
    normalized = torch.exp(logp - lse)
    after_route_scale = normalized * route_scale
    final = after_route_scale * global_scale.detach().to(torch.bfloat16)
    return {
        "selected_logits": selected.detach().cpu(),
        "logsigmoid": logp.detach().cpu(),
        "logsumexp": lse.detach().cpu(),
        "normalized": normalized.detach().cpu(),
        "after_route_scale": after_route_scale.detach().cpu(),
        "final": final.detach().cpu(),
    }


def top_policy_records(
    official: torch.Tensor,
    helper: FixedWeightHelper,
    selected: torch.Tensor,
    route_scale: float,
    global_scale: float,
    *,
    limit: int = 12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    exact: list[dict[str, Any]] = []
    for family in (0, 1):
        for flags in range(1 << FLAG_COUNT):
            candidate = helper.evaluate(
                selected,
                route_scale,
                global_scale,
                family,
                flags,
            )
            value = metrics(official, candidate)
            record = {
                **policy_description(family, flags),
                "metrics": value,
                "semantic_penalty": list(semantic_penalty(family, flags)),
            }
            records.append(record)
            if value["raw_exact_fraction"] == 1.0:
                exact.append(record)
    records.sort(
        key=lambda item: (
            -item["metrics"]["raw_exact_fraction"],
            -item["metrics"]["bfloat16_exact_fraction"],
            item["metrics"]["raw_max_abs"],
            tuple(item["semantic_penalty"]),
        )
    )
    exact.sort(
        key=lambda item: (
            tuple(item["semantic_penalty"]),
            len(item["boundaries"]),
            item["key"],
        )
    )
    return records[:limit], exact


def split_metrics(
    official: torch.Tensor, candidate: torch.Tensor, top_k: int
) -> dict[str, Any]:
    return {
        "all": metrics(official, candidate),
        "routed": metrics(official[:top_k], candidate[:top_k]),
        "shared": metrics(official[top_k:], candidate[top_k:]),
    }


def classify_layer(
    anchors: dict[str, dict[str, Any]],
    fixed_current: dict[str, Any],
    exact_policies: list[dict[str, Any]],
) -> dict[str, Any]:
    failed = [
        name
        for name, value in anchors.items()
        if value["raw_exact_fraction"] != 1.0
    ]
    if failed:
        return {
            "classification": "router_weight_anchor_failed",
            "failed_anchors": failed,
            "preferred_policy": None,
        }
    if fixed_current["all"]["raw_exact_fraction"] == 1.0:
        return {
            "classification": "router_weight_mismatch_is_route_choice_only",
            "failed_anchors": [],
            "preferred_policy": {
                "key": "ratio:000",
                "family": "ratio",
                "flags": 0,
                "boundaries": [],
            },
        }
    if exact_policies:
        preferred = {
            key: exact_policies[0][key]
            for key in ("key", "family", "flags", "boundaries")
        }
        return {
            "classification": "portable_router_weight_policy_identified",
            "failed_anchors": [],
            "preferred_policy": preferred,
            "exact_policy_count": len(exact_policies),
        }
    return {
        "classification": "native_or_unmodeled_router_reduction_required",
        "failed_anchors": [],
        "preferred_policy": None,
        "exact_policy_count": 0,
    }


def analyze_layer(
    helper: FixedWeightHelper,
    actual_router: ActualCRouter,
    fixture: Any,
    config: Any,
    layer: int,
    inputs: torch.Tensor,
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
    with torch.no_grad():
        logits = F.linear(normalized, module.mlp.gate.weight)[0]
    gate = CanonicalOfficialGate(module.mlp.gate, [route])
    _routed_logits, routed_weights, routed_ids, shared_gammas = gate(normalized)
    if not gate.applied:
        raise RouterWeightLatticeError("canonical official gate was not applied")

    n_routed = int(module.mlp.gate.num_experts)
    n_shared = int(module.mlp.gate.n_shared_experts)
    top_k = int(module.mlp.gate.top_k)
    route_scale = float(module.mlp.gate.route_scale)
    global_scale_tensor = module.mlp.gate.global_scale.detach()
    global_scale = float(global_scale_tensor.float().cpu().reshape(-1)[0])
    correction_bias = module.mlp.gate.e_score_correction_bias.detach()

    official_all = torch.cat(
        (
            routed_weights[0].detach().cpu(),
            shared_gammas[0].detach().cpu(),
        ),
        dim=0,
    )
    canonical_ids = torch.tensor(route.indices, dtype=torch.int64)
    canonical_selected = selected_logits(
        logits.detach().cpu(), canonical_ids, n_routed
    )
    manual = official_manual_weights(
        canonical_selected,
        route_scale,
        global_scale_tensor.cpu(),
    )

    actual = actual_router.evaluate(
        logits.detach().cpu(),
        correction_bias.cpu(),
        n_routed,
        n_shared,
        top_k,
        route_scale,
        global_scale,
    )
    actual_selected = selected_logits(
        logits.detach().cpu(), actual["indices"], n_routed
    )
    actual_helper = helper.evaluate(
        actual_selected,
        route_scale,
        global_scale,
        0,
        0,
    )
    actual_all = torch.cat((actual["routed"], actual["shared"]), dim=0)

    fixed_current = helper.evaluate(
        canonical_selected,
        route_scale,
        global_scale,
        0,
        0,
    )
    top_records, exact = top_policy_records(
        official_all,
        helper,
        canonical_selected,
        route_scale,
        global_scale,
    )

    anchors = {
        "official_manual_final": metrics(official_all, manual["final"]),
        "actual_c_formula": metrics(actual_all, actual_helper),
    }
    fixed_metrics = split_metrics(official_all, fixed_current, top_k)
    actual_metrics = split_metrics(official_all, actual_all, top_k)
    decision = classify_layer(anchors, fixed_metrics, exact)
    preferred_metrics = None
    if decision["preferred_policy"] is not None:
        policy = decision["preferred_policy"]
        preferred_candidate = helper.evaluate(
            canonical_selected,
            route_scale,
            global_scale,
            0 if policy["family"] == "ratio" else 1,
            int(policy["flags"]),
        )
        preferred_metrics = split_metrics(
            official_all, preferred_candidate, top_k
        )

    return {
        "decision": decision,
        "anchors": anchors,
        "dtypes": {
            "router_logits": str(logits.dtype).removeprefix("torch."),
            "correction_bias": str(correction_bias.dtype).removeprefix(
                "torch."
            ),
            "global_scale": str(global_scale_tensor.dtype).removeprefix(
                "torch."
            ),
            "official_routed_weights": str(routed_weights.dtype).removeprefix(
                "torch."
            ),
            "official_shared_gammas": str(shared_gammas.dtype).removeprefix(
                "torch."
            ),
        },
        "route_scale": route_scale,
        "global_scale": global_scale,
        "canonical_indices": canonical_ids.tolist(),
        "actual_c_indices": actual["indices"].tolist(),
        "actual_c_vs_official": actual_metrics,
        "fixed_canonical_current_formula": fixed_metrics,
        "preferred_policy_metrics": preferred_metrics,
        "exact_policy_count": len(exact),
        "exact_policies": exact[:24],
        "top_policies": top_records,
        "official_stages": {
            name: value.detach().float().cpu().reshape(-1).tolist()
            for name, value in manual.items()
        },
        "official_routed_weights": routed_weights[0]
        .detach()
        .float()
        .cpu()
        .tolist(),
        "official_shared_gammas": shared_gammas[0]
        .detach()
        .float()
        .cpu()
        .tolist(),
        "actual_c_routed_weights": actual["routed"].tolist(),
        "actual_c_shared_weights": actual["shared"].tolist(),
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
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(input_values, list) or len(input_values) != 8:
            raise RouterWeightLatticeError(
                "--inputs must contain the eight source-bound deterministic rows"
            )
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise RouterWeightLatticeError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection, fixture, input_values, layers
        )
        device = torch.device(args.device)
        if device.type != "cpu":
            raise RouterWeightLatticeError("router-weight lattice requires CPU")
        dtype = DTYPES[args.dtype]

        helper = FixedWeightHelper(
            build_helper(Path(args.out).with_suffix(".weights.so"))
        )
        source_root = Path(__file__).resolve().parents[2] / "inkling" / "src"
        actual_router = ActualCRouter(
            build_library(
                source_root,
                Path(args.out).with_suffix(".runtime.so"),
            )
        )
        analyses = {
            str(layer): analyze_layer(
                helper,
                actual_router,
                fixture,
                config,
                layer,
                inputs,
                routes[layer][0],
                device=device,
                dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-bfloat16-router-weight-lattice",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "source_tokens": len(input_values),
            "executed_position": 0,
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
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
        RouterWeightLatticeError,
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
