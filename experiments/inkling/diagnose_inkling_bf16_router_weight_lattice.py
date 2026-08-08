#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Classify Inkling BF16 router-weight normalization with fixed expert IDs.

The aggregation lattice exposed a shared-weight difference, but the existing C
router had selected cutoff-tie alternative routed IDs before producing those
weights. This probe separates route-set choice from normalization arithmetic.
It compares the checked-in C router twice: once with the real correction bias,
and once with an evidence-only bias that forces the source-bound canonical IDs
without changing the logits used for normalization. A standalone C helper then
enumerates explicit BF16 completion policies on those fixed logits.

Expert values are never evaluated and production sources remain unchanged.
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

from diagnose_inkling_pre_router import PreRouterDiagnosisError, _capture_official_pre_router
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import CanonicalRouteError, RouteRow, load_canonical_routes
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import DTYPES, FixtureReferenceError, build_layer_from_fixture, verify_config_binding
from inkling_layer_parity import build_library
from inkling_release_config import build_transformers_text_config
from run_inkling_fixture_reference_canonical import CanonicalOfficialGate, CanonicalOfficialRouteError


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
    uint32_t bits; memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(value)); return value;
}
static float logsigmoid_probe(float x) {
    if (x >= 0.0f) return -log1pf(expf(-x));
    return x - log1pf(expf(x));
}
int inkling_probe_fixed_router_weights(const float *selected_logits, size_t n,
    float route_scale, float global_scale, int family, unsigned flags, float *out) {
    if (!selected_logits || !out || n == 0 || n > 16 || (family != 0 && family != 1)) return -1;
    float logp[16], terms[16], max_logp = -FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        float v = logsigmoid_probe(selected_logits[i]);
        if (flags & F_LOGP_BF16) v = bf16_round_probe(v);
        logp[i] = v; if (v > max_logp) max_logp = v;
    }
    float denom = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = logp[i] - max_logp;
        if (flags & F_REDUCE_DELTA_BF16) d = bf16_round_probe(d);
        float term = expf(d);
        if (flags & F_REDUCE_EXP_BF16) term = bf16_round_probe(term);
        terms[i] = term; denom += term;
        if (flags & F_DENOM_BF16) denom = bf16_round_probe(denom);
    }
    if (!(denom > 0.0f) || !isfinite(denom)) return -1;
    float lse = 0.0f;
    if (family == 1) {
        float ld = logf(denom);
        if (flags & F_LOGDENOM_BF16) ld = bf16_round_probe(ld);
        lse = max_logp + ld;
        if (flags & F_LSE_BF16) lse = bf16_round_probe(lse);
    }
    for (size_t i = 0; i < n; i++) {
        float v;
        if (family == 0) v = terms[i] / denom;
        else {
            float d = logp[i] - lse;
            if (flags & F_OUTPUT_DELTA_BF16) d = bf16_round_probe(d);
            v = expf(d);
        }
        if (flags & F_NORMALIZED_BF16) v = bf16_round_probe(v);
        v *= route_scale;
        if (flags & F_ROUTE_SCALE_BF16) v = bf16_round_probe(v);
        v *= global_scale;
        if (flags & F_GLOBAL_SCALE_BF16) v = bf16_round_probe(v);
        if (flags & F_FINAL_BF16) v = bf16_round_probe(v);
        out[i] = v;
    }
    return 0;
}
'''


def build_helper(out: Path) -> Path:
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        raise RouterWeightLatticeError("no C compiler available")
    root = Path(tempfile.mkdtemp(prefix="inkling-router-weight-lattice-"))
    source = root / "probe.c"
    source.write_text(_C_SOURCE, encoding="utf-8")
    result = subprocess.run(
        [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", str(source), "-o", str(out), "-lm"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RouterWeightLatticeError(f"helper build failed:\n{result.stderr}")
    return out


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu().contiguous()
    candidate = candidate.detach().float().cpu().contiguous()
    if tuple(reference.shape) != tuple(candidate.shape):
        raise RouterWeightLatticeError(f"shape mismatch {tuple(reference.shape)} != {tuple(candidate.shape)}")
    if not reference.numel():
        raise RouterWeightLatticeError("cannot compare empty router weights")
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


def policy_description(family: int, flags: int) -> dict[str, Any]:
    return {
        "key": f"{'ratio' if family == 0 else 'logsumexp'}:{flags:03x}",
        "family": "ratio" if family == 0 else "logsumexp",
        "flags": int(flags),
        "boundaries": [name for bit, name in FLAG_NAMES if flags & bit],
    }


def semantic_penalty(family: int, flags: int) -> tuple[int, int, int]:
    expected = F_LOGP_BF16 | F_LSE_BF16 | F_OUTPUT_DELTA_BF16 | F_NORMALIZED_BF16 | F_ROUTE_SCALE_BF16 | F_GLOBAL_SCALE_BF16 | F_FINAL_BF16
    return (0 if family == 1 else 1, (expected & ~flags).bit_count(), (flags & ~expected).bit_count())


class FixedWeightHelper:
    def __init__(self, library: Path) -> None:
        self.lib = ctypes.CDLL(str(library))
        self.fn = self.lib.inkling_probe_fixed_router_weights
        fp = ctypes.POINTER(ctypes.c_float)
        self.fn.argtypes = [fp, ctypes.c_size_t, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_uint, fp]
        self.fn.restype = ctypes.c_int

    def evaluate(self, logits: torch.Tensor, route_scale: float, global_scale: float, family: int, flags: int) -> torch.Tensor:
        values = logits.detach().float().cpu().contiguous().reshape(-1)
        n = int(values.numel())
        if not 0 < n <= 16:
            raise RouterWeightLatticeError(f"selected width {n} outside helper bounds")
        ib = (ctypes.c_float * n)(*values.tolist())
        ob = (ctypes.c_float * n)()
        if self.fn(ib, n, route_scale, global_scale, family, flags, ob):
            raise RouterWeightLatticeError("fixed router helper failed")
        return torch.tensor([float(ob[i]) for i in range(n)], dtype=torch.float32)


class ActualCRouter:
    def __init__(self, library: Path) -> None:
        self.lib = ctypes.CDLL(str(library))
        self.fn = self.lib.waste_inkling_route
        fp = ctypes.POINTER(ctypes.c_float)
        ip = ctypes.POINTER(ctypes.c_int)
        self.fn.argtypes = [fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ip, fp, fp]
        self.fn.restype = ctypes.c_int

    def evaluate(self, logits: torch.Tensor, bias: torch.Tensor, n_routed: int, n_shared: int, top_k: int, route_scale: float, global_scale: float) -> dict[str, torch.Tensor]:
        logits = logits.detach().float().cpu().contiguous().reshape(-1)
        bias = bias.detach().float().cpu().contiguous().reshape(-1)
        lb = (ctypes.c_float * logits.numel())(*logits.tolist())
        bb = (ctypes.c_float * n_routed)(*bias.tolist())
        ids = (ctypes.c_int * top_k)()
        routed = (ctypes.c_float * top_k)()
        shared = (ctypes.c_float * max(n_shared, 1))()
        if self.fn(lb, bb, n_routed, n_shared, top_k, route_scale, global_scale, ids, routed, shared):
            raise RouterWeightLatticeError("checked-in C router failed")
        return {
            "indices": torch.tensor([int(ids[i]) for i in range(top_k)], dtype=torch.int64),
            "routed": torch.tensor([float(routed[i]) for i in range(top_k)]),
            "shared": torch.tensor([float(shared[i]) for i in range(n_shared)]),
        }


def force_bias_for_order(n_routed: int, ids: torch.Tensor) -> torch.Tensor:
    bias = torch.full((n_routed,), -1000.0, dtype=torch.float32)
    for position, expert in enumerate(ids.tolist()):
        bias[int(expert)] = 1000.0 - float(position)
    return bias


def selected_logits(logits: torch.Tensor, ids: torch.Tensor, n_routed: int) -> torch.Tensor:
    return torch.cat((logits[:n_routed].index_select(0, ids.long()), logits[n_routed:]), 0).contiguous()


def official_manual_weights(selected: torch.Tensor, route_scale: float, global_scale: torch.Tensor) -> dict[str, torch.Tensor]:
    selected = selected.detach().to(torch.bfloat16)
    logp = F.logsigmoid(selected)
    lse = torch.logsumexp(logp, dim=-1, keepdim=True)
    normalized = torch.exp(logp - lse)
    after_route = normalized * route_scale
    final = after_route * global_scale.detach().to(torch.bfloat16)
    return {
        "selected_logits": selected.cpu(),
        "logsigmoid": logp.cpu(),
        "logsumexp": lse.cpu(),
        "normalized": normalized.cpu(),
        "after_route_scale": after_route.cpu(),
        "final": final.cpu(),
    }


def split_metrics(official: torch.Tensor, candidate: torch.Tensor, top_k: int) -> dict[str, Any]:
    return {
        "all": metrics(official, candidate),
        "routed": metrics(official[:top_k], candidate[:top_k]),
        "shared": metrics(official[top_k:], candidate[top_k:]),
    }


def enumerate_policies(official: torch.Tensor, helper: FixedWeightHelper, selected: torch.Tensor, route_scale: float, global_scale: float, limit: int = 12) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    exact: list[dict[str, Any]] = []
    for family in (0, 1):
        for flags in range(1 << FLAG_COUNT):
            candidate = helper.evaluate(selected, route_scale, global_scale, family, flags)
            record = {
                **policy_description(family, flags),
                "metrics": metrics(official, candidate),
                "semantic_penalty": list(semantic_penalty(family, flags)),
            }
            records.append(record)
            if record["metrics"]["raw_exact_fraction"] == 1.0:
                exact.append(record)
    records.sort(key=lambda r: (-r["metrics"]["raw_exact_fraction"], -r["metrics"]["bfloat16_exact_fraction"], r["metrics"]["raw_max_abs"], tuple(r["semantic_penalty"])))
    exact.sort(key=lambda r: (tuple(r["semantic_penalty"]), len(r["boundaries"]), r["key"]))
    return records[:limit], exact


def classify(anchor: dict[str, Any], fixed_current: dict[str, Any], exact: list[dict[str, Any]]) -> dict[str, Any]:
    if anchor["raw_exact_fraction"] != 1.0:
        return {"classification": "router_weight_anchor_failed", "preferred_policy": None}
    if fixed_current["all"]["raw_exact_fraction"] == 1.0:
        return {"classification": "router_weight_mismatch_is_route_choice_only", "preferred_policy": None, "exact_policy_count": len(exact)}
    if exact:
        preferred = {k: exact[0][k] for k in ("key", "family", "flags", "boundaries")}
        return {"classification": "portable_router_weight_policy_identified", "preferred_policy": preferred, "exact_policy_count": len(exact)}
    return {"classification": "native_or_unmodeled_router_reduction_required", "preferred_policy": None, "exact_policy_count": 0}


def analyze_layer(helper: FixedWeightHelper, c_router: ActualCRouter, fixture: Any, config: Any, layer: int, inputs: torch.Tensor, route: RouteRow, *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    pre, _, _, _ = _capture_official_pre_router(fixture, config, layer, inputs, device=device, dtype=dtype)
    module = build_layer_from_fixture(fixture, config, layer, device=device, dtype=dtype)
    normalized = pre["post_attention_norm"][0].to(device=device, dtype=torch.bfloat16).reshape(1, -1)
    with torch.no_grad():
        logits = F.linear(normalized, module.mlp.gate.weight)[0]
    gate = CanonicalOfficialGate(module.mlp.gate, [route])
    _, routed_weights, routed_ids, shared_gammas = gate(normalized)
    if not gate.applied:
        raise RouterWeightLatticeError("canonical official gate not applied")

    n_routed = int(module.mlp.gate.num_experts)
    n_shared = int(module.mlp.gate.n_shared_experts)
    top_k = int(module.mlp.gate.top_k)
    route_scale = float(module.mlp.gate.route_scale)
    global_tensor = module.mlp.gate.global_scale.detach()
    global_scale = float(global_tensor.float().cpu().reshape(-1)[0])
    real_bias = module.mlp.gate.e_score_correction_bias.detach().cpu()
    canonical_ids = torch.tensor(route.indices, dtype=torch.int64)
    official = torch.cat((routed_weights[0].detach().cpu(), shared_gammas[0].detach().cpu()))
    canonical_selected = selected_logits(logits.detach().cpu(), canonical_ids, n_routed)
    manual = official_manual_weights(canonical_selected, route_scale, global_tensor.cpu())
    anchor = metrics(official, manual["final"])

    actual = c_router.evaluate(logits.cpu(), real_bias, n_routed, n_shared, top_k, route_scale, global_scale)
    forced = c_router.evaluate(logits.cpu(), force_bias_for_order(n_routed, canonical_ids), n_routed, n_shared, top_k, route_scale, global_scale)
    if not torch.equal(forced["indices"], canonical_ids):
        raise RouterWeightLatticeError(f"forced C route {forced['indices'].tolist()} != canonical {canonical_ids.tolist()}")
    actual_all = torch.cat((actual["routed"], actual["shared"]))
    forced_all = torch.cat((forced["routed"], forced["shared"]))
    fixed_current = split_metrics(official, forced_all, top_k)
    top, exact = enumerate_policies(official, helper, canonical_selected, route_scale, global_scale)
    decision = classify(anchor, fixed_current, exact)
    preferred_metrics = None
    if decision["preferred_policy"]:
        p = decision["preferred_policy"]
        candidate = helper.evaluate(canonical_selected, route_scale, global_scale, 0 if p["family"] == "ratio" else 1, int(p["flags"]))
        preferred_metrics = split_metrics(official, candidate, top_k)

    return {
        "decision": decision,
        "official_manual_anchor": anchor,
        "dtypes": {
            "router_logits": str(logits.dtype).removeprefix("torch."),
            "correction_bias": str(module.mlp.gate.e_score_correction_bias.dtype).removeprefix("torch."),
            "global_scale": str(global_tensor.dtype).removeprefix("torch."),
            "routed_weights": str(routed_weights.dtype).removeprefix("torch."),
            "shared_gammas": str(shared_gammas.dtype).removeprefix("torch."),
        },
        "route_scale": route_scale,
        "global_scale": global_scale,
        "canonical_indices": canonical_ids.tolist(),
        "actual_c_indices": actual["indices"].tolist(),
        "actual_c_vs_official": split_metrics(official, actual_all, top_k),
        "fixed_canonical_current_c": fixed_current,
        "preferred_policy_metrics": preferred_metrics,
        "exact_policy_count": len(exact),
        "exact_policies": exact[:24],
        "top_policies": top,
        "official_stages": {k: v.detach().float().reshape(-1).tolist() for k, v in manual.items()},
        "official_routed_weights": routed_weights[0].detach().float().cpu().tolist(),
        "official_shared_gammas": shared_gammas[0].detach().float().cpu().tolist(),
        "actual_c_routed_weights": actual["routed"].tolist(),
        "actual_c_shared_weights": actual["shared"].tolist(),
        "fixed_canonical_routed_weights": forced["routed"].tolist(),
        "fixed_canonical_shared_weights": forced["shared"].tolist(),
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
            raise RouterWeightLatticeError("--inputs must contain eight source-bound rows")
        inputs = torch.tensor(values, dtype=torch.float32)
        layers = [int(v) for v in args.layers.split(",") if v]
        if not layers:
            raise RouterWeightLatticeError("--layers must be nonempty")
        routes = load_canonical_routes(args.selection, fixture, values, layers)
        device = torch.device(args.device)
        if device.type != "cpu":
            raise RouterWeightLatticeError("router-weight lattice requires CPU")
        dtype = DTYPES[args.dtype]
        helper = FixedWeightHelper(build_helper(Path(args.out).with_suffix(".weights.so")))
        source_root = Path(__file__).resolve().parents[2] / "inkling" / "src"
        c_router = ActualCRouter(build_library(source_root, Path(args.out).with_suffix(".runtime.so")))
        analyses = {
            str(layer): analyze_layer(helper, c_router, fixture, config, layer, inputs, routes[layer][0], device=device, dtype=dtype)
            for layer in layers
        }
        result = {
            "format": "inkling-bfloat16-router-weight-lattice",
            "version": 2,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "source_tokens": len(values),
            "executed_position": 0,
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "production_source_modified": False,
            "layers": analyses,
        }
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (FixtureError, FixtureReferenceError, PreRouterDiagnosisError, CanonicalRouteError, CanonicalOfficialRouteError, RouterWeightLatticeError, OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc)); return 2
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
