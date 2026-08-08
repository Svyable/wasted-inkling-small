#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the mixed-precision denominator policy implied by #50.

#50 proves that upstream BF16 ``logsumexp`` uses BF16 stabilized deltas and BF16
exp terms, while a float32 reduction over those exact terms followed by a single
BF16 denominator completion matches Torch at all eight source-bound positions.
The prior policy lattice had no boundary for that operation.

This evidence-only probe fixes that newly observed reduction semantics, then
enumerates only the downstream completion points (log, final add, normalized
exp, route scale, and global scale) and requires complete routed/shared weight
parity across all eight positions. Experts are never executed. Production source
and public APIs remain unchanged.
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

from diagnose_inkling_bf16_router_weight_lattice import metrics, selected_logits
from diagnose_inkling_pre_router import PreRouterDiagnosisError, _capture_official_pre_router
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import CanonicalRouteError, load_canonical_routes
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import DTYPES, FixtureReferenceError, build_layer_from_fixture, verify_config_binding
from inkling_release_config import build_transformers_text_config
from run_inkling_fixture_reference_canonical import CanonicalOfficialGate, CanonicalOfficialRouteError


class PostReductionDenomError(RuntimeError):
    """The post-reduction denominator policy cannot be verified safely."""


F_LOGDENOM_BF16 = 1 << 0
F_LSE_BF16 = 1 << 1
F_OUTPUT_DELTA_BF16 = 1 << 2
F_NORMALIZED_BF16 = 1 << 3
F_ROUTE_SCALE_BF16 = 1 << 4
F_GLOBAL_SCALE_BF16 = 1 << 5
F_FINAL_BF16 = 1 << 6
FLAG_COUNT = 7
EXPECTED_FLAGS = (1 << FLAG_COUNT) - 1
STAGES = (
    "denom",
    "logdenom",
    "lse",
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
#define F_LOGDENOM_BF16 (1u << 0)
#define F_LSE_BF16 (1u << 1)
#define F_OUTPUT_DELTA_BF16 (1u << 2)
#define F_NORMALIZED_BF16 (1u << 3)
#define F_ROUTE_SCALE_BF16 (1u << 4)
#define F_GLOBAL_SCALE_BF16 (1u << 5)
#define F_FINAL_BF16 (1u << 6)

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

int inkling_probe_post_reduction_denom(
    const float *selected_logits, size_t n,
    float route_scale, float global_scale, unsigned flags,
    float *out_denom, float *out_logdenom, float *out_lse,
    float *out_normalized, float *out_route, float *out_final) {
    if (!selected_logits || !out_denom || !out_logdenom || !out_lse ||
        !out_normalized || !out_route || !out_final || n == 0 || n > 16)
        return -1;

    float logp[16], max_logp = -FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        const float v = bf16_round_probe(logsigmoid_probe(selected_logits[i]));
        logp[i] = v;
        if (v > max_logp) max_logp = v;
    }

    /* #50: BF16 delta, BF16 exp term, F32 accumulation, then one BF16
     * denominator completion after the complete reduction. */
    float denom_f32 = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float delta = bf16_round_probe(logp[i] - max_logp);
        const float term = bf16_round_probe(expf(delta));
        denom_f32 += term;
    }
    const float denom = bf16_round_probe(denom_f32);
    *out_denom = denom;

    float logdenom = logf(denom);
    if (flags & F_LOGDENOM_BF16)
        logdenom = bf16_round_probe(logdenom);
    *out_logdenom = logdenom;

    float lse = max_logp + logdenom;
    if (flags & F_LSE_BF16)
        lse = bf16_round_probe(lse);
    *out_lse = lse;

    for (size_t i = 0; i < n; i++) {
        float delta = logp[i] - lse;
        if (flags & F_OUTPUT_DELTA_BF16)
            delta = bf16_round_probe(delta);
        float normalized = expf(delta);
        if (flags & F_NORMALIZED_BF16)
            normalized = bf16_round_probe(normalized);
        out_normalized[i] = normalized;

        float route = normalized * route_scale;
        if (flags & F_ROUTE_SCALE_BF16)
            route = bf16_round_probe(route);
        out_route[i] = route;

        float final = route * global_scale;
        if (flags & F_GLOBAL_SCALE_BF16)
            final = bf16_round_probe(final);
        if (flags & F_FINAL_BF16)
            final = bf16_round_probe(final);
        out_final[i] = final;
    }
    return 0;
}
'''


def build_helper(out: Path) -> Path:
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        raise PostReductionDenomError("no C compiler available")
    root = Path(tempfile.mkdtemp(prefix="inkling-post-reduction-denom-"))
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
        raise PostReductionDenomError(
            f"post-reduction helper build failed:\n{result.stderr}"
        )
    return out


class PostReductionHelper:
    def __init__(self, library: Path) -> None:
        self.lib = ctypes.CDLL(str(library))
        self.fn = self.lib.inkling_probe_post_reduction_denom
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
        selected: torch.Tensor,
        route_scale: float,
        global_scale: float,
        flags: int,
    ) -> dict[str, torch.Tensor]:
        values = selected.detach().to(torch.bfloat16).float().cpu().contiguous().reshape(-1)
        n = int(values.numel())
        if not 0 < n <= 16:
            raise PostReductionDenomError(f"selected width {n} outside helper bounds")
        ib = (ctypes.c_float * n)(*values.tolist())
        denom = (ctypes.c_float * 1)()
        logdenom = (ctypes.c_float * 1)()
        lse = (ctypes.c_float * 1)()
        normalized = (ctypes.c_float * n)()
        route = (ctypes.c_float * n)()
        final = (ctypes.c_float * n)()
        if self.fn(
            ib,
            n,
            route_scale,
            global_scale,
            flags,
            denom,
            logdenom,
            lse,
            normalized,
            route,
            final,
        ):
            raise PostReductionDenomError("post-reduction helper failed")
        vector = lambda buf: torch.tensor(
            [float(buf[i]) for i in range(n)], dtype=torch.float32
        )
        scalar = lambda buf: torch.tensor([float(buf[0])], dtype=torch.float32)
        return {
            "denom": scalar(denom),
            "logdenom": scalar(logdenom),
            "lse": scalar(lse),
            "normalized": vector(normalized),
            "after_route_scale": vector(route),
            "final": vector(final),
        }


def policy_description(flags: int) -> dict[str, Any]:
    names = []
    for bit, name in (
        (F_LOGDENOM_BF16, "logdenom_bf16"),
        (F_LSE_BF16, "lse_bf16"),
        (F_OUTPUT_DELTA_BF16, "output_delta_bf16"),
        (F_NORMALIZED_BF16, "normalized_bf16"),
        (F_ROUTE_SCALE_BF16, "route_scale_bf16"),
        (F_GLOBAL_SCALE_BF16, "global_scale_bf16"),
        (F_FINAL_BF16, "final_bf16"),
    ):
        if flags & bit:
            names.append(name)
    return {
        "flags": flags,
        "key": f"post-reduction:{flags:02x}",
        "boundaries": names,
    }


def torch_stages(
    selected: torch.Tensor,
    route_scale: float,
    global_scale_tensor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    selected = selected.detach().to(torch.bfloat16)
    logp = F.logsigmoid(selected)
    maximum = torch.max(logp).reshape(1)
    delta = logp - maximum
    terms = torch.exp(delta)
    denom = torch.sum(terms).reshape(1)
    logdenom = torch.log(denom)
    lse = maximum + logdenom
    normalized = torch.exp(logp - lse)
    after_route = normalized * route_scale
    final = after_route * global_scale_tensor.detach().to(torch.bfloat16)
    return {
        "denom": denom.detach().float().cpu(),
        "logdenom": logdenom.detach().float().cpu(),
        "lse": lse.detach().float().cpu(),
        "normalized": normalized.detach().float().cpu(),
        "after_route_scale": after_route.detach().float().cpu(),
        "final": final.detach().float().cpu(),
    }


def evaluate_policies(
    helper: PostReductionHelper,
    selected: torch.Tensor,
    official_stages: dict[str, torch.Tensor],
    official_weights: torch.Tensor,
    route_scale: float,
    global_scale: float,
) -> dict[str, Any]:
    exact: list[int] = []
    records: list[dict[str, Any]] = []
    denom_anchor = None
    for flags in range(1 << FLAG_COUNT):
        candidate = helper.evaluate(
            selected, route_scale, global_scale, flags
        )
        stage_metrics = {
            stage: metrics(official_stages[stage], candidate[stage])
            for stage in STAGES
        }
        weight_metrics = metrics(official_weights, candidate["final"])
        if denom_anchor is None:
            denom_anchor = stage_metrics["denom"]
        if stage_metrics["denom"]["raw_exact_fraction"] != 1.0:
            raise PostReductionDenomError(
                "new post-reduction denominator is not raw-exact"
            )
        all_exact = (
            all(
                stage_metrics[stage]["raw_exact_fraction"] == 1.0
                for stage in STAGES
            )
            and weight_metrics["raw_exact_fraction"] == 1.0
        )
        if all_exact:
            exact.append(flags)
        records.append(
            {
                "policy": policy_description(flags),
                "stages": stage_metrics,
                "official_weights": weight_metrics,
                "all_semantic_stages_raw_exact": all_exact,
            }
        )
    records.sort(
        key=lambda record: (
            0 if record["all_semantic_stages_raw_exact"] else 1,
            bin(record["policy"]["flags"]).count("1"),
            record["policy"]["flags"],
        )
    )
    return {
        "denom_anchor": denom_anchor,
        "exact_flags": exact,
        "exact_policy_count": len(exact),
        "expected_policy_exact": EXPECTED_FLAGS in exact,
        "top_policies": records[:8],
    }


def classify(
    positions: list[dict[str, Any]],
    intersection: list[int],
) -> dict[str, Any]:
    first_denom_failure = next(
        (
            item["position"]
            for item in positions
            if item["denom_anchor"]["raw_exact_fraction"] != 1.0
        ),
        None,
    )
    if first_denom_failure is not None:
        return {
            "classification": "post_reduction_denominator_policy_failed",
            "first_failure_position": first_denom_failure,
            "preferred_policy": None,
        }
    if EXPECTED_FLAGS in intersection:
        return {
            "classification": "expected_post_reduction_policy_is_raw_exact",
            "first_failure_position": None,
            "preferred_policy": policy_description(EXPECTED_FLAGS),
        }
    if intersection:
        return {
            "classification": "alternate_post_reduction_policy_is_raw_exact",
            "first_failure_position": None,
            "preferred_policy": policy_description(intersection[0]),
        }
    return {
        "classification": "post_reduction_pipeline_still_unexplained",
        "first_failure_position": next(
            (
                item["position"]
                for item in positions
                if item["exact_policy_count"] == 0
            ),
            None,
        ),
        "preferred_policy": None,
    }


def analyze_layer(
    helper: PostReductionHelper,
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
        raise PostReductionDenomError("canonical official gate was not applied")

    n_routed = int(module.mlp.gate.num_experts)
    route_scale = float(module.mlp.gate.route_scale)
    global_tensor = module.mlp.gate.global_scale.detach().cpu()
    global_scale = float(global_tensor.float().reshape(-1)[0])

    positions: list[dict[str, Any]] = []
    exact_sets: list[set[int]] = []
    for position, row in enumerate(rows):
        if routed_ids[position].detach().cpu().tolist() != list(row.indices):
            raise PostReductionDenomError(
                f"layer {layer} position {position} canonical routed IDs changed"
            )
        ids = torch.tensor(row.indices, dtype=torch.int64)
        selected = selected_logits(logits[position], ids, n_routed)
        official = torch.cat(
            (
                routed_weights[position].detach().cpu(),
                shared_gammas[position].detach().cpu(),
            )
        ).float()
        stage_targets = torch_stages(selected, route_scale, global_tensor)
        final_anchor = metrics(official, stage_targets["final"])
        if final_anchor["raw_exact_fraction"] != 1.0:
            raise PostReductionDenomError(
                f"layer {layer} position {position} Torch stage anchor failed"
            )
        result = evaluate_policies(
            helper,
            selected,
            stage_targets,
            official,
            route_scale,
            global_scale,
        )
        exact_sets.append(set(result["exact_flags"]))
        positions.append(
            {
                "position": position,
                "canonical_indices": list(row.indices),
                "torch_final_anchor": final_anchor,
                **result,
            }
        )

    intersection = sorted(set.intersection(*exact_sets)) if exact_sets else []
    return {
        "decision": classify(positions, intersection),
        "positions": positions,
        "all_position_exact_flags": intersection,
        "all_position_exact_policies": [
            policy_description(flags) for flags in intersection
        ],
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
            raise PostReductionDenomError(
                "--inputs must contain eight source-bound deterministic rows"
            )
        inputs = torch.tensor(values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise PostReductionDenomError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection, fixture, values, layers
        )
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise PostReductionDenomError(
                "post-reduction denominator probe requires CPU"
            )
        helper = PostReductionHelper(
            build_helper(Path(args.out).with_suffix(".post-reduction.so"))
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
            "format": "inkling-bfloat16-router-post-reduction-denom-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "source_tokens": len(values),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "expected_flags": EXPECTED_FLAGS,
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
        PostReductionDenomError,
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
