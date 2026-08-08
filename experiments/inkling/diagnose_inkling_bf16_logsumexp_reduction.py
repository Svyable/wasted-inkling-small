#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dissect the position-3 BF16 ``torch.logsumexp`` boundary.

#49 proves that layer-2 position 3 is exact through BF16 ``logsigmoid`` and
then loses every modeled scalar policy at ``logsumexp``. This evidence-only
probe starts from that already-exact BF16 ``logp`` vector and separates:

* the public Torch stabilized decomposition from the native ``torch.logsumexp``;
* scalar max/subtract arithmetic;
* the BF16 ``exp`` primitive;
* denominator reduction/association;
* the scalar logarithm;
* the final max-plus-log add.

Expert values are never executed. Production source and public APIs remain
unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from diagnose_inkling_bf16_router_weight_lattice import (
    metrics,
    selected_logits,
)
from diagnose_inkling_pre_router import PreRouterDiagnosisError, _capture_official_pre_router
from discover_inkling_router_experts import input_sha256
from inkling_canonical_route_layer_parity import CanonicalRouteError, load_canonical_routes
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import DTYPES, FixtureReferenceError, build_layer_from_fixture, verify_config_binding
from inkling_release_config import build_transformers_text_config
from run_inkling_fixture_reference_canonical import CanonicalOfficialGate, CanonicalOfficialRouteError


class LogsumexpReductionError(RuntimeError):
    """The logsumexp reduction cannot be classified without exact anchors."""


_C_SOURCE = r'''
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static float bf16_round_probe(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

int inkling_probe_logsumexp_primitives(
    const float *logp, size_t n,
    float *out_max, float *out_delta, float *out_terms,
    float *out_left_bf16, float *out_right_bf16,
    float *out_left_f32_bf16, float *out_balanced_bf16,
    float *out_log_left_bf16, float *out_lse_left_bf16) {
    if (!logp || !out_max || !out_delta || !out_terms ||
        !out_left_bf16 || !out_right_bf16 || !out_left_f32_bf16 ||
        !out_balanced_bf16 || !out_log_left_bf16 || !out_lse_left_bf16 ||
        n == 0 || n > 16)
        return -1;

    float m = -FLT_MAX;
    for (size_t i = 0; i < n; i++) if (logp[i] > m) m = logp[i];
    *out_max = m;

    for (size_t i = 0; i < n; i++) {
        const float d = bf16_round_probe(logp[i] - m);
        out_delta[i] = d;
        out_terms[i] = bf16_round_probe(expf(d));
    }

    float left_bf16 = bf16_round_probe(0.0f);
    float left_f32 = 0.0f;
    for (size_t i = 0; i < n; i++) {
        left_bf16 = bf16_round_probe(left_bf16 + out_terms[i]);
        left_f32 += out_terms[i];
    }
    *out_left_bf16 = left_bf16;
    *out_left_f32_bf16 = bf16_round_probe(left_f32);

    float right_bf16 = bf16_round_probe(0.0f);
    for (size_t i = n; i-- > 0;)
        right_bf16 = bf16_round_probe(right_bf16 + out_terms[i]);
    *out_right_bf16 = right_bf16;

    float work[16];
    for (size_t i = 0; i < n; i++) work[i] = out_terms[i];
    size_t width = n;
    while (width > 1) {
        size_t next = 0;
        size_t i = 0;
        for (; i + 1 < width; i += 2)
            work[next++] = bf16_round_probe(work[i] + work[i + 1]);
        if (i < width) work[next++] = work[i];
        width = next;
    }
    *out_balanced_bf16 = work[0];

    const float ld = bf16_round_probe(logf(left_bf16));
    *out_log_left_bf16 = ld;
    *out_lse_left_bf16 = bf16_round_probe(m + ld);
    return 0;
}
'''


def build_helper(out: Path) -> Path:
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        raise LogsumexpReductionError("no C compiler available")
    root = Path(tempfile.mkdtemp(prefix="inkling-logsumexp-reduction-"))
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
        raise LogsumexpReductionError(
            f"logsumexp helper build failed:\n{result.stderr}"
        )
    return out


class PrimitiveHelper:
    def __init__(self, library: Path) -> None:
        self.lib = ctypes.CDLL(str(library))
        self.fn = self.lib.inkling_probe_logsumexp_primitives
        fp = ctypes.POINTER(ctypes.c_float)
        self.fn.argtypes = [
            fp,
            ctypes.c_size_t,
            fp,
            fp,
            fp,
            fp,
            fp,
            fp,
            fp,
            fp,
            fp,
        ]
        self.fn.restype = ctypes.c_int

    def evaluate(self, logp: torch.Tensor) -> dict[str, torch.Tensor]:
        values = logp.detach().to(torch.bfloat16).float().cpu().contiguous().reshape(-1)
        n = int(values.numel())
        if not 0 < n <= 16:
            raise LogsumexpReductionError(f"logp width {n} outside helper bounds")
        ib = (ctypes.c_float * n)(*values.tolist())
        maximum = (ctypes.c_float * 1)()
        delta = (ctypes.c_float * n)()
        terms = (ctypes.c_float * n)()
        left = (ctypes.c_float * 1)()
        right = (ctypes.c_float * 1)()
        left_f32 = (ctypes.c_float * 1)()
        balanced = (ctypes.c_float * 1)()
        log_left = (ctypes.c_float * 1)()
        lse_left = (ctypes.c_float * 1)()
        if self.fn(
            ib,
            n,
            maximum,
            delta,
            terms,
            left,
            right,
            left_f32,
            balanced,
            log_left,
            lse_left,
        ):
            raise LogsumexpReductionError("logsumexp primitive helper failed")
        vector = lambda buf: torch.tensor(
            [float(buf[i]) for i in range(n)], dtype=torch.float32
        )
        scalar = lambda buf: torch.tensor([float(buf[0])], dtype=torch.float32)
        return {
            "max": scalar(maximum),
            "delta": vector(delta),
            "terms": vector(terms),
            "sum_left_bf16": scalar(left),
            "sum_right_bf16": scalar(right),
            "sum_left_f32_then_bf16": scalar(left_f32),
            "sum_balanced_bf16": scalar(balanced),
            "log_left_bf16": scalar(log_left),
            "lse_left_bf16": scalar(lse_left),
        }


def bf16_round_scalar(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if (bits & 0x7F800000) != 0x7F800000:
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def any_bf16_tree_sum(
    values: torch.Tensor,
    target: float,
) -> dict[str, Any]:
    """Search every binary association/order under BF16 completion per add."""
    items = [bf16_round_scalar(float(v)) for v in values.reshape(-1).tolist()]
    n = len(items)
    if not 0 < n <= 10:
        raise LogsumexpReductionError(f"tree search width {n} is unsupported")
    target = bf16_round_scalar(target)
    # mask -> value -> witness expression. Restrict split enumeration to the
    # least-significant element on the left to remove mirrored duplicates.
    dp: list[dict[float, Any]] = [dict() for _ in range(1 << n)]
    for i, value in enumerate(items):
        dp[1 << i][value] = i
    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue
        anchor = mask & -mask
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if other and (sub & anchor):
                for left, left_witness in dp[sub].items():
                    for right, right_witness in dp[other].items():
                        value = bf16_round_scalar(left + right)
                        dp[mask].setdefault(
                            value,
                            [left_witness, right_witness],
                        )
            sub = (sub - 1) & mask
    full = dp[(1 << n) - 1]
    return {
        "target": target,
        "reachable": target in full,
        "possible_result_count": len(full),
        "witness": full.get(target),
        "possible_results": sorted(full)[:64],
    }


def torch_decompositions(logp: torch.Tensor) -> dict[str, Any]:
    logp = logp.detach().to(torch.bfloat16).cpu().contiguous().reshape(-1)
    native_lse = torch.logsumexp(logp, dim=-1, keepdim=True)

    maximum = torch.max(logp).reshape(1)
    delta = logp - maximum
    terms = torch.exp(delta)
    denom = torch.sum(terms).reshape(1)
    logdenom = torch.log(denom)
    staged_lse = maximum + logdenom

    f = logp.float()
    fmax = torch.max(f).reshape(1)
    fdelta = f - fmax
    fterms = torch.exp(fdelta)
    fdenom = torch.sum(fterms).reshape(1)
    flogdenom = torch.log(fdenom)
    flse = fmax + flogdenom
    fp32_to_bf16 = flse.to(torch.bfloat16)

    unstable = torch.log(torch.sum(torch.exp(logp))).reshape(1)

    def logaddexp_reduce(reverse: bool) -> torch.Tensor:
        values = list(logp)
        if reverse:
            values.reverse()
        current = values[0]
        for value in values[1:]:
            current = torch.logaddexp(current, value)
        return current.reshape(1)

    return {
        "native_lse": native_lse,
        "maximum": maximum,
        "delta": delta,
        "terms": terms,
        "denom": denom,
        "logdenom": logdenom,
        "staged_lse": staged_lse,
        "fp32_max": fmax,
        "fp32_delta": fdelta,
        "fp32_terms": fterms,
        "fp32_denom": fdenom,
        "fp32_logdenom": flogdenom,
        "fp32_lse": flse,
        "fp32_lse_to_bfloat16": fp32_to_bf16,
        "unstable_lse": unstable,
        "logaddexp_left": logaddexp_reduce(False),
        "logaddexp_right": logaddexp_reduce(True),
        "dtypes": {
            "logp": str(logp.dtype).removeprefix("torch."),
            "native_lse": str(native_lse.dtype).removeprefix("torch."),
            "maximum": str(maximum.dtype).removeprefix("torch."),
            "delta": str(delta.dtype).removeprefix("torch."),
            "terms": str(terms.dtype).removeprefix("torch."),
            "denom": str(denom.dtype).removeprefix("torch."),
            "logdenom": str(logdenom.dtype).removeprefix("torch."),
            "staged_lse": str(staged_lse.dtype).removeprefix("torch."),
        },
    }


def classify_position(
    torch_values: dict[str, Any],
    scalar: dict[str, torch.Tensor],
    comparisons: dict[str, dict[str, Any]],
    tree_search: dict[str, Any],
) -> dict[str, Any]:
    if comparisons["torch_staged_lse"]["raw_exact_fraction"] != 1.0:
        return {
            "classification": "torch_logsumexp_internal_kernel_boundary",
            "first_boundary": "torch_native_vs_public_decomposition",
        }
    if comparisons["scalar_max"]["raw_exact_fraction"] != 1.0:
        return {
            "classification": "logsumexp_max_mismatch",
            "first_boundary": "max",
        }
    if comparisons["scalar_delta"]["raw_exact_fraction"] != 1.0:
        return {
            "classification": "logsumexp_delta_arithmetic_mismatch",
            "first_boundary": "delta",
        }
    if comparisons["scalar_terms"]["raw_exact_fraction"] != 1.0:
        return {
            "classification": "logsumexp_exp_primitive_mismatch",
            "first_boundary": "exp",
        }
    if comparisons["scalar_left_sum"]["raw_exact_fraction"] != 1.0:
        if tree_search["reachable"]:
            return {
                "classification": "logsumexp_reduction_order_mismatch",
                "first_boundary": "denominator_reduction",
            }
        if comparisons["scalar_f32_sum"]["raw_exact_fraction"] == 1.0:
            return {
                "classification": "logsumexp_reduction_precision_mismatch",
                "first_boundary": "denominator_reduction",
            }
        return {
            "classification": "logsumexp_sum_kernel_unmodeled",
            "first_boundary": "denominator_reduction",
        }
    if comparisons["scalar_logdenom"]["raw_exact_fraction"] != 1.0:
        return {
            "classification": "logsumexp_log_primitive_mismatch",
            "first_boundary": "log",
        }
    if comparisons["scalar_lse"]["raw_exact_fraction"] != 1.0:
        return {
            "classification": "logsumexp_final_add_mismatch",
            "first_boundary": "final_add",
        }
    return {
        "classification": "scalar_logsumexp_decomposition_is_raw_exact",
        "first_boundary": None,
    }


def analyze_position(
    helper: PrimitiveHelper,
    logp: torch.Tensor,
) -> dict[str, Any]:
    torch_values = torch_decompositions(logp)
    scalar = helper.evaluate(logp)
    comparisons = {
        "torch_staged_lse": metrics(
            torch_values["native_lse"], torch_values["staged_lse"]
        ),
        "torch_fp32_lse_to_bfloat16": metrics(
            torch_values["native_lse"], torch_values["fp32_lse_to_bfloat16"]
        ),
        "torch_unstable_lse": metrics(
            torch_values["native_lse"], torch_values["unstable_lse"]
        ),
        "torch_logaddexp_left": metrics(
            torch_values["native_lse"], torch_values["logaddexp_left"]
        ),
        "torch_logaddexp_right": metrics(
            torch_values["native_lse"], torch_values["logaddexp_right"]
        ),
        "scalar_max": metrics(torch_values["maximum"], scalar["max"]),
        "scalar_delta": metrics(torch_values["delta"], scalar["delta"]),
        "scalar_terms": metrics(torch_values["terms"], scalar["terms"]),
        "scalar_left_sum": metrics(
            torch_values["denom"], scalar["sum_left_bf16"]
        ),
        "scalar_right_sum": metrics(
            torch_values["denom"], scalar["sum_right_bf16"]
        ),
        "scalar_f32_sum": metrics(
            torch_values["denom"], scalar["sum_left_f32_then_bf16"]
        ),
        "scalar_balanced_sum": metrics(
            torch_values["denom"], scalar["sum_balanced_bf16"]
        ),
        "scalar_logdenom": metrics(
            torch_values["logdenom"], scalar["log_left_bf16"]
        ),
        "scalar_lse": metrics(
            torch_values["native_lse"], scalar["lse_left_bf16"]
        ),
    }
    tree_search = any_bf16_tree_sum(
        torch_values["terms"].float(),
        float(torch_values["denom"].float().item()),
    )
    return {
        "decision": classify_position(
            torch_values, scalar, comparisons, tree_search
        ),
        "comparisons": comparisons,
        "tree_search": tree_search,
        "torch_dtypes": torch_values["dtypes"],
        "torch_values": {
            key: value.detach().float().reshape(-1).tolist()
            for key, value in torch_values.items()
            if isinstance(value, torch.Tensor)
        },
        "scalar_values": {
            key: value.detach().float().reshape(-1).tolist()
            for key, value in scalar.items()
        },
    }


def analyze_layer(
    helper: PrimitiveHelper,
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
    _, _routed_weights, routed_ids, _shared_gammas = gate(normalized)
    if not gate.applied:
        raise LogsumexpReductionError("canonical official gate was not applied")
    n_routed = int(module.mlp.gate.num_experts)

    positions: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if routed_ids[position].detach().cpu().tolist() != list(row.indices):
            raise LogsumexpReductionError(
                f"layer {layer} position {position} canonical routed IDs changed"
            )
        ids = torch.tensor(row.indices, dtype=torch.int64)
        selected = selected_logits(logits[position], ids, n_routed).to(torch.bfloat16)
        logp = F.logsigmoid(selected)
        # #49's hard semantic anchor: scalar logsigmoid with BF16 completion was
        # already proven exact. Here we start directly from this official value.
        positions.append(
            {
                "position": position,
                "canonical_indices": list(row.indices),
                "selected_logits": selected.detach().float().tolist(),
                "logp": logp.detach().float().tolist(),
                **analyze_position(helper, logp),
            }
        )
    first_position = next(
        (
            item["position"]
            for item in positions
            if item["decision"]["classification"]
            != "scalar_logsumexp_decomposition_is_raw_exact"
        ),
        None,
    )
    return {
        "first_nonexact_position": first_position,
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
            raise LogsumexpReductionError(
                "--inputs must contain eight source-bound deterministic rows"
            )
        inputs = torch.tensor(values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise LogsumexpReductionError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection, fixture, values, layers
        )
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise LogsumexpReductionError("logsumexp reduction probe requires CPU")
        helper = PrimitiveHelper(
            build_helper(Path(args.out).with_suffix(".primitives.so"))
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
            "format": "inkling-bfloat16-logsumexp-reduction-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "source_tokens": len(values),
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
        LogsumexpReductionError,
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
