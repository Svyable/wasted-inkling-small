#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Classify portable BF16 SiLU and gate-product completion boundaries.

The single-token sparse-MoE probe proves that routed/shared gate and up
projections are exact and that the first mismatch is ``SiLU(gate) * up``.  This
primitive probe removes every later MoE operation.  It obtains source-bound
BF16 gate/up tensors for the first routed expert and first shared expert, then
compares the pinned official BF16 result against standalone C11/``expf``
variants with explicit BF16 completion points.

No production source or public ABI is compiled or modified.
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
    load_canonical_routes,
)
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    build_layer_from_fixture,
    verify_config_binding,
)
from inkling_release_config import build_transformers_text_config


class SiluProductError(RuntimeError):
    """The BF16 SiLU/product boundary cannot be classified safely."""


VARIANTS = (
    "silu_raw",
    "silu_bfloat16",
    "product_raw",
    "product_result_bfloat16",
    "product_silu_bfloat16_result_bfloat16",
)

_C_SOURCE = r"""
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static float bf16_round_probe(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float silu_probe(float x)
{
    return x / (1.0f + expf(-x));
}

void inkling_probe_silu_raw(const float *gate, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = silu_probe(gate[i]);
}

void inkling_probe_silu_bfloat16(const float *gate, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = bf16_round_probe(silu_probe(gate[i]));
}

void inkling_probe_product_raw(const float *gate, const float *up,
                               float *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = silu_probe(gate[i]) * up[i];
}

void inkling_probe_product_result_bfloat16(const float *gate, const float *up,
                                           float *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = bf16_round_probe(silu_probe(gate[i]) * up[i]);
}

void inkling_probe_product_silu_bfloat16_result_bfloat16(
    const float *gate, const float *up, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        const float activated = bf16_round_probe(silu_probe(gate[i]));
        out[i] = bf16_round_probe(activated * bf16_round_probe(up[i]));
    }
}
"""


def build_helper(out: Path) -> Path:
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        raise SiluProductError("no C compiler available")
    root = Path(tempfile.mkdtemp(prefix="inkling-bf16-silu-product-"))
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
        raise SiluProductError(f"C helper build failed:\n{result.stderr}")
    return out


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu().contiguous()
    candidate = candidate.detach().float().cpu().contiguous()
    if tuple(reference.shape) != tuple(candidate.shape):
        raise SiluProductError(
            f"shape mismatch {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
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


def classify(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if variants["silu_bfloat16"]["bfloat16_exact_fraction"] != 1.0:
        return {
            "classification": "portable_expf_does_not_reproduce_native_bfloat16_silu",
            "sufficient_policy": None,
        }
    if (
        variants["product_silu_bfloat16_result_bfloat16"][
            "bfloat16_exact_fraction"
        ]
        == 1.0
    ):
        return {
            "classification": "bfloat16_complete_silu_before_multiply_is_sufficient",
            "sufficient_policy": "product_silu_bfloat16_result_bfloat16",
        }
    if variants["product_result_bfloat16"]["bfloat16_exact_fraction"] == 1.0:
        return {
            "classification": "bfloat16_complete_product_is_sufficient",
            "sufficient_policy": "product_result_bfloat16",
        }
    return {
        "classification": "bfloat16_multiply_boundary_remains_unexplained",
        "sufficient_policy": None,
    }


class CVariants:
    def __init__(self, library: Path) -> None:
        self.lib = ctypes.CDLL(str(library))
        fp = ctypes.POINTER(ctypes.c_float)
        for name in ("silu_raw", "silu_bfloat16"):
            fn = getattr(self.lib, f"inkling_probe_{name}")
            fn.argtypes = [fp, fp, ctypes.c_size_t]
            fn.restype = None
        for name in (
            "product_raw",
            "product_result_bfloat16",
            "product_silu_bfloat16_result_bfloat16",
        ):
            fn = getattr(self.lib, f"inkling_probe_{name}")
            fn.argtypes = [fp, fp, fp, ctypes.c_size_t]
            fn.restype = None

    @staticmethod
    def _array(value: torch.Tensor) -> ctypes.Array:
        flat = value.detach().float().cpu().contiguous().reshape(-1).tolist()
        return (ctypes.c_float * len(flat))(*flat)

    @staticmethod
    def _tensor(value: ctypes.Array, n: int) -> torch.Tensor:
        return torch.tensor([float(value[i]) for i in range(n)])

    def evaluate(
        self, gate: torch.Tensor, up: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        gate = gate.detach().float().cpu().contiguous().reshape(-1)
        up = up.detach().float().cpu().contiguous().reshape(-1)
        if gate.shape != up.shape or not gate.numel():
            raise SiluProductError("gate/up vectors must be nonempty and aligned")
        n = int(gate.numel())
        gate_buf = self._array(gate)
        up_buf = self._array(up)
        results: dict[str, torch.Tensor] = {}
        for name in VARIANTS:
            out = (ctypes.c_float * n)()
            fn = getattr(self.lib, f"inkling_probe_{name}")
            if name.startswith("silu_"):
                fn(gate_buf, out, n)
            else:
                fn(gate_buf, up_buf, out, n)
            results[name] = self._tensor(out, n)
        return results


def official_gate_up(
    module: Any,
    normalized: torch.Tensor,
    expert: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    normalized = normalized.detach().to(torch.bfloat16).reshape(1, -1)
    fused = module.mlp.experts.gate_up[str(expert)]
    routed_gate, routed_up = F.linear(normalized, fused).chunk(2, dim=-1)
    shared_gate = F.linear(
        normalized, module.mlp.shared_experts.gate_proj[0]
    )
    shared_up = F.linear(
        normalized, module.mlp.shared_experts.up_proj[0]
    )
    return {
        "routed": (routed_gate[0], routed_up[0]),
        "shared": (shared_gate[0], shared_up[0]),
    }


def analyze_path(
    helper: CVariants,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> dict[str, Any]:
    gate_bf16 = gate.detach().to(torch.bfloat16)
    up_bf16 = up.detach().to(torch.bfloat16)
    official_silu = F.silu(gate_bf16).float()
    official_product = (F.silu(gate_bf16) * up_bf16).float()
    candidates = helper.evaluate(gate_bf16.float(), up_bf16.float())
    comparisons = {
        "silu_raw": metrics(official_silu, candidates["silu_raw"]),
        "silu_bfloat16": metrics(
            official_silu, candidates["silu_bfloat16"]
        ),
        "product_raw": metrics(
            official_product, candidates["product_raw"]
        ),
        "product_result_bfloat16": metrics(
            official_product, candidates["product_result_bfloat16"]
        ),
        "product_silu_bfloat16_result_bfloat16": metrics(
            official_product,
            candidates["product_silu_bfloat16_result_bfloat16"],
        ),
    }
    return {
        "decision": classify(comparisons),
        "variants": comparisons,
        "gate_dtype": str(gate_bf16.dtype).removeprefix("torch."),
        "up_dtype": str(up_bf16.dtype).removeprefix("torch."),
        "official_silu_dtype": str(
            F.silu(gate_bf16).dtype
        ).removeprefix("torch."),
        "official_product_dtype": str(
            (F.silu(gate_bf16) * up_bf16).dtype
        ).removeprefix("torch."),
    }


def analyze_layer(
    helper: CVariants,
    fixture: Any,
    config: Any,
    layer: int,
    inputs: torch.Tensor,
    expert: int,
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
    values = official_gate_up(
        module, pre_router["post_attention_norm"][0], expert
    )
    return {
        "expert": int(expert),
        "routed": analyze_path(helper, *values["routed"]),
        "shared": analyze_path(helper, *values["shared"]),
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
            raise SiluProductError(
                "--inputs must contain the eight source-bound deterministic rows"
            )
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise SiluProductError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection, fixture, input_values, layers
        )
        experts = {layer: int(routes[layer][0].indices[0]) for layer in layers}
        for layer, expert in experts.items():
            declared = list(fixture.experts.get(layer, ()))
            if declared != [expert]:
                raise SiluProductError(
                    f"layer {layer} fixture experts {declared} != first route expert {expert}"
                )
        device = torch.device(args.device)
        if device.type != "cpu":
            raise SiluProductError("the C primitive probe requires CPU")
        dtype = DTYPES[args.dtype]
        helper_path = build_helper(Path(args.out).with_suffix(".so"))
        helper = CVariants(helper_path)
        analyses = {
            str(layer): analyze_layer(
                helper, fixture, config, layer, inputs, experts[layer],
                device=device, dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-bfloat16-silu-product-probe",
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
        SiluProductError,
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
