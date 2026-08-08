#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply the proven BF16 residual rule and close the pre-expert path.

This evidence-only probe composes the previously proven temporary policies:

* official layer RMSNorm BF16 ordering;
* the complete portable BF16 attention policy;
* BF16-quantized attention branch plus BF16 residual addition.

Q/K/V/R/O/router matrices continue to use the pinned native BF16 backend.  The
probe requires source-integrity and reconstruction contracts, compares every
named pre-router activation, and audits route choices against the official BF16
cutoff-tie equivalence class.  Production sources remain unchanged.
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

from audit_inkling_router_cutoff_ties import audit_layer
from diagnose_inkling_native_bf16_backend import (
    NativeBfloat16BackendError,
    NativeMatrixProvider,
    capture_c_with_backend,
    transform_rms_source,
)
from diagnose_inkling_portable_bf16_attention import (
    PortableBfloat16AttentionError,
    transform_attention_source,
)
from diagnose_inkling_pre_router import (
    POINTS,
    PreRouterDiagnosisError,
    _capture_official_pre_router,
    compare_pre_router,
)
from discover_inkling_router_experts import input_sha256
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    verify_config_binding,
)
from inkling_layer_parity import (
    LAYER_SOURCES,
    LayerParityError,
    build_library,
    configure_library,
)
from inkling_release_config import build_transformers_text_config


class PortableBfloat16PreExpertError(RuntimeError):
    """The portable BF16 pre-expert candidate cannot proceed without guessing."""


_RESIDUAL_OLD = "    for (int i = 0; i < hidden; i++) x[i] += s->branch[i];\n"
_RESIDUAL_NEW = """    for (int i = 0; i < hidden; i++) {
        const float residual = bf16_round_probe(x[i]);
        const float branch = bf16_round_probe(s->branch[i]);
        x[i] = bf16_round_probe(residual + branch);
    }
"""


def transform_residual_source(source: str) -> str:
    """Apply the proven BF16 branch/residual boundary exactly once."""
    if source.count("static float bf16_round_probe(float value)") != 1:
        raise PortableBfloat16PreExpertError(
            "the RMS transform must provide exactly one BF16 helper"
        )
    count = source.count(_RESIDUAL_OLD)
    if count != 1:
        raise PortableBfloat16PreExpertError(
            f"expected exactly one post-attention residual anchor; found {count}"
        )
    return source.replace(_RESIDUAL_OLD, _RESIDUAL_NEW, 1)


def build_portable_preexpert_library(out: Path) -> tuple[Path, dict[str, Any]]:
    source_root = Path(__file__).resolve().parents[2] / "inkling" / "src"
    temporary = Path(tempfile.mkdtemp(prefix="inkling-portable-bf16-preexpert-"))
    probe_root = temporary / "src"
    shutil.copytree(source_root, probe_root)

    layer_path = probe_root / "inkling_layer.c"
    attention_path = probe_root / "inkling_attention.c"
    layer_original = layer_path.read_text(encoding="utf-8")
    attention_original = attention_path.read_text(encoding="utf-8")

    layer_transformed = transform_residual_source(
        transform_rms_source(layer_original)
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
        "transforms": [
            "official layer RMSNorm BF16 ordering",
            "complete portable BF16 attention policy",
            "BF16 attention branch before residual addition",
            "BF16 residual input and BF16 completed residual",
        ],
    }


def exact_prefix(comparison: dict[str, Any]) -> list[str]:
    prefix: list[str] = []
    for point in POINTS:
        if comparison["stages"][point]["quantized_exact_fraction"] != 1.0:
            break
        prefix.append(point)
    return prefix


def classify_preexpert(
    comparison: dict[str, Any], route_audit: dict[str, Any]
) -> dict[str, Any]:
    prefix = exact_prefix(comparison)
    if len(prefix) != len(POINTS):
        return {
            "classification": "preexpert_activation_mismatch_remains",
            "exact_bfloat16_prefix": prefix,
            "first_nonexact_stage": POINTS[len(prefix)],
        }
    if not route_audit["passed"]:
        return {
            "classification": "preexpert_activations_exact_but_route_audit_failed",
            "exact_bfloat16_prefix": prefix,
            "first_nonexact_stage": None,
        }
    return {
        "classification": "complete_preexpert_path_is_bfloat16_exact",
        "exact_bfloat16_prefix": prefix,
        "first_nonexact_stage": None,
    }


def diagnose_layer(
    lib: Any,
    fixture: Any,
    config: Any,
    c_config: dict[str, Any],
    layer: int,
    inputs: torch.Tensor,
    input_values: list[list[float]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    weight_atol: float,
) -> dict[str, Any]:
    official, module, official_indices, _official_weights = (
        _capture_official_pre_router(
            fixture,
            config,
            layer,
            inputs,
            device=device,
            dtype=dtype,
        )
    )
    provider = NativeMatrixProvider(
        layer,
        module,
        official_norm=official["input_norm"],
    )
    candidate, routed_index, routed_weight = capture_c_with_backend(
        lib,
        fixture,
        c_config,
        layer,
        input_values,
        provider,
    )
    comparison = compare_pre_router(
        official,
        candidate,
        compare_dtype=dtype,
    )

    with torch.no_grad():
        official_router_logits = F.linear(
            official["post_attention_norm"].to(device=device, dtype=dtype),
            module.mlp.gate.weight.detach().to(device=device, dtype=dtype),
        ).detach()
    route_audit = audit_layer(
        module,
        official_router_logits,
        official_indices,
        {
            "topk_indices": routed_index.tolist(),
            "topk_weights": routed_weight.tolist(),
        },
        weight_atol=weight_atol,
    )
    decision = classify_preexpert(comparison, route_audit)
    return {
        "stages": comparison["stages"],
        "decision": decision,
        "route_audit": route_audit,
        "official_routed_index": official_indices.detach().cpu().tolist(),
        "candidate_routed_index": routed_index.tolist(),
        "candidate_routed_weight": routed_weight.tolist(),
        "backend_calls": provider.calls,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--c-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--weight-atol", type=float, default=1e-3)
    args = parser.parse_args(argv)
    try:
        if args.weight_atol < 0.0:
            raise PortableBfloat16PreExpertError(
                "--weight-atol must be nonnegative"
            )
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        c_config = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise PortableBfloat16PreExpertError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise PortableBfloat16PreExpertError(
                "the C candidate requires --device cpu"
            )
        library, source = build_portable_preexpert_library(
            Path(args.out).with_suffix(".so")
        )
        if not source["production_source_unchanged"]:
            raise PortableBfloat16PreExpertError("production source changed")
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        analyses = {
            str(layer): diagnose_layer(
                lib,
                fixture,
                config,
                c_config,
                layer,
                inputs,
                input_values,
                device=device,
                dtype=dtype,
                weight_atol=args.weight_atol,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-portable-bfloat16-preexpert-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "weight_atol": args.weight_atol,
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
