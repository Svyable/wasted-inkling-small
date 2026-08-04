#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test whether official BF16 Q/K head RMSNorm fixes attention output.

The native matrix backend has made input RMSNorm, Q/K/V/R projections, and K/V
short convolution exact. This diagnostic changes only the per-head Q/K RMSNorm
inside a temporary ``inkling_attention.c`` copy and compares it with the prior
native-matrix candidate on the same immutable one-expert fixture.
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

from diagnose_inkling_native_bf16_backend import (
    NativeBfloat16BackendError,
    build_rms_probe_library,
    diagnose_layer,
    transform_rms_source,
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


class AttentionHeadNormError(RuntimeError):
    """The Q/K head RMSNorm boundary cannot be classified exactly."""


_INCLUDE_ANCHOR = "#include <math.h>\n#include <stddef.h>\n#include <string.h>\n"
_INCLUDE_REPLACEMENT = """#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static float bf16_round_attention_probe(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof bits);
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        bits += 0x7fffu + ((bits >> 16) & 1u);
        bits &= 0xffff0000u;
        memcpy(&value, &bits, sizeof value);
    }
    return value;
}
"""
_HEAD_OLD = "    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * weight[i];\n"
_HEAD_NEW = """    for (int i = 0; i < dim; i++) {
        const float normalized = bf16_round_attention_probe(x[i] * scale);
        out[i] = bf16_round_attention_probe(normalized * weight[i]);
    }
"""


def transform_attention_head_rms(source: str) -> str:
    transformed = source
    for old, new, label in (
        (_INCLUDE_ANCHOR, _INCLUDE_REPLACEMENT, "attention include anchor"),
        (_HEAD_OLD, _HEAD_NEW, "head RMSNorm output"),
    ):
        count = transformed.count(old)
        if count != 1:
            raise AttentionHeadNormError(
                f"expected exactly one {label}; found {count}"
            )
        transformed = transformed.replace(old, new, 1)
    return transformed


def build_head_norm_probe_library(out: Path) -> tuple[Path, dict[str, Any]]:
    source_root = Path(__file__).resolve().parents[1] / "inkling" / "src"
    temporary = Path(tempfile.mkdtemp(prefix="inkling-bf16-head-norm-"))
    probe_root = temporary / "src"
    shutil.copytree(source_root, probe_root)

    layer_path = probe_root / "inkling_layer.c"
    layer_original = layer_path.read_text(encoding="utf-8")
    layer_transformed = transform_rms_source(layer_original)
    layer_path.write_text(layer_transformed, encoding="utf-8")

    attention_path = probe_root / "inkling_attention.c"
    attention_original = attention_path.read_text(encoding="utf-8")
    attention_transformed = transform_attention_head_rms(attention_original)
    attention_path.write_text(attention_transformed, encoding="utf-8")

    library = build_library(probe_root, out)
    return library, {
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
        "production_source_unchanged": (
            (source_root / "inkling_layer.c").read_text(encoding="utf-8")
            == layer_original
            and (source_root / "inkling_attention.c").read_text(encoding="utf-8")
            == attention_original
        ),
        "compiled_sources": list(LAYER_SOURCES),
        "transforms": [
            "layer RMSNorm official BF16 ordering",
            "Q/K head RMSNorm official BF16 ordering",
        ],
    }


def classify_attention_boundary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_metric = baseline["stage_comparison"]["stages"]["attention_out"]
    candidate_metric = candidate["stage_comparison"]["stages"]["attention_out"]
    for label, metric in (("baseline", baseline_metric), ("candidate", candidate_metric)):
        exact = float(metric["quantized_exact_fraction"])
        maximum = float(metric["max_abs"])
        mean = float(metric["mean_abs"])
        if not (0.0 <= exact <= 1.0 and maximum >= 0.0 and mean >= 0.0):
            raise AttentionHeadNormError(f"invalid {label} attention metrics")
    if candidate_metric["quantized_exact_fraction"] == 1.0:
        classification = "head_rmsnorm_is_sufficient"
    elif (
        candidate_metric["quantized_exact_fraction"]
        > baseline_metric["quantized_exact_fraction"]
        and candidate_metric["max_abs"] < baseline_metric["max_abs"]
        and candidate_metric["mean_abs"] < baseline_metric["mean_abs"]
    ):
        classification = "head_rmsnorm_improves_but_attention_math_still_differs"
    elif (
        candidate_metric["quantized_exact_fraction"]
        == baseline_metric["quantized_exact_fraction"]
        and candidate_metric["max_abs"] == baseline_metric["max_abs"]
        and candidate_metric["mean_abs"] == baseline_metric["mean_abs"]
    ):
        classification = "head_rmsnorm_has_no_effect"
    else:
        classification = "head_rmsnorm_changes_attention_without_monotonic_improvement"
    return {
        "classification": classification,
        "baseline_attention_out": baseline_metric,
        "candidate_attention_out": candidate_metric,
        "exact_fraction_delta": (
            candidate_metric["quantized_exact_fraction"]
            - baseline_metric["quantized_exact_fraction"]
        ),
        "max_abs_delta": candidate_metric["max_abs"] - baseline_metric["max_abs"],
        "mean_abs_delta": candidate_metric["mean_abs"] - baseline_metric["mean_abs"],
        "candidate_exact_prefix": candidate["exact_bfloat16_prefix"],
        "candidate_next_stage": candidate["next_stage_after_exact_prefix"],
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
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    args = parser.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        c_config = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise AttentionHeadNormError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)

        baseline_library, baseline_source = build_rms_probe_library(
            Path(args.out).with_name(Path(args.out).stem + "-baseline.so")
        )
        candidate_library, candidate_source = build_head_norm_probe_library(
            Path(args.out).with_name(Path(args.out).stem + "-candidate.so")
        )
        if not baseline_source["production_source_unchanged"]:
            raise AttentionHeadNormError("baseline changed production source")
        if not candidate_source["production_source_unchanged"]:
            raise AttentionHeadNormError("candidate changed production source")
        baseline_lib = ctypes.CDLL(str(baseline_library))
        candidate_lib = ctypes.CDLL(str(candidate_library))
        configure_library(baseline_lib)
        configure_library(candidate_lib)

        analyses: dict[str, Any] = {}
        for layer in layers:
            baseline = diagnose_layer(
                baseline_lib,
                fixture,
                config,
                c_config,
                layer,
                inputs,
                input_values,
                device=device,
                dtype=dtype,
            )["variants"]["native-token-step"]
            candidate = diagnose_layer(
                candidate_lib,
                fixture,
                config,
                c_config,
                layer,
                inputs,
                input_values,
                device=device,
                dtype=dtype,
            )["variants"]["native-token-step"]
            analyses[str(layer)] = classify_attention_boundary(
                baseline, candidate
            )

        result = {
            "format": "inkling-bfloat16-attention-head-rmsnorm-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "baseline_source": baseline_source,
            "candidate_source": candidate_source,
            "layers": analyses,
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        AttentionHeadNormError,
        NativeBfloat16BackendError,
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
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
