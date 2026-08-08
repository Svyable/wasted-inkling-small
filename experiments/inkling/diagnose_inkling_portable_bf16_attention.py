#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile and run the proven portable BF16 attention policy.

This evidence-only probe leaves the checked-in F32 decoder untouched. It copies
the four layer translation units, applies the already-proven layer RMSNorm cast
ordering, and patches only the temporary ``inkling_attention.c`` with the
portable policy established by the arithmetic lattice and reduction probe:

* Q/K head inputs and outputs are BF16-bound around RMSNorm;
* completed QK and relative reductions, scaling, and score addition are BF16;
* FP32 softmax probabilities are rounded to BF16 before value aggregation;
* value operands use BF16 values, float32 left-to-right accumulation, and a
  BF16 completed output.

Q/K/V/R/O/router matrices continue to use the existing backend ABI with the
pinned native BF16 linear kernel. The result identifies the first remaining
activation boundary after the complete tested attention policy.
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
    NativeMatrixProvider,
    capture_c_with_backend,
    transform_rms_source,
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


class PortableBfloat16AttentionError(RuntimeError):
    """The portable BF16 attention candidate cannot be built without guessing."""


_INCLUDE_ANCHOR = "#include <math.h>\n#include <stddef.h>\n#include <string.h>\n"
_HELPER = """#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static float bf16_round_probe(float value)
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
_HEAD_OLD = """    double ss = 0.0;
    for (int i = 0; i < dim; i++) ss += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(ss / dim) + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * weight[i];
"""
_HEAD_NEW = """    double ss = 0.0;
    for (int i = 0; i < dim; i++) {
        const float source = bf16_round_probe(x[i]);
        ss += (double)source * (double)source;
    }
    const float scale = 1.0f / sqrtf((float)(ss / dim) + eps);
    for (int i = 0; i < dim; i++) {
        const float source = bf16_round_probe(x[i]);
        const float normalized = bf16_round_probe(source * scale);
        out[i] = bf16_round_probe(normalized * weight[i]);
    }
"""
_VALUE_CACHE_OLD = "    memcpy(vdst, v, kv_stride * sizeof(float));\n"
_VALUE_CACHE_NEW = """    for (size_t i = 0; i < kv_stride; i++)
        vdst[i] = bf16_round_probe(v[i]);
"""
_SCORE_OLD = """            float score = 0.0f;
            for (int d = 0; d < D; d++) score += qh[d] * kc[d];
            score *= dot_scale;
            const int distance = position - key_pos;
            if (distance >= 0 && distance < s->relative_extent) {
                const float *rs = relative_state + (size_t)h * s->d_rel;
                float bias = 0.0f;
                for (int r = 0; r < s->d_rel; r++)
                    bias += rs[r] * relative_proj[(size_t)r * s->relative_extent + distance];
                score += tau * bias;
            }
"""
_SCORE_NEW = """            float score = 0.0f;
            for (int d = 0; d < D; d++)
                score += bf16_round_probe(qh[d]) * bf16_round_probe(kc[d]);
            score = bf16_round_probe(score);
            score = bf16_round_probe(score * dot_scale);
            const int distance = position - key_pos;
            if (distance >= 0 && distance < s->relative_extent) {
                const float *rs = relative_state + (size_t)h * s->d_rel;
                float bias = 0.0f;
                for (int r = 0; r < s->d_rel; r++)
                    bias += bf16_round_probe(rs[r]) *
                            bf16_round_probe(relative_proj[
                                (size_t)r * s->relative_extent + distance]);
                bias = bf16_round_probe(bias);
                bias = bf16_round_probe(tau * bias);
                score = bf16_round_probe(score + bias);
            }
"""
_VALUE_OLD = """            const float weight = row[j] * inv_sum;
            for (int d = 0; d < D; d++) oh[d] += weight * vc[d];
        }
"""
_VALUE_NEW = """            const float weight = bf16_round_probe(row[j] * inv_sum);
            for (int d = 0; d < D; d++)
                oh[d] += weight * bf16_round_probe(vc[d]);
        }
        for (int d = 0; d < D; d++)
            oh[d] = bf16_round_probe(oh[d]);
"""


def transform_attention_source(source: str) -> str:
    """Apply every reviewed portable boundary exactly once and fail on drift."""
    transformed = source
    replacements = (
        (_INCLUDE_ANCHOR, _HELPER, "include anchor"),
        (_HEAD_OLD, _HEAD_NEW, "head RMSNorm"),
        (_VALUE_CACHE_OLD, _VALUE_CACHE_NEW, "value cache boundary"),
        (_SCORE_OLD, _SCORE_NEW, "score and relative boundary"),
        (_VALUE_OLD, _VALUE_NEW, "probability and value boundary"),
    )
    for old, new, label in replacements:
        count = transformed.count(old)
        if count != 1:
            raise PortableBfloat16AttentionError(
                f"expected exactly one {label}; found {count}"
            )
        transformed = transformed.replace(old, new, 1)
    return transformed


def build_portable_attention_library(out: Path) -> tuple[Path, dict[str, Any]]:
    source_root = Path(__file__).resolve().parents[2] / "inkling" / "src"
    temporary = Path(tempfile.mkdtemp(prefix="inkling-portable-bf16-attention-"))
    probe_root = temporary / "src"
    shutil.copytree(source_root, probe_root)

    layer_path = probe_root / "inkling_layer.c"
    attention_path = probe_root / "inkling_attention.c"
    layer_original = layer_path.read_text(encoding="utf-8")
    attention_original = attention_path.read_text(encoding="utf-8")
    layer_transformed = transform_rms_source(layer_original)
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
        "production_attention_sha256": hashlib.sha256(
            attention_original.encode()
        ).hexdigest(),
        "probe_attention_sha256": hashlib.sha256(
            attention_transformed.encode()
        ).hexdigest(),
        "compiled_sources": list(LAYER_SOURCES),
        "transforms": [
            "official layer RMSNorm BF16 ordering",
            "BF16-bound Q/K head RMSNorm inputs and outputs",
            "BF16-bound QK and relative reductions, scaling, and score addition",
            "FP32 softmax probability rounded to BF16",
            "BF16 probability/value operands with float32 accumulation and BF16 output",
        ],
    }


def _exact_prefix(result: dict[str, Any]) -> list[str]:
    prefix: list[str] = []
    for point in POINTS:
        if result["stages"][point]["quantized_exact_fraction"] != 1.0:
            break
        prefix.append(point)
    return prefix


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
) -> dict[str, Any]:
    official, module, _, _ = _capture_official_pre_router(
        fixture,
        config,
        layer,
        inputs,
        device=device,
        dtype=dtype,
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
    prefix = _exact_prefix(comparison)
    comparison["exact_bfloat16_prefix"] = prefix
    comparison["next_stage_after_exact_prefix"] = (
        POINTS[len(prefix)] if len(prefix) < len(POINTS) else None
    )
    comparison["backend_calls"] = provider.calls
    comparison["candidate_routed_index"] = routed_index.tolist()
    comparison["candidate_routed_weight"] = routed_weight.tolist()
    return comparison


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
            raise PortableBfloat16AttentionError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise PortableBfloat16AttentionError("the C candidate requires --device cpu")
        library, source = build_portable_attention_library(
            Path(args.out).with_suffix(".so")
        )
        if not source["production_source_unchanged"]:
            raise PortableBfloat16AttentionError("production source changed")
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
            )
            for layer in layers
        }
        result = {
            "format": "inkling-portable-bfloat16-attention-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "log_scaling_exercised": any(
                (
                    layer >= int(c_config["dense_layers"])
                    and int(c_config.get("log_scaling_n_floor", 0) or 0) > 0
                    and len(input_values) >= int(c_config["log_scaling_n_floor"])
                )
                for layer in layers
            ),
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
