#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile an evidence-only C layer with proven BF16 cast boundaries.

The production F32 source is never modified. This tool copies the small layer
translation-unit set to a temporary directory, applies two exact source
transforms that are independently proven by the primitive oracle, compiles the
temporary library, and locates the next official-vs-C activation mismatch:

* RMSNorm casts the normalized value to BF16 before the scale-weight multiply
  and rounds the product to BF16;
* every layer-level matvec rounds its completed double-accumulation output to
  BF16.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "inkling" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from diagnose_inkling_pre_router import (
    POINTS,
    PreRouterDiagnosisError,
    _capture_c_pre_router,
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


class Bfloat16LayerProbeError(RuntimeError):
    """The BF16 candidate cannot be produced without exact source binding."""


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
_MATVEC_OLD = "        out[r] = (float)sum;\n"
_MATVEC_NEW = "        out[r] = bf16_round_probe((float)sum);\n"
_RMS_OLD = "    for (int i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];\n"
_RMS_NEW = """    for (int i = 0; i < n; i++) {
        const float normalized = bf16_round_probe(x[i] * scale);
        out[i] = bf16_round_probe(normalized * weight[i]);
    }
"""


def transform_layer_source(source: str) -> str:
    """Apply each reviewed transform exactly once and fail closed on drift."""
    replacements = (
        (_INCLUDE_ANCHOR, _HELPER, "include anchor"),
        (_MATVEC_OLD, _MATVEC_NEW, "matvec output"),
        (_RMS_OLD, _RMS_NEW, "RMSNorm output"),
    )
    transformed = source
    for old, new, label in replacements:
        count = transformed.count(old)
        if count != 1:
            raise Bfloat16LayerProbeError(
                f"expected exactly one {label}; found {count}"
            )
        transformed = transformed.replace(old, new, 1)
    return transformed


def build_probe_library(out: Path) -> tuple[Path, dict[str, str]]:
    """Copy the layer source set, patch only the copy, and compile it."""
    source_root = REPO / "inkling" / "src"
    temporary = Path(tempfile.mkdtemp(prefix="inkling-bf16-probe-"))
    probe_root = temporary / "src"
    shutil.copytree(source_root, probe_root)
    layer_path = probe_root / "inkling_layer.c"
    original = layer_path.read_text(encoding="utf-8")
    transformed = transform_layer_source(original)
    layer_path.write_text(transformed, encoding="utf-8")
    library = build_library(probe_root, out)
    return library, {
        "production_layer_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "probe_layer_sha256": hashlib.sha256(transformed.encode()).hexdigest(),
        "production_source_unchanged": (
            (source_root / "inkling_layer.c").read_text(encoding="utf-8")
            == original
        ),
        "compiled_sources": list(LAYER_SOURCES),
    }


def diagnose_layer(
    lib,
    fixture,
    config,
    c_config: dict[str, Any],
    layer: int,
    inputs: torch.Tensor,
    input_values: list[list[float]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    official, _, _, _ = _capture_official_pre_router(
        fixture,
        config,
        layer,
        inputs,
        device=device,
        dtype=dtype,
    )
    candidate, _, _ = _capture_c_pre_router(
        lib,
        fixture,
        c_config,
        layer,
        input_values,
    )
    result = compare_pre_router(
        official,
        candidate,
        compare_dtype=dtype,
    )
    exact_prefix: list[str] = []
    for point in POINTS:
        if result["stages"][point]["quantized_exact_fraction"] != 1.0:
            break
        exact_prefix.append(point)
    result["exact_bfloat16_prefix"] = exact_prefix
    result["next_stage_after_exact_prefix"] = (
        POINTS[len(exact_prefix)] if len(exact_prefix) < len(POINTS) else None
    )
    return result


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
            raise Bfloat16LayerProbeError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        library, source = build_probe_library(
            Path(args.out).with_suffix(".so")
        )
        if not source["production_source_unchanged"]:
            raise Bfloat16LayerProbeError("production layer source changed")
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
            "format": "inkling-bfloat16-layer-source-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "source": source,
            "transforms": [
                "RMSNorm cast normalized value to BF16 before weight multiply",
                "RMSNorm round weighted output to BF16",
                "round completed layer-level matvec output to BF16",
            ],
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
        Bfloat16LayerProbeError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
