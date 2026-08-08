#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply the proven BF16 SiLU/product rule to one sparse MoE token.

This evidence-only composition probe reuses the source-bound 12-expert fixture
and native BF16 matrix backend from the single-token MoE diagnostic.  The only
new temporary-source change is the primitive policy proven independently in
#32: complete SiLU to BF16 before multiplying by BF16 ``up``, then complete the
product to BF16.

Production sources and the public matrix-backend ABI remain unchanged.
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

from diagnose_inkling_native_bf16_backend import transform_rms_source
from diagnose_inkling_portable_bf16_attention import transform_attention_source
from diagnose_inkling_portable_bf16_preexpert import transform_residual_source
from diagnose_inkling_portable_bf16_single_token_moe import (
    SingleTokenMoeError,
    analyze_layer,
    transform_moe_source,
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
    verify_config_binding,
)
from inkling_layer_parity import (
    LAYER_SOURCES,
    LayerParityError,
    build_library,
    configure_library,
)
from inkling_release_config import build_transformers_text_config


class MoeActivationError(RuntimeError):
    """The composed BF16 expert activation probe cannot proceed safely."""


_ROUTED_OLD = """                for (int i = 0; i < inter; i++)
                    s->gate[i] = silu(s->gate[i]) * s->up[i];
"""
_ROUTED_NEW = """                for (int i = 0; i < inter; i++) {
                    const float activated =
                        bf16_round_probe(silu(s->gate[i]));
                    s->gate[i] = bf16_round_probe(
                        activated * bf16_round_probe(s->up[i]));
                }
"""
_SHARED_OLD = """                for (int j = 0; j < inter; j++)
                    s->gate[j] = silu(s->gate[j]) * s->up[j];
"""
_SHARED_NEW = """                for (int j = 0; j < inter; j++) {
                    const float activated =
                        bf16_round_probe(silu(s->gate[j]));
                    s->gate[j] = bf16_round_probe(
                        activated * bf16_round_probe(s->up[j]));
                }
"""


def transform_activation_source(source: str) -> str:
    """Apply the two proven BF16 completion points exactly once per path."""
    transformed = source
    for old, new, label in (
        (_ROUTED_OLD, _ROUTED_NEW, "routed activation loop"),
        (_SHARED_OLD, _SHARED_NEW, "shared activation loop"),
    ):
        count = transformed.count(old)
        if count != 1:
            raise MoeActivationError(
                f"expected exactly one {label}; found {count}"
            )
        transformed = transformed.replace(old, new, 1)
    if transformed.count("bf16_round_probe(silu(") < 2:
        raise MoeActivationError("activation transform omitted a BF16 SiLU boundary")
    return transformed


def build_activation_library(out: Path) -> tuple[Path, dict[str, Any]]:
    source_root = Path(__file__).resolve().parents[2] / "inkling" / "src"
    temporary = Path(tempfile.mkdtemp(prefix="inkling-bf16-moe-activation-"))
    probe_root = temporary / "src"
    shutil.copytree(source_root, probe_root)

    layer_path = probe_root / "inkling_layer.c"
    attention_path = probe_root / "inkling_attention.c"
    layer_original = layer_path.read_text(encoding="utf-8")
    attention_original = attention_path.read_text(encoding="utf-8")

    layer_transformed = transform_activation_source(
        transform_moe_source(
            transform_residual_source(transform_rms_source(layer_original))
        )
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
        "activation_policy": {
            "silu_completion": "bfloat16",
            "up_operand_completion": "bfloat16",
            "product_completion": "bfloat16",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--c-config", required=True)
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
        cfg = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(input_values, list) or len(input_values) != 8:
            raise MoeActivationError(
                "--inputs must contain the eight source-bound deterministic rows"
            )
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise MoeActivationError("--layers must be nonempty")
        routes = load_canonical_routes(
            args.selection, fixture, input_values, layers
        )
        position_zero = {layer: routes[layer][0] for layer in layers}
        for layer, row in position_zero.items():
            declared = set(fixture.experts.get(layer, ()))
            if declared != set(row.indices):
                raise MoeActivationError(
                    f"layer {layer} fixture experts {sorted(declared)} differ from "
                    f"position-zero route {sorted(row.indices)}"
                )
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise MoeActivationError("the C probe requires --device cpu")

        library, source = build_activation_library(
            Path(args.out).with_suffix(".so")
        )
        if not source["production_source_unchanged"]:
            raise MoeActivationError("production source changed")
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        analyses = {
            str(layer): analyze_layer(
                lib,
                fixture,
                config,
                cfg,
                layer,
                inputs,
                input_values,
                position_zero[layer],
                device=device,
                dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-portable-bfloat16-single-token-moe-activation-probe",
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
        CanonicalRouteError,
        SingleTokenMoeError,
        MoeActivationError,
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
