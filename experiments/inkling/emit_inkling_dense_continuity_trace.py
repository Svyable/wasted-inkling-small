#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit and compare a chained dense 0 -> 1 trace from the continuity fixture.

This is the cheap arithmetic preflight for the full 0 -> 1 -> 2 gate. It uses
the same official and checked-C execution helpers as the full continuity runner,
but stops before sparse expert execution. Expected and actual traces contain the
canonical source-bound input plus each dense layer's exact F32 output. The
architecture-neutral port_trace localizer therefore reports the first
(layer,boundary,index,raw-bits) divergence and verifies byte-exact inter-layer
chaining before the expensive sparse stage runs.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import struct
from pathlib import Path
from typing import Any

import torch

from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    build_layer_from_fixture,
    verify_config_binding,
)
from inkling_layer_parity import LayerParityError
from inkling_release_config import build_transformers_text_config
from measure_inkling_dense_bf16_profile import (
    DenseBfloat16ProbeError,
    DenseMatrixProvider,
    run_candidate as run_dense_candidate,
)
from port_trace import TraceError, boundary_manifest, compare, load_trace
from run_inkling_checked_bf16_cross_layer import (
    CrossLayerContinuityError,
    official_dense_prefix,
    tensor_rows,
)
from run_inkling_checked_bf16_sparse_layer import (
    build_checked_library,
    configure_profile_library,
)


class DenseContinuityTraceError(RuntimeError):
    pass


def _f32_payload(tensor: torch.Tensor) -> bytes:
    values = tensor.detach().float().cpu().contiguous().reshape(-1).tolist()
    return struct.pack(f"<{len(values)}f", *values)


def _layer(
    root: Path,
    layer: int,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> dict[str, Any]:
    prefix = f"layer-{layer}"
    return {
        "layer": layer,
        "structural_class": "dense",
        "boundaries": [
            boundary_manifest(
                root,
                f"{prefix}/input.f32.bin",
                "input",
                "F32",
                list(input_tensor.shape),
                _f32_payload(input_tensor),
            ),
            boundary_manifest(
                root,
                f"{prefix}/output.f32.bin",
                "output",
                "F32",
                list(output_tensor.shape),
                _f32_payload(output_tensor),
            ),
        ],
    }


def _manifest(
    root: Path,
    *,
    model: str,
    revision: str,
    trace_name: str,
    first_input: torch.Tensor,
    layer0: torch.Tensor,
    layer1: torch.Tensor,
) -> Path:
    manifest = {
        "schema_version": 1,
        "model": model,
        "revision": revision,
        "trace_name": trace_name,
        "layers": [
            _layer(root, 0, first_input, layer0),
            _layer(root, 1, layer0, layer1),
        ],
    }
    path = root / "trace.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Loading here turns producer mistakes into an invalid trace before compare.
    load_trace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--c-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        cfg = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(input_values, list) or len(input_values) != 8:
            raise DenseContinuityTraceError(
                "--inputs must contain eight source-bound rows"
            )
        dtype = DTYPES["bfloat16"]
        device = torch.device("cpu")
        official = official_dense_prefix(
            fixture, config, input_values, device=device, dtype=dtype
        )

        library, source = build_checked_library(
            Path(args.out).with_suffix(".runtime.so")
        )
        if source.get("source_rewriting") is not False:
            raise DenseContinuityTraceError(
                "dense trace must compile checked-in C unchanged"
            )
        lib = ctypes.CDLL(str(library))
        configure_profile_library(lib)
        modules = {
            layer: build_layer_from_fixture(
                fixture, config, layer, device=device, dtype=dtype
            )
            for layer in (0, 1)
        }
        dense0 = run_dense_candidate(
            lib, fixture, cfg, 0, input_values, DenseMatrixProvider(0, modules[0])
        )
        dense0_tensor = torch.stack(
            [value.detach().float().cpu() for value in dense0["outputs"]]
        ).contiguous()
        dense1 = run_dense_candidate(
            lib,
            fixture,
            cfg,
            1,
            tensor_rows(dense0["outputs"]),
            DenseMatrixProvider(1, modules[1]),
        )
        dense1_tensor = torch.stack(
            [value.detach().float().cpu() for value in dense1["outputs"]]
        ).contiguous()

        # This canonical first boundary is the BF16 hidden state seen by the
        # official stack, represented losslessly as F32. It is shared between
        # expected/actual traces; the localizer is for execution/chaining, not
        # for re-litigating source-input serialization.
        first_input = torch.tensor(input_values, dtype=torch.bfloat16).float()
        expected_root = Path(args.out_dir) / "expected"
        actual_root = Path(args.out_dir) / "actual"
        expected_root.mkdir(parents=True, exist_ok=True)
        actual_root.mkdir(parents=True, exist_ok=True)
        expected_manifest = _manifest(
            expected_root,
            model=fixture.model_id,
            revision=str(fixture.source.get("revision")),
            trace_name="official-dense-0-1",
            first_input=first_input,
            layer0=official[0],
            layer1=official[1],
        )
        actual_manifest = _manifest(
            actual_root,
            model=fixture.model_id,
            revision=str(fixture.source.get("revision")),
            trace_name="checked-c-dense-0-1",
            first_input=first_input,
            layer0=dense0_tensor,
            layer1=dense1_tensor,
        )
        decision = compare(load_trace(expected_manifest), load_trace(actual_manifest))
        result = {
            "format": "inkling-checked-in-dense-continuity-trace",
            "version": 1,
            "source": source,
            "decision": decision,
            "expected_manifest": str(expected_manifest),
            "actual_manifest": str(actual_manifest),
            "backend_calls": {
                "0": len(dense0["backend_calls"]),
                "1": len(dense1["backend_calls"]),
            },
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        CrossLayerContinuityError,
        DenseBfloat16ProbeError,
        DenseContinuityTraceError,
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
        TraceError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"]["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
