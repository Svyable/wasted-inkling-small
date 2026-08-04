#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe Inkling with native BF16 matrix callbacks and exact projections.

This evidence-only tool leaves the checked-in F32 decoder untouched. It copies
and compiles the layer source with only the independently proven RMSNorm cast
ordering, then supplies selected matrices through the existing backend ABI.

Two variants are run:

* ``native-token-step`` uses the pinned PyTorch BF16 linear kernel for
  Q/K/V/R/O/router on each C token step;
* ``official-prefill-projection-oracle`` returns the exact official full-prefill
  Q/K/V/R rows, while O/router still use the native BF16 kernel.

The oracle variant makes the next mismatch identifiable even when prefill and
token-step GEMM kernels differ. Every oracle call verifies that the C RMSNorm
input is BF16-identical to the official row before returning an activation.
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

from diagnose_inkling_bf16_projection_granularity import (
    PROJECTIONS,
    _rowwise_official_projection,
    classify_projection,
)
from diagnose_inkling_pre_router import (
    POINTS,
    PreRouterDiagnosisError,
    _capture_official_pre_router,
    compare_pre_router,
)
from discover_inkling_c_router_experts import _RejectExperts
from discover_inkling_router_experts import input_sha256
from inkling_compact_layer_parity import bind_sparse_layer_compact
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    verify_config_binding,
)
from inkling_layer_parity import (
    FP,
    LAYER_SOURCES,
    LayerParityError,
    LayerScratch,
    LayerState,
    MatrixBackend,
    TraceCollector,
    _build_config,
    build_library,
    configure_library,
)
from inkling_release_config import build_transformers_text_config


class NativeBfloat16BackendError(RuntimeError):
    """The native BF16 backend probe cannot continue without guessing."""


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
_RMS_OLD = "    for (int i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];\n"
_RMS_NEW = """    for (int i = 0; i < n; i++) {
        const float normalized = bf16_round_probe(x[i] * scale);
        out[i] = bf16_round_probe(normalized * weight[i]);
    }
"""

# waste_inkling_matrix_kind values from inkling_layer.h.
MAT_Q = 0
MAT_K = 1
MAT_V = 2
MAT_R = 3
MAT_O = 4
MAT_ROUTER = 8
KIND_NAMES = {
    MAT_Q: "q_proj",
    MAT_K: "k_proj",
    MAT_V: "v_proj",
    MAT_R: "relative_proj_input",
    MAT_O: "o_proj",
    MAT_ROUTER: "router",
}
NULL_WEIGHT_FIELDS = (
    "wq",
    "wk",
    "wv",
    "wr",
    "wo",
    "router_weight",
)

MatvecCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    FP,
    FP,
    ctypes.c_int,
    ctypes.c_int,
)


def transform_rms_source(source: str) -> str:
    """Apply only the proven RMSNorm ordering to a temporary source copy."""
    transformed = source
    for old, new, label in (
        (_INCLUDE_ANCHOR, _HELPER, "include anchor"),
        (_RMS_OLD, _RMS_NEW, "RMSNorm output"),
    ):
        count = transformed.count(old)
        if count != 1:
            raise NativeBfloat16BackendError(
                f"expected exactly one {label}; found {count}"
            )
        transformed = transformed.replace(old, new, 1)
    return transformed


def build_rms_probe_library(out: Path) -> tuple[Path, dict[str, Any]]:
    source_root = Path(__file__).resolve().parents[1] / "inkling" / "src"
    temporary = Path(tempfile.mkdtemp(prefix="inkling-native-bf16-backend-"))
    probe_root = temporary / "src"
    shutil.copytree(source_root, probe_root)
    layer_path = probe_root / "inkling_layer.c"
    original = layer_path.read_text(encoding="utf-8")
    transformed = transform_rms_source(original)
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
        "transforms": [
            "RMSNorm cast normalized value to BF16 before weight multiply",
            "RMSNorm round weighted output to BF16",
        ],
    }


def native_bfloat16_linear(weight: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Run the exact one-row BF16 kernel used by the backend callback."""
    if weight.ndim != 2 or vector.ndim != 1 or weight.shape[1] != vector.shape[0]:
        raise NativeBfloat16BackendError(
            f"invalid linear geometry {tuple(weight.shape)} x {tuple(vector.shape)}"
        )
    with torch.no_grad():
        return F.linear(
            vector.to(dtype=torch.bfloat16).reshape(1, -1),
            weight.to(dtype=torch.bfloat16),
        )[0].detach().float().cpu().contiguous()


class NativeMatrixProvider:
    """Serve selected official matrices through the C backend callback."""

    def __init__(
        self,
        layer: int,
        module: Any,
        *,
        official_norm: torch.Tensor,
        projection_oracle: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.layer = int(layer)
        self.official_norm = official_norm.detach().to(torch.bfloat16).float().cpu()
        self.projection_oracle = {
            point: value.detach().to(torch.bfloat16).float().cpu().contiguous()
            for point, value in (projection_oracle or {}).items()
        }
        self.weights = {
            MAT_Q: module.self_attn.q_proj.weight.detach().cpu().contiguous(),
            MAT_K: module.self_attn.k_proj.weight.detach().cpu().contiguous(),
            MAT_V: module.self_attn.v_proj.weight.detach().cpu().contiguous(),
            MAT_R: module.self_attn.r_proj.weight.detach().cpu().contiguous(),
            MAT_O: module.self_attn.o_proj.weight.detach().cpu().contiguous(),
            MAT_ROUTER: module.mlp.gate.weight.detach().cpu().contiguous(),
        }
        self.call_counts = {kind: 0 for kind in self.weights}
        self.calls: list[dict[str, Any]] = []
        self.error: str | None = None
        self.callback = MatvecCallback(self._call)
        self.backend = MatrixBackend(
            ctypes.cast(self.callback, ctypes.c_void_p),
            None,
        )

    def _vector(self, pointer: FP, cols: int) -> torch.Tensor:
        return torch.tensor(
            [float(pointer[index]) for index in range(cols)],
            dtype=torch.float32,
        )

    def _copy_out(self, pointer: FP, value: torch.Tensor, rows: int) -> None:
        if value.ndim != 1 or value.numel() != rows:
            raise NativeBfloat16BackendError(
                f"backend output has {tuple(value.shape)}; expected [{rows}]"
            )
        for index, item in enumerate(value.tolist()):
            pointer[index] = float(item)

    def _call(
        self,
        _ctx: int,
        layer: int,
        kind: int,
        index: int,
        x: FP,
        out: FP,
        rows: int,
        cols: int,
    ) -> int:
        try:
            if layer != self.layer or index != 0 or kind not in self.weights:
                raise NativeBfloat16BackendError(
                    f"unexpected backend request layer={layer} kind={kind} index={index}"
                )
            weight = self.weights[kind]
            if tuple(weight.shape) != (rows, cols):
                raise NativeBfloat16BackendError(
                    f"{KIND_NAMES[kind]} weight {tuple(weight.shape)} != {(rows, cols)}"
                )
            position = self.call_counts[kind]
            self.call_counts[kind] += 1
            vector = self._vector(x, cols)
            point = KIND_NAMES[kind]
            if point in self.projection_oracle:
                if position >= self.official_norm.shape[0]:
                    raise NativeBfloat16BackendError(
                        f"{point} oracle has no position {position}"
                    )
                normalized = vector.to(torch.bfloat16).float()
                if not torch.equal(normalized, self.official_norm[position]):
                    raise NativeBfloat16BackendError(
                        f"{point} position {position} received non-official RMSNorm input"
                    )
                oracle = self.projection_oracle[point]
                if position >= oracle.shape[0] or oracle.shape[1] != rows:
                    raise NativeBfloat16BackendError(
                        f"{point} oracle shape {tuple(oracle.shape)} cannot serve {(position, rows)}"
                    )
                value = oracle[position]
                mode = "official-prefill-oracle"
            else:
                value = native_bfloat16_linear(weight, vector)
                mode = "native-token-step"
            self._copy_out(out, value, rows)
            self.calls.append(
                {
                    "position": position,
                    "kind": point,
                    "mode": mode,
                    "rows": rows,
                    "cols": cols,
                }
            )
            return 0
        except Exception as exc:  # ctypes callbacks cannot propagate safely.
            self.error = str(exc)
            return -1


def capture_c_with_backend(
    lib: Any,
    fixture: Any,
    cfg: dict[str, Any],
    layer: int,
    inputs: list[list[float]],
    provider: NativeMatrixProvider,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    compact = bind_sparse_layer_compact(fixture, cfg, layer)
    for field in NULL_WEIGHT_FIELDS:
        setattr(compact.weights, field, FP())
    config = _build_config(lib, cfg)
    hidden = int(cfg["hidden"])
    is_local = layer in set(cfg.get("local_layer_ids", []))
    kv_heads = int(cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"])
    head_dim = int(cfg["local_head_dim"] if is_local else cfg["global_head_dim"])
    conv_k = int(cfg["conv_kernel"])
    cap = (
        min(max(len(inputs), 1), int(cfg["sliding_window"]))
        if is_local
        else max(len(inputs), 1)
    )
    nfloat = lib.waste_inkling_layer_scratch_floats(
        ctypes.byref(config), layer, cap
    )
    nint = lib.waste_inkling_layer_scratch_ints(ctypes.byref(config))
    if not nfloat:
        raise NativeBfloat16BackendError("scratch sizing failed")
    fbuf = (ctypes.c_float * nfloat)()
    ibuf = (ctypes.c_int * max(nint, 1))()
    scratch = LayerScratch()
    if lib.waste_inkling_layer_scratch_init(
        ctypes.byref(scratch),
        ctypes.byref(config),
        layer,
        cap,
        fbuf,
        ctypes.c_size_t(nfloat),
        ibuf,
        ctypes.c_size_t(nint),
    ):
        raise NativeBfloat16BackendError("scratch init failed")
    kv = (ctypes.c_float * (cap * kv_heads * head_dim))()
    vv = (ctypes.c_float * (cap * kv_heads * head_dim))()
    kconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    vconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    aconv = (ctypes.c_float * (hidden * conv_k))()
    mconv = (ctypes.c_float * (hidden * conv_k))()
    state = LayerState()
    if lib.waste_inkling_layer_state_init(
        ctypes.byref(state),
        ctypes.byref(config),
        layer,
        cap,
        kv,
        vv,
        kconv,
        vconv,
        aconv,
        mconv,
    ):
        raise NativeBfloat16BackendError("state init failed")

    collector = TraceCollector()
    reject = _RejectExperts(layer)
    reject_ptr = ctypes.cast(reject.callback, ctypes.c_void_p)
    for position, row in enumerate(inputs):
        if len(row) != hidden:
            raise NativeBfloat16BackendError(
                f"input {position} has {len(row)} values; hidden is {hidden}"
            )
        collector.position = position
        x = (ctypes.c_float * hidden)(*row)
        before = len(reject.requests)
        rc = lib.waste_inkling_layer_step_backend_trace(
            ctypes.byref(config),
            layer,
            ctypes.byref(compact.weights),
            ctypes.byref(provider.backend),
            ctypes.byref(state),
            x,
            position,
            ctypes.byref(scratch),
            reject_ptr,
            None,
            ctypes.byref(collector.c_trace),
        )
        if provider.error:
            raise NativeBfloat16BackendError(provider.error)
        if rc == 0 or len(reject.requests) != before + 1:
            raise NativeBfloat16BackendError(
                f"layer {layer} position {position} did not stop at one expert lookup"
            )

    stages: dict[str, torch.Tensor] = {}
    route_indices = []
    route_weights = []
    for position in range(len(inputs)):
        for point, rows in (
            ("routed_index", route_indices),
            ("routed_weight", route_weights),
        ):
            name = f"token.{position}.layer.{layer}.{point}"
            if name not in collector.values:
                raise NativeBfloat16BackendError(f"C trace missed {name}")
            rows.append(collector.values[name])
    for point in POINTS:
        rows = []
        for position in range(len(inputs)):
            name = f"token.{position}.layer.{layer}.{point}"
            if name not in collector.values:
                raise NativeBfloat16BackendError(f"C trace missed {name}")
            rows.append(collector.values[name])
        stages[point] = torch.tensor(rows, dtype=torch.float32)
    return (
        stages,
        torch.tensor(route_indices, dtype=torch.int64),
        torch.tensor(route_weights, dtype=torch.float32),
    )


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
    variants: dict[str, Any] = {}
    for name, oracle in (
        ("native-token-step", None),
        (
            "official-prefill-projection-oracle",
            {point: official[point] for point in PROJECTIONS},
        ),
    ):
        provider = NativeMatrixProvider(
            layer,
            module,
            official_norm=official["input_norm"],
            projection_oracle=oracle,
        )
        candidate, indices, weights = capture_c_with_backend(
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
        projection_kernel = {}
        for point in PROJECTIONS:
            token_step = _rowwise_official_projection(
                module,
                point,
                official["input_norm"],
                device=device,
                dtype=dtype,
            )
            projection_kernel[point] = classify_projection(
                official[point],
                token_step,
                candidate[point],
                dtype=dtype,
            )
        variants[name] = {
            "exact_bfloat16_prefix": prefix,
            "next_stage_after_exact_prefix": (
                POINTS[len(prefix)] if len(prefix) < len(POINTS) else None
            ),
            "stage_comparison": comparison,
            "projection_kernel": projection_kernel,
            "router_indices": indices.tolist(),
            "router_weights": weights.tolist(),
            "backend_calls": provider.calls,
        }
    return {"variants": variants}


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
            raise NativeBfloat16BackendError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        library, source = build_rms_probe_library(Path(args.out).with_suffix(".so"))
        if not source["production_source_unchanged"]:
            raise NativeBfloat16BackendError("production layer source changed")
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
            "format": "inkling-native-bfloat16-matrix-backend-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
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
