#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove the checked-in dense BF16 layer profile on official layer-0 bytes.

The dense completion policy was first measured in an evidence-only source copy
and was exact across all eight source-bound layer-0 outputs.  The policy is now
checked into ``inkling/src``.  This gate compiles that source unchanged and
repeats the same stateful official-weight comparison, so future edits cannot
pass by keeping the old temporary rewrite exact while the shipped C drifts.

The promoted policy is deliberately narrow:

* reuse the BF16 norm/attention/residual policy proven on sparse layers;
* force dense gate/up/down through the native PyTorch BF16 matrix backend;
* complete SiLU and the gated product to BF16;
* complete the dense down result and global-scale multiply to BF16; and
* retain the existing F32 short-convolution state and final BF16 residual.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

import torch

from diagnose_inkling_native_bf16_backend import (
    MatvecCallback,
    NativeBfloat16BackendError,
    native_bfloat16_linear,
)
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    build_layer_from_fixture,
    run_layer_reference,
    verify_config_binding,
)
from inkling_layer_parity import (
    FP,
    LayerParityError,
    LayerScratch,
    LayerState,
    MatrixBackend,
    TraceCollector,
    _build_config,
    bind_layer,
)
from inkling_release_config import build_transformers_text_config
from run_inkling_checked_bf16_sparse_layer import (
    BF16_REFERENCE,
    build_checked_library,
    configure_profile_library,
)


class DenseBfloat16ProbeError(RuntimeError):
    pass


MAT_Q = 0
MAT_K = 1
MAT_V = 2
MAT_R = 3
MAT_O = 4
MAT_DENSE_GATE = 5
MAT_DENSE_UP = 6
MAT_DENSE_DOWN = 7

KIND_NAMES = {
    MAT_Q: "q_proj",
    MAT_K: "k_proj",
    MAT_V: "v_proj",
    MAT_R: "relative_proj_input",
    MAT_O: "o_proj",
    MAT_DENSE_GATE: "dense_gate",
    MAT_DENSE_UP: "dense_up",
    MAT_DENSE_DOWN: "dense_down",
}

FINAL_POINTS = (
    "input_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "relative_proj_input",
    "k_sconv",
    "v_sconv",
    "attention_out",
    "attention_branch",
    "post_attention_residual",
    "post_attention_norm",
    "dense_mlp_out",
    "mlp_branch",
    "layer_out",
)


def bf16_round_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(torch.bfloat16).float().cpu()


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu().reshape(-1)
    candidate = candidate.detach().float().cpu().reshape(-1)
    if reference.shape != candidate.shape:
        raise DenseBfloat16ProbeError(
            f"shape mismatch {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    raw = (reference - candidate).abs()
    ref_bf16 = bf16_round_tensor(reference)
    cand_bf16 = bf16_round_tensor(candidate)
    bf16 = (ref_bf16 - cand_bf16).abs()
    return {
        "count": int(reference.numel()),
        "raw_exact_fraction": float(torch.eq(reference, candidate).float().mean()),
        "raw_max_abs": float(raw.max()) if raw.numel() else 0.0,
        "bfloat16_exact_fraction": float(torch.eq(ref_bf16, cand_bf16).float().mean()),
        "bfloat16_max_abs": float(bf16.max()) if bf16.numel() else 0.0,
    }


class DenseMatrixProvider:
    def __init__(self, layer: int, module: Any) -> None:
        self.layer = int(layer)
        self.weights = {
            MAT_Q: module.self_attn.q_proj.weight.detach().cpu().contiguous(),
            MAT_K: module.self_attn.k_proj.weight.detach().cpu().contiguous(),
            MAT_V: module.self_attn.v_proj.weight.detach().cpu().contiguous(),
            MAT_R: module.self_attn.r_proj.weight.detach().cpu().contiguous(),
            MAT_O: module.self_attn.o_proj.weight.detach().cpu().contiguous(),
            MAT_DENSE_GATE: module.mlp.gate_proj.weight.detach().cpu().contiguous(),
            MAT_DENSE_UP: module.mlp.up_proj.weight.detach().cpu().contiguous(),
            MAT_DENSE_DOWN: module.mlp.down_proj.weight.detach().cpu().contiguous(),
        }
        self.call_counts = {kind: 0 for kind in self.weights}
        self.calls: list[dict[str, Any]] = []
        self.error: str | None = None
        self.callback = MatvecCallback(self._call)
        self.backend = MatrixBackend(ctypes.cast(self.callback, ctypes.c_void_p), None)

    @staticmethod
    def _vector(pointer: FP, cols: int) -> torch.Tensor:
        return torch.tensor([float(pointer[index]) for index in range(cols)], dtype=torch.float32)

    @staticmethod
    def _copy(pointer: FP, value: torch.Tensor, rows: int) -> None:
        value = value.detach().float().cpu().reshape(-1)
        if value.numel() != rows:
            raise DenseBfloat16ProbeError(
                f"backend returned {value.numel()} values; expected {rows}"
            )
        for index, item in enumerate(value.tolist()):
            pointer[index] = float(item)

    def _call(self, _ctx, layer, kind, index, x, out, rows, cols) -> int:
        try:
            if layer != self.layer or index != 0 or kind not in self.weights:
                raise DenseBfloat16ProbeError(
                    f"unexpected backend request layer={layer} kind={kind} index={index}"
                )
            weight = self.weights[kind]
            if tuple(weight.shape) != (rows, cols):
                raise DenseBfloat16ProbeError(
                    f"{KIND_NAMES[kind]} weight {tuple(weight.shape)} != {(rows, cols)}"
                )
            position = self.call_counts[kind]
            self.call_counts[kind] += 1
            value = native_bfloat16_linear(weight, self._vector(x, cols))
            self._copy(out, value, rows)
            self.calls.append(
                {
                    "position": position,
                    "kind": KIND_NAMES[kind],
                    "rows": rows,
                    "cols": cols,
                    "mode": "native-token-step",
                }
            )
            return 0
        except Exception as exc:
            self.error = str(exc)
            return -1


def run_candidate(
    lib: Any,
    fixture: Any,
    cfg: dict[str, Any],
    layer: int,
    input_values: list[list[float]],
    provider: DenseMatrixProvider,
) -> dict[str, Any]:
    binding = bind_layer(fixture, cfg, layer)
    for field in ("wq", "wk", "wv", "wr", "wo", "dense_gate", "dense_up", "dense_down"):
        setattr(binding.weights, field, FP())
    config = _build_config(lib, cfg)
    hidden = int(cfg["hidden"])
    is_local = layer in set(cfg.get("local_layer_ids", []))
    kv_heads = int(cfg["local_kv_heads"] if is_local else cfg["global_kv_heads"])
    head_dim = int(cfg["local_head_dim"] if is_local else cfg["global_head_dim"])
    conv_k = int(cfg["conv_kernel"])
    capacity = min(len(input_values), int(cfg["sliding_window"])) if is_local else len(input_values)

    nfloat = lib.waste_inkling_layer_scratch_floats(ctypes.byref(config), layer, capacity)
    nint = lib.waste_inkling_layer_scratch_ints(ctypes.byref(config))
    fbuf = (ctypes.c_float * nfloat)()
    ibuf = (ctypes.c_int * max(nint, 1))()
    scratch = LayerScratch()
    if lib.waste_inkling_layer_scratch_init(
        ctypes.byref(scratch), ctypes.byref(config), layer, capacity,
        fbuf, ctypes.c_size_t(nfloat), ibuf, ctypes.c_size_t(nint),
    ):
        raise DenseBfloat16ProbeError("scratch init failed")
    kv = (ctypes.c_float * (capacity * kv_heads * head_dim))()
    vv = (ctypes.c_float * (capacity * kv_heads * head_dim))()
    kconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    vconv = (ctypes.c_float * (kv_heads * head_dim * conv_k))()
    aconv = (ctypes.c_float * (hidden * conv_k))()
    mconv = (ctypes.c_float * (hidden * conv_k))()
    state = LayerState()
    if lib.waste_inkling_layer_state_init(
        ctypes.byref(state), ctypes.byref(config), layer, capacity,
        kv, vv, kconv, vconv, aconv, mconv,
    ):
        raise DenseBfloat16ProbeError("state init failed")

    collector = TraceCollector()
    outputs: list[torch.Tensor] = []
    for position, row in enumerate(input_values):
        collector.position = position
        x = (ctypes.c_float * hidden)(*row)
        rc = lib.waste_inkling_layer_step_backend_trace_profile(
            ctypes.byref(config), layer, ctypes.byref(binding.weights),
            ctypes.byref(provider.backend), ctypes.byref(state), x, position,
            ctypes.byref(scratch), None, None, ctypes.byref(collector.c_trace),
            BF16_REFERENCE,
        )
        if provider.error:
            raise DenseBfloat16ProbeError(provider.error)
        if rc:
            raise DenseBfloat16ProbeError(
                f"dense candidate failed at position {position} with status {rc}"
            )
        output = torch.tensor([float(x[index]) for index in range(hidden)], dtype=torch.float32)
        outputs.append(output)
        collector.values[f"token.{position}.layer.{layer}.output"] = output.tolist()
    return {"collector": collector, "outputs": outputs, "backend_calls": provider.calls}


def first_nonexact(stages: dict[str, dict[str, Any]]) -> str | None:
    for point in FINAL_POINTS:
        if stages[point]["bfloat16_exact_fraction"] != 1.0:
            return point
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--c-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        cfg = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(input_values, list) or len(input_values) != 8:
            raise DenseBfloat16ProbeError("--inputs must contain eight source-bound rows")
        dtype = DTYPES["bfloat16"]
        device = torch.device("cpu")
        reference = run_layer_reference(
            fixture,
            config,
            0,
            input_values,
            device=device,
            dtype=dtype,
        )
        module = build_layer_from_fixture(fixture, config, 0, device=device, dtype=dtype)
        library, source = build_checked_library(Path(args.out).with_suffix(".so"))
        if source["source_rewriting"] is not False:
            raise DenseBfloat16ProbeError("dense gate must compile checked-in source unchanged")
        lib = ctypes.CDLL(str(library))
        configure_profile_library(lib)
        provider = DenseMatrixProvider(0, module)
        candidate = run_candidate(lib, fixture, cfg, 0, input_values, provider)

        output_metrics = []
        first_output = None
        for position in range(8):
            key = f"token.{position}.layer.0.output"
            if key not in reference:
                raise DenseBfloat16ProbeError(f"official trace omitted {key}")
            value = metrics(reference[key], candidate["outputs"][position])
            output_metrics.append(value)
            if first_output is None and value["bfloat16_exact_fraction"] != 1.0:
                first_output = position

        final_position = 7
        stage_metrics: dict[str, dict[str, Any]] = {}
        for point in FINAL_POINTS:
            key = f"token.{final_position}.layer.0.{point}"
            if key not in reference:
                raise DenseBfloat16ProbeError(f"official trace omitted {key}")
            values = candidate["collector"].values
            if key not in values:
                raise DenseBfloat16ProbeError(f"candidate trace omitted {key}")
            stage_metrics[point] = metrics(
                reference[key], torch.tensor(values[key], dtype=torch.float32)
            )
        first_stage = first_nonexact(stage_metrics)
        result = {
            "format": "inkling-checked-in-dense-bfloat16-profile",
            "version": 1,
            "source": source,
            "execution_profile": {
                "torch": str(torch.__version__),
                "onednn_max_cpu_isa": __import__("os").environ.get("ONEDNN_MAX_CPU_ISA"),
            },
            "decision": {
                "classification": (
                    "checked_in_dense_bf16_stateful_exact"
                    if first_output is None and first_stage is None
                    else "checked_in_dense_bf16_stateful_mismatch"
                ),
                "first_nonexact_output_position": first_output,
                "first_nonexact_final_token_stage": first_stage,
            },
            "outputs": output_metrics,
            "final_token_stages": stage_metrics,
            "backend_calls": candidate["backend_calls"],
        }
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (
        DenseBfloat16ProbeError,
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
        NativeBfloat16BackendError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())