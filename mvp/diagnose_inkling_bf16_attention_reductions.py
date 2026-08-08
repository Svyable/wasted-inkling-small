#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Determine whether Inkling attention needs native BF16 reductions.

The attention arithmetic lattice proves that Q/K head normalization, QK score
matmul, relative-bias matmul, and probability/value matmul are all required.
This probe asks the implementation question left by that result: can each
matmul be reproduced by a portable scalar accumulation with explicit BF16
operand/result boundaries, or is the pinned native BF16 kernel itself required?

Two scalar candidates are compared against PyTorch 5.14.1's CPU BF16 matmul:

* left-to-right float32 multiplication and accumulation;
* left-to-right double multiplication and accumulation.

All operands and completed outputs are rounded to BF16 at the same semantic
boundaries as the official implementation. The eight-token fixture does not
cross the configured global log-scaling floor, so this probe classifies the
reduction kernels only; long-context tau scaling remains a separate gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch

from diagnose_inkling_bf16_attention_lattice import (
    AttentionArithmetic,
    AttentionInputs,
    AttentionLatticeError,
    bf16,
    f32,
    official_value_sum,
    tensor_metrics,
    valid_keys,
)
from diagnose_inkling_pre_router import (
    PreRouterDiagnosisError,
    _capture_official_pre_router,
)
from discover_inkling_router_experts import input_sha256
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    verify_config_binding,
)
from inkling_release_config import build_transformers_text_config


class AttentionReductionError(RuntimeError):
    """An attention reduction cannot be classified without exact geometry."""


REDUCTIONS = ("float32_left_to_right", "double_left_to_right")


def bf16_scalar(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32).to(torch.bfloat16).float())


def reduce_dot(
    left: torch.Tensor,
    right: torch.Tensor,
    mode: str,
) -> float:
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise AttentionReductionError(
            f"invalid reduction geometry {tuple(left.shape)} x {tuple(right.shape)}"
        )
    if mode == "float32_left_to_right":
        total = f32(0.0)
        for lhs, rhs in zip(left.tolist(), right.tolist()):
            product = f32(float(lhs) * float(rhs))
            total = f32(total + product)
        return total
    if mode == "double_left_to_right":
        total = 0.0
        for lhs, rhs in zip(left.tolist(), right.tolist()):
            total += float(lhs) * float(rhs)
        return total
    raise AttentionReductionError(f"unknown reduction mode {mode!r}")


def completed_bf16_dot(
    left: torch.Tensor,
    right: torch.Tensor,
    mode: str,
) -> float:
    """Round completed scalar reduction to BF16, as a BF16 matmul output."""
    return bf16_scalar(reduce_dot(bf16(left), bf16(right), mode))


def bf16_product(left: float, right: float) -> float:
    """Multiply BF16 operands and retain a BF16 result."""
    lhs = torch.tensor(left, dtype=torch.bfloat16)
    rhs = torch.tensor(right, dtype=torch.bfloat16)
    return float((lhs * rhs).to(torch.bfloat16).float())


def software_qk_scores(
    arithmetic: AttentionArithmetic,
    position: int,
    head: int,
    keys: list[int],
    mode: str,
) -> torch.Tensor:
    q = arithmetic.q_bf16[position, head]
    tau = arithmetic._tau_official(position)
    if tau != 1.0:
        q = bf16(q.float() * tau)
    kv_head = head // arithmetic.groups
    values = []
    scaling = bf16_scalar(1.0 / arithmetic.head_dim)
    for key in keys:
        dot = completed_bf16_dot(q, arithmetic.k_bf16[key, kv_head], mode)
        values.append(bf16_product(dot, scaling))
    return torch.tensor(values, dtype=torch.float32)


def software_relative_scores(
    arithmetic: AttentionArithmetic,
    position: int,
    head: int,
    keys: list[int],
    mode: str,
) -> torch.Tensor:
    state = arithmetic.r_source[position, head]
    projection = bf16(arithmetic.rel_projection)
    tau = arithmetic._tau_official(position)
    tau_bf16 = bf16_scalar(tau)
    values = []
    for key in keys:
        distance = position - key
        if distance < 0 or distance >= arithmetic.extent:
            value = 0.0
        else:
            value = completed_bf16_dot(
                state,
                projection[:, distance],
                mode,
            )
        if tau != 1.0:
            value = bf16_product(value, tau_bf16)
        values.append(value)
    return torch.tensor(values, dtype=torch.float32)


def software_value_sum(
    probabilities: torch.Tensor,
    values: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if probabilities.ndim != 1 or values.ndim != 2:
        raise AttentionReductionError("invalid value reduction rank")
    if probabilities.numel() != values.shape[0]:
        raise AttentionReductionError("probability/value row mismatch")
    probability = bf16(probabilities)
    source = bf16(values)
    output = [
        completed_bf16_dot(probability, source[:, column], mode)
        for column in range(source.shape[1])
    ]
    return torch.tensor(output, dtype=torch.float32)


def row_mismatches(
    reference_rows: list[tuple[int, int, torch.Tensor]],
    candidate_rows: list[tuple[int, int, torch.Tensor]],
) -> dict[str, Any]:
    if len(reference_rows) != len(candidate_rows):
        raise AttentionReductionError("reference/candidate row counts differ")
    positions: set[int] = set()
    rows = []
    for expected, actual in zip(reference_rows, candidate_rows):
        ep, eh, reference = expected
        ap, ah, candidate = actual
        if (ep, eh) != (ap, ah):
            raise AttentionReductionError("reference/candidate row identity differs")
        if not torch.equal(bf16(reference), bf16(candidate)):
            positions.add(ep)
            rows.append({"position": ep, "head": eh})
    return {
        "mismatch_positions": sorted(positions),
        "mismatch_rows": rows,
        "mismatch_row_count": len(rows),
    }


def compare_rows(
    reference_rows: list[tuple[int, int, torch.Tensor]],
    candidate_rows: list[tuple[int, int, torch.Tensor]],
) -> dict[str, Any]:
    reference = torch.cat([row for _, _, row in reference_rows])
    candidate = torch.cat([row for _, _, row in candidate_rows])
    result = tensor_metrics(reference, candidate)
    result.update(row_mismatches(reference_rows, candidate_rows))
    return result


def classify_operation(variants: dict[str, dict[str, Any]]) -> str:
    for mode in REDUCTIONS:
        if variants[mode]["quantized_exact_fraction"] == 1.0:
            return f"{mode}_matches_native_bf16"
    return "native_bf16_kernel_required"


def analyze_operation(
    reference_rows: list[tuple[int, int, torch.Tensor]],
    builder: Callable[[str], list[tuple[int, int, torch.Tensor]]],
) -> dict[str, Any]:
    variants = {
        mode: compare_rows(reference_rows, builder(mode))
        for mode in REDUCTIONS
    }
    return {
        "classification": classify_operation(variants),
        "variants": variants,
    }


def analyze_layer(
    fixture: Any,
    config: Any,
    layer: int,
    inputs: torch.Tensor,
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
    values = AttentionInputs(
        q_candidate=official["q_proj"],
        k_candidate=official["k_sconv"],
        v_candidate=official["v_sconv"],
        relative_candidate=official["relative_proj_input"],
        q_official=official["q_proj"],
        k_official=official["k_sconv"],
        v_official=official["v_sconv"],
        relative_official=official["relative_proj_input"],
    )
    arithmetic = AttentionArithmetic(module, config, layer, values)
    all_official = arithmetic.run(frozenset((
        "head_norm", "qk_score", "relative_bias", "softmax", "value"
    )))
    anchor = tensor_metrics(
        official["attention_out"],
        all_official["attention_out"],
    )
    if anchor["quantized_exact_fraction"] != 1.0:
        raise AttentionReductionError(
            f"layer {layer} all-official attention reconstruction drifted"
        )

    qk_reference: list[tuple[int, int, torch.Tensor]] = []
    relative_reference: list[tuple[int, int, torch.Tensor]] = []
    value_reference: list[tuple[int, int, torch.Tensor]] = []
    key_rows: dict[int, list[int]] = {}
    for position in range(arithmetic.tokens):
        keys = list(valid_keys(
            position,
            sliding=arithmetic.sliding,
            window=arithmetic.window,
        ))
        key_rows[position] = keys
        for head in range(arithmetic.heads):
            qk_reference.append((
                position,
                head,
                arithmetic._qk_scores(
                    position,
                    head,
                    keys,
                    official_head=True,
                    official_qk=True,
                ),
            ))
            relative_reference.append((
                position,
                head,
                arithmetic._relative_scores(
                    position,
                    head,
                    keys,
                    official=True,
                ),
            ))
            kv_head = head // arithmetic.groups
            probability = all_official["probabilities"][position, head, keys]
            value_reference.append((
                position,
                head,
                official_value_sum(
                    probability,
                    arithmetic.v_source[keys, kv_head],
                ),
            ))

    qk = analyze_operation(
        qk_reference,
        lambda mode: [
            (
                position,
                head,
                software_qk_scores(
                    arithmetic,
                    position,
                    head,
                    key_rows[position],
                    mode,
                ),
            )
            for position in range(arithmetic.tokens)
            for head in range(arithmetic.heads)
        ],
    )
    relative = analyze_operation(
        relative_reference,
        lambda mode: [
            (
                position,
                head,
                software_relative_scores(
                    arithmetic,
                    position,
                    head,
                    key_rows[position],
                    mode,
                ),
            )
            for position in range(arithmetic.tokens)
            for head in range(arithmetic.heads)
        ],
    )
    value = analyze_operation(
        value_reference,
        lambda mode: [
            (
                position,
                head,
                software_value_sum(
                    all_official["probabilities"][
                        position,
                        head,
                        key_rows[position],
                    ],
                    arithmetic.v_source[
                        key_rows[position],
                        head // arithmetic.groups,
                    ],
                    mode,
                ),
            )
            for position in range(arithmetic.tokens)
            for head in range(arithmetic.heads)
        ],
    )
    operations = {
        "qk_score": qk,
        "relative_bias": relative,
        "value": value,
    }
    native_required = sorted(
        name
        for name, result in operations.items()
        if result["classification"] == "native_bf16_kernel_required"
    )
    portable = sorted(set(operations).difference(native_required))
    return {
        "geometry": {
            "tokens": arithmetic.tokens,
            "heads": arithmetic.heads,
            "kv_heads": arithmetic.kv_heads,
            "head_dim": arithmetic.head_dim,
            "d_rel": arithmetic.d_rel,
            "maximum_kv_length": max(len(value) for value in key_rows.values()),
            "sliding": arithmetic.sliding,
            "log_scaling_floor": arithmetic.floor,
            "log_scaling_exercised": any(
                arithmetic._tau_official(position) != 1.0
                for position in range(arithmetic.tokens)
            ),
        },
        "anchor": anchor,
        "operations": operations,
        "decision": {
            "native_bf16_required": native_required,
            "portable_scalar_reduction": portable,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
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
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        inputs = torch.tensor(input_values, dtype=torch.float32)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise AttentionReductionError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise AttentionReductionError("the pinned reduction probe requires CPU")
        analyses = {
            str(layer): analyze_layer(
                fixture,
                config,
                layer,
                inputs,
                device=device,
                dtype=dtype,
            )
            for layer in layers
        }
        result = {
            "format": "inkling-bfloat16-attention-reduction-probe",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "reduction_candidates": list(REDUCTIONS),
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
        AttentionLatticeError,
        AttentionReductionError,
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
