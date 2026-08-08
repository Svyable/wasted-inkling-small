#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Classify every BF16 arithmetic boundary inside Inkling attention.

The native matrix backend already proves exact Q/K/V/R projections and exact
BF16-visible K/V short convolution. This diagnostic therefore starts at those
captured tensors and compares the attention core as an arithmetic lattice:

* per-head Q/K RMSNorm;
* QK score matmul and global query scaling;
* relative-position projection and global bias scaling;
* FP32 softmax followed by BF16 conversion;
* probability-by-value matmul.

It first proves that an all-C reconstruction matches the actual C trace and an
all-official reconstruction matches the pinned Transformers trace at BF16 bit
precision. Counterfactual conclusions are rejected unless both anchors pass.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from diagnose_inkling_native_bf16_backend import (
    NativeBfloat16BackendError,
    NativeMatrixProvider,
    build_rms_probe_library,
    capture_c_with_backend,
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
from inkling_layer_parity import LayerParityError, configure_library
from inkling_release_config import build_transformers_text_config


class AttentionLatticeError(RuntimeError):
    """The attention arithmetic lattice cannot be classified without guessing."""


BOUNDARIES = ("head_norm", "qk_score", "relative_bias", "softmax", "value")
CUMULATIVE = (
    ("c_baseline", frozenset()),
    ("head_norm_bf16", frozenset(("head_norm",))),
    ("qk_score_bf16", frozenset(("head_norm", "qk_score"))),
    (
        "relative_bias_bf16",
        frozenset(("head_norm", "qk_score", "relative_bias")),
    ),
    (
        "softmax_bf16",
        frozenset(("head_norm", "qk_score", "relative_bias", "softmax")),
    ),
    ("all_official", frozenset(BOUNDARIES)),
)


def f32(value: float) -> float:
    """Round one scalar exactly as a C float assignment."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def bf16(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(torch.bfloat16).float().cpu().contiguous()


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise AttentionLatticeError(
            f"tensor shape mismatch {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    ref = bf16(reference)
    cand = bf16(candidate)
    count = ref.numel()
    if count == 0:
        raise AttentionLatticeError("cannot compare empty tensors")
    delta = (ref - cand).abs().float()
    exact = ref.eq(cand)
    return {
        "count": count,
        "quantized_exact": int(exact.sum().item()),
        "quantized_exact_fraction": float(exact.float().mean().item()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def _c_head_norm_row(
    values: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if values.ndim != 1 or weight.ndim != 1 or values.shape != weight.shape:
        raise AttentionLatticeError("invalid C head RMSNorm geometry")
    raw = [float(item) for item in values.tolist()]
    scales = [float(item) for item in weight.tolist()]
    ss = 0.0
    for item in raw:
        ss += item * item
    mean = f32(ss / len(raw))
    root = f32(math.sqrt(f32(mean + f32(eps))))
    scale = f32(1.0 / root)
    out = [f32(f32(item * scale) * scales[index]) for index, item in enumerate(raw)]
    return torch.tensor(out, dtype=torch.float32)


def c_head_norm(
    values: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if values.ndim != 3 or weight.ndim != 1 or values.shape[-1] != weight.numel():
        raise AttentionLatticeError("invalid batched C head RMSNorm geometry")
    rows = []
    for token in range(values.shape[0]):
        heads = []
        for head in range(values.shape[1]):
            heads.append(_c_head_norm_row(values[token, head], weight, eps))
        rows.append(torch.stack(heads))
    return torch.stack(rows)


def c_tau(position: int, floor: int, alpha: float) -> float:
    if position < 0 or floor <= 0:
        raise AttentionLatticeError("invalid global log-scaling geometry")
    ratio = f32(f32(position + 1) / f32(floor))
    logarithm = f32(math.log(ratio if ratio > 1.0 else 1.0))
    return f32(1.0 + f32(f32(alpha) * logarithm))


def c_dot(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise AttentionLatticeError("invalid C dot-product geometry")
    total = f32(0.0)
    for lhs, rhs in zip(left.tolist(), right.tolist()):
        total = f32(total + f32(float(lhs) * float(rhs)))
    return total


def c_relative(
    state: torch.Tensor,
    projection: torch.Tensor,
    distance: int,
) -> float:
    if state.ndim != 1 or projection.ndim != 2 or state.numel() != projection.shape[0]:
        raise AttentionLatticeError("invalid C relative-bias geometry")
    if distance < 0 or distance >= projection.shape[1]:
        return 0.0
    total = f32(0.0)
    for index, item in enumerate(state.tolist()):
        total = f32(
            total + f32(float(item) * float(projection[index, distance]))
        )
    return total


def c_softmax(scores: Iterable[float]) -> torch.Tensor:
    row = [f32(value) for value in scores]
    if not row:
        raise AttentionLatticeError("cannot softmax an empty score row")
    maximum = max(row)
    exponentials = [f32(math.exp(f32(value - maximum))) for value in row]
    denominator = sum(float(value) for value in exponentials)
    if not denominator > 0.0 or not math.isfinite(denominator):
        raise AttentionLatticeError("invalid C softmax denominator")
    inverse = f32(1.0 / denominator)
    return torch.tensor(
        [f32(value * inverse) for value in exponentials],
        dtype=torch.float32,
    )


def c_value_sum(probabilities: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 1 or values.ndim != 2 or probabilities.numel() != values.shape[0]:
        raise AttentionLatticeError("invalid C value-aggregation geometry")
    output = []
    for column in range(values.shape[1]):
        total = f32(0.0)
        for row in range(values.shape[0]):
            total = f32(
                total
                + f32(float(probabilities[row]) * float(values[row, column]))
            )
        output.append(total)
    return torch.tensor(output, dtype=torch.float32)


def official_softmax(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 1 or not scores.numel():
        raise AttentionLatticeError("invalid official softmax geometry")
    with torch.no_grad():
        return (
            torch.softmax(scores.to(torch.bfloat16), dim=-1, dtype=torch.float32)
            .to(torch.bfloat16)
            .float()
            .cpu()
        )


def official_value_sum(probabilities: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 1 or values.ndim != 2 or probabilities.numel() != values.shape[0]:
        raise AttentionLatticeError("invalid official value-aggregation geometry")
    with torch.no_grad():
        return (
            torch.matmul(
                probabilities.to(torch.bfloat16).reshape(1, -1),
                values.to(torch.bfloat16),
            )[0]
            .to(torch.bfloat16)
            .float()
            .cpu()
        )


def valid_keys(position: int, *, sliding: bool, window: int) -> range:
    if position < 0:
        raise AttentionLatticeError("position must be nonnegative")
    begin = max(0, position - window + 1) if sliding else 0
    return range(begin, position + 1)


@dataclass(frozen=True)
class AttentionInputs:
    q_candidate: torch.Tensor
    k_candidate: torch.Tensor
    v_candidate: torch.Tensor
    relative_candidate: torch.Tensor
    q_official: torch.Tensor
    k_official: torch.Tensor
    v_official: torch.Tensor
    relative_official: torch.Tensor


class AttentionArithmetic:
    def __init__(
        self,
        module: Any,
        config: Any,
        layer: int,
        values: AttentionInputs,
    ) -> None:
        self.module = module
        self.config = config
        self.layer = int(layer)
        self.attn = module.self_attn
        self.tokens = int(values.q_candidate.shape[0])
        self.heads = int(self.attn.num_heads)
        self.kv_heads = int(self.attn.num_key_value_heads)
        self.head_dim = int(self.attn.head_dim)
        self.groups = self.heads // self.kv_heads
        self.d_rel = int(config.d_rel)
        self.sliding = bool(self.attn.is_sliding)
        self.window = int(self.attn.sliding_window or self.tokens)
        self.extent = int(self.attn.rel_extent)
        self.eps = float(config.rms_norm_eps)
        self.floor = int(getattr(config, "log_scaling_n_floor", 0) or 0)
        self.alpha = float(getattr(config, "log_scaling_alpha", 0.0) or 0.0)
        self.values = values
        self._validate_geometry()

        self.q_candidate = values.q_candidate.reshape(
            self.tokens, self.heads, self.head_dim
        ).float().cpu()
        self.k_candidate = values.k_candidate.reshape(
            self.tokens, self.kv_heads, self.head_dim
        ).float().cpu()
        self.v_candidate = values.v_candidate.reshape(
            self.tokens, self.kv_heads, self.head_dim
        ).float().cpu()
        self.r_candidate = values.relative_candidate.reshape(
            self.tokens, self.heads, self.d_rel
        ).float().cpu()

        self.q_source = bf16(values.q_candidate).reshape(
            self.tokens, self.heads, self.head_dim
        )
        self.k_source = bf16(values.k_candidate).reshape(
            self.tokens, self.kv_heads, self.head_dim
        )
        self.v_source = bf16(values.v_candidate).reshape(
            self.tokens, self.kv_heads, self.head_dim
        )
        self.r_source = bf16(values.relative_candidate).reshape(
            self.tokens, self.heads, self.d_rel
        )

        self.q_weight = self.attn.q_norm.weight.detach().float().cpu()
        self.k_weight = self.attn.k_norm.weight.detach().float().cpu()
        self.rel_projection = self.attn.rel_logits_proj.proj.detach().float().cpu()

        self.q_c = c_head_norm(self.q_candidate, self.q_weight, self.eps)
        self.k_c = c_head_norm(self.k_candidate, self.k_weight, self.eps)
        with torch.no_grad():
            self.q_bf16 = (
                self.attn.q_norm(
                    self.q_source.to(
                        device=self.q_weight.device,
                        dtype=torch.bfloat16,
                    )
                )
                .detach()
                .float()
                .cpu()
            )
            self.k_bf16 = (
                self.attn.k_norm(
                    self.k_source.to(
                        device=self.k_weight.device,
                        dtype=torch.bfloat16,
                    )
                )
                .detach()
                .float()
                .cpu()
            )
            positions = torch.arange(self.tokens, device=self.r_source.device)
            self.relative_bf16 = (
                self.attn.rel_logits_proj(
                    self.r_source.reshape(
                        1, self.tokens, self.heads, self.d_rel
                    ).to(torch.bfloat16),
                    positions,
                    positions,
                )[0]
                .detach()
                .float()
                .cpu()
            )

    def _validate_geometry(self) -> None:
        if self.heads <= 0 or self.kv_heads <= 0 or self.head_dim <= 0:
            raise AttentionLatticeError("invalid attention geometry")
        if self.heads % self.kv_heads:
            raise AttentionLatticeError("attention heads are not grouped evenly")
        expected = {
            "q_candidate": (self.tokens, self.heads * self.head_dim),
            "k_candidate": (self.tokens, self.kv_heads * self.head_dim),
            "v_candidate": (self.tokens, self.kv_heads * self.head_dim),
            "relative_candidate": (self.tokens, self.heads * self.d_rel),
            "q_official": (self.tokens, self.heads * self.head_dim),
            "k_official": (self.tokens, self.kv_heads * self.head_dim),
            "v_official": (self.tokens, self.kv_heads * self.head_dim),
            "relative_official": (self.tokens, self.heads * self.d_rel),
        }
        for name, shape in expected.items():
            value = getattr(self.values, name)
            if tuple(value.shape) != shape:
                raise AttentionLatticeError(
                    f"{name} has {tuple(value.shape)}; expected {shape}"
                )

    def _tau_c(self, position: int) -> float:
        if self.sliding or not self.floor:
            return 1.0
        return c_tau(position, self.floor, self.alpha)

    def _tau_official(self, position: int) -> float:
        if self.sliding or not self.floor:
            return 1.0
        ratio = max((position + 1) / self.floor, 1.0)
        return 1.0 + self.alpha * math.log(ratio)

    def _head_values(self, official: bool):
        return (self.q_bf16, self.k_bf16) if official else (self.q_c, self.k_c)

    def _qk_scores(
        self,
        position: int,
        head: int,
        keys: list[int],
        *,
        official_head: bool,
        official_qk: bool,
    ) -> torch.Tensor:
        q_values, k_values = self._head_values(official_head)
        kv_head = head // self.groups
        q = q_values[position, head]
        key_rows = k_values[keys, kv_head]
        if official_qk:
            q_native = q.to(torch.bfloat16)
            tau = self._tau_official(position)
            if tau != 1.0:
                q_native = (q_native.float() * tau).to(torch.bfloat16)
            with torch.no_grad():
                score = torch.matmul(
                    q_native.reshape(1, -1),
                    key_rows.to(torch.bfloat16).transpose(0, 1),
                )[0]
                score = score * (1.0 / self.head_dim)
            return score.to(torch.bfloat16).float().cpu()
        tau = self._tau_c(position)
        dot_scale = f32(tau / self.head_dim)
        return torch.tensor(
            [f32(c_dot(q, key_rows[index]) * dot_scale) for index in range(len(keys))],
            dtype=torch.float32,
        )

    def _relative_scores(
        self,
        position: int,
        head: int,
        keys: list[int],
        *,
        official: bool,
    ) -> torch.Tensor:
        if official:
            row = self.relative_bf16[head, position, keys]
            tau = self._tau_official(position)
            if tau != 1.0:
                row = (row.float() * tau).to(torch.bfloat16).float()
            return row.cpu()
        tau = self._tau_c(position)
        state = self.r_candidate[position, head]
        return torch.tensor(
            [
                f32(
                    tau
                    * c_relative(
                        state,
                        self.rel_projection,
                        position - key_position,
                    )
                )
                for key_position in keys
            ],
            dtype=torch.float32,
        )

    def run(self, official_boundaries: frozenset[str]) -> dict[str, torch.Tensor]:
        unknown = sorted(set(official_boundaries).difference(BOUNDARIES))
        if unknown:
            raise AttentionLatticeError(
                "unknown attention boundaries: " + ", ".join(unknown)
            )
        q_official = "head_norm" in official_boundaries
        qk_official = "qk_score" in official_boundaries
        relative_official = "relative_bias" in official_boundaries
        softmax_is_official = "softmax" in official_boundaries
        value_official = "value" in official_boundaries
        q_norm, k_norm = self._head_values(q_official)

        scores = torch.full(
            (self.tokens, self.heads, self.tokens),
            float("nan"),
            dtype=torch.float32,
        )
        probabilities = torch.full_like(scores, float("nan"))
        outputs = []
        for position in range(self.tokens):
            keys = list(
                valid_keys(
                    position,
                    sliding=self.sliding,
                    window=self.window,
                )
            )
            head_outputs = []
            for head in range(self.heads):
                qk = self._qk_scores(
                    position,
                    head,
                    keys,
                    official_head=q_official,
                    official_qk=qk_official,
                )
                relative = self._relative_scores(
                    position,
                    head,
                    keys,
                    official=relative_official,
                )
                if qk_official and relative_official:
                    combined = (qk.to(torch.bfloat16) + relative.to(torch.bfloat16))
                    combined = combined.to(torch.bfloat16).float()
                else:
                    combined = torch.tensor(
                        [f32(float(a) + float(b)) for a, b in zip(qk, relative)],
                        dtype=torch.float32,
                    )
                probability = (
                    official_softmax(combined)
                    if softmax_is_official
                    else c_softmax(combined.tolist())
                )
                kv_head = head // self.groups
                value_rows = (
                    self.v_source[keys, kv_head]
                    if value_official
                    else self.v_candidate[keys, kv_head]
                )
                output = (
                    official_value_sum(probability, value_rows)
                    if value_official
                    else c_value_sum(probability, value_rows)
                )
                scores[position, head, keys] = combined
                probabilities[position, head, keys] = probability
                head_outputs.append(output)
            outputs.append(torch.stack(head_outputs).reshape(-1))
        return {
            "q_norm": q_norm,
            "k_norm": k_norm,
            "scores": scores,
            "probabilities": probabilities,
            "attention_out": torch.stack(outputs),
        }

    def valid_values(self, tensor: torch.Tensor) -> torch.Tensor:
        rows = []
        for position in range(self.tokens):
            keys = list(
                valid_keys(
                    position,
                    sliding=self.sliding,
                    window=self.window,
                )
            )
            rows.append(tensor[position, :, keys].reshape(-1))
        return torch.cat(rows)


def classify_lattice(
    cumulative: dict[str, dict[str, Any]],
    ablations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    first_exact = None
    for name, _boundaries in CUMULATIVE:
        metric = cumulative[name]["attention_out"]
        if metric["quantized_exact_fraction"] == 1.0:
            first_exact = name
            break
    necessary = sorted(
        boundary
        for boundary, values in ablations.items()
        if values["attention_out"]["quantized_exact_fraction"] < 1.0
    )
    if first_exact is None:
        classification = "official_reconstruction_failed"
    elif first_exact == "c_baseline":
        classification = "c_attention_already_exact"
    elif first_exact == "all_official" and necessary:
        classification = "multiple_attention_boundaries_are_required"
    else:
        classification = "cumulative_attention_boundary_identified"
    return {
        "classification": classification,
        "first_exact_cumulative_variant": first_exact,
        "necessary_boundaries_by_leave_one_c": necessary,
    }


def analyze_layer(
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
    candidate, _, _ = capture_c_with_backend(
        lib,
        fixture,
        c_config,
        layer,
        input_values,
        provider,
    )
    values = AttentionInputs(
        q_candidate=candidate["q_proj"],
        k_candidate=candidate["k_sconv"],
        v_candidate=candidate["v_sconv"],
        relative_candidate=candidate["relative_proj_input"],
        q_official=official["q_proj"],
        k_official=official["k_sconv"],
        v_official=official["v_sconv"],
        relative_official=official["relative_proj_input"],
    )
    arithmetic = AttentionArithmetic(module, config, layer, values)

    variants = {
        name: arithmetic.run(boundaries)
        for name, boundaries in CUMULATIVE
    }
    full = variants["all_official"]
    baseline = variants["c_baseline"]
    official_anchor = tensor_metrics(
        official["attention_out"], full["attention_out"]
    )
    c_anchor = tensor_metrics(
        candidate["attention_out"], baseline["attention_out"]
    )
    if official_anchor["quantized_exact_fraction"] != 1.0:
        raise AttentionLatticeError(
            f"layer {layer} official arithmetic reconstruction is not BF16-exact"
        )
    if c_anchor["quantized_exact_fraction"] != 1.0:
        raise AttentionLatticeError(
            f"layer {layer} C arithmetic reconstruction is not BF16-exact"
        )

    cumulative_metrics: dict[str, dict[str, Any]] = {}
    for name, tensors in variants.items():
        cumulative_metrics[name] = {
            "q_norm": tensor_metrics(full["q_norm"], tensors["q_norm"]),
            "k_norm": tensor_metrics(full["k_norm"], tensors["k_norm"]),
            "scores": tensor_metrics(
                arithmetic.valid_values(full["scores"]),
                arithmetic.valid_values(tensors["scores"]),
            ),
            "probabilities": tensor_metrics(
                arithmetic.valid_values(full["probabilities"]),
                arithmetic.valid_values(tensors["probabilities"]),
            ),
            "attention_out": tensor_metrics(
                official["attention_out"], tensors["attention_out"]
            ),
        }

    ablation_metrics: dict[str, dict[str, Any]] = {}
    all_boundaries = frozenset(BOUNDARIES)
    for boundary in BOUNDARIES:
        tensors = arithmetic.run(all_boundaries.difference((boundary,)))
        ablation_metrics[boundary] = {
            "scores": tensor_metrics(
                arithmetic.valid_values(full["scores"]),
                arithmetic.valid_values(tensors["scores"]),
            ),
            "probabilities": tensor_metrics(
                arithmetic.valid_values(full["probabilities"]),
                arithmetic.valid_values(tensors["probabilities"]),
            ),
            "attention_out": tensor_metrics(
                official["attention_out"], tensors["attention_out"]
            ),
        }

    return {
        "geometry": {
            "tokens": arithmetic.tokens,
            "heads": arithmetic.heads,
            "kv_heads": arithmetic.kv_heads,
            "head_dim": arithmetic.head_dim,
            "d_rel": arithmetic.d_rel,
            "sliding": arithmetic.sliding,
            "window": arithmetic.window,
            "relative_extent": arithmetic.extent,
        },
        "anchors": {
            "official_reconstruction": official_anchor,
            "c_reconstruction": c_anchor,
            "backend_calls": provider.calls,
        },
        "cumulative": cumulative_metrics,
        "leave_one_c": ablation_metrics,
        "decision": classify_lattice(cumulative_metrics, ablation_metrics),
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
            raise AttentionLatticeError("--layers must be nonempty")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise AttentionLatticeError("the C reconstruction requires --device cpu")
        library, source = build_rms_probe_library(Path(args.out).with_suffix(".so"))
        if not source["production_source_unchanged"]:
            raise AttentionLatticeError("production layer source changed")
        lib = ctypes.CDLL(str(library))
        configure_library(lib)
        analyses = {
            str(layer): analyze_layer(
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
            "format": "inkling-bfloat16-attention-arithmetic-lattice",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": int(inputs.shape[0]),
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "source": source,
            "boundaries": list(BOUNDARIES),
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
        AttentionLatticeError,
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
