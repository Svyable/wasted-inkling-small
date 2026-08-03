#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run official Transformers Inkling decoder layers from bounded fixtures.

This is the missing official side of G0. It deliberately lives outside the
published WASTE patch until it has run against an official fixture: the bundle's
source/tree invariant remains untouched while this evidence harness is tested.

A sparse layer is instantiated on the meta device, its full expert bank is
removed before materialization, and only fixture-selected experts are loaded.
If routing selects an expert absent from the fixture the run fails with the
missing expert IDs instead of silently substituting zeros.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
INKLING_TOOLS = REPO / "inkling" / "tools"
if str(INKLING_TOOLS) not in sys.path:
    sys.path.insert(0, str(INKLING_TOOLS))


class FixtureReferenceError(RuntimeError):
    """A bounded reference run cannot be performed without guessing."""


def _repo_modules() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Import repository tools lazily so adapter unit tests need only torch."""
    try:
        from inkling_fixture import FixtureError, load_fixture
        from inkling_parity import write_activation_archive
        from inkling_plan import (
            layer_attention_names,
            layer_dense_mlp_names,
            layer_sparse_mlp_names,
        )
        from inkling_reference import register_reference_hooks
    except ImportError as exc:
        raise FixtureReferenceError(
            f"cannot import repository Inkling tools from {INKLING_TOOLS}: {exc}"
        ) from exc
    return (
        FixtureError,
        load_fixture,
        write_activation_archive,
        layer_attention_names,
        layer_dense_mlp_names,
        layer_sparse_mlp_names,
        register_reference_hooks,
    )


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _source_tensor(fixture: Any, entry: Any) -> torch.Tensor:
    """Read one fixture entry in its published dtype, preserving source bits."""
    dtype = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
    }.get(entry.dtype)
    if dtype is None:
        raise FixtureReferenceError(f"unsupported fixture dtype {entry.dtype!r}")
    payload = bytearray(fixture.raw(entry.name, entry.axis0))
    return torch.frombuffer(payload, dtype=dtype).reshape(entry.shape).clone()


def split_provider_gate_up(fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split provider row-interleaved [gate0, up0, gate1, up1, ...] rows."""
    if fused.ndim < 2 or fused.shape[-2] % 2:
        raise FixtureReferenceError(
            "provider gate/up tensor must have an even row dimension, "
            f"got {tuple(fused.shape)}"
        )
    return fused[..., 0::2, :].contiguous(), fused[..., 1::2, :].contiguous()


def official_gate_up(fused: torch.Tensor) -> torch.Tensor:
    """Convert provider row interleaving to Transformers [all gate; all up]."""
    gate, up = split_provider_gate_up(fused)
    return torch.cat((gate, up), dim=-2).contiguous()


def verify_config_binding(fixture: Any, config_path: Path | str) -> dict[str, Any]:
    """Read config JSON and prove it is the fixture's extraction source."""
    path = Path(config_path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureReferenceError(f"cannot read model config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FixtureReferenceError("model config is not a JSON object")
    expected = fixture.source.get("config_sha256")
    actual = hashlib.sha256(raw).hexdigest()
    if not isinstance(expected, str) or actual != expected:
        raise FixtureReferenceError(
            f"config hash mismatch: fixture expects {expected!r}, {path} is {actual}"
        )
    return value


class FixtureSparseExperts(nn.Module):
    """Transformers-compatible expert bank containing only selected experts."""

    def __init__(
        self,
        *,
        num_experts: int,
        gate_up: dict[int, torch.Tensor],
        down: dict[int, torch.Tensor],
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise FixtureReferenceError("num_experts must be positive")
        if set(gate_up) != set(down) or not gate_up:
            raise FixtureReferenceError(
                "gate/up and down expert selections must match and be nonempty"
            )
        if any(i < 0 or i >= num_experts for i in gate_up):
            raise FixtureReferenceError(
                "fixture expert id is outside the configured expert bank"
            )
        self.num_experts = int(num_experts)
        self.activation = activation
        self.gate_up = nn.ParameterDict(
            {
                str(i): nn.Parameter(t.detach().contiguous(), requires_grad=False)
                for i, t in gate_up.items()
            }
        )
        self.down = nn.ParameterDict(
            {
                str(i): nn.Parameter(t.detach().contiguous(), requires_grad=False)
                for i, t in down.items()
            }
        )

    @property
    def expert_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(key) for key in self.gate_up))

    def _activation(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation == "silu":
            return F.silu(value)
        if self.activation == "gelu":
            return F.gelu(value)
        raise FixtureReferenceError(
            f"unsupported expert activation {self.activation!r}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise FixtureReferenceError(
                "expert input must be [tokens, hidden], "
                f"got {tuple(hidden_states.shape)}"
            )
        used = {
            int(value)
            for value in torch.unique(top_k_index).detach().cpu().tolist()
        }
        missing = sorted(used - set(self.expert_ids))
        if missing:
            raise FixtureReferenceError(
                "router selected experts absent from the bounded fixture: "
                + ",".join(str(value) for value in missing)
                + "; re-extract the fixture with those expert IDs"
            )

        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()
        for item in expert_hit:
            expert = int(item[0])
            top_k_pos, token_idx = torch.where(expert_mask[expert])
            current = hidden_states[token_idx]
            gate, up = F.linear(
                current, self.gate_up[str(expert)]
            ).chunk(2, dim=-1)
            current = self._activation(gate) * up
            current = F.linear(current, self.down[str(expert)])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


class _UnmaterializedExperts(nn.Module):
    def forward(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise FixtureReferenceError("sparse expert adapter was not installed")


def _copy_parameter(
    parameters: dict[str, nn.Parameter],
    assigned: set[str],
    target: str,
    value: torch.Tensor,
) -> None:
    try:
        parameter = parameters[target]
    except KeyError:
        raise FixtureReferenceError(
            f"installed Transformers Inkling layer has no parameter {target!r}; "
            "API drift"
        ) from None
    if tuple(parameter.shape) != tuple(value.shape):
        raise FixtureReferenceError(
            f"shape mismatch for {target}: module {tuple(parameter.shape)}, "
            f"fixture {tuple(value.shape)}"
        )
    with torch.no_grad():
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    assigned.add(target)


def _fixture_value(
    fixture: Any, name: str, axis0: int | None = None
) -> torch.Tensor:
    return _source_tensor(fixture, fixture.entry(name, axis0))


def _provider_assignments(
    fixture: Any,
    layer: int,
    *,
    sparse: bool,
) -> tuple[dict[str, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Map provider-raw fixture names to official layer parameter names."""
    (
        _,
        _,
        _,
        layer_attention_names,
        layer_dense_mlp_names,
        layer_sparse_mlp_names,
        _,
    ) = _repo_modules()
    attention = layer_attention_names(layer, "provider_raw")
    values: dict[str, torch.Tensor] = {
        "input_layernorm.weight": _fixture_value(
            fixture, attention["input_norm"]
        ),
        "post_attention_layernorm.weight": _fixture_value(
            fixture, attention["post_attention_norm"]
        ),
        "self_attn.q_proj.weight": _fixture_value(fixture, attention["q"]),
        "self_attn.k_proj.weight": _fixture_value(fixture, attention["k"]),
        "self_attn.v_proj.weight": _fixture_value(fixture, attention["v"]),
        "self_attn.r_proj.weight": _fixture_value(fixture, attention["r"]),
        "self_attn.o_proj.weight": _fixture_value(fixture, attention["o"]),
        "self_attn.q_norm.weight": _fixture_value(
            fixture, attention["q_norm"]
        ),
        "self_attn.k_norm.weight": _fixture_value(
            fixture, attention["k_norm"]
        ),
        "self_attn.rel_logits_proj.proj": _fixture_value(
            fixture, attention["rel_proj"]
        ),
        "self_attn.k_sconv.conv1d.weight": _fixture_value(
            fixture, attention["k_sconv"]
        ),
        "self_attn.v_sconv.conv1d.weight": _fixture_value(
            fixture, attention["v_sconv"]
        ),
        "attn_sconv.conv1d.weight": _fixture_value(
            fixture, attention["attn_sconv"]
        ),
        "mlp_sconv.conv1d.weight": _fixture_value(
            fixture, attention["mlp_sconv"]
        ),
    }
    if not sparse:
        names = layer_dense_mlp_names(layer, "provider_raw")
        gate, up = split_provider_gate_up(
            _fixture_value(fixture, names["fused_gate_up"])
        )
        values.update(
            {
                "mlp.gate_proj.weight": gate,
                "mlp.up_proj.weight": up,
                "mlp.down_proj.weight": _fixture_value(fixture, names["down"]),
                "mlp.global_scale": _fixture_value(
                    fixture, names["global_scale"]
                ),
            }
        )
        return values, {}, {}

    names = layer_sparse_mlp_names(layer, "provider_raw")
    router = names["router"]
    shared = names["shared"]
    shared_gate, shared_up = split_provider_gate_up(
        _fixture_value(fixture, shared["fused_gate_up"])
    )
    values.update(
        {
            "mlp.gate.weight": _fixture_value(fixture, router["weight"]),
            "mlp.gate.e_score_correction_bias": _fixture_value(
                fixture, router["correction_bias"]
            ),
            "mlp.gate.global_scale": _fixture_value(
                fixture, router["global_scale"]
            ),
            "mlp.shared_experts.gate_proj": shared_gate,
            "mlp.shared_experts.up_proj": shared_up,
            "mlp.shared_experts.down_proj": _fixture_value(
                fixture, shared["down"]
            ),
        }
    )
    selected = fixture.experts.get(layer, ())
    if not selected:
        raise FixtureReferenceError(
            f"layer {layer} is sparse but the fixture declares no routed experts"
        )
    routed = names["routed"]
    gate_up: dict[int, torch.Tensor] = {}
    down: dict[int, torch.Tensor] = {}
    for expert in selected:
        gate_up[expert] = official_gate_up(
            _fixture_value(fixture, routed["fused_gate_up"], expert)
        )
        down[expert] = _fixture_value(fixture, routed["down"], expert)
    return values, gate_up, down


def _import_transformers() -> tuple[Any, Any, Any, str]:
    try:
        import transformers
        from transformers import InklingConfig, InklingTextConfig
        from transformers.cache_utils import DynamicCache
        from transformers.models.inkling.modeling_inkling import (
            InklingDecoderLayer,
        )
    except (ImportError, AttributeError) as exc:
        raise FixtureReferenceError(
            "Transformers with official InklingDecoderLayer support is required "
            "(tested API: transformers 5.14.x)"
        ) from exc
    return (
        InklingConfig,
        InklingTextConfig,
        (InklingDecoderLayer, DynamicCache),
        transformers.__version__,
    )


def _text_config(config_json: dict[str, Any]) -> Any:
    InklingConfig, InklingTextConfig, _, _ = _import_transformers()
    if isinstance(config_json.get("text_config"), dict):
        config = InklingConfig.from_dict(config_json).get_text_config()
    else:
        config = InklingTextConfig.from_dict(config_json)
    config._attn_implementation = "eager"
    return config


def _is_sparse(config: Any, layer: int) -> bool:
    kinds = getattr(config, "mlp_layer_types", None)
    if not isinstance(kinds, (list, tuple)) or layer >= len(kinds):
        raise FixtureReferenceError(
            "Inkling config has no usable mlp_layer_types"
        )
    return kinds[layer] == "sparse"


def build_layer_from_fixture(
    fixture: Any,
    config: Any,
    layer: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Instantiate one official layer without allocating the full expert bank."""
    _, _, classes, _ = _import_transformers()
    InklingDecoderLayer, _ = classes
    fixture.require_layers([layer])
    sparse = _is_sparse(config, layer)

    with torch.device("meta"):
        module = InklingDecoderLayer(config, layer)
    if sparse:
        module.mlp.experts = _UnmaterializedExperts()
    module.to_empty(device=device)
    module.to(dtype=dtype)

    # Official checkpoints keep all four short-convolution modules in FP32.
    for path in (
        "self_attn.k_sconv.conv1d",
        "self_attn.v_sconv.conv1d",
        "attn_sconv.conv1d",
        "mlp_sconv.conv1d",
    ):
        current: Any = module
        for part in path.split("."):
            current = getattr(current, part)
        current.float()

    values, gate_up, down = _provider_assignments(
        fixture, layer, sparse=sparse
    )
    parameters = dict(module.named_parameters())
    assigned: set[str] = set()
    for target, value in values.items():
        _copy_parameter(parameters, assigned, target, value)

    if sparse:
        module.mlp.experts = FixtureSparseExperts(
            num_experts=int(config.n_routed_experts),
            gate_up={
                key: value.to(device=device, dtype=dtype)
                for key, value in gate_up.items()
            },
            down={
                key: value.to(device=device, dtype=dtype)
                for key, value in down.items()
            },
            activation=getattr(config, "hidden_act", "silu"),
        )

    remaining = sorted(
        name
        for name, _ in module.named_parameters()
        if not name.startswith("mlp.experts.") and name not in assigned
    )
    if remaining:
        raise FixtureReferenceError(
            "fixture did not initialize official layer parameters: "
            + ", ".join(remaining)
        )
    module.eval()
    return module


class _LayerHost(nn.Module):
    """Minimal shape consumed by register_reference_hooks()."""

    def __init__(self, layer: nn.Module, layer_id: int, config: Any) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Identity() for _ in range(layer_id)] + [layer]
        )
        self.config = config


def run_layer_reference(
    fixture: Any,
    config: Any,
    layer: int,
    inputs: Iterable[Iterable[float]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Run successive hidden states through one official decoder layer."""
    _, _, classes, _ = _import_transformers()
    _, DynamicCache = classes
    rows = [list(map(float, row)) for row in inputs]
    if not rows:
        raise FixtureReferenceError("input hidden-state list is empty")
    hidden = int(config.hidden_size)
    for index, row in enumerate(rows):
        if len(row) != hidden:
            raise FixtureReferenceError(
                f"input {index} has {len(row)} values; hidden_size is {hidden}"
            )

    module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    host = _LayerHost(module, layer, config)
    store: dict[str, torch.Tensor] = {}
    position = [0]
    *_, register_reference_hooks = _repo_modules()
    handles = register_reference_hooks(
        host, [layer], store, lambda: position[0]
    )
    def _mask(length: int) -> torch.Tensor:
        """Causal, and windowed on a sliding layer.

        `attention_mask=None` does NOT mean causal in the eager path — it means
        unmasked, so the layer attends to future tokens. Passing None produced
        reference activations that differed from a correct decoder by ~4e-1 and
        decayed with position, which reads exactly like a bug in the candidate
        rather than in the reference.
        """
        neg = torch.finfo(dtype).min
        m = torch.triu(torch.full((length, length), neg, dtype=dtype), diagonal=1)
        window = int(getattr(config, "sliding_window_size", 0) or 0)
        if window and config.layer_types[layer] == "hybrid_sliding":
            for row in range(length):
                low = row - window + 1
                if low > 0:
                    m[row, :low] = neg
        return m.view(1, 1, length, length).to(device=device)

    # One prefill over the whole sequence, not one cached step per token.
    # The official short convolution only trims its output back to the query
    # length when a previous conv state exists, so a fresh cache fed a short
    # prefix emits kernel-length states and the attention reshape fails. A
    # single prefill sidesteps that entirely and is what a reference is for.
    #
    # Consequence worth knowing: the hooks in register_reference_hooks collapse
    # to the last token, so hook-captured *intermediate* activations describe
    # the final position. The per-position `.output` entries below come from
    # the returned tensor and are exact for every position.
    if len(rows) < int(getattr(config, "sconv_kernel_size", 1) or 1):
        raise FixtureReferenceError(
            f"need at least sconv_kernel_size={config.sconv_kernel_size} input "
            f"positions to prefill; got {len(rows)}"
        )
    try:
        with torch.no_grad():
            position[0] = len(rows) - 1
            prefix = torch.tensor(rows, device=device, dtype=dtype).view(
                1, len(rows), hidden
            )
            output = module(
                prefix,
                attention_mask=_mask(len(rows)),
                conv_mask=None,
                past_key_values=DynamicCache(config=config),
            )
            for index in range(len(rows)):
                store[f"token.{index}.layer.{layer}.output"] = (
                    output[0, index].detach().float().cpu().contiguous()
                )
    finally:
        for handle in handles:
            handle.remove()
    return store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "run official Inkling decoder layers from a bounded parity fixture"
        )
    )
    ap.add_argument("--fixture", required=True)
    ap.add_argument(
        "--model-config",
        required=True,
        help="official config.json used to extract fixture",
    )
    ap.add_argument(
        "--inputs", required=True, help="JSON list of hidden-state vectors"
    )
    ap.add_argument("--layers", required=True, help="comma-separated layer ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    args = ap.parse_args(argv)

    try:
        _, load_fixture, write_activation_archive, *_ = _repo_modules()
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = _text_config(config_json)
        inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(inputs, list):
            raise FixtureReferenceError("--inputs must contain a JSON list")
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise FixtureReferenceError("--layers must be nonempty")
        device = torch.device(args.device)
        dtype = DTYPES[args.dtype]
        _, _, _, transformers_version = _import_transformers()

        merged: dict[str, torch.Tensor] = {}
        for layer in layers:
            traced = run_layer_reference(
                fixture, config, layer, inputs, device=device, dtype=dtype
            )
            overlap = sorted(set(merged).intersection(traced))
            if overlap:
                raise FixtureReferenceError(
                    "duplicate activation names while merging layers: "
                    + ", ".join(overlap)
                )
            merged.update(traced)
        write_activation_archive(
            args.out,
            merged,
            metadata={
                "runtime": "transformers-fixture-layer",
                "transformers": transformers_version,
                "layers": layers,
                "positions": len(inputs),
                "dtype": args.dtype,
                "fixture": str(args.fixture),
                "config_sha256": fixture.source.get("config_sha256"),
            },
        )
    except (FixtureReferenceError, OSError, ValueError, KeyError, RuntimeError) as exc:
        ap.error(str(exc))
        return 2
    print(
        f"wrote official fixture-backed layer trace for {layers} to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
