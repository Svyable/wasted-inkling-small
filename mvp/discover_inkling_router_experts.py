#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record official Inkling router choices from a bounded seed fixture.

The seed fixture may contain any nonempty routed-expert selection required to
construct a sparse official layer. Immediately after construction, the loaded
expert bank is replaced by a capture sentinel. The sentinel records the exact
``topk(sorted=False)`` expert IDs and their attached weights, then aborts before
expert computation. Seed expert values therefore cannot affect discovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
INKLING_TOOLS = REPO / "inkling" / "tools"
if str(INKLING_TOOLS) not in sys.path:
    sys.path.insert(0, str(INKLING_TOOLS))

from inkling_fixture import load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    _import_transformers,
    _is_sparse,
    build_layer_from_fixture,
    verify_config_binding,
)
from inkling_release_config import build_transformers_text_config


class RouterDiscoveryError(RuntimeError):
    """Router discovery cannot continue without guessing."""


class _RouterCaptured(BaseException):
    def __init__(self, indices: torch.Tensor, weights: torch.Tensor) -> None:
        self.indices = indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
        self.weights = weights.detach().float().cpu().contiguous()


class _CaptureExperts(nn.Module):
    def forward(
        self,
        _hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        raise _RouterCaptured(top_k_index, top_k_weights)


def deterministic_hidden_states(tokens: int, hidden: int, seed: int) -> torch.Tensor:
    """Generate version-independent hidden states, rounded exactly to BF16."""
    if tokens < 1 or hidden < 1:
        raise RouterDiscoveryError("tokens and hidden must be positive")
    state = seed & 0xFFFFFFFFFFFFFFFF
    if state == 0:
        state = 0x9E3779B97F4A7C15
    mask = 0xFFFFFFFFFFFFFFFF
    values: list[float] = []
    for _ in range(tokens * hidden):
        state ^= (state << 13) & mask
        state ^= state >> 7
        state ^= (state << 17) & mask
        signed = ((state >> 48) & 0xFFFF) - 32768
        values.append(signed / 32768.0)
    return (
        torch.tensor(values, dtype=torch.float32)
        .reshape(tokens, hidden)
        .to(torch.bfloat16)
        .float()
    )


def input_sha256(inputs: torch.Tensor, dtype: torch.dtype) -> str:
    """Hash tensor bytes without depending on NumPy being installed."""
    octets = inputs.to(dtype=dtype).contiguous().view(torch.uint8).tolist()
    return hashlib.sha256(bytes(octets)).hexdigest()


def _causal_mask(
    config: Any,
    layer: int,
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    neg = torch.finfo(dtype).min
    mask = torch.triu(torch.full((length, length), neg, dtype=dtype), diagonal=1)
    window = int(getattr(config, "sliding_window_size", 0) or 0)
    if window and config.layer_types[layer] == "hybrid_sliding":
        for row in range(length):
            low = row - window + 1
            if low > 0:
                mask[row, :low] = neg
    return mask.view(1, 1, length, length).to(device=device)


def discover_layer_router(
    fixture: Any,
    config: Any,
    layer: int,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Return exact official top-k IDs and weights for each supplied position."""
    if not _is_sparse(config, layer):
        raise RouterDiscoveryError(f"layer {layer} is not sparse")
    if inputs.ndim != 2 or inputs.shape[1] != int(config.hidden_size):
        raise RouterDiscoveryError(
            f"inputs must be [tokens,{config.hidden_size}], got {tuple(inputs.shape)}"
        )
    kernel = int(getattr(config, "sconv_kernel_size", 1) or 1)
    if inputs.shape[0] < kernel:
        raise RouterDiscoveryError(
            f"need at least sconv_kernel_size={kernel} positions; got {inputs.shape[0]}"
        )

    _, _, classes, _ = _import_transformers()
    _, DynamicCache = classes
    module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    # Construction proves the seed fixture has valid official expert geometry.
    # From this point onward those values are deliberately unreachable.
    module.mlp.experts = _CaptureExperts()
    prefix = inputs.to(device=device, dtype=dtype).unsqueeze(0)
    try:
        with torch.no_grad():
            module(
                prefix,
                attention_mask=_causal_mask(
                    config, layer, inputs.shape[0], device=device, dtype=dtype
                ),
                conv_mask=None,
                past_key_values=DynamicCache(config=config),
            )
    except _RouterCaptured as captured:
        indices, weights = captured.indices, captured.weights
    else:
        raise RouterDiscoveryError(
            "official layer completed without invoking routed experts"
        )

    expected = (inputs.shape[0], int(config.num_experts_per_tok))
    if tuple(indices.shape) != expected or tuple(weights.shape) != expected:
        raise RouterDiscoveryError(
            f"captured router shape mismatch: {tuple(indices.shape)} / "
            f"{tuple(weights.shape)}, expected {expected}"
        )
    return {
        "seed_fixture_experts": list(fixture.experts.get(layer, ())),
        "selected_experts": sorted(
            {int(value) for value in indices.flatten().tolist()}
        ),
        "topk_indices": [
            [int(value) for value in row] for row in indices.tolist()
        ],
        "topk_weights": [
            [float(value) for value in row] for row in weights.tolist()
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    args = ap.parse_args(argv)

    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise RouterDiscoveryError("--layers must be nonempty")
        fixture.require_layers(layers)
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        inputs = deterministic_hidden_states(
            args.tokens, int(config.hidden_size), args.seed
        )
        discoveries = {
            str(layer): discover_layer_router(
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
            "format": "inkling-router-discovery",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "tokens": args.tokens,
            "seed": args.seed,
            "input_dtype": args.dtype,
            "inputs_sha256": input_sha256(inputs, dtype),
            "layers": discoveries,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (
        FixtureReferenceError,
        RouterDiscoveryError,
        ValueError,
        OSError,
    ) as exc:
        ap.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
