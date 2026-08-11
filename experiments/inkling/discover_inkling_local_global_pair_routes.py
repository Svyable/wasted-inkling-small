#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Discover official routes across one consecutive local -> global sparse pair.

The supplied fixture must contain the full routed-expert union for the local
layer and at least one seed expert for the following global layer.  The local
route is first re-captured and required to equal the fixture expert union.  The
local layer then executes normally, using only those captured experts.  Its
actual returned hidden tensor is fed directly into the global layer router,
whose expert bank is replaced by the existing capture sentinel before expert
math.

This proves that the second route is conditioned on the real first-layer output
rather than on the original source-bound rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from discover_inkling_router_experts import (
    RouterDiscoveryError,
    _causal_mask,
    deterministic_hidden_states,
    discover_layer_router,
    input_sha256,
)
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    _import_transformers,
    build_layer_from_fixture,
    verify_config_binding,
)
from inkling_release_config import build_transformers_text_config


class PairRouteError(RuntimeError):
    pass


def _hidden_tensor(
    value: Any, *, positions: int, hidden: int, layer: int
) -> torch.Tensor:
    tensor = value
    if isinstance(tensor, (tuple, list)):
        tensor = next((item for item in tensor if isinstance(item, torch.Tensor)), None)
    if not isinstance(tensor, torch.Tensor):
        raise PairRouteError(f"layer {layer} returned no hidden-state tensor")
    expected = (1, positions, hidden)
    if tuple(tensor.shape) != expected:
        raise PairRouteError(
            f"layer {layer} returned {tuple(tensor.shape)}; expected {expected}"
        )
    return tensor


def execute_first_layer(
    fixture: Any,
    config: Any,
    layer: int,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Execute one official sparse layer with a fresh live cache/state."""
    positions, hidden = int(inputs.shape[0]), int(inputs.shape[1])
    module = build_layer_from_fixture(
        fixture, config, layer, device=device, dtype=dtype
    )
    _, _, classes, _ = _import_transformers()
    _, DynamicCache = classes
    with torch.no_grad():
        result = module(
            inputs.to(device=device, dtype=dtype).unsqueeze(0),
            attention_mask=_causal_mask(
                config, layer, positions, device=device, dtype=dtype
            ),
            conv_mask=None,
            past_key_values=DynamicCache(config=config),
        )
    return _hidden_tensor(
        result, positions=positions, hidden=hidden, layer=layer
    )[0].detach().float().cpu().contiguous()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--first-layer", type=int, required=True)
    parser.add_argument("--second-layer", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    try:
        if args.second_layer != args.first_layer + 1:
            raise PairRouteError("layers must be consecutive")
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        first, second = args.first_layer, args.second_layer
        fixture.require_layers((first, second))
        local_ids = set(int(value) for value in config.local_layer_ids)
        if first not in local_ids or second in local_ids:
            raise PairRouteError(
                f"layers {first}->{second} are not local->global in pinned config"
            )
        dtype = DTYPES["bfloat16"]
        device = torch.device("cpu")
        inputs = deterministic_hidden_states(
            args.tokens, int(config.hidden_size), args.seed
        )

        first_route = discover_layer_router(
            fixture,
            config,
            first,
            inputs,
            device=device,
            dtype=dtype,
        )
        selected_first = [int(value) for value in first_route["selected_experts"]]
        fixture_first = sorted(int(value) for value in fixture.experts.get(first, ()))
        if selected_first != fixture_first:
            raise PairRouteError(
                f"layer {first} fixture experts {fixture_first} differ from "
                f"recaptured route union {selected_first}"
            )

        first_output = execute_first_layer(
            fixture,
            config,
            first,
            inputs,
            device=device,
            dtype=dtype,
        )
        second_route = discover_layer_router(
            fixture,
            config,
            second,
            first_output,
            device=device,
            dtype=dtype,
        )
        if not second_route["selected_experts"]:
            raise PairRouteError(f"layer {second} route union is empty")

        result = {
            "format": "inkling-local-global-pair-route-discovery",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "input_dtype": "bfloat16",
            "tokens": args.tokens,
            "seed": args.seed,
            "inputs_sha256": input_sha256(inputs, dtype),
            "pair": [first, second],
            "structural_classes": ["local", "global"],
            "first_layer": first_route,
            "first_output_sha256": input_sha256(first_output, dtype),
            "second_layer": second_route,
            "reference_execution": (
                f"layer{first}-router-capture -> layer{first}-official-full -> "
                f"direct-hidden-handoff -> layer{second}-router-capture"
            ),
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        FixtureError,
        FixtureReferenceError,
        PairRouteError,
        RouterDiscoveryError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        AttributeError,
    ) as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
