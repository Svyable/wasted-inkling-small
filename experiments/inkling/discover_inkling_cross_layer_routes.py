#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the route reached by the official consecutive 0 -> 1 -> 2 slice.

Representative single-layer route archives cannot size a consecutive fixture:
layer 2 consumes layer-1 output, not the original source-bound rows. This tool
runs official dense layers 0 and 1 with a direct hidden-state handoff, then uses
the repository's existing sparse router-capture contract at layer 2.

The layer-2 fixture needs one seed expert only so the official sparse module can
be constructed. The discovery helper replaces the expert bank with a capture
sentinel before expert arithmetic, so seed expert values cannot affect routing.
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


class CrossLayerRouteError(RuntimeError):
    pass


def _hidden_tensor(value: Any, *, positions: int, hidden: int, layer: int) -> torch.Tensor:
    tensor = value
    if isinstance(tensor, (tuple, list)):
        tensor = next((item for item in tensor if isinstance(item, torch.Tensor)), None)
    if not isinstance(tensor, torch.Tensor):
        raise CrossLayerRouteError(f"layer {layer} returned no hidden-state tensor")
    expected = (1, positions, hidden)
    if tuple(tensor.shape) != expected:
        raise CrossLayerRouteError(
            f"layer {layer} returned {tuple(tensor.shape)}; expected {expected}"
        )
    return tensor


def run_dense_prefix(
    fixture: Any,
    config: Any,
    input_values: list[list[float]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, str]]:
    """Execute official dense layers 0 and 1 with the real tensor handoff."""
    kinds = getattr(config, "mlp_layer_types", None)
    if (
        not isinstance(kinds, (list, tuple))
        or len(kinds) < 2
        or any(kinds[layer] == "sparse" for layer in (0, 1))
    ):
        raise CrossLayerRouteError(
            "official config does not declare layers 0 and 1 as dense MLP layers"
        )

    hidden_size = int(config.hidden_size)
    positions = len(input_values)
    current = torch.tensor(input_values, device=device, dtype=dtype).view(
        1, positions, hidden_size
    )
    _, _, classes, _ = _import_transformers()
    _, DynamicCache = classes
    cache = DynamicCache(config=config)
    hashes: dict[str, str] = {}

    with torch.no_grad():
        for layer in (0, 1):
            module = build_layer_from_fixture(
                fixture, config, layer, device=device, dtype=dtype
            )
            output = module(
                current,
                attention_mask=_causal_mask(
                    config, layer, positions, device=device, dtype=dtype
                ),
                conv_mask=None,
                past_key_values=cache,
            )
            current = _hidden_tensor(
                output, positions=positions, hidden=hidden_size, layer=layer
            )
            hashes[str(layer)] = input_sha256(
                current[0].detach().float().cpu(), dtype
            )

    return current[0].detach().float().cpu().contiguous(), hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(input_values, list) or len(input_values) != 8:
            raise CrossLayerRouteError("--inputs must contain eight source-bound rows")

        inputs = torch.tensor(input_values, dtype=torch.float32)
        dtype = DTYPES["bfloat16"]
        device = torch.device("cpu")
        layer1_output, dense_hashes = run_dense_prefix(
            fixture, config, input_values, device=device, dtype=dtype
        )
        discovery = discover_layer_router(
            fixture,
            config,
            2,
            layer1_output,
            device=device,
            dtype=dtype,
        )
        union = [int(value) for value in discovery["selected_experts"]]
        if not union:
            raise CrossLayerRouteError("layer-2 route union is empty")

        result: dict[str, Any] = {
            "format": "inkling-cross-layer-route-discovery",
            "version": 2,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "positions": len(input_values),
            "input_dtype": "bfloat16",
            "inputs_sha256": input_sha256(inputs, dtype),
            "layers": [0, 1, 2],
            "dense_output_sha256": dense_hashes,
            "layer2_seed_fixture_experts": discovery["seed_fixture_experts"],
            "layer2_topk_indices": discovery["topk_indices"],
            "layer2_topk_weights": discovery["topk_weights"],
            "layer2_expert_union": union,
            "layer2_expert_union_size": len(union),
            "reference_execution": (
                "layer0-prefill -> direct-hidden-handoff -> layer1-prefill -> "
                "direct-hidden-handoff -> layer2-router-capture"
            ),
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        CrossLayerRouteError,
        FixtureError,
        FixtureReferenceError,
        RouterDiscoveryError,
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
