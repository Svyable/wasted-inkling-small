#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export exact BF16 Inkling router-choice bit patterns for platform audits.

The archive is generated only on the source-bound Linux reference job.  It
stores each 256-wide routed-expert choice row as signed int16 BF16 payload bits,
plus the exact committed official route IDs.  Downstream platform jobs can then
reconstruct the *identical* choice tensors without model weights, GEMM, sigmoid,
or JSON decimal round-trips and run only ``torch.topk``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from analyze_inkling_router_mismatch import capture_official_router
from discover_inkling_router_experts import input_sha256
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import DTYPES, FixtureReferenceError, verify_config_binding
from inkling_release_config import build_transformers_text_config


class ChoiceArchiveError(RuntimeError):
    """The source-bound choice archive cannot be emitted safely."""


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def choice_bits(choice: torch.Tensor) -> list[list[int]]:
    choice = choice.detach().to(torch.bfloat16).cpu().contiguous()
    if choice.ndim != 2:
        raise ChoiceArchiveError(f"choice tensor must be 2-D, got {tuple(choice.shape)}")
    return choice.view(torch.int16).tolist()


def export_layer(
    fixture: Any,
    config: Any,
    layer: int,
    inputs: torch.Tensor,
    expected_rows: list[list[int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    module, _router_input, router_logits, official_indices, _weights = capture_official_router(
        fixture, config, layer, inputs, device=device, dtype=dtype
    )
    gate = module.mlp.gate
    routed_n = int(gate.num_experts)
    top_k = int(gate.top_k)
    choice = router_logits[..., :routed_n].sigmoid() + gate.e_score_correction_bias.detach().to(
        router_logits.dtype
    )
    if choice.shape != (inputs.shape[0], routed_n):
        raise ChoiceArchiveError(
            f"layer {layer} choice shape {tuple(choice.shape)} != "
            f"{(int(inputs.shape[0]), routed_n)}"
        )
    captured = official_indices.detach().to(torch.int64).cpu().tolist()
    if captured != expected_rows:
        raise ChoiceArchiveError(
            f"layer {layer} captured route archive differs from committed selection"
        )
    # Reconstruct from bits immediately and require exact top-k reproduction.
    bits = choice_bits(choice)
    reconstructed = torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)
    replay = torch.topk(reconstructed, top_k, dim=-1, sorted=False).indices.to(torch.int64)
    if replay.tolist() != expected_rows:
        raise ChoiceArchiveError(
            f"layer {layer} BF16 bit replay changed the official route archive"
        )
    return {
        "top_k": top_k,
        "n_routed_experts": routed_n,
        "choice_dtype": "bfloat16",
        "choice_bits_int16": bits,
        "official_indices": expected_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
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
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        inputs = torch.tensor(input_values, dtype=torch.float32)
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise ChoiceArchiveError("--layers must be nonempty")
        if selection.get("revision") != fixture.source.get("revision"):
            raise ChoiceArchiveError("selection revision differs from fixture")
        if selection.get("config_sha256") != fixture.source.get("config_sha256"):
            raise ChoiceArchiveError("selection config hash differs from fixture")
        if selection.get("index_sha256") != fixture.source.get("index_sha256"):
            raise ChoiceArchiveError("selection index hash differs from fixture")
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        if device.type != "cpu":
            raise ChoiceArchiveError("the choice archive requires the pinned CPU reference")
        actual_hash = input_sha256(inputs, dtype)
        if actual_hash != selection.get("inputs_sha256"):
            raise ChoiceArchiveError("input hash differs from committed selection")
        archived_layers: dict[str, Any] = {}
        for layer in layers:
            layer_selection = selection.get("layers", {}).get(str(layer))
            if not isinstance(layer_selection, dict):
                raise ChoiceArchiveError(f"selection has no layer {layer}")
            expected = layer_selection.get("topk_indices")
            if not isinstance(expected, list):
                raise ChoiceArchiveError(f"selection layer {layer} lacks topk_indices")
            archived_layers[str(layer)] = export_layer(
                fixture,
                config,
                layer,
                inputs,
                expected,
                device=device,
                dtype=dtype,
            )
        archive = {
            "format": "inkling-bfloat16-router-choice-bits",
            "version": 1,
            "model_id": fixture.model_id,
            "revision": fixture.source.get("revision"),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "inputs_sha256": actual_hash,
            "tokens": int(inputs.shape[0]),
            "layers": archived_layers,
        }
        archive["archive_sha256"] = hashlib.sha256(_canonical_bytes(archive)).hexdigest()
        Path(args.out).write_text(json.dumps(archive, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (
        FixtureError,
        FixtureReferenceError,
        ChoiceArchiveError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(archive, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
