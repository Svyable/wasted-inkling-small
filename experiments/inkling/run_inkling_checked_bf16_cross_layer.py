#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove checked-in BF16 continuity across dense 0 -> dense 1 -> sparse 2.

This composes only already-promoted execution paths. Dense layers 0 and 1 run
through the checked-in BF16 profile with the native matrix backend and one live
state per layer across all eight positions. Their real C outputs feed the next
layer. Sparse layer 2 then reuses the checked stateful sparse analyzer and its
backend-only routed expert contract.

The archived layer-2 route is independently reproduced from the official dense
0 -> 1 hidden state before any canonical downstream adapter is allowed to act.
Raw C routing remains visible in the sparse analysis; archived canonical IDs
and exact post-reduction weights are used only for downstream arithmetic, the
same bounded tie contract as the checked single-layer evidence.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

import diagnose_inkling_bf16_stateful_mlp_boundary as sparse_boundary
import diagnose_inkling_portable_bf16_stateful_sparse_layer as sparse_stateful
from checked_bf16_profile_collector import (
    CheckedPostReductionWeightHelper,
    build_checked_weight_helper,
)
from diagnose_inkling_pre_router import _capture_official_pre_router
from discover_inkling_router_experts import _causal_mask, input_sha256
from inkling_fixture import FixtureError, load_fixture
from inkling_fixture_reference import (
    DTYPES,
    FixtureReferenceError,
    _import_transformers,
    build_layer_from_fixture,
    verify_config_binding,
)
from inkling_layer_parity import LayerParityError
from inkling_release_config import build_transformers_text_config
from measure_inkling_dense_bf16_profile import (
    DenseBfloat16ProbeError,
    DenseMatrixProvider,
    metrics,
    run_candidate as run_dense_candidate,
)
from run_inkling_checked_bf16_sparse_layer import (
    build_checked_library,
    configure_profile_library,
)
from run_inkling_checked_bf16_stateful_layer import (
    CheckedStatefulError,
    run_c_stateful_profile,
)


class CrossLayerContinuityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchivedRouteRow:
    indices: tuple[int, ...]
    weights: tuple[float, ...]


def _hidden_tensor(value: Any, *, positions: int, hidden: int, layer: int) -> torch.Tensor:
    tensor = value
    if isinstance(tensor, (tuple, list)):
        tensor = next((item for item in tensor if isinstance(item, torch.Tensor)), None)
    if not isinstance(tensor, torch.Tensor):
        raise CrossLayerContinuityError(
            f"official layer {layer} returned no hidden-state tensor"
        )
    expected = (1, positions, hidden)
    if tuple(tensor.shape) != expected:
        raise CrossLayerContinuityError(
            f"official layer {layer} returned {tuple(tensor.shape)}; expected {expected}"
        )
    return tensor


def official_dense_prefix(
    fixture: Any,
    config: Any,
    input_values: list[list[float]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """Return the actual official hidden tensor after dense layers 0 and 1."""
    hidden = int(config.hidden_size)
    positions = len(input_values)
    current = torch.tensor(input_values, device=device, dtype=dtype).view(
        1, positions, hidden
    )
    _, _, classes, _ = _import_transformers()
    _, DynamicCache = classes
    cache = DynamicCache(config=config)
    outputs: dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for layer in (0, 1):
            module = build_layer_from_fixture(
                fixture, config, layer, device=device, dtype=dtype
            )
            result = module(
                current,
                attention_mask=_causal_mask(
                    config, layer, positions, device=device, dtype=dtype
                ),
                conv_mask=None,
                past_key_values=cache,
            )
            current = _hidden_tensor(
                result, positions=positions, hidden=hidden, layer=layer
            )
            outputs[layer] = current[0].detach().float().cpu().contiguous()
    return outputs


def archived_rows(archive: dict[str, Any]) -> list[ArchivedRouteRow]:
    indices = archive.get("layer2_topk_indices")
    weights = archive.get("layer2_topk_weights")
    if not isinstance(indices, list) or not isinstance(weights, list):
        raise CrossLayerContinuityError("cross-layer route archive is incomplete")
    if len(indices) != 8 or len(weights) != 8:
        raise CrossLayerContinuityError("cross-layer route archive must contain 8 rows")
    rows: list[ArchivedRouteRow] = []
    for position, (ids, values) in enumerate(zip(indices, weights)):
        if not isinstance(ids, list) or not isinstance(values, list) or len(ids) != len(values):
            raise CrossLayerContinuityError(
                f"cross-layer route row {position} is malformed"
            )
        rows.append(
            ArchivedRouteRow(
                tuple(int(value) for value in ids),
                tuple(float(value) for value in values),
            )
        )
    return rows


def tensor_rows(values: list[torch.Tensor]) -> list[list[float]]:
    return [value.detach().float().cpu().reshape(-1).tolist() for value in values]


def compare_dense_layer(
    official: torch.Tensor,
    candidate: list[torch.Tensor],
) -> list[dict[str, Any]]:
    if official.ndim != 2 or official.shape[0] != len(candidate):
        raise CrossLayerContinuityError(
            f"dense comparison shape {tuple(official.shape)} vs {len(candidate)} rows"
        )
    return [metrics(official[position], candidate[position]) for position in range(len(candidate))]


def first_dense_boundary(
    dense: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    for layer in ("0", "1"):
        for position, value in enumerate(dense[layer]):
            if value["raw_exact_fraction"] != 1.0:
                return {"layer": int(layer), "position": position, "stage": "layer_out"}
    return None


def verify_archived_route(
    fixture: Any,
    config: Any,
    official_layer1: torch.Tensor,
    archive: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    _pre, _module, indices, weights = _capture_official_pre_router(
        fixture,
        config,
        2,
        official_layer1,
        device=device,
        dtype=dtype,
    )
    actual_indices = indices.detach().to(torch.int64).cpu().tolist()
    actual_weights = weights.detach().float().cpu().tolist()
    if actual_indices != archive["layer2_topk_indices"]:
        raise CrossLayerContinuityError(
            "official layer-2 route no longer matches the archived consecutive route"
        )
    if actual_weights != archive["layer2_topk_weights"]:
        raise CrossLayerContinuityError(
            "official layer-2 weights no longer match the archived consecutive route"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--c-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        cfg = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        archive = json.loads(Path(args.routes).read_text(encoding="utf-8"))
        if not isinstance(input_values, list) or len(input_values) != 8:
            raise CrossLayerContinuityError("--inputs must contain eight source-bound rows")
        if archive.get("layers") != [0, 1, 2] or archive.get("positions") != 8:
            raise CrossLayerContinuityError("cross-layer route archive has the wrong scope")

        inputs = torch.tensor(input_values, dtype=torch.float32)
        dtype = DTYPES["bfloat16"]
        device = torch.device("cpu")
        if input_sha256(inputs, dtype) != archive.get("inputs_sha256"):
            raise CrossLayerContinuityError("input hash differs from cross-layer route archive")
        if fixture.source.get("revision") != archive.get("revision"):
            raise CrossLayerContinuityError("fixture revision differs from route archive")
        if fixture.source.get("config_sha256") != archive.get("config_sha256"):
            raise CrossLayerContinuityError("fixture config hash differs from route archive")
        if fixture.source.get("index_sha256") != archive.get("index_sha256"):
            raise CrossLayerContinuityError("fixture index hash differs from route archive")
        required_experts = set(int(value) for value in archive["layer2_expert_union"])
        if set(fixture.experts.get(2, ())) != required_experts:
            raise CrossLayerContinuityError("fixture layer-2 expert union differs from archive")

        official = official_dense_prefix(
            fixture, config, input_values, device=device, dtype=dtype
        )
        for layer in (0, 1):
            digest = input_sha256(official[layer], dtype)
            if digest != archive["dense_output_sha256"][str(layer)]:
                raise CrossLayerContinuityError(
                    f"official dense layer {layer} output hash differs from archive"
                )
        verify_archived_route(
            fixture,
            config,
            official[1],
            archive,
            device=device,
            dtype=dtype,
        )
        rows = archived_rows(archive)

        library, source = build_checked_library(Path(args.out).with_suffix(".runtime.so"))
        if source.get("source_rewriting") is not False:
            raise CrossLayerContinuityError("continuity gate must compile checked-in C unchanged")
        lib = ctypes.CDLL(str(library))
        configure_profile_library(lib)

        dense_modules = {
            layer: build_layer_from_fixture(
                fixture, config, layer, device=device, dtype=dtype
            )
            for layer in (0, 1)
        }
        dense0_provider = DenseMatrixProvider(0, dense_modules[0])
        dense0 = run_dense_candidate(
            lib, fixture, cfg, 0, input_values, dense0_provider
        )
        dense0_values = tensor_rows(dense0["outputs"])
        dense1_provider = DenseMatrixProvider(1, dense_modules[1])
        dense1 = run_dense_candidate(
            lib, fixture, cfg, 1, dense0_values, dense1_provider
        )
        dense1_values = tensor_rows(dense1["outputs"])
        dense_metrics = {
            "0": compare_dense_layer(official[0], dense0["outputs"]),
            "1": compare_dense_layer(official[1], dense1["outputs"]),
        }

        helper = CheckedPostReductionWeightHelper(
            build_checked_weight_helper(Path(args.out).with_suffix(".weights.so"))
        )
        original_runner = sparse_stateful.run_c_stateful
        sparse_stateful.run_c_stateful = run_c_stateful_profile
        try:
            sparse = sparse_boundary.analyze_layer(
                lib,
                helper,
                fixture,
                config,
                cfg,
                2,
                official[1],
                dense1_values,
                rows,
                device=device,
                dtype=dtype,
            )
        finally:
            sparse_stateful.run_c_stateful = original_runner

        first = first_dense_boundary(dense_metrics)
        if first is None and sparse["decision"]["classification"] != "stateful_sparse_mlp_ladder_exact":
            first = {
                "layer": 2,
                "position": sparse["decision"].get("first_boundary_position"),
                "stage": sparse["decision"].get("first_boundary_stage"),
            }
        classification = (
            "checked_in_bf16_cross_layer_0_1_2_exact"
            if first is None
            else "checked_in_bf16_cross_layer_0_1_2_mismatch"
        )
        result = {
            "format": "inkling-checked-in-bfloat16-cross-layer-continuity",
            "version": 1,
            "positions": 8,
            "layers": [0, 1, 2],
            "inputs_sha256": archive["inputs_sha256"],
            "source": source,
            "decision": {
                "classification": classification,
                "first_boundary": first,
            },
            "dense": {
                "0": {
                    "outputs": dense_metrics["0"],
                    "backend_calls": dense0["backend_calls"],
                },
                "1": {
                    "outputs": dense_metrics["1"],
                    "backend_calls": dense1["backend_calls"],
                },
            },
            "sparse_layer_2": sparse,
            "route_archive": {
                "expert_union_size": archive["layer2_expert_union_size"],
                "expert_union": archive["layer2_expert_union"],
                "official_route_reproduced_before_adapter": True,
            },
        }
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        CheckedStatefulError,
        CrossLayerContinuityError,
        DenseBfloat16ProbeError,
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
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
