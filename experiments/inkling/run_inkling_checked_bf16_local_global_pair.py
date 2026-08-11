#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove checked BF16 continuity across sparse local layer 4 -> global layer 5."""
from __future__ import annotations

import argparse
import ctypes
import json
import struct
from pathlib import Path
from typing import Any

import torch

import diagnose_inkling_bf16_stateful_mlp_boundary as boundary
import diagnose_inkling_portable_bf16_composed_moe as composed
import diagnose_inkling_portable_bf16_stateful_sparse_layer as stateful
from checked_bf16_profile_collector import (
    CheckedPostReductionWeightHelper,
    build_checked_weight_helper,
)
from diagnose_inkling_pre_router import _capture_official_pre_router
from discover_inkling_router_experts import _causal_mask, input_sha256
from inkling_canonical_route_layer_parity import RouteRow
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
from port_trace import TraceError, boundary_manifest, compare, load_trace
from run_inkling_checked_bf16_sparse_layer import build_checked_library, configure_profile_library
from run_inkling_checked_bf16_stateful_layer import CheckedStatefulError, run_c_stateful_profile
from run_inkling_fixture_reference_canonical import run_layer_reference_canonical


class PairContinuityError(RuntimeError):
    pass


def _rows(section: dict[str, Any], layer: int) -> list[RouteRow]:
    indices = section.get("topk_indices")
    weights = section.get("topk_weights")
    if not isinstance(indices, list) or not isinstance(weights, list) or len(indices) != len(weights):
        raise PairContinuityError(f"layer {layer} route archive is malformed")
    rows: list[RouteRow] = []
    for position, (ids, values) in enumerate(zip(indices, weights)):
        if not isinstance(ids, list) or not isinstance(values, list) or len(ids) != len(values):
            raise PairContinuityError(f"layer {layer} route row {position} is malformed")
        rows.append(
            RouteRow(
                indices=tuple(int(value) for value in ids),
                weights=tuple(float(value) for value in values),
            )
        )
    return rows


def _tensor_rows(values: list[torch.Tensor]) -> list[list[float]]:
    return [value.detach().float().cpu().reshape(-1).tolist() for value in values]


def _stack(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([value.detach().float().cpu() for value in values]).contiguous()


def _verify_route(
    fixture: Any,
    config: Any,
    layer: int,
    inputs: torch.Tensor,
    section: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    _pre, _module, indices, weights = _capture_official_pre_router(
        fixture, config, layer, inputs, device=device, dtype=dtype
    )
    if indices.detach().to(torch.int64).cpu().tolist() != section["topk_indices"]:
        raise PairContinuityError(f"official layer {layer} route differs from archive")
    if weights.detach().float().cpu().tolist() != section["topk_weights"]:
        raise PairContinuityError(f"official layer {layer} route weights differ from archive")


def _execute_official_layer(
    fixture: Any,
    config: Any,
    layer: int,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if inputs.ndim != 2:
        raise PairContinuityError(f"layer {layer} official input must be rank-2")
    positions, hidden = int(inputs.shape[0]), int(inputs.shape[1])
    module = build_layer_from_fixture(fixture, config, layer, device=device, dtype=dtype)
    _, _, classes, _ = _import_transformers()
    _, DynamicCache = classes
    with torch.no_grad():
        output = module(
            inputs.to(device=device, dtype=dtype).unsqueeze(0),
            attention_mask=_causal_mask(config, layer, positions, device=device, dtype=dtype),
            conv_mask=None,
            past_key_values=DynamicCache(config=config),
        )
    tensor = output
    if isinstance(tensor, (tuple, list)):
        tensor = next((item for item in tensor if isinstance(item, torch.Tensor)), None)
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (1, positions, hidden):
        raise PairContinuityError(f"layer {layer} official output has unexpected shape")
    return tensor[0].detach().float().cpu().contiguous()


def _analyze_layer_once(
    lib: Any,
    helper: Any,
    fixture: Any,
    config: Any,
    cfg: dict[str, Any],
    layer: int,
    official_inputs: torch.Tensor,
    candidate_inputs: list[list[float]],
    rows: list[RouteRow],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    official_values = official_inputs.detach().float().cpu().tolist()
    official_pre, _, _, _ = _capture_official_pre_router(
        fixture, config, layer, official_inputs, device=device, dtype=dtype
    )
    canonical_outputs = run_layer_reference_canonical(
        fixture,
        config,
        layer,
        official_values,
        rows,
        device=device,
        dtype=dtype,
    )
    official = boundary.official_mlp_sequence(
        fixture,
        config,
        layer,
        official_pre["post_attention_norm"],
        official_pre["post_attention_residual"],
        rows,
        canonical_outputs,
        device=device,
        dtype=dtype,
    )

    provider_module = build_layer_from_fixture(fixture, config, layer, device=device, dtype=dtype)
    provider = composed.CapturingMoeProvider(layer, provider_module)
    collector = stateful.StatefulExactWeightCollector(
        rows,
        provider,
        helper,
        int(provider_module.mlp.gate.num_experts),
        float(provider_module.mlp.gate.route_scale),
        float(provider_module.mlp.gate.global_scale.detach().float().cpu().reshape(-1)[0]),
    )
    candidate = run_c_stateful_profile(
        lib, fixture, cfg, layer, candidate_inputs, rows, provider, collector
    )

    stages: dict[str, list[dict[str, Any]]] = {
        "routed_weight": [],
        "shared_weight": [],
        "moe_out": [],
        "mlp_branch": [],
        "layer_out": [],
    }
    for position in range(len(candidate_inputs)):
        stages["routed_weight"].append(
            boundary.metrics(
                official["routed_weight"][position],
                boundary._trace_tensor(collector, layer, position, "routed_weight"),
            )
        )
        stages["shared_weight"].append(
            boundary.metrics(
                official["shared_weight"][position],
                boundary._trace_tensor(collector, layer, position, "shared_weight"),
            )
        )
        stages["moe_out"].append(
            boundary.metrics(
                official["moe_out"][position],
                boundary._trace_tensor(collector, layer, position, "moe_out"),
            )
        )
        stages["mlp_branch"].append(
            boundary.metrics(
                official["mlp_branch"][position],
                boundary._trace_tensor(collector, layer, position, "mlp_branch"),
            )
        )
        stages["layer_out"].append(
            boundary.metrics(official["layer_out"][position], candidate["outputs"][position])
        )

    preexpert = [
        boundary.metrics(
            official_pre["post_attention_norm"][position],
            candidate["post_attention_norm"][position],
        )
        for position in range(len(candidate_inputs))
    ]
    if any(value["bfloat16_exact_fraction"] != 1.0 for value in preexpert):
        raise PairContinuityError(f"layer {layer} pre-expert boundary is not BF16-exact")

    analysis = {
        "decision": boundary.classify_ladder(stages),
        "preexpert": preexpert,
        "stages": stages,
        "official_layer_out_anchor": official["layer_out_anchor"],
        "candidate_routes": candidate["candidate_routes"],
        "canonical_routes": candidate["canonical_routes"],
        "backend_calls": candidate["backend_calls"],
    }
    return candidate["outputs"], analysis


def _bf16_payload(tensor: torch.Tensor) -> bytes:
    bits = tensor.detach().to(torch.bfloat16).cpu().contiguous().view(torch.uint16).reshape(-1).tolist()
    return struct.pack(f"<{len(bits)}H", *bits)


def _trace_manifest(
    root: Path,
    *,
    model: str,
    revision: str,
    name: str,
    input4: torch.Tensor,
    output4: torch.Tensor,
    output5: torch.Tensor,
    observed4: list[list[int]],
    observed5: list[list[int]],
    reference4: list[list[int]],
    reference5: list[list[int]],
) -> Path:
    reason = "official route reproduced before canonical downstream adapter"
    layers = []
    for layer, structural_class, inp, out, observed, reference in (
        (4, "sparse-local", input4, output4, observed4, reference4),
        (5, "sparse-global", output4, output5, observed5, reference5),
    ):
        prefix = f"layer-{layer}"
        layers.append(
            {
                "layer": layer,
                "structural_class": structural_class,
                "boundaries": [
                    boundary_manifest(
                        root,
                        f"{prefix}/input.bf16.bin",
                        "input",
                        "BF16",
                        list(inp.shape),
                        _bf16_payload(inp),
                    ),
                    boundary_manifest(
                        root,
                        f"{prefix}/output.bf16.bin",
                        "output",
                        "BF16",
                        list(out.shape),
                        _bf16_payload(out),
                    ),
                ],
                "routing": {
                    "observed_route": observed,
                    "reference_route": reference,
                    "downstream_route_used": "reference",
                    "route_equivalence_reason": reason,
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "model": model,
        "revision": revision,
        "trace_name": name,
        "layers": layers,
    }
    path = root / "trace.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_trace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--c-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    try:
        fixture = load_fixture(args.fixture)
        config_json = verify_config_binding(fixture, args.model_config)
        config = build_transformers_text_config(config_json)
        cfg = json.loads(Path(args.c_config).read_text(encoding="utf-8"))
        input_values = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        routes = json.loads(Path(args.routes).read_text(encoding="utf-8"))
        if routes.get("pair") != [4, 5] or routes.get("structural_classes") != ["local", "global"]:
            raise PairContinuityError("route archive is not the pinned local->global pair")
        if not isinstance(input_values, list) or len(input_values) != 8:
            raise PairContinuityError("--inputs must contain eight source-bound rows")
        inputs = torch.tensor(input_values, dtype=torch.float32)
        dtype = DTYPES["bfloat16"]
        device = torch.device("cpu")
        if input_sha256(inputs, dtype) != routes["inputs_sha256"]:
            raise PairContinuityError("input hash differs from route archive")
        if fixture.source.get("revision") != routes["revision"] or \
           fixture.source.get("config_sha256") != routes["config_sha256"] or \
           fixture.source.get("index_sha256") != routes["index_sha256"]:
            raise PairContinuityError("fixture provenance differs from route archive")
        if sorted(fixture.experts.get(4, ())) != routes["first_layer"]["selected_experts"]:
            raise PairContinuityError("fixture layer-4 expert union differs from route archive")
        if sorted(fixture.experts.get(5, ())) != routes["second_layer"]["selected_experts"]:
            raise PairContinuityError("fixture layer-5 expert union differs from route archive")

        rows4 = _rows(routes["first_layer"], 4)
        rows5 = _rows(routes["second_layer"], 5)
        _verify_route(fixture, config, 4, inputs, routes["first_layer"], device=device, dtype=dtype)
        official4 = _execute_official_layer(fixture, config, 4, inputs, device=device, dtype=dtype)
        if input_sha256(official4, dtype) != routes["first_output_sha256"]:
            raise PairContinuityError("official layer-4 output hash differs from route archive")
        _verify_route(fixture, config, 5, official4, routes["second_layer"], device=device, dtype=dtype)
        official5 = _execute_official_layer(fixture, config, 5, official4, device=device, dtype=dtype)

        library, source = build_checked_library(Path(args.out).with_suffix(".runtime.so"))
        if source.get("source_rewriting") is not False:
            raise PairContinuityError("pair gate must compile checked-in C unchanged")
        lib = ctypes.CDLL(str(library))
        configure_profile_library(lib)
        helper = CheckedPostReductionWeightHelper(
            build_checked_weight_helper(Path(args.out).with_suffix(".weights.so"))
        )

        candidate4, analysis4 = _analyze_layer_once(
            lib,
            helper,
            fixture,
            config,
            cfg,
            4,
            inputs,
            input_values,
            rows4,
            device=device,
            dtype=dtype,
        )
        candidate4_tensor = _stack(candidate4)
        candidate5, analysis5 = _analyze_layer_once(
            lib,
            helper,
            fixture,
            config,
            cfg,
            5,
            official4,
            _tensor_rows(candidate4),
            rows5,
            device=device,
            dtype=dtype,
        )
        candidate5_tensor = _stack(candidate5)

        trace_root = Path(args.trace_dir)
        expected_root = trace_root / "expected"
        actual_root = trace_root / "actual"
        expected_root.mkdir(parents=True, exist_ok=True)
        actual_root.mkdir(parents=True, exist_ok=True)
        reference4 = routes["first_layer"]["topk_indices"]
        reference5 = routes["second_layer"]["topk_indices"]
        expected_manifest = _trace_manifest(
            expected_root,
            model=fixture.model_id,
            revision=str(fixture.source.get("revision")),
            name="official-local-global-4-5",
            input4=inputs,
            output4=official4,
            output5=official5,
            observed4=reference4,
            observed5=reference5,
            reference4=reference4,
            reference5=reference5,
        )
        actual_manifest = _trace_manifest(
            actual_root,
            model=fixture.model_id,
            revision=str(fixture.source.get("revision")),
            name="checked-local-global-4-5",
            input4=inputs,
            output4=candidate4_tensor,
            output5=candidate5_tensor,
            observed4=analysis4["candidate_routes"],
            observed5=analysis5["candidate_routes"],
            reference4=reference4,
            reference5=reference5,
        )
        trace_decision = compare(load_trace(expected_manifest), load_trace(actual_manifest))

        first = None
        for layer, analysis in ((4, analysis4), (5, analysis5)):
            if analysis["decision"]["classification"] != "stateful_sparse_mlp_ladder_exact":
                first = {
                    "layer": layer,
                    "position": analysis["decision"].get("first_boundary_position"),
                    "stage": analysis["decision"].get("first_boundary_stage"),
                }
                break
        if first is None and trace_decision["status"] != "match":
            first = {
                "layer": trace_decision.get("layer"),
                "position": None,
                "stage": trace_decision.get("boundary"),
                "index": trace_decision.get("index"),
            }
        classification = (
            "checked_in_bf16_local_global_4_5_exact"
            if first is None
            else "checked_in_bf16_local_global_4_5_mismatch"
        )
        result = {
            "format": "inkling-checked-in-bfloat16-local-global-continuity",
            "version": 1,
            "pair": [4, 5],
            "positions": 8,
            "source": source,
            "adapter_contract": "explicit_candidate_runner_no_module_rebinding",
            "route_contract": "observed_preserved_reference_used_downstream",
            "decision": {"classification": classification, "first_boundary": first},
            "trace": trace_decision,
            "official": {
                "layer4_output_sha256": input_sha256(official4, dtype),
                "layer5_output_sha256": input_sha256(official5, dtype),
            },
            "layers": {"4": analysis4, "5": analysis5},
        }
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (
        CheckedStatefulError,
        FixtureError,
        FixtureReferenceError,
        LayerParityError,
        PairContinuityError,
        TraceError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        AttributeError,
    ) as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"]["first_boundary"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
