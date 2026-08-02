#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Build a deterministic text-only Inkling conversion plan.

This tool is the second implementation milestone.  It does not write a WASTE
container and does not load tensor payloads.  It turns the official config,
sidecars, index, and safetensors headers into a complete, reviewable plan for:

* resident trunk tensors;
* streamed routed-expert banks;
* resident shared experts;
* exact per-layer attention/MLP descriptors;
* source tensor transforms and exclusions.

No tensor is converted until every required source shape has passed the probes
below.  The actual converter will consume this plan rather than rediscovering
architecture from names while writing output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from inspect_inkling import InspectError, TensorCatalog, detect_dialect, inspect, load_json
from inkling_release import ReleaseError, inspect_release


class PlanError(RuntimeError):
    pass


def _value(report: dict[str, Any], name: str) -> Any:
    item = report.get("fields", {}).get(name)
    return item.get("value") if isinstance(item, dict) else None


def _require_int(report: dict[str, Any], name: str, *, minimum: int = 1) -> int:
    value = _value(report, name)
    if not isinstance(value, int) or value < minimum:
        raise PlanError(f"required positive integer field is unavailable: {name}")
    return value


def _shape(catalog: TensorCatalog, name: str) -> list[int] | None:
    meta = catalog.meta(name)
    return meta.shape if meta else None


def _dtype(catalog: TensorCatalog, name: str) -> str | None:
    meta = catalog.meta(name)
    return meta.dtype if meta else None


def _probe(
    catalog: TensorCatalog,
    name: str,
    expected: Iterable[int] | None,
    reason: str,
) -> dict[str, Any]:
    actual = _shape(catalog, name)
    want = list(expected) if expected is not None else None
    if actual is None:
        status = "unknown"
    elif want is None or actual == want:
        status = "ok"
    else:
        status = "mismatch"
    return {
        "name": name,
        "expected_shape": want,
        "actual_shape": actual,
        "dtype": _dtype(catalog, name),
        "status": status,
        "reason": reason,
    }


def _raw_layer_prefix(layer: int) -> str:
    return f"model.llm.layers.{layer}"


def _hf_layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}"


DIALECTS = ("provider_raw", "transformers_normalized")


def _dialect(dialect: str) -> str:
    if dialect not in DIALECTS:
        raise PlanError(f"unsupported checkpoint dialect: {dialect!r}")
    return dialect


def layer_prefix(layer: int, dialect: str) -> str:
    return (_raw_layer_prefix(layer) if _dialect(dialect) == "provider_raw"
            else _hf_layer_prefix(layer))


def layer_attention_names(layer: int, dialect: str) -> dict[str, str]:
    """Source tensor names for one layer's norms and attention projections.

    Split out of build_plan so a consumer that already has the tensors — a
    parity fixture, say — can find them by the same names the planner uses,
    rather than keeping a second copy of this table that drifts.
    """
    p = layer_prefix(layer, dialect)
    if dialect == "provider_raw":
        return {
            "input_norm": f"{p}.attn_norm.weight",
            "post_attention_norm": f"{p}.mlp_norm.weight",
            "q": f"{p}.attn.wq_du.weight",
            "k": f"{p}.attn.wk_dv.weight",
            "v": f"{p}.attn.wv_dv.weight",
            "r": f"{p}.attn.wr_du.weight",
            "o": f"{p}.attn.wo_ud.weight",
            "q_norm": f"{p}.attn.q_norm.weight",
            "k_norm": f"{p}.attn.k_norm.weight",
            "rel_proj": f"{p}.attn.rel_logits_proj.proj",
            "k_sconv": f"{p}.attn.k_sconv.weight",
            "v_sconv": f"{p}.attn.v_sconv.weight",
            "attn_sconv": f"{p}.attn_sconv.weight",
            "mlp_sconv": f"{p}.mlp_sconv.weight",
        }
    return {
        "input_norm": f"{p}.input_layernorm.weight",
        "post_attention_norm": f"{p}.post_attention_layernorm.weight",
        "q": f"{p}.self_attn.q_proj.weight",
        "k": f"{p}.self_attn.k_proj.weight",
        "v": f"{p}.self_attn.v_proj.weight",
        "r": f"{p}.self_attn.r_proj.weight",
        "o": f"{p}.self_attn.o_proj.weight",
        "q_norm": f"{p}.self_attn.q_norm.weight",
        "k_norm": f"{p}.self_attn.k_norm.weight",
        "rel_proj": f"{p}.self_attn.rel_logits_proj.proj",
        "k_sconv": f"{p}.self_attn.k_sconv.conv1d.weight",
        "v_sconv": f"{p}.self_attn.v_sconv.conv1d.weight",
        "attn_sconv": f"{p}.attn_sconv.conv1d.weight",
        "mlp_sconv": f"{p}.mlp_sconv.conv1d.weight",
    }


def layer_dense_mlp_names(layer: int, dialect: str) -> dict[str, str]:
    """Source names for a dense layer's MLP. The provider ships gate and up
    fused into one row-interleaved tensor; the normalized dialect does not."""
    p = layer_prefix(layer, dialect)
    if dialect == "provider_raw":
        return {
            "fused_gate_up": f"{p}.mlp.w13_dn.weight",
            "down": f"{p}.mlp.w2_md.weight",
            "global_scale": f"{p}.mlp.global_scale",
        }
    return {
        "gate": f"{p}.mlp.gate_proj.weight",
        "up": f"{p}.mlp.up_proj.weight",
        "down": f"{p}.mlp.down_proj.weight",
        "global_scale": f"{p}.mlp.global_scale",
    }


def layer_sparse_mlp_names(layer: int, dialect: str) -> dict[str, dict[str, str]]:
    """Source names for a sparse layer, grouped as routed / router / shared."""
    p = layer_prefix(layer, dialect)
    if dialect == "provider_raw":
        return {
            "routed": {
                "fused_gate_up": f"{p}.mlp.experts.w13_weight",
                "down": f"{p}.mlp.experts.w2_weight",
            },
            "router": {
                "weight": f"{p}.mlp.gate.weight",
                "correction_bias": f"{p}.mlp.gate.bias",
                "global_scale": f"{p}.mlp.gate.global_scale",
            },
            "shared": {
                "fused_gate_up": f"{p}.mlp.shared_experts.shared_w13_weight",
                "down": f"{p}.mlp.shared_experts.shared_w2_weight",
            },
        }
    return {
        "routed": {
            "fused_gate_up": f"{p}.mlp.experts.gate_up_proj",
            "down": f"{p}.mlp.experts.down_proj",
        },
        "router": {
            "weight": f"{p}.mlp.gate.weight",
            "correction_bias": f"{p}.mlp.gate.e_score_correction_bias",
            "global_scale": f"{p}.mlp.gate.global_scale",
        },
        "shared": {
            "gate": f"{p}.mlp.shared_experts.gate_proj",
            "up": f"{p}.mlp.shared_experts.up_proj",
            "down": f"{p}.mlp.shared_experts.down_proj",
        },
    }


def build_plan(src: Path, *, require_payloads: bool = False) -> dict[str, Any]:
    source_report = inspect(src).data
    if not source_report["architecture"]["is_inkling"]:
        raise PlanError("source config does not explicitly identify Inkling")
    if not source_report["ready"]["text_metadata"]:
        raise PlanError("text metadata inventory has blockers; run inspect_inkling.py for details")
    if require_payloads and not source_report["ready"]["text_payloads"]:
        raise PlanError("required text shards or tensor headers are not present locally")

    cfg = load_json(src / "config.json", required=True)
    assert cfg is not None
    text = cfg.get("text_config")
    if not isinstance(text, dict):
        raise PlanError("Inkling text_config is missing")

    catalog = TensorCatalog(src)
    dialect = detect_dialect(set(catalog.names()))
    if dialect not in ("provider_raw", "transformers_normalized"):
        raise PlanError(f"unsupported tensor naming dialect: {dialect}")

    n_layers = _require_int(source_report, "num_hidden_layers")
    hidden = _require_int(source_report, "hidden_size")
    dense_inter = _require_int(source_report, "dense_intermediate_size")
    routed_inter = _require_int(source_report, "routed_intermediate_size")
    n_experts = _require_int(source_report, "num_routed_experts")
    top_k = _require_int(source_report, "num_experts_per_token")
    n_shared = _require_int(source_report, "num_shared_experts")
    dense_count = _require_int(source_report, "dense_mlp_idx", minimum=0)
    vocab = _require_int(source_report, "vocab_size")
    unpadded_vocab = _value(source_report, "unpadded_vocab_size")
    if not isinstance(unpadded_vocab, int) or unpadded_vocab <= 0:
        unpadded_vocab = vocab
    d_rel = _require_int(source_report, "relative_state_dim")
    rel_extent = _require_int(source_report, "relative_extent")
    global_heads = _require_int(source_report, "global_num_attention_heads")
    global_kv = _require_int(source_report, "global_num_key_value_heads")
    global_dim = _require_int(source_report, "global_head_dim")
    local_heads = _require_int(source_report, "local_num_attention_heads")
    local_kv = _require_int(source_report, "local_num_key_value_heads")
    local_dim = _require_int(source_report, "local_head_dim")
    local_window = _require_int(source_report, "sliding_window_size")
    conv_kernel = _require_int(source_report, "short_conv_kernel_size")
    local_ids = _value(source_report, "local_layer_ids")
    if not isinstance(local_ids, list) or any(not isinstance(x, int) for x in local_ids):
        raise PlanError("local_layer_ids is unavailable")
    local_set = set(local_ids)

    if dense_count > n_layers:
        raise PlanError(f"dense_mlp_idx={dense_count} exceeds num_hidden_layers={n_layers}")
    if top_k > n_experts:
        raise PlanError(f"top-k {top_k} exceeds routed expert count {n_experts}")
    if global_heads % global_kv or local_heads % local_kv:
        raise PlanError("attention head counts are not divisible by KV head counts")

    probes: list[dict[str, Any]] = []
    trunk: list[dict[str, Any]] = []
    expert_banks: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []

    if dialect == "provider_raw":
        core_names = {
            "embed": "model.llm.embed.weight",
            "embed_norm": "model.llm.embed_norm.weight",
            "final_norm": "model.llm.norm.weight",
            "unembed": "model.llm.unembed.weight",
        }
    else:
        core_names = {
            "embed": "model.language_model.embed_tokens.weight",
            "embed_norm": "model.language_model.embed_norm.weight",
            "final_norm": "model.language_model.norm.weight",
            "unembed": "lm_head.weight",
        }

    core_expected = {
        "embed": [vocab, hidden],
        "embed_norm": [hidden],
        "final_norm": [hidden],
        "unembed": [unpadded_vocab, hidden],
    }
    for role, name in core_names.items():
        probes.append(_probe(catalog, name, core_expected[role], f"official Inkling {role} module shape"))
        trunk.append({"role": role, "source": name, "storage": "resident trunk"})

    for layer in range(n_layers):
        is_local = layer in local_set
        heads = local_heads if is_local else global_heads
        kv_heads = local_kv if is_local else global_kv
        head_dim = local_dim if is_local else global_dim
        extent = local_window if is_local else rel_extent
        mlp_kind = "dense" if layer < dense_count else "sparse"
        p = _raw_layer_prefix(layer) if dialect == "provider_raw" else _hf_layer_prefix(layer)

        attn = layer_attention_names(layer, dialect)

        expected = {
            "input_norm": [hidden],
            "post_attention_norm": [hidden],
            "q": [heads * head_dim, hidden],
            "k": [kv_heads * head_dim, hidden],
            "v": [kv_heads * head_dim, hidden],
            "r": [heads * d_rel, hidden],
            "o": [hidden, heads * head_dim],
            "q_norm": [head_dim],
            "k_norm": [head_dim],
            "rel_proj": [d_rel, extent],
            "k_sconv": [kv_heads * head_dim, 1, conv_kernel],
            "v_sconv": [kv_heads * head_dim, 1, conv_kernel],
            "attn_sconv": [hidden, 1, conv_kernel],
            "mlp_sconv": [hidden, 1, conv_kernel],
        }
        for role, name in attn.items():
            probes.append(_probe(catalog, name, expected[role], f"layer {layer} {role}"))
            trunk.append({"role": f"layer.{layer}.{role}", "source": name, "storage": "resident trunk"})

        descriptor: dict[str, Any] = {
            "layer": layer,
            "attention": {
                "kind": "hybrid_sliding" if is_local else "hybrid_global",
                "num_heads": heads,
                "num_key_value_heads": kv_heads,
                "head_dim": head_dim,
                "relative_state_dim": d_rel,
                "relative_extent": extent,
                "sliding_window": local_window if is_local else None,
                "scale": "1/head_dim after per-head q/k RMSNorm",
                "log_scaling": not is_local and _value(source_report, "log_scaling_n_floor") is not None,
                "short_conv_kernel": conv_kernel,
                "short_conv_states": 4,
            },
            "mlp": {"kind": mlp_kind},
        }

        if mlp_kind == "dense":
            mlp = layer_dense_mlp_names(layer, dialect)
            if dialect == "provider_raw":
                expected_mlp = {
                    "fused_gate_up": [2 * dense_inter, hidden],
                    "down": [hidden, dense_inter],
                    "global_scale": [1],
                }
                transform = "interleave rows, then split rows into gate/up (official checkpoint mapping)"
            else:
                expected_mlp = {
                    "gate": [dense_inter, hidden],
                    "up": [dense_inter, hidden],
                    "down": [hidden, dense_inter],
                    "global_scale": [1],
                }
                transform = "already normalized"
            for role, name in mlp.items():
                probes.append(_probe(catalog, name, expected_mlp[role], f"layer {layer} dense MLP {role}"))
                trunk.append({"role": f"layer.{layer}.mlp.{role}", "source": name, "storage": "resident trunk"})
            descriptor["mlp"].update({"intermediate_size": dense_inter, "source_transform": transform})
        else:
            sparse_names = layer_sparse_mlp_names(layer, dialect)
            routed = sparse_names["routed"]
            router = sparse_names["router"]
            shared = sparse_names["shared"]

            routed_expected = {
                "fused_gate_up": [n_experts, 2 * routed_inter, hidden],
                "down": [n_experts, hidden, routed_inter],
            }
            router_expected = {
                "weight": [n_experts + n_shared, hidden],
                "correction_bias": [n_experts],
                "global_scale": [1],
            }
            shared_expected = (
                {
                    "fused_gate_up": [n_shared, 2 * routed_inter, hidden],
                    "down": [n_shared, hidden, routed_inter],
                }
                if "fused_gate_up" in shared
                else {
                    "gate": [n_shared, routed_inter, hidden],
                    "up": [n_shared, routed_inter, hidden],
                    "down": [n_shared, hidden, routed_inter],
                }
            )
            for role, name in routed.items():
                probes.append(_probe(catalog, name, routed_expected[role], f"layer {layer} routed experts {role}"))
            for role, name in router.items():
                probes.append(_probe(catalog, name, router_expected[role], f"layer {layer} router {role}"))
                trunk.append({"role": f"layer.{layer}.router.{role}", "source": name, "storage": "resident trunk"})
            for role, name in shared.items():
                probes.append(_probe(catalog, name, shared_expected[role], f"layer {layer} shared experts {role}"))
                trunk.append({"role": f"layer.{layer}.shared.{role}", "source": name, "storage": "resident trunk"})

            expert_banks.append(
                {
                    "layer": layer,
                    "file": f"experts-L{layer}.bin",
                    "experts": n_experts,
                    "top_k": top_k,
                    "source": routed,
                    "source_transform": (
                        "interleave fused gate/up rows on axis 1, split per expert into WASTE gate/up/down"
                        if dialect == "provider_raw"
                        else "split normalized gate_up projection per expert into WASTE gate/up/down"
                    ),
                    "storage": "streamed routed experts; one aligned record per expert",
                }
            )
            descriptor["mlp"].update(
                {
                    "intermediate_size": routed_inter,
                    "routed_experts": n_experts,
                    "top_k": top_k,
                    "shared_experts": n_shared,
                    "shared_storage": "resident trunk",
                    "routing": "sigmoid choice bias; joint log-sigmoid normalization over selected routed + all shared",
                }
            )
        layers.append(descriptor)

    mismatches = [item for item in probes if item["status"] == "mismatch"]
    unknown_shapes = [item for item in probes if item["status"] == "unknown"]
    if mismatches:
        first = mismatches[0]
        raise PlanError(
            f"{len(mismatches)} tensor shape probes failed; first is {first['name']}: "
            f"expected {first['expected_shape']}, got {first['actual_shape']}"
        )
    if require_payloads and unknown_shapes:
        raise PlanError(f"{len(unknown_shapes)} required tensor shapes are unavailable")

    # Exclusions are explicit and prefix-based.  A future converter must audit
    # every source tensor left over after applying these rules.
    exclusions = [
        {"pattern": "model.mtp.*", "reason": "exclude until base-model text logits match"},
        {"pattern": "model.visual.* / model.vision_tower.*", "reason": "defer vision milestone"},
        {"pattern": "model.audio.* / model.audio_tower.*", "reason": "defer audio milestone"},
    ]

    sidecars = {
        "tokenizer": source_report["assets"].get("tokenizer"),
        "chat_template": source_report["assets"].get("chat_template"),
        "special_token_ids": {
            name: _value(source_report, name)
            for name in (
                "eos_token_id",
                "image_token_id",
                "audio_token_id",
                "image_bos_token_id",
                "audio_bos_token_id",
            )
        },
        "policy": "copy official assets verbatim; raw mode when no official chat template exists",
    }

    try:
        release = inspect_release(src)
    except ReleaseError as exc:
        release = {"official_small": False, "error": str(exc)}

    plan = {
        "schema": "waste.inkling-conversion-plan.v2",
        "source": str(src),
        "source_dialect": dialect,
        "mode": "text-only-bf16-parity-first",
        "status": {
            "metadata_complete": source_report["ready"]["text_metadata"],
            "payload_headers_complete": source_report["ready"]["text_payloads"],
            "shape_probes": {
                "total": len(probes),
                "ok": sum(item["status"] == "ok" for item in probes),
                "unknown": len(unknown_shapes),
                "mismatch": 0,
            },
            "container_written": False,
            "runtime_supported": False,
            "reference_parity": False,
        },
        "release": release,
        "model": {
            "family": "inkling",
            "profile": (
                "official-inkling-small"
                if release.get("profile", {}).get("match")
                else "generic-inkling"
            ),
            "num_hidden_layers": n_layers,
            "hidden_size": hidden,
            "vocab_size": vocab,
            "unpadded_vocab_size": unpadded_vocab,
            "maximum_configured_context": _value(source_report, "model_max_length"),
            "routed_experts": n_experts,
            "top_k": top_k,
            "shared_experts": n_shared,
            "dense_mlp_layers": list(range(dense_count)),
            "sparse_mlp_layers": list(range(dense_count, n_layers)),
            "local_attention_layers": sorted(local_set),
            "global_attention_layers": [i for i in range(n_layers) if i not in local_set],
        },
        "manifest_preview": {
            "format_version": 0,
            "arch": "inkling",
            "tensor_prefix": "",
            "config": cfg,
            "inkling": {
                "schema_version": 1,
                "layers": layers,
                "routed_expert_storage": "streamed",
                "shared_expert_storage": "resident",
            },
            "note": "preview only; the current WASTE loader must reject this until WASTE_ARCH_INKLING is implemented",
        },
        "trunk": trunk,
        "expert_banks": expert_banks,
        "shape_probes": probes,
        "sidecars": sidecars,
        "exclusions": exclusions,
        "next_gate": "implement payload conversion only after this plan has zero unknown/mismatched shapes; implement C runtime before claiming load/inference support",
    }
    return plan


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="write conversion-plan JSON")
    ap.add_argument(
        "--require-payloads",
        action="store_true",
        help="fail unless all required base-text shards and headers are local",
    )
    args = ap.parse_args(argv)
    try:
        plan = build_plan(args.src, require_payloads=args.require_payloads)
    except (InspectError, PlanError) as exc:
        print(f"inkling_plan: {exc}", file=sys.stderr)
        return 2
    if args.out:
        write_json_atomic(args.out, plan)
    status = plan["status"]
    print(
        f"Inkling plan: {plan['model']['num_hidden_layers']} layers, "
        f"{len(plan['expert_banks'])} streamed expert banks, "
        f"shape probes {status['shape_probes']['ok']}/{status['shape_probes']['total']} confirmed, "
        f"{status['shape_probes']['unknown']} unknown"
    )
    print("container written: no; runtime supported: no; parity established: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
