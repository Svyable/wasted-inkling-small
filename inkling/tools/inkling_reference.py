#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture named activations from the official Transformers Inkling model.

This is the trusted-reference side of the parity protocol.  It intentionally
uses the official Transformers implementation and therefore may require a
large-memory or provider-supported runtime for the complete checkpoint.  The
output is the same CRC-protected archive consumed by inkling_parity.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import torch

from inkling_parity import write_activation_archive


def _last_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _last_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _token_vector(value: Any) -> torch.Tensor | None:
    tensor = _last_tensor(value)
    if tensor is None:
        return None
    tensor = tensor.detach().float().cpu()
    while tensor.ndim > 1:
        tensor = tensor[-1]
    return tensor.contiguous()


def _text_model(model: Any) -> Any:
    root = getattr(model, "model", model)
    for name in ("llm", "language_model", "text_model"):
        child = getattr(root, name, None)
        if child is not None and hasattr(child, "layers"):
            return child
    if hasattr(root, "layers"):
        return root
    raise RuntimeError("cannot locate Inkling text decoder layers")


def register_reference_hooks(model: Any, layers: list[int], store: dict[str, torch.Tensor],
                             position_getter: Callable[[], int]) -> list[Any]:
    text = _text_model(model)
    handles = []

    def store_tensor(layer: int, point: str, value: Any,
                     transform: Callable[[Any], torch.Tensor | None]) -> None:
        tensor = transform(value)
        if tensor is not None:
            scope = "model" if layer < 0 else f"layer.{layer}"
            store[f"token.{position_getter()}.{scope}.{point}"] = tensor

    def add(module: Any, layer: int, point: str,
            transform: Callable[[Any], torch.Tensor | None] = _token_vector):
        if module is None:
            return
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            store_tensor(layer, point, output, transform)
        handles.append(module.register_forward_hook(hook))

    def add_pre(module: Any, layer: int, point: str,
                transform: Callable[[Any], torch.Tensor | None] = _token_vector):
        if module is None:
            return
        def hook(_module: Any, inputs: Any) -> None:
            if inputs:
                store_tensor(layer, point, inputs[0], transform)
        handles.append(module.register_forward_pre_hook(hook))

    add(getattr(text, "embed_norm", None), -1, "embedding_norm")
    add(getattr(text, "norm", None), -1, "final_norm")
    for layer_id in layers:
        layer = text.layers[layer_id]
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        add(getattr(layer, "input_layernorm", None), layer_id, "input_norm")
        for attr, point in (("q_proj", "q_proj"), ("k_proj", "k_proj"),
                            ("v_proj", "v_proj"), ("r_proj", "relative_proj_input"),
                            ("k_sconv", "k_sconv"), ("v_sconv", "v_sconv")):
            add(getattr(attn, attr, None), layer_id, point)
        # C's attention_out is the value entering o_proj, not the output of the
        # whole attention module. Keep the trusted-reference name on that exact
        # arithmetic boundary so probes cannot compare different tensors under
        # the same label.
        add_pre(getattr(attn, "o_proj", None), layer_id, "attention_out")
        add(getattr(layer, "attn_sconv", None), layer_id, "attention_branch")
        # The outer attention residual is the input to post-attention RMSNorm.
        # Capture it explicitly; a forward hook on the norm only sees the
        # normalized value and cannot reconstruct this boundary exactly.
        add_pre(getattr(layer, "post_attention_layernorm", None), layer_id,
                "post_attention_residual")
        add(getattr(layer, "post_attention_layernorm", None), layer_id, "post_attention_norm")

        gate = getattr(mlp, "gate", None)
        if gate is not None:
            def router_transform(output: Any, lid: int = layer_id) -> torch.Tensor | None:
                if not isinstance(output, (tuple, list)) or len(output) < 4:
                    return None
                names = ("router_logits", "routed_weight", "routed_index", "shared_weight")
                for name, value in zip(names, output[:4]):
                    tensor = _token_vector(value)
                    if tensor is not None:
                        if name == "routed_index": tensor = tensor.to(torch.int32)
                        store[f"token.{position_getter()}.layer.{lid}.{name}"] = tensor
                return None
            add(gate, layer_id, "router", router_transform)
        add(mlp, layer_id, "dense_mlp_out" if layer_id < getattr(text.config, "dense_mlp_idx", 2) else "moe_out")
        add(getattr(layer, "mlp_sconv", None), layer_id, "mlp_branch")
        add(layer, layer_id, "layer_out")
    return handles


def run_reference(src: Path | str, tokens: list[int], out: Path | str, *,
                  layers: list[int], device_map: str = "auto",
                  dtype: str = "bfloat16") -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers with Inkling support is required") from exc
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=torch_dtype, device_map=device_map,
        low_cpu_mem_usage=True, trust_remote_code=False,
    )
    model.eval()
    store: dict[str, torch.Tensor] = {}
    position = [0]
    handles = register_reference_hooks(model, layers, store, lambda: position[0])
    past = None
    try:
        with torch.no_grad():
            for i, token in enumerate(tokens):
                position[0] = i
                device = next(model.parameters()).device
                result = model(input_ids=torch.tensor([[token]], device=device),
                               past_key_values=past, use_cache=True)
                past = result.past_key_values
                store[f"token.{i}.model.logits"] = result.logits[0, -1].detach().float().cpu().contiguous()
                # The official forward divides by this factor immediately before lm_head.
                final = store.get(f"token.{i}.model.final_norm")
                width = float(model.config.get_text_config().logits_mup_width_multiplier)
                if final is not None:
                    store[f"token.{i}.model.final_norm_scaled"] = final / width
    finally:
        for handle in handles:
            handle.remove()
    return write_activation_archive(
        out, store, metadata={"runtime": "transformers-official", "tokens": tokens,
                              "layers": layers, "dtype": dtype},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--layers", default="0,1,2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = ap.parse_args()
    tokens = [int(x) for x in args.tokens.split(",") if x]
    layers = [int(x) for x in args.layers.split(",") if x]
    if not tokens or not layers:
        ap.error("--tokens and --layers must be nonempty")
    run_reference(args.src, tokens, args.out, layers=layers,
                  device_map=args.device_map, dtype=args.dtype)
    print(f"wrote official reference trace to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())