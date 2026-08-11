#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Write a synthetic *public* Inkling container.

The converter turns a 495 GiB checkpoint into one of these. That is the right
way to get a real container and the wrong way to test the loader: nothing in
CI can spend a datacenter to find out whether an offset was read as the wrong
width. So this writes a container of the same shape at a geometry that fits in
a few kilobytes — same manifest schema, same `trunk.bin` layout, same
canonical tensor names, same quantized widths and scale tables.

It is also the executable half of the format description. If you want to know
what a public Inkling container is, this file says so in less space than prose
can, and unlike prose it is checked by the suite that reads what it writes.

The weights are deterministic nonsense. This produces a container that loads,
plans and binds; it does not produce a model, and nothing downstream should
read a token out of it.

    python3 tools/make_inkling_container.py --out /tmp/ink --trunk-bits 8

Dependency-free on purpose: it runs in the fast CI tier, where torch does not.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

# fmt codes as `trunk[].fmt` records them, matching src/model.c's load_trunk.
FMT_F32, FMT_Q8G, FMT_Q4G = 0, 2, 3

# The two vocabulary tables the engine reads one row of per token. The loader
# leaves them on disk when they are quantized and the planner leaves them out
# of the resident floor under the same condition — see waste_arch_row_backed.
ROW_BACKED = ("inkling.embed", "inkling.unembed")

# Tensors that must be resident f32 because inkling_bind.c dereferences them
# directly rather than going through a matrix backend: every norm, bias and
# scalar, and — less obviously — the short-convolution kernels and the
# relative-position projection, which are 2-D but are not matmuls.
def resident_f32(name: str) -> bool:
    tail = name.rsplit(".", 1)[-1]
    return tail in (
        "embed_norm", "final_norm", "input_norm", "post_attention_norm",
        "q_norm", "k_norm", "rel_proj", "k_sconv", "v_sconv", "attn_sconv",
        "mlp_sconv", "global_scale", "correction_bias",
    )


class Rng:
    """A tiny deterministic PRNG, so the container is reproducible byte for
    byte without pulling in numpy for values nobody interprets."""

    def __init__(self, seed: int) -> None:
        self.s = seed & 0xFFFFFFFF or 1

    def uniform(self) -> float:
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s / 0x7FFFFFFF * 2.0 - 1.0


def f32_to_f16_bits(x: float) -> int:
    """IEEE binary16 with round-to-nearest-even, as the engine's f16_to_f32
    expects to read back. Group scales are always positive here, so the
    subnormal and infinity paths exist for completeness rather than use."""
    if x == 0.0:
        return 0
    (bits,) = struct.unpack("<I", struct.pack("<f", x))
    sign = (bits >> 16) & 0x8000
    exp = ((bits >> 23) & 0xFF) - 127 + 15
    man = bits & 0x7FFFFF
    if exp >= 0x1F:
        return sign | 0x7C00
    if exp <= 0:
        return sign
    half = sign | (exp << 10) | (man >> 13)
    if (man & 0x1FFF) > 0x1000 or ((man & 0x1FFF) == 0x1000 and (half & 1)):
        half += 1
    return half & 0xFFFF


def quantize_rows(values, rows: int, cols: int, group: int, bits: int):
    """One row-major tensor into the packed payload and its fp16 group scales.

    The layout is the container's, not a convenience: `ng` groups of exactly
    `group` weights per row, padded past the real width, because that is the
    stride waste_deq_row indexes with. A tail group holding fewer real values
    still occupies a whole one.
    """
    ng = (cols + group - 1) // group
    payload = bytearray()
    scales = bytearray()
    for r in range(rows):
        row = values[r * cols:(r + 1) * cols]
        packed = bytearray()
        for g in range(ng):
            chunk = row[g * group:(g + 1) * group]
            chunk = list(chunk) + [0.0] * (group - len(chunk))
            peak = max((abs(v) for v in chunk), default=0.0)
            lim = 127.0 if bits == 8 else 7.0
            scale = peak / lim if peak > 0 else 1.0
            scales += struct.pack("<H", f32_to_f16_bits(scale))
            codes = [max(-int(lim) - (1 if bits == 4 else 0),
                         min(int(lim), int(round(v / scale)))) for v in chunk]
            if bits == 8:
                packed += struct.pack("<%db" % group, *codes)
            else:
                # Two weights per byte, low nibble first, stored biased by 8
                # so the unpacker's `(byte & 0x0F) - 8` recovers the code.
                for i in range(0, group, 2):
                    lo = (codes[i] + 8) & 0x0F
                    hi = (codes[i + 1] + 8) & 0x0F
                    packed.append(lo | (hi << 4))
        payload += packed
    return bytes(payload), bytes(scales), ng


class Trunk:
    """trunk.bin, accumulated tensor by tensor."""

    def __init__(self, group: int) -> None:
        self.blob = bytearray()
        self.entries: list[dict] = []
        self.group = group

    def _place(self, data: bytes) -> int:
        while len(self.blob) % 64:
            self.blob.append(0)
        off = len(self.blob)
        self.blob += data
        return off

    def add(self, name: str, shape, values, bits: int) -> None:
        n = math.prod(shape)
        assert len(values) == n, name
        if bits == 32:
            off = self._place(struct.pack("<%df" % n, *values))
            self.entries.append({"name": name, "fmt": FMT_F32, "off": off,
                                 "shape": list(shape), "bytes": n * 4})
            return
        cols = shape[-1]
        rows = n // cols
        payload, scales, ng = quantize_rows(values, rows, cols, self.group, bits)
        off = self._place(payload)
        soff = self._place(scales)
        self.entries.append({
            "name": name, "fmt": FMT_Q8G if bits == 8 else FMT_Q4G,
            "off": off, "shape": list(shape), "group": self.group,
            "scale_off": soff, "bytes": len(payload) + len(scales),
        })
        assert ng * self.group * bits // 8 * rows == len(payload)


def build(out: Path, cfg: dict, trunk_bits: int, group: int, seed: int) -> dict:
    rng = Rng(seed)

    def vals(n: int):
        return [rng.uniform() for _ in range(n)]

    n_layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    dense = cfg["dense_mlp_idx"]
    local = set(cfg["local_layer_ids"])
    conv_k = cfg["sconv_kernel_size"]
    d_rel = cfg["d_rel"]
    routed, shared = cfg["n_routed_experts"], cfg["n_shared_experts"]
    moe_inter, dense_inter = cfg["intermediate_size"], cfg["dense_intermediate_size"]

    trunk = Trunk(group)

    def add(name, shape):
        # A matrix keeps its stored width; everything the binder dereferences
        # directly has to be f32 whatever --trunk-bits says.
        bits = 32 if resident_f32(name) else trunk_bits
        trunk.add(name, shape, vals(math.prod(shape)), bits)

    add("inkling.embed", (cfg["vocab_size"], hidden))
    add("inkling.embed_norm", (hidden,))
    add("inkling.final_norm", (hidden,))
    add("inkling.unembed", (cfg["unpadded_vocab_size"], hidden))

    for i in range(n_layers):
        is_local = i in local
        heads = cfg["swa_num_attention_heads"] if is_local else cfg["num_attention_heads"]
        kv = cfg["swa_num_key_value_heads"] if is_local else cfg["num_key_value_heads"]
        hd = cfg["swa_head_dim"] if is_local else cfg["head_dim"]
        extent = cfg["sliding_window_size"] if is_local else cfg["rel_extent"]
        p = "inkling.layer.%d." % i
        add(p + "input_norm", (hidden,))
        add(p + "post_attention_norm", (hidden,))
        add(p + "q", (heads * hd, hidden))
        add(p + "k", (kv * hd, hidden))
        add(p + "v", (kv * hd, hidden))
        add(p + "r", (heads * d_rel, hidden))
        add(p + "o", (hidden, heads * hd))
        add(p + "q_norm", (hd,))
        add(p + "k_norm", (hd,))
        add(p + "rel_proj", (d_rel, extent))
        add(p + "k_sconv", (kv * hd, conv_k))
        add(p + "v_sconv", (kv * hd, conv_k))
        add(p + "attn_sconv", (hidden, conv_k))
        add(p + "mlp_sconv", (hidden, conv_k))
        if i < dense:
            add(p + "mlp.gate", (dense_inter, hidden))
            add(p + "mlp.up", (dense_inter, hidden))
            add(p + "mlp.down", (hidden, dense_inter))
            add(p + "mlp.global_scale", (1,))
        else:
            add(p + "router.weight", (routed + shared, hidden))
            add(p + "router.correction_bias", (routed,))
            add(p + "router.global_scale", (1,))
            add(p + "shared.gate", (shared, moe_inter, hidden))
            add(p + "shared.up", (shared, moe_inter, hidden))
            add(p + "shared.down", (shared, hidden, moe_inter))

    out.mkdir(parents=True, exist_ok=True)
    (out / "trunk.bin").write_bytes(bytes(trunk.blob))

    # Expert quantization. The records themselves are placeholders sized
    # exactly as the manifest declares: the loader validates that geometry and
    # opens nothing, and the promotion that reads them is the next one. Sized
    # rather than omitted so the container is the right shape on disk.
    stages, vec_dim, entries, index_block = 3, 8, 256, 16
    rec_bytes = 4096
    layers = []
    for L in range(dense, n_layers):
        name = "experts-L%d.bin" % L
        (out / name).write_bytes(b"\0" * (rec_bytes * routed))
        layers.append({"file": name, "layer": L, "experts": routed,
                       "bytes": rec_bytes * routed, "codebook_base": 3 * stages * (L - dense)})

    books = 3 * stages * (n_layers - dense)
    book = bytearray()
    for _ in range(books):
        book += b"\0" * 16
        book += b"".join(struct.pack("<H", f32_to_f16_bits(rng.uniform()))
                         for _ in range(entries * vec_dim))
    (out / "codebooks.bin").write_bytes(bytes(book))

    manifest = {
        "format_version": 0,
        "arch": "inkling",
        "tensor_prefix": "",
        "config": cfg,
        "expert_quant": {"stages": stages, "vec_dim": vec_dim,
                         "entries": entries, "index_block": index_block,
                         "bits_per_weight": 3.0},
        "trunk": trunk.entries,
        "layers": layers,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


TINY = {
    # Small enough to write in a few hundred kilobytes, wide enough that a
    # quantized matvec and its f32 reference differ by rounding rather than
    # by luck. Every layer class the binder distinguishes is present: dense
    # (0, 1), sparse local (2), sparse global (3).
    "num_hidden_layers": 4,
    "hidden_size": 32,
    "vocab_size": 48,
    "unpadded_vocab_size": 40,
    # The CLI defaults to a 4096-token context and both the planner and the
    # loader refuse one the model cannot hold, so a toy container still has
    # to declare a real one.
    "model_max_length": 4096,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 8,
    "sliding_window_size": 8,
    "d_rel": 4,
    "rel_extent": 16,
    "sconv_kernel_size": 4,
    "dense_mlp_idx": 2,
    "local_layer_ids": [2],
    "dense_intermediate_size": 64,
    "intermediate_size": 16,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 2,
    "rms_norm_eps": 1e-6,
    "route_scale": 8.0,
    "logits_mup_width_multiplier": 16.0,
    "log_scaling_n_floor": 128000,
    "log_scaling_alpha": 0.1,
    "eos_token_id": 11,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--trunk-bits", type=int, default=8, choices=(4, 8, 32),
                    help="stored width of the matrices; 32 keeps them f32")
    ap.add_argument("--group", type=int, default=8,
                    help="weights per quantization scale")
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--config", type=Path,
                    help="a text config to use instead of the tiny default")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text()) if args.config else dict(TINY)
    if "text_config" in cfg:
        cfg = cfg["text_config"]
    m = build(args.out, cfg, args.trunk_bits, args.group, args.seed)
    print("%s: %d tensors, %d expert banks, trunk %d bytes"
          % (args.out, len(m["trunk"]), len(m["layers"]),
             (args.out / "trunk.bin").stat().st_size))


if __name__ == "__main__":
    main()
