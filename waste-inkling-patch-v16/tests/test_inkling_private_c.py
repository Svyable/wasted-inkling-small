#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import ctypes
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

from inkling_runtime_stage import _bank_bytes, _header_bytes, _layer_bytes, _tensor_bytes, _qtensor_bytes
from inkling_qtrunk import FMT_Q4G, FMT_Q8G, HEADER as QHEADER, _align as qalign, _header as qheader, _policy as qpolicy, _quantize_rows
from inkling_vq import (
    VQSpec, dequantize_record, quantize_matrix, read_codebooks,
    train_codebooks, write_codebooks, write_expert_record,
)
from test_inkling_layer_c import FP, ExpertCallback, InklingLayerCDifferentialTest, Weights as LayerWeights, array
from test_inkling_model_c import ModelScratch, ModelState, ModelWeights


class PrivateOptions(ctypes.Structure):
    _fields_ = [
        ("context_capacity", ctypes.c_int),
        ("verify_crc", ctypes.c_int),
        ("require_official_small", ctypes.c_int),
    ]


def rounded(t):
    return t.detach().to(torch.bfloat16).float().contiguous()


def bf16_bytes(t):
    return rounded(t).to(torch.bfloat16).view(torch.uint16).numpy().tobytes()


def write_tensor(root, ordinal, name, value):
    value = rounded(value)
    payload = bf16_bytes(value)
    shape = list(value.shape)
    padded = shape + [0] * (4 - len(shape))
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = struct.pack(
        "<4sHBBHH4IQII20s", b"IKTN", 1, 1, len(shape), 0, 0,
        *padded, len(payload), crc, zlib.crc32(name.encode()) & 0xFFFFFFFF,
        b"\0" * 20,
    )
    stored = (64 + len(payload) + 4095) // 4096 * 4096
    file_name = f"{ordinal:05d}.tensor.stage"
    path = root / "trunk-stage" / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + payload + b"\0" * (stored - 64 - len(payload)))
    return {
        "target": name,
        "file": file_name,
        "dtype": "BF16",
        "shape": shape,
        "payload_bytes": len(payload),
        "stored_bytes": stored,
        "crc32": crc,
    }


def write_bank(root, layer, cfg, extra):
    matrix = cfg.hidden * cfg.moe_intermediate
    payload_bytes = matrix * 6
    record_bytes = (64 + payload_bytes + 4095) // 4096 * 4096
    path = root / f"experts-L{layer}.bf16.stage"
    records = []
    for expert in range(cfg.n_routed_experts):
        gate = bf16_bytes(extra["routed_gate"][expert])
        up = bf16_bytes(extra["routed_up"][expert])
        down = bf16_bytes(extra["routed_down"][expert])
        payload = gate + up + down
        header = struct.pack(
            "<IHHHBBIIQQQQI8x",
            0x46424B49, 1, layer, expert, 1, 0,
            cfg.hidden, cfg.moe_intermediate,
            64, 64 + len(gate), 64 + len(gate) + len(up), len(payload),
            zlib.crc32(payload) & 0xFFFFFFFF,
        )
        records.append(header + payload + b"\0" * (record_bytes - 64 - len(payload)))
    path.write_bytes(b"".join(records))
    return {
        "file": path.name,
        "layer": layer,
        "experts": cfg.n_routed_experts,
        "hidden_size": cfg.hidden,
        "intermediate_size": cfg.moe_intermediate,
        "record_bytes": record_bytes,
        "bytes": path.stat().st_size,
    }


def write_vq_bank(root, layer, cfg, extra, spec):
    kinds = (("gate", "routed_gate"), ("up", "routed_up"), ("down", "routed_down"))
    books = {}
    for kind_index, (book_name, source_name) in enumerate(kinds):
        chunks = []
        for matrix in extra[source_name]:
            matrix = matrix.float().contiguous()
            scale = matrix.abs().amax(1, keepdim=True).clamp_min(1e-8)
            chunks.append((matrix / scale).reshape(-1, spec.vec_dim))
        books[book_name] = train_codebooks(
            torch.cat(chunks), spec, iterations=3, assign_chunk=64,
            seed=1000 + layer * 10 + kind_index,
        )
    base = (layer - cfg.dense_layers) * 3 * spec.stages
    codebook_path = root / f"codebooks-L{layer}.bin"
    write_codebooks(codebook_path, base, books, spec)
    bank_path = root / f"experts-L{layer}.bin"
    record_bytes = 0
    with bank_path.open("wb") as f:
        for expert in range(cfg.n_routed_experts):
            matrices = [
                quantize_matrix(extra[source_name][expert], books[book_name], spec,
                                assign_chunk=64)
                for book_name, source_name in kinds
            ]
            record_bytes = write_expert_record(f, layer, expert, base, matrices, spec)
    loaded = read_codebooks(codebook_path, spec)
    shapes = (
        (cfg.moe_intermediate, cfg.hidden),
        (cfg.moe_intermediate, cfg.hidden),
        (cfg.hidden, cfg.moe_intermediate),
    )
    decoded = [[], [], []]
    raw = bank_path.read_bytes()
    for expert in range(cfg.n_routed_experts):
        weights = dequantize_record(
            raw[expert * record_bytes : (expert + 1) * record_bytes],
            loaded, shapes, spec, expected_layer=layer,
            expected_expert=expert, verify_crc=True,
        )
        for target, value in zip(decoded, (weights.gate, weights.up, weights.down)):
            target.append(value)
    item = {
        "file": bank_path.name,
        "codebooks_file": codebook_path.name,
        "layer": layer,
        "experts": cfg.n_routed_experts,
        "hidden_size": cfg.hidden,
        "intermediate_size": cfg.moe_intermediate,
        "record_bytes": record_bytes,
        "bytes": bank_path.stat().st_size,
        "codebook_base": base,
    }
    return item, tuple(torch.stack(values) for values in decoded)


def write_qtensor(root, ordinal, name, value, bits=4, group=4):
    value = rounded(value).float().contiguous()
    shape = list(value.shape)
    rows = int(torch.tensor(shape[:-1]).prod().item())
    cols = shape[-1]
    fmt = FMT_Q8G if name in ("inkling.embed", "inkling.unembed") or name.endswith("router.weight") else (FMT_Q8G if bits == 8 else FMT_Q4G)
    qb, sb = _quantize_rows(value.reshape(rows, cols), fmt, group)
    qcrc = zlib.crc32(qb) & 0xFFFFFFFF
    scrc = zlib.crc32(sb) & 0xFFFFFFFF
    file_name = f"{ordinal:05d}.qtensor"
    path = root / "qtrunk-stage" / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    head = qheader(name, fmt, group, shape, rows, cols, len(qb), len(sb), qcrc, scrc)
    stored = qalign(len(head) + len(qb) + len(sb))
    path.write_bytes(head + qb + sb + b"\0" * (stored - len(head) - len(qb) - len(sb)))
    scales = torch.frombuffer(bytearray(sb), dtype=torch.float16).float().reshape(rows, -1)
    padded = scales.shape[1] * group
    if fmt == FMT_Q8G:
        q = torch.frombuffer(bytearray(qb), dtype=torch.int8).reshape(rows, padded).float()
    else:
        packed = torch.frombuffer(bytearray(qb), dtype=torch.uint8).reshape(rows, padded // 2)
        q = torch.empty(rows, padded, dtype=torch.float32)
        q[:, 0::2] = (packed & 15).float() - 8
        q[:, 1::2] = (packed >> 4).float() - 8
    deq = (q.reshape(rows, -1, group) * scales.unsqueeze(-1)).reshape(rows, padded)[:, :cols].reshape(shape).contiguous()
    item = {
        "target": name, "file": file_name, "shape": shape,
        "bits": 8 if fmt == FMT_Q8G else 4, "group": group,
        "rows": rows, "cols": cols, "qbytes": len(qb),
        "scale_bytes": len(sb), "stored_bytes": stored,
        "q_crc32": qcrc, "scale_crc32": scrc,
    }
    return item, deq


def overwrite_pointer(ptr, value):
    a = array(value)
    ctypes.memmove(ptr, a, value.numel() * ctypes.sizeof(ctypes.c_float))
    return a


class InklingPrivateOpenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shutil.which("cc")
        if not cc:
            raise unittest.SkipTest("C compiler unavailable")
        cls.build = tempfile.TemporaryDirectory()
        so = Path(cls.build.name) / "libinkling_private.so"
        subprocess.run(
            [
                cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                f"-I{REPO / 'src'}",
                *[str(REPO / "src" / name) for name in (
                    "inkling_private.c", "inkling_stage_reader.c", "inkling_wexp.c", "inkling_qtensor.c", "inkling_bind.c",
                    "inkling_model.c", "inkling_layer.c", "inkling_attention.c",
                    "inkling_config.c", "inkling.c",
                )],
                "-lm", "-o", str(so),
            ],
            check=True, capture_output=True,
        )
        cls.lib = ctypes.CDLL(str(so))
        cls.lib.waste_inkling_config_build.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls.lib.waste_inkling_config_build.restype = ctypes.c_int
        cls.lib.waste_inkling_model_state_floats.argtypes = [ctypes.c_void_p, ctypes.c_int]
        cls.lib.waste_inkling_model_state_floats.restype = ctypes.c_size_t
        cls.lib.waste_inkling_model_scratch_floats.argtypes = [ctypes.c_void_p, ctypes.c_int]
        cls.lib.waste_inkling_model_scratch_floats.restype = ctypes.c_size_t
        cls.lib.waste_inkling_model_scratch_ints.argtypes = [ctypes.c_void_p]
        cls.lib.waste_inkling_model_scratch_ints.restype = ctypes.c_size_t
        cls.lib.waste_inkling_model_state_init.argtypes = [
            ctypes.POINTER(ModelState), ctypes.c_void_p, ctypes.c_int, FP, ctypes.c_size_t,
        ]
        cls.lib.waste_inkling_model_state_init.restype = ctypes.c_int
        cls.lib.waste_inkling_model_scratch_init.argtypes = [
            ctypes.POINTER(ModelScratch), ctypes.c_void_p, ctypes.c_int,
            FP, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
        ]
        cls.lib.waste_inkling_model_scratch_init.restype = ctypes.c_int
        cls.lib.waste_inkling_model_step.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ModelWeights), ctypes.POINTER(ModelState),
            ctypes.POINTER(ModelScratch), ctypes.c_int, ctypes.c_int, FP,
            ctypes.c_size_t, ExpertCallback, ctypes.c_void_p,
        ]
        cls.lib.waste_inkling_model_step.restype = ctypes.c_int
        cls.lib.waste_inkling_model_reset.argtypes = [ctypes.POINTER(ModelState), ctypes.c_void_p]
        cls.lib.waste_inkling_private_open.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(PrivateOptions),
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t,
        ]
        cls.lib.waste_inkling_private_open.restype = ctypes.c_int
        cls.lib.waste_inkling_private_step.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, FP, ctypes.c_size_t,
        ]
        cls.lib.waste_inkling_private_step.restype = ctypes.c_int
        cls.lib.waste_inkling_private_reset.argtypes = [ctypes.c_void_p]
        cls.lib.waste_inkling_private_close.argtypes = [ctypes.c_void_p]
        cls.lib.waste_inkling_private_quantized_trunk_bytes.argtypes = [ctypes.c_void_p]
        cls.lib.waste_inkling_private_quantized_trunk_bytes.restype = ctypes.c_size_t
        cls.lib.waste_inkling_private_resident_f32_bytes.argtypes = [ctypes.c_void_p]
        cls.lib.waste_inkling_private_resident_f32_bytes.restype = ctypes.c_size_t

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def fixture(self):
        # Reuse the already differential-tested tiny architecture and generate
        # one BF16-rounded copy for both the direct and staged runtimes.
        helper = InklingLayerCDifferentialTest(methodName="runTest")
        helper.lib = self.lib
        cfg, local_keep = helper.build_config()
        layer_weights = (LayerWeights * cfg.n_layers)()
        all_common = []
        all_extra = []
        keep = [local_keep, layer_weights]
        tensors = {}
        for layer in range(cfg.n_layers):
            w, common, layer_keep = helper.make_common(layer, cfg)
            common = {k: rounded(v) for k, v in common.items()}
            layer_keep = {k: array(v) for k, v in common.items()}
            for field, ptr in layer_keep.items():
                setattr(w, field, ptr)
            torch.manual_seed(900 + layer)
            if layer < cfg.dense_layers:
                extra = {
                    "dense_gate": rounded(torch.randn(cfg.dense_intermediate, cfg.hidden) * 0.09),
                    "dense_up": rounded(torch.randn(cfg.dense_intermediate, cfg.hidden) * 0.09),
                    "dense_down": rounded(torch.randn(cfg.hidden, cfg.dense_intermediate) * 0.09),
                    "dense_scale": rounded(torch.tensor([0.85])),
                }
                for name in ("dense_gate", "dense_up", "dense_down"):
                    layer_keep[name] = array(extra[name])
                    setattr(w, name, layer_keep[name])
                layer_keep["dense_scale"] = array(extra["dense_scale"])
                w.dense_global_scale = layer_keep["dense_scale"]
                w.sparse = 0
            else:
                total = cfg.n_routed_experts + cfg.n_shared_experts
                extra = {
                    "router_weight": rounded(torch.randn(total, cfg.hidden) * 0.09),
                    "router_bias": rounded(torch.randn(cfg.n_routed_experts) * 0.02),
                    "router_scale": rounded(torch.tensor([0.95])),
                    "shared_gate": rounded(torch.randn(cfg.n_shared_experts, cfg.moe_intermediate, cfg.hidden) * 0.09),
                    "shared_up": rounded(torch.randn(cfg.n_shared_experts, cfg.moe_intermediate, cfg.hidden) * 0.09),
                    "shared_down": rounded(torch.randn(cfg.n_shared_experts, cfg.hidden, cfg.moe_intermediate) * 0.09),
                    "routed_gate": rounded(torch.randn(cfg.n_routed_experts, cfg.moe_intermediate, cfg.hidden) * 0.09),
                    "routed_up": rounded(torch.randn(cfg.n_routed_experts, cfg.moe_intermediate, cfg.hidden) * 0.09),
                    "routed_down": rounded(torch.randn(cfg.n_routed_experts, cfg.hidden, cfg.moe_intermediate) * 0.09),
                }
                for name, value in extra.items():
                    layer_keep[name] = array(value)
                    setattr(w, "router_global_scale" if name == "router_scale" else name,
                            layer_keep[name])
                w.sparse = 1
            layer_weights[layer] = w
            all_common.append(common)
            all_extra.append(extra)
            keep.append(layer_keep)

        torch.manual_seed(1200)
        tables = {
            "embedding": rounded(torch.randn(cfg.vocab, cfg.hidden) * 0.1),
            "embed_norm": rounded(torch.rand(cfg.hidden) + 0.5),
            "final_norm": rounded(torch.rand(cfg.hidden) + 0.5),
            "unembedding": rounded(torch.randn(cfg.unpadded_vocab, cfg.hidden) * 0.1),
        }
        carr = {name: array(value) for name, value in tables.items()}
        keep.append(carr)
        direct = ModelWeights()
        direct.embedding = carr["embedding"]
        direct.embed_norm = carr["embed_norm"]
        direct.final_norm = carr["final_norm"]
        direct.unembedding = carr["unembedding"]
        direct.unembedding_rows = cfg.unpadded_vocab
        direct.layer = layer_weights

        tensors.update({
            "inkling.embed": tables["embedding"],
            "inkling.embed_norm": tables["embed_norm"],
            "inkling.final_norm": tables["final_norm"],
            "inkling.unembed": tables["unembedding"],
        })
        common_map = {
            "input_norm": "input_norm", "post_attention_norm": "post_attention_norm",
            "wq": "q", "wk": "k", "wv": "v", "wr": "r", "wo": "o",
            "q_norm": "q_norm", "k_norm": "k_norm", "relative_proj": "rel_proj",
            "k_sconv": "k_sconv", "v_sconv": "v_sconv",
            "attn_sconv": "attn_sconv", "mlp_sconv": "mlp_sconv",
        }
        for layer, (common, extra) in enumerate(zip(all_common, all_extra)):
            for source_name, canonical in common_map.items():
                tensors[f"inkling.layer.{layer}.{canonical}"] = common[source_name]
            if layer < cfg.dense_layers:
                tensors[f"inkling.layer.{layer}.mlp.gate"] = extra["dense_gate"]
                tensors[f"inkling.layer.{layer}.mlp.up"] = extra["dense_up"]
                tensors[f"inkling.layer.{layer}.mlp.down"] = extra["dense_down"]
                tensors[f"inkling.layer.{layer}.mlp.global_scale"] = extra["dense_scale"]
            else:
                tensors[f"inkling.layer.{layer}.router.weight"] = extra["router_weight"]
                tensors[f"inkling.layer.{layer}.router.correction_bias"] = extra["router_bias"]
                tensors[f"inkling.layer.{layer}.router.global_scale"] = extra["router_scale"]
                tensors[f"inkling.layer.{layer}.shared.gate"] = extra["shared_gate"]
                tensors[f"inkling.layer.{layer}.shared.up"] = extra["shared_up"]
                tensors[f"inkling.layer.{layer}.shared.down"] = extra["shared_down"]
        return cfg, direct, tensors, all_extra, keep

    def write_stage(self, cfg, tensors, extras):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        items = [write_tensor(root, i, name, value)
                 for i, (name, value) in enumerate(sorted(tensors.items()))]
        banks = [write_bank(root, layer, cfg, extras[layer])
                 for layer in range(cfg.dense_layers, cfg.n_layers)]
        config = {
            "n_layers": cfg.n_layers, "hidden": cfg.hidden, "vocab": cfg.vocab,
            "unpadded_vocab": cfg.unpadded_vocab, "max_context": cfg.max_context,
            "global_heads": cfg.global_heads, "global_kv_heads": cfg.global_kv_heads,
            "global_head_dim": cfg.global_head_dim, "local_heads": cfg.local_heads,
            "local_kv_heads": cfg.local_kv_heads, "local_head_dim": cfg.local_head_dim,
            "sliding_window": cfg.sliding_window, "d_rel": cfg.d_rel,
            "rel_extent": cfg.rel_extent, "conv_kernel": cfg.conv_kernel,
            "dense_layers": cfg.dense_layers, "dense_intermediate": cfg.dense_intermediate,
            "moe_intermediate": cfg.moe_intermediate,
            "n_routed_experts": cfg.n_routed_experts, "top_k": cfg.top_k,
            "n_shared_experts": cfg.n_shared_experts, "rms_eps": cfg.rms_eps,
            "route_scale": cfg.route_scale,
            "logits_width_multiplier": cfg.logits_width_multiplier,
            "log_scaling_n_floor": cfg.log_scaling_n_floor,
            "log_scaling_alpha": cfg.log_scaling_alpha,
            "layers": [
                {
                    "layer": i, "is_local": bool(cfg.layer[i].is_local),
                    "num_heads": cfg.layer[i].num_heads,
                    "num_kv_heads": cfg.layer[i].num_kv_heads,
                    "head_dim": cfg.layer[i].head_dim,
                    "relative_extent": cfg.layer[i].relative_extent,
                    "sparse": i >= cfg.dense_layers,
                }
                for i in range(cfg.n_layers)
            ],
        }
        release = {
            "official_small": False, "config_sha256": "00" * 32,
            "index_sha256": "11" * 32, "model_id": "synthetic-inkling",
            "release_upload_commit": "", "package": {"total_size": 0},
        }
        payload = bytearray(_header_bytes(config, release, len(items), len(banks)))
        for layer in config["layers"]:
            payload += _layer_bytes(layer)
        for item in items:
            payload += _tensor_bytes(root, item)
        for item in banks:
            payload += _bank_bytes(item)
        (root / "runtime-stage.bin").write_bytes(payload)
        return root

    def write_vq_stage(self, cfg, tensors, extras, direct, keep):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        items = [write_tensor(root, i, name, value)
                 for i, (name, value) in enumerate(sorted(tensors.items()))]
        spec = VQSpec(stages=2, vec_dim=2, entries=4, index_block=4)
        banks = []
        for layer in range(cfg.dense_layers, cfg.n_layers):
            item, decoded = write_vq_bank(root, layer, cfg, extras[layer], spec)
            banks.append(item)
            refs = [array(value) for value in decoded]
            direct.layer[layer].routed_gate = refs[0]
            direct.layer[layer].routed_up = refs[1]
            direct.layer[layer].routed_down = refs[2]
            keep.append(refs)
        config = {
            "n_layers": cfg.n_layers, "hidden": cfg.hidden, "vocab": cfg.vocab,
            "unpadded_vocab": cfg.unpadded_vocab, "max_context": cfg.max_context,
            "global_heads": cfg.global_heads, "global_kv_heads": cfg.global_kv_heads,
            "global_head_dim": cfg.global_head_dim, "local_heads": cfg.local_heads,
            "local_kv_heads": cfg.local_kv_heads, "local_head_dim": cfg.local_head_dim,
            "sliding_window": cfg.sliding_window, "d_rel": cfg.d_rel,
            "rel_extent": cfg.rel_extent, "conv_kernel": cfg.conv_kernel,
            "dense_layers": cfg.dense_layers, "dense_intermediate": cfg.dense_intermediate,
            "moe_intermediate": cfg.moe_intermediate,
            "n_routed_experts": cfg.n_routed_experts, "top_k": cfg.top_k,
            "n_shared_experts": cfg.n_shared_experts, "rms_eps": cfg.rms_eps,
            "route_scale": cfg.route_scale,
            "logits_width_multiplier": cfg.logits_width_multiplier,
            "log_scaling_n_floor": cfg.log_scaling_n_floor,
            "log_scaling_alpha": cfg.log_scaling_alpha,
            "layers": [
                {
                    "layer": i, "is_local": bool(cfg.layer[i].is_local),
                    "num_heads": cfg.layer[i].num_heads,
                    "num_kv_heads": cfg.layer[i].num_kv_heads,
                    "head_dim": cfg.layer[i].head_dim,
                    "relative_extent": cfg.layer[i].relative_extent,
                    "sparse": i >= cfg.dense_layers,
                }
                for i in range(cfg.n_layers)
            ],
        }
        release = {
            "official_small": False, "config_sha256": "00" * 32,
            "index_sha256": "11" * 32, "model_id": "synthetic-inkling-vq",
            "release_upload_commit": "", "package": {"total_size": 0},
        }
        quant = {
            "stages": spec.stages, "vec_dim": spec.vec_dim,
            "entries": spec.entries, "index_block": spec.index_block,
        }
        payload = bytearray(_header_bytes(
            config, release, len(items), len(banks),
            expert_format="vq", quant=quant,
        ))
        for layer in config["layers"]:
            payload += _layer_bytes(layer)
        for item in items:
            payload += _tensor_bytes(root, item)
        for item in banks:
            payload += _bank_bytes(item, final_vq=True)
        (root / "runtime-stage.bin").write_bytes(payload)
        return root

    def write_qtrunk_vq_stage(self, cfg, tensors, extras, direct, keep):
        # Start from the independently dequantized final-WEXP fixture, then
        # replace all eligible trunk matrices with Q8/Q4 artifacts and a v3
        # private index. The direct model's resident matrices are overwritten
        # with the independently decoded Q values, making byte-to-logit parity
        # exact rather than comparing against the unquantized source.
        root = self.write_vq_stage(cfg, tensors, extras, direct, keep)
        qitems = []
        canonical = []
        role_to_field = {
            "q": "wq", "k": "wk", "v": "wv", "r": "wr", "o": "wo",
            "mlp.gate": "dense_gate", "mlp.up": "dense_up", "mlp.down": "dense_down",
            "router.weight": "router_weight", "shared.gate": "shared_gate",
            "shared.up": "shared_up", "shared.down": "shared_down",
        }
        for ordinal, (name, value) in enumerate(sorted(tensors.items())):
            if value.ndim >= 2 and qpolicy(name, 4) is not None:
                item, deq = write_qtensor(root, ordinal, name, value, bits=4, group=4)
                qitems.append(item)
                if name == "inkling.embed":
                    keep.append(overwrite_pointer(direct.embedding, deq))
                elif name == "inkling.unembed":
                    keep.append(overwrite_pointer(direct.unembedding, deq))
                else:
                    parts = name.split(".")
                    layer = int(parts[2])
                    role = ".".join(parts[3:])
                    keep.append(overwrite_pointer(getattr(direct.layer[layer], role_to_field[role]), deq))
            else:
                canonical.append(write_tensor(root, ordinal, name, value))
        banks = []
        spec = VQSpec(stages=2, vec_dim=2, entries=4, index_block=4)
        for layer in range(cfg.dense_layers, cfg.n_layers):
            path = root / f"experts-L{layer}.bin"
            banks.append({
                "file": path.name, "layer": layer,
                "experts": cfg.n_routed_experts, "hidden_size": cfg.hidden,
                "intermediate_size": cfg.moe_intermediate,
                "record_bytes": path.stat().st_size // cfg.n_routed_experts,
                "bytes": path.stat().st_size,
                "codebook_base": (layer - cfg.dense_layers) * 3 * spec.stages,
            })
        config = {
            "n_layers": cfg.n_layers, "hidden": cfg.hidden, "vocab": cfg.vocab,
            "unpadded_vocab": cfg.unpadded_vocab, "max_context": cfg.max_context,
            "global_heads": cfg.global_heads, "global_kv_heads": cfg.global_kv_heads,
            "global_head_dim": cfg.global_head_dim, "local_heads": cfg.local_heads,
            "local_kv_heads": cfg.local_kv_heads, "local_head_dim": cfg.local_head_dim,
            "sliding_window": cfg.sliding_window, "d_rel": cfg.d_rel,
            "rel_extent": cfg.rel_extent, "conv_kernel": cfg.conv_kernel,
            "dense_layers": cfg.dense_layers, "dense_intermediate": cfg.dense_intermediate,
            "moe_intermediate": cfg.moe_intermediate,
            "n_routed_experts": cfg.n_routed_experts, "top_k": cfg.top_k,
            "n_shared_experts": cfg.n_shared_experts, "rms_eps": cfg.rms_eps,
            "route_scale": cfg.route_scale,
            "logits_width_multiplier": cfg.logits_width_multiplier,
            "log_scaling_n_floor": cfg.log_scaling_n_floor,
            "log_scaling_alpha": cfg.log_scaling_alpha,
            "layers": [{
                "layer": i, "is_local": bool(cfg.layer[i].is_local),
                "num_heads": cfg.layer[i].num_heads,
                "num_kv_heads": cfg.layer[i].num_kv_heads,
                "head_dim": cfg.layer[i].head_dim,
                "relative_extent": cfg.layer[i].relative_extent,
                "sparse": i >= cfg.dense_layers,
            } for i in range(cfg.n_layers)],
        }
        release = {
            "official_small": False, "config_sha256": "00" * 32,
            "index_sha256": "11" * 32, "model_id": "synthetic-inkling-qtrunk",
            "release_upload_commit": "", "package": {"total_size": 0},
        }
        quant = {"stages": 2, "vec_dim": 2, "entries": 4, "index_block": 4}
        payload = bytearray(_header_bytes(
            config, release, len(qitems) + len(canonical), len(banks),
            expert_format="vq", quant=quant, quantized_trunk=True,
        ))
        for layer in config["layers"]:
            payload += _layer_bytes(layer)
        for item in sorted(canonical, key=lambda x: x["target"]):
            payload += _tensor_bytes(root, item)
        for item in sorted(qitems, key=lambda x: x["target"]):
            payload += _qtensor_bytes(item)
        # Runtime registry order is irrelevant, but canonical names must be
        # unique. Sorting the two groups separately deliberately exercises it.
        for item in banks:
            payload += _bank_bytes(item, final_vq=True)
        (root / "runtime-stage.bin").write_bytes(payload)
        return root

    def direct_runtime(self, cfg):
        capacity = 5
        ns = self.lib.waste_inkling_model_state_floats(ctypes.byref(cfg), capacity)
        nf = self.lib.waste_inkling_model_scratch_floats(ctypes.byref(cfg), capacity)
        ni = self.lib.waste_inkling_model_scratch_ints(ctypes.byref(cfg))
        sbuf = (ctypes.c_float * ns)()
        fbuf = (ctypes.c_float * nf)()
        ibuf = (ctypes.c_int * ni)()
        state = ModelState()
        scratch = ModelScratch()
        self.assertEqual(self.lib.waste_inkling_model_state_init(
            ctypes.byref(state), ctypes.byref(cfg), capacity, sbuf, ns), 0)
        self.assertEqual(self.lib.waste_inkling_model_scratch_init(
            ctypes.byref(scratch), ctypes.byref(cfg), capacity,
            fbuf, nf, ibuf, ni), 0)
        return state, scratch, (sbuf, fbuf, ibuf), capacity

    def test_open_step_and_reset_match_direct_model(self):
        cfg, direct, tensors, extras, keep = self.fixture()
        root = self.write_stage(cfg, tensors, extras)
        state, scratch, runtime_keep, capacity = self.direct_runtime(cfg)
        opts = PrivateOptions(capacity, 1, 0)
        private = ctypes.c_void_p()
        detail = ctypes.create_string_buffer(256)
        self.assertEqual(self.lib.waste_inkling_private_open(
            str(root).encode(), ctypes.byref(opts), ctypes.byref(private),
            detail, len(detail)), 0, detail.value.decode())
        self.addCleanup(lambda: self.lib.waste_inkling_private_close(private) if private.value else None)
        null_expert = ExpertCallback()
        tokens = [3, 7, 1, 12]
        for pos, token in enumerate(tokens):
            a = (ctypes.c_float * cfg.unpadded_vocab)()
            b = (ctypes.c_float * cfg.unpadded_vocab)()
            self.assertEqual(self.lib.waste_inkling_model_step(
                ctypes.byref(cfg), ctypes.byref(direct), ctypes.byref(state),
                ctypes.byref(scratch), token, pos, a, len(a),
                null_expert, None), 0)
            self.assertEqual(self.lib.waste_inkling_private_step(
                private, token, pos, b, len(b)), 0)
            torch.testing.assert_close(torch.tensor(list(b)), torch.tensor(list(a)), rtol=0, atol=0)
        self.lib.waste_inkling_private_reset(private)
        self.lib.waste_inkling_model_reset(ctypes.byref(state), ctypes.byref(cfg))
        a = (ctypes.c_float * cfg.unpadded_vocab)()
        b = (ctypes.c_float * cfg.unpadded_vocab)()
        self.assertEqual(self.lib.waste_inkling_model_step(
            ctypes.byref(cfg), ctypes.byref(direct), ctypes.byref(state),
            ctypes.byref(scratch), tokens[0], 0, a, len(a), null_expert, None), 0)
        self.assertEqual(self.lib.waste_inkling_private_step(private, tokens[0], 0, b, len(b)), 0)
        self.assertEqual(bytes(a), bytes(b))
        self.assertTrue(keep and runtime_keep)

    def test_final_vq_runtime_stage_matches_independent_dequantized_model(self):
        cfg, direct, tensors, extras, keep = self.fixture()
        root = self.write_vq_stage(cfg, tensors, extras, direct, keep)
        state, scratch, runtime_keep, capacity = self.direct_runtime(cfg)
        opts = PrivateOptions(capacity, 1, 0)
        private = ctypes.c_void_p()
        detail = ctypes.create_string_buffer(256)
        self.assertEqual(self.lib.waste_inkling_private_open(
            str(root).encode(), ctypes.byref(opts), ctypes.byref(private),
            detail, len(detail)), 0, detail.value.decode())
        self.addCleanup(lambda: self.lib.waste_inkling_private_close(private) if private.value else None)
        null_expert = ExpertCallback()
        for pos, token in enumerate((3, 7, 1, 12)):
            expected = (ctypes.c_float * cfg.unpadded_vocab)()
            actual = (ctypes.c_float * cfg.unpadded_vocab)()
            self.assertEqual(self.lib.waste_inkling_model_step(
                ctypes.byref(cfg), ctypes.byref(direct), ctypes.byref(state),
                ctypes.byref(scratch), token, pos, expected, len(expected),
                null_expert, None), 0)
            self.assertEqual(self.lib.waste_inkling_private_step(
                private, token, pos, actual, len(actual)), 0)
            torch.testing.assert_close(
                torch.tensor(list(actual)), torch.tensor(list(expected)), rtol=0, atol=0
            )
        self.assertTrue(keep and runtime_keep)

    def test_quantized_trunk_v3_matches_independent_dequantized_model(self):
        cfg, direct, tensors, extras, keep = self.fixture()
        root = self.write_qtrunk_vq_stage(cfg, tensors, extras, direct, keep)
        state, scratch, runtime_keep, capacity = self.direct_runtime(cfg)
        opts = PrivateOptions(capacity, 1, 0)
        private = ctypes.c_void_p()
        detail = ctypes.create_string_buffer(256)
        self.assertEqual(self.lib.waste_inkling_private_open(
            str(root).encode(), ctypes.byref(opts), ctypes.byref(private),
            detail, len(detail)), 0, detail.value.decode())
        self.addCleanup(lambda: self.lib.waste_inkling_private_close(private) if private.value else None)
        self.assertGreater(self.lib.waste_inkling_private_quantized_trunk_bytes(private), 0)
        self.assertLess(self.lib.waste_inkling_private_resident_f32_bytes(private),
                        sum(v.numel() for v in tensors.values()) * 4)
        null_expert = ExpertCallback()
        for pos, token in enumerate((3, 7, 1, 12)):
            expected = (ctypes.c_float * cfg.unpadded_vocab)()
            actual = (ctypes.c_float * cfg.unpadded_vocab)()
            self.assertEqual(self.lib.waste_inkling_model_step(
                ctypes.byref(cfg), ctypes.byref(direct), ctypes.byref(state),
                ctypes.byref(scratch), token, pos, expected, len(expected),
                null_expert, None), 0)
            self.assertEqual(self.lib.waste_inkling_private_step(
                private, token, pos, actual, len(actual)), 0)
            torch.testing.assert_close(torch.tensor(list(actual)),
                                       torch.tensor(list(expected)), rtol=0, atol=0)
        self.assertTrue(keep and runtime_keep)

    def test_official_requirement_and_corrupt_index_fail_closed(self):
        cfg, direct, tensors, extras, keep = self.fixture()
        root = self.write_stage(cfg, tensors, extras)
        detail = ctypes.create_string_buffer(256)
        p = ctypes.c_void_p()
        opts = PrivateOptions(4, 1, 1)
        self.assertEqual(self.lib.waste_inkling_private_open(
            str(root).encode(), ctypes.byref(opts), ctypes.byref(p), detail, len(detail)), -5)
        self.assertFalse(p.value)
        raw = bytearray((root / "runtime-stage.bin").read_bytes())
        raw[0] ^= 1
        (root / "runtime-stage.bin").write_bytes(raw)
        opts.require_official_small = 0
        detail.value = b""
        self.assertEqual(self.lib.waste_inkling_private_open(
            str(root).encode(), ctypes.byref(opts), ctypes.byref(p), detail, len(detail)), -3)
        self.assertFalse(p.value)
        self.assertTrue(direct and keep)


    def test_exact_official_small_geometry_reaches_artifact_validation(self):
        import json
        cfg_json = json.loads((REPO / "tests" / "data" / "inkling-small-config.json").read_text())
        text = cfg_json["text_config"]
        local = set(text["local_layer_ids"])
        config = {
            "n_layers": text["num_hidden_layers"], "hidden": text["hidden_size"],
            "vocab": text["vocab_size"], "unpadded_vocab": text["unpadded_vocab_size"],
            "max_context": text["model_max_length"],
            "global_heads": text["num_attention_heads"],
            "global_kv_heads": text["num_key_value_heads"],
            "global_head_dim": text["head_dim"],
            "local_heads": text["swa_num_attention_heads"],
            "local_kv_heads": text["swa_num_key_value_heads"],
            "local_head_dim": text["swa_head_dim"],
            "sliding_window": text["sliding_window_size"], "d_rel": text["d_rel"],
            "rel_extent": text["rel_extent"], "conv_kernel": text["sconv_kernel_size"],
            "dense_layers": text["dense_mlp_idx"],
            "dense_intermediate": text["dense_intermediate_size"],
            "moe_intermediate": text["intermediate_size"],
            "n_routed_experts": text["n_routed_experts"],
            "top_k": text["num_experts_per_tok"],
            "n_shared_experts": text["n_shared_experts"],
            "rms_eps": text["rms_norm_eps"], "route_scale": text["route_scale"],
            "logits_width_multiplier": text["logits_mup_width_multiplier"],
            "log_scaling_n_floor": text["log_scaling_n_floor"],
            "log_scaling_alpha": text["log_scaling_alpha"],
            "layers": [],
        }
        for i in range(config["n_layers"]):
            is_local = i in local
            config["layers"].append({
                "layer": i, "is_local": is_local,
                "num_heads": config["local_heads"] if is_local else config["global_heads"],
                "num_kv_heads": config["local_kv_heads"] if is_local else config["global_kv_heads"],
                "head_dim": config["local_head_dim"] if is_local else config["global_head_dim"],
                "relative_extent": config["sliding_window"] if is_local else config["rel_extent"],
                "sparse": i >= config["dense_layers"],
            })
        td = tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup)
        root = Path(td.name)
        release = {
            "official_small": True, "config_sha256": "22" * 32,
            "index_sha256": "33" * 32, "model_id": "thinkingmachines/Inkling-Small",
            "release_upload_commit": "21152b5", "package": {"total_size": 531912898740},
        }
        # One deliberately incomplete tensor entry is enough to distinguish
        # accepted official geometry (-3 later) from unsupported geometry (-5).
        payload = bytearray(_header_bytes(config, release, 1, 40))
        for layer in config["layers"]: payload += _layer_bytes(layer)
        payload += b"\0" * 256
        payload += b"\0" * (40 * 128)
        (root / "runtime-stage.bin").write_bytes(payload)
        pctx = ctypes.c_void_p(); detail = ctypes.create_string_buffer(256)
        opts = PrivateOptions(4096, 1, 1)
        self.assertEqual(self.lib.waste_inkling_private_open(
            str(root).encode(), ctypes.byref(opts), ctypes.byref(pctx), detail, len(detail)), -3)
        self.assertFalse(pctx.value)
        self.assertIn(b"tensor registry", detail.value)

    def test_corrupt_tensor_crc_fails_before_runtime_allocation(self):
        cfg, direct, tensors, extras, keep = self.fixture()
        root = self.write_stage(cfg, tensors, extras)
        victim = next((root / "trunk-stage").glob("*.tensor.stage"))
        raw = bytearray(victim.read_bytes())
        raw[70] ^= 1
        victim.write_bytes(raw)
        p = ctypes.c_void_p()
        detail = ctypes.create_string_buffer(256)
        opts = PrivateOptions(4, 1, 0)
        self.assertEqual(self.lib.waste_inkling_private_open(
            str(root).encode(), ctypes.byref(opts), ctypes.byref(p), detail, len(detail)), -3)
        self.assertFalse(p.value)
        self.assertIn(b"staged trunk tensor", detail.value)
        self.assertTrue(direct and keep)


if __name__ == "__main__":
    unittest.main()
