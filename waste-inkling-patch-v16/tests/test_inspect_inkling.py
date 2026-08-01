#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.

import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
SPEC = importlib.util.spec_from_file_location("inspect_inkling", REPO / "tools" / "inspect_inkling.py")
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)

from inkling_source import config_identifies_inkling


def write_shard(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header = {
        name: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}
        for name, (dtype, shape) in tensors.items()
    }
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def inkling_config(layers=3):
    local = list(range(max(0, layers - 1)))
    return {
        "architectures": ["InklingForConditionalGeneration"],
        "model_type": "inkling_mm_model",
        "eos_token_id": 200006,
        "text_config": {
            "model_max_length": 1048576,
            "torch_dtype": "bfloat16",
            "hidden_size": 16,
            "num_hidden_layers": layers,
            "vocab_size": 64,
            "unpadded_vocab_size": 60,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "d_rel": 2,
            "rel_extent": 8,
            "rms_norm_eps": 1e-6,
            "use_embed_norm": True,
            "local_layer_ids": local,
            "dense_mlp_idx": 1,
            "use_sconv": True,
            "sconv_kernel_size": 4,
            "swa_head_dim": 4,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "sliding_window_size": 8,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 2,
            "shared_expert_sink": True,
            "dense_intermediate_size": 32,
            "intermediate_size": 8,
            "route_scale": 8.0,
            "use_gate_bias": True,
            "gate_activation": "sigmoid",
            "norm_after_topk": True,
            "use_global_scale": True,
            "logits_mup_width_multiplier": 2.0,
            "log_scaling_n_floor": 128000,
            "log_scaling_alpha": 0.1,
        },
        "vision_config": {
            "vision_encoder_type": "hmlp",
            "decoder_dmodel": 16,
            "patch_size": 40,
            "temporal_patch_size": 2,
            "n_channels": 3,
            "n_layers": 4,
            "use_vision_norm": True,
        },
        "audio_config": {
            "decoder_dmodel": 16,
            "n_mel_bins": 80,
            "mel_vocab_size": 16,
            "bias": False,
            "dmel_min_value": -7.0,
            "dmel_max_value": 2.0,
            "use_audio_norm": True,
            "audio_mode": "dmel",
        },
        "mtp_config": {"num_nextn_predict_layers": 8},
    }


def official_tokenizer_config():
    values = {
        200005: "<|content_image|>",
        200006: "<|content_model_end_sampling|>",
        200020: "<|content_audio_input|>",
        200053: "<|unused_200053|>",
        200054: "<|unused_200054|>",
    }
    return {
        "added_tokens_decoder": {
            str(token_id): {"content": text, "special": True}
            for token_id, text in values.items()
        }
    }


def processor_config():
    return {
        "image_token": "<|unused_200054|>",
        "audio_token": "<|unused_200053|>",
        "image_bos_token": "<|content_image|>",
        "audio_bos_token": "<|content_audio_input|>",
        "num_dmel_bins": 16,
        "dmel_min_value": -7.0,
        "dmel_max_value": 2.0,
        "image_processor": {
            "size": {"height": 40, "width": 40},
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
        },
        "feature_extractor": {
            "sampling_rate": 16000,
            "hop_length": 800,
            "n_fft": 1600,
            "window_size": 1600,
            "feature_size": 80,
        },
    }


def raw_tensors(config=None):
    cfg = config or inkling_config()
    text = cfg["text_config"]
    h = text["hidden_size"]
    layers = text["num_hidden_layers"]
    drel = text["d_rel"]
    kernel = text["sconv_kernel_size"]
    dense = text["dense_intermediate_size"]
    inter = text["intermediate_size"]
    experts = text["n_routed_experts"]
    shared = text["n_shared_experts"]
    local = set(text["local_layer_ids"])
    tensors = {
        "model.llm.embed.weight": ("BF16", [text["vocab_size"], h]),
        "model.llm.embed_norm.weight": ("BF16", [h]),
        "model.llm.norm.weight": ("BF16", [h]),
        "model.llm.unembed.weight": ("BF16", [text["unpadded_vocab_size"], h]),
        "model.audio.encoder.weight": ("BF16", [80 * 16, h]),
        "model.audio.final_norm.weight": ("BF16", [h]),
        "model.visual.layers.linear_0.weight": ("BF16", [32, 3 * 40 * 40 * 2]),
        "model.visual.layers.linear_1.weight": ("BF16", [32, 32]),
        "model.visual.layers.linear_2.weight": ("BF16", [32, 32]),
        "model.visual.layers.linear_3.weight": ("BF16", [h, 32]),
        "model.visual.layers.norm_0.weight": ("BF16", [32]),
        "model.visual.layers.norm_1.weight": ("BF16", [32]),
        "model.visual.layers.norm_2.weight": ("BF16", [32]),
        "model.visual.final_norm.weight": ("BF16", [h]),
    }
    for layer in range(layers):
        p = f"model.llm.layers.{layer}"
        is_local = layer in local
        heads = text["swa_num_attention_heads"] if is_local else text["num_attention_heads"]
        kv = text["swa_num_key_value_heads"] if is_local else text["num_key_value_heads"]
        dim = text["swa_head_dim"] if is_local else text["head_dim"]
        extent = text["sliding_window_size"] if is_local else text["rel_extent"]
        tensors.update(
            {
                f"{p}.attn_norm.weight": ("BF16", [h]),
                f"{p}.mlp_norm.weight": ("BF16", [h]),
                f"{p}.attn.wq_du.weight": ("BF16", [heads * dim, h]),
                f"{p}.attn.wk_dv.weight": ("BF16", [kv * dim, h]),
                f"{p}.attn.wv_dv.weight": ("BF16", [kv * dim, h]),
                f"{p}.attn.wr_du.weight": ("BF16", [heads * drel, h]),
                f"{p}.attn.wo_ud.weight": ("BF16", [h, heads * dim]),
                f"{p}.attn.q_norm.weight": ("BF16", [dim]),
                f"{p}.attn.k_norm.weight": ("BF16", [dim]),
                f"{p}.attn.rel_logits_proj.proj": ("BF16", [drel, extent]),
                f"{p}.attn.k_sconv.weight": ("F32", [kv * dim, 1, kernel]),
                f"{p}.attn.v_sconv.weight": ("F32", [kv * dim, 1, kernel]),
                f"{p}.attn_sconv.weight": ("F32", [h, 1, kernel]),
                f"{p}.mlp_sconv.weight": ("F32", [h, 1, kernel]),
            }
        )
        if layer < text["dense_mlp_idx"]:
            tensors.update(
                {
                    f"{p}.mlp.w13_dn.weight": ("BF16", [2 * dense, h]),
                    f"{p}.mlp.w2_md.weight": ("BF16", [h, dense]),
                    f"{p}.mlp.global_scale": ("F32", [1]),
                }
            )
        else:
            tensors.update(
                {
                    f"{p}.mlp.experts.w13_weight": ("BF16", [experts, 2 * inter, h]),
                    f"{p}.mlp.experts.w2_weight": ("BF16", [experts, h, inter]),
                    f"{p}.mlp.gate.weight": ("BF16", [experts + shared, h]),
                    f"{p}.mlp.gate.bias": ("F32", [experts]),
                    f"{p}.mlp.gate.global_scale": ("F32", [1]),
                    f"{p}.mlp.shared_experts.shared_w13_weight": ("BF16", [shared, 2 * inter, h]),
                    f"{p}.mlp.shared_experts.shared_w2_weight": ("BF16", [shared, h, inter]),
                }
            )
    return tensors


class InspectInklingTest(unittest.TestCase):
    def make_checkpoint(self, omit=(), *, with_payloads=True):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cfg = inkling_config()
        (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (root / "tiktoken").mkdir()
        (root / "tiktoken" / "tokenizer.model").write_bytes(b"tokenizer")
        (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        (root / "tokenizer_config.json").write_text(
            json.dumps(official_tokenizer_config()), encoding="utf-8"
        )
        (root / "processor_config.json").write_text(
            json.dumps(processor_config()), encoding="utf-8"
        )
        tensors = raw_tensors(cfg)
        for name in omit:
            tensors.pop(name, None)
        shard = "model-00001-of-00001.safetensors"
        if with_payloads:
            write_shard(root / shard, tensors)
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 531912898740},
                    "weight_map": {name: shard for name in tensors},
                }
            ),
            encoding="utf-8",
        )
        return td, root

    def test_architecture_detection_is_explicit(self):
        self.assertTrue(config_identifies_inkling(inkling_config()))
        self.assertFalse(config_identifies_inkling({"text_config": {"num_hidden_layers": 42}}))

    def test_official_raw_dialect_nested_tokenizer_and_token_ids(self):
        td, root = self.make_checkpoint()
        self.addCleanup(td.cleanup)
        report = MOD.inspect(root).data
        self.assertEqual(report["tensor_layout"]["dialect"]["value"], "provider_raw")
        self.assertEqual(report["assets"]["tokenizer"]["path"], "tiktoken/tokenizer.model")
        self.assertTrue(report["ready"]["text_metadata"])
        self.assertTrue(report["ready"]["text_payloads"])
        self.assertTrue(report["ready"]["text_source"])
        self.assertTrue(report["ready"]["vision_source"])
        self.assertTrue(report["ready"]["audio_source"])
        self.assertEqual(report["fields"]["global_layer_ids"]["value"], [2])
        self.assertEqual(report["fields"]["image_token_id"]["value"], 200054)
        self.assertEqual(report["fields"]["audio_token_id"]["value"], 200053)
        self.assertEqual(report["fields"]["positional_encoding"]["provenance"], MOD.PROV_WEIGHTS)

    def test_index_only_is_metadata_ready_but_not_convertible(self):
        td, root = self.make_checkpoint(with_payloads=False)
        self.addCleanup(td.cleanup)
        report = MOD.inspect(root).data
        self.assertTrue(report["ready"]["text_metadata"])
        self.assertFalse(report["ready"]["text_payloads"])
        self.assertFalse(report["ready"]["text_source"])
        self.assertTrue(report["assets"]["text_payloads"]["missing_shards"])

    def test_quantized_and_mtp_assets_are_deferred_not_silently_enabled(self):
        td, root = self.make_checkpoint()
        self.addCleanup(td.cleanup)
        index = json.loads((root / "model.safetensors.index.json").read_text())
        shard = "model-00001-of-00001.safetensors"
        index["weight_map"]["model.llm.layers.2.mlp.experts.w13_weight.scale"] = shard
        index["weight_map"]["model.mtp.layers.0.input_proj.weight"] = "mtp.safetensors"
        (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
        report = MOD.inspect(root).data
        self.assertEqual(report["assets"]["source_quantization"]["kind"], "packed/quantized")
        self.assertTrue(report["tensor_layout"]["mtp"]["present"])
        self.assertTrue(any("BF16 parity" in x for x in report["deferred"]))
        self.assertTrue(any("MTP tensors" in x for x in report["deferred"]))
        self.assertTrue(report["ready"]["text_source"], "MTP is excluded from the base-text payload gate")

    def test_missing_sparse_tensor_is_a_text_blocker(self):
        missing = "model.llm.layers.2.mlp.experts.w2_weight"
        td, root = self.make_checkpoint(omit={missing})
        self.addCleanup(td.cleanup)
        report = MOD.inspect(root).data
        self.assertFalse(report["ready"]["text_metadata"])
        self.assertFalse(report["ready"]["text_source"])
        self.assertTrue(any("layers have missing" in x for x in report["blocking"]["text"]))
        layer2 = next(x for x in report["tensor_layout"]["layers"]["missing"] if x["layer"] == 2)
        self.assertIn(missing, layer2["tensors"])

    def test_unresolved_processor_token_blocks_only_that_media_path(self):
        td, root = self.make_checkpoint()
        self.addCleanup(td.cleanup)
        tok = official_tokenizer_config()
        tok["added_tokens_decoder"].pop("200054")
        (root / "tokenizer_config.json").write_text(json.dumps(tok), encoding="utf-8")
        report = MOD.inspect(root).data
        self.assertTrue(report["ready"]["text_source"])
        self.assertFalse(report["ready"]["vision_source"])
        self.assertTrue(report["ready"]["audio_source"])
        self.assertIsNone(report["fields"]["image_token_id"]["value"])

    def test_no_chat_template_stays_raw_instead_of_guessing(self):
        td, root = self.make_checkpoint()
        self.addCleanup(td.cleanup)
        (root / "chat_template.jinja").unlink()
        report = MOD.inspect(root).data
        self.assertTrue(report["ready"]["text_source"])
        self.assertTrue(any("do not guess" in x for x in report["deferred"]))

    def test_strict_returns_two_for_missing_requested_capability(self):
        td, root = self.make_checkpoint()
        self.addCleanup(td.cleanup)
        os.remove(root / "processor_config.json")
        self.assertEqual(MOD.main(["--src", str(root), "--strict", "vision"]), 2)
        self.assertEqual(MOD.main(["--src", str(root), "--strict", "text"]), 0)


if __name__ == "__main__":
    unittest.main()
