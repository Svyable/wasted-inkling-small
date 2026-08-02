#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

from inkling_plan import PlanError, build_plan
from test_inspect_inkling import (
    inkling_config,
    official_tokenizer_config,
    processor_config,
    raw_tensors,
    write_shard,
)


def official_small_config():
    return json.loads((REPO / "tests" / "data" / "inkling-small-config.json").read_text(encoding="utf-8"))


def make_checkpoint(cfg, *, with_payload=True, mutate=None):
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (root / "tiktoken").mkdir()
    (root / "tiktoken" / "tokenizer.model").write_bytes(b"tokenizer")
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps(official_tokenizer_config()), encoding="utf-8")
    (root / "processor_config.json").write_text(json.dumps(processor_config()), encoding="utf-8")
    tensors = raw_tensors(cfg)
    if mutate:
        mutate(tensors)
    shard = "model-00001-of-00001.safetensors"
    if with_payload:
        write_shard(root / shard, tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 531912898740}, "weight_map": {n: shard for n in tensors}}),
        encoding="utf-8",
    )
    return td, root


class InklingPlanTest(unittest.TestCase):
    def test_plan_separates_streamed_and_resident_experts(self):
        td, root = make_checkpoint(inkling_config())
        self.addCleanup(td.cleanup)
        plan = build_plan(root, require_payloads=True)
        self.assertEqual(plan["source_dialect"], "provider_raw")
        self.assertEqual(plan["model"]["global_attention_layers"], [2])
        self.assertEqual(plan["model"]["dense_mlp_layers"], [0])
        self.assertEqual(plan["model"]["sparse_mlp_layers"], [1, 2])
        self.assertEqual(len(plan["expert_banks"]), 2)
        self.assertTrue(all(bank["experts"] == 4 for bank in plan["expert_banks"]))
        shared = [item for item in plan["trunk"] if ".shared." in item["role"]]
        self.assertEqual(len(shared), 4)
        self.assertEqual(plan["status"]["shape_probes"]["unknown"], 0)
        self.assertFalse(plan["status"]["container_written"])
        self.assertFalse(plan["status"]["runtime_supported"])

    def test_shape_mismatch_fails_before_conversion(self):
        def break_q(tensors):
            tensors["model.llm.layers.0.attn.wq_du.weight"] = ("BF16", [15, 16])

        td, root = make_checkpoint(inkling_config(), mutate=break_q)
        self.addCleanup(td.cleanup)
        with self.assertRaisesRegex(PlanError, "shape probes failed"):
            build_plan(root, require_payloads=True)

    def test_index_only_plan_is_reviewable_but_not_payload_ready(self):
        td, root = make_checkpoint(inkling_config(), with_payload=False)
        self.addCleanup(td.cleanup)
        plan = build_plan(root)
        self.assertTrue(plan["status"]["metadata_complete"])
        self.assertFalse(plan["status"]["payload_headers_complete"])
        self.assertGreater(plan["status"]["shape_probes"]["unknown"], 0)
        with self.assertRaisesRegex(PlanError, "not present locally"):
            build_plan(root, require_payloads=True)

    def test_official_small_profile_is_parameterized_not_hardcoded(self):
        cfg = official_small_config()
        td, root = make_checkpoint(cfg)
        self.addCleanup(td.cleanup)
        plan = build_plan(root, require_payloads=True)
        self.assertEqual(plan["model"]["profile"], "official-inkling-small")
        self.assertTrue(plan["release"]["profile"]["match"])
        self.assertFalse(plan["release"]["package"]["match"])
        self.assertEqual(plan["model"]["num_hidden_layers"], 42)
        self.assertEqual(plan["model"]["hidden_size"], 4096)
        self.assertEqual(plan["model"]["routed_experts"], 256)
        self.assertEqual(plan["model"]["top_k"], 6)
        self.assertEqual(plan["model"]["shared_experts"], 2)
        self.assertEqual(plan["model"]["global_attention_layers"], [5, 11, 17, 23, 29, 35, 41])
        self.assertEqual(len(plan["expert_banks"]), 40)
        last = plan["manifest_preview"]["inkling"]["layers"][41]
        self.assertEqual(last["attention"]["kind"], "hybrid_global")
        self.assertEqual(last["attention"]["relative_extent"], 1024)
        self.assertEqual(last["mlp"]["intermediate_size"], 2048)


if __name__ == "__main__":
    unittest.main()
