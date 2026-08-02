#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from inkling_release import (
    CHECKPOINT_SHARDS,
    CHECKPOINT_TOTAL_SIZE,
    ReleaseError,
    inspect_release,
    required_text_tensor_names,
    require_official_small,
)


def official_config():
    return json.loads((REPO / "tests" / "data" / "inkling-small-config.json").read_text())


def make_source(*, mutate_config=None, missing_name=None, assets=True):
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    cfg = official_config()
    if mutate_config:
        mutate_config(cfg)
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    names = required_text_tensor_names(cfg)
    if missing_name:
        names.discard(missing_name)
    wm = {name: f"model-{i % CHECKPOINT_SHARDS + 1:05d}-of-{CHECKPOINT_SHARDS:05d}.safetensors"
          for i, name in enumerate(sorted(names))}
    # Ensure every official shard name is represented even in the metadata-only fixture.
    for i in range(CHECKPOINT_SHARDS):
        wm[f"fixture.nontext.{i}"] = f"model-{i + 1:05d}-of-{CHECKPOINT_SHARDS:05d}.safetensors"
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": CHECKPOINT_TOTAL_SIZE}, "weight_map": wm}),
        encoding="utf-8",
    )
    if assets:
        for name in (
            "chat_template.jinja", "processor_config.json", "special_tokens_map.json",
            "tokenizer.json", "tokenizer_config.json",
        ):
            (root / name).write_text("{}" if name.endswith(".json") else "template", encoding="utf-8")
        (root / "tiktoken").mkdir()
        (root / "tiktoken" / "tokenizer.model").write_bytes(b"tokenizer")
    return td, root


class InklingReleaseTest(unittest.TestCase):
    def test_exact_official_profile_and_package_pass(self):
        td, root = make_source()
        self.addCleanup(td.cleanup)
        report = require_official_small(root)
        self.assertTrue(report["official_small"])
        self.assertEqual(report["package"]["shard_count"], 32)
        self.assertEqual(report["package"]["required_text_tensor_count"], 878)

    def test_profile_change_is_not_silently_labelled_small(self):
        td, root = make_source(mutate_config=lambda cfg: cfg["text_config"].__setitem__("hidden_size", 4097))
        self.addCleanup(td.cleanup)
        report = inspect_release(root)
        self.assertFalse(report["profile"]["match"])
        with self.assertRaisesRegex(ReleaseError, "hidden_size"):
            require_official_small(root)

    def test_missing_tensor_and_sidecar_fail_closed(self):
        missing = "model.llm.layers.41.mlp.experts.w2_weight"
        td, root = make_source(missing_name=missing, assets=False)
        self.addCleanup(td.cleanup)
        report = inspect_release(root)
        self.assertIn(missing, report["package"]["missing_text_tensors"])
        self.assertFalse(report["assets"]["complete"])
        with self.assertRaises(ReleaseError):
            require_official_small(root)


if __name__ == "__main__":
    unittest.main()
