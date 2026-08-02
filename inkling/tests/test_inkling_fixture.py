#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light tests for the bounded parity fixture reader.

No torch: validating a fixture is a bytes-and-CRC problem, and this suite runs
in CI on a machine that has never seen a checkpoint.
"""

import json
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from inkling_fixture import Fixture, FixtureError, decode_f32, load_fixture, main


CONFIG_SHA = "a" * 64
INDEX_SHA = "b" * 64


def payload(nbytes: int, seed: int = 0) -> bytes:
    return bytes((seed + i * 7) & 0xFF for i in range(nbytes))


class FixtureBuilder:
    """Writes a minimal but structurally exact parity fixture."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: list[dict] = []
        self.layers = [0, 1]
        self.experts = {1: [3, 9]}

    def add_tensor(self, name: str, shape: list[int], dtype: str = "BF16",
                   *, seed: int = 1, corrupt_crc: bool = False,
                   truncate: bool = False) -> "FixtureBuilder":
        item = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
        nbytes = item
        for dim in shape:
            nbytes *= dim
        blob = payload(nbytes, seed)
        rel = f"{name.replace('.', '_')}.bin"
        (self.root / rel).write_bytes(blob[:-1] if truncate else blob)
        crc = zlib.crc32(blob) & 0xFFFFFFFF
        self.entries.append({
            "name": name, "kind": "tensor", "dtype": dtype, "shape": list(shape),
            "bytes": nbytes, "crc32": (crc ^ 0xFFFF) & 0xFFFFFFFF if corrupt_crc else crc,
            "path": rel,
        })
        return self

    def add_slice(self, name: str, expert: int, shape: list[int],
                  dtype: str = "BF16", *, seed: int = 2) -> "FixtureBuilder":
        item = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
        nbytes = item
        for dim in shape:
            nbytes *= dim
        blob = payload(nbytes, seed + expert)
        rel = f"{name.replace('.', '_')}_{expert}.bin"
        (self.root / rel).write_bytes(blob)
        self.entries.append({
            "name": name, "kind": "axis0-slice", "axis0": expert, "dtype": dtype,
            "shape": list(shape), "bytes": nbytes,
            "crc32": zlib.crc32(blob) & 0xFFFFFFFF, "path": rel,
        })
        return self

    def manifest(self, **overrides) -> dict:
        value = {
            "format": "inkling-parity-fixture",
            "version": 1,
            "model_id": "thinkingmachines/Inkling-Small",
            "layers": self.layers,
            "experts": {str(k): v for k, v in self.experts.items()},
            "source": {"config_sha256": CONFIG_SHA, "index_sha256": INDEX_SHA},
            "total_payload_bytes": sum(e["bytes"] for e in self.entries),
            "entries": self.entries,
        }
        value.update(overrides)
        return value

    def write(self, **overrides) -> Path:
        (self.root / "fixture.json").write_text(
            json.dumps(self.manifest(**overrides), indent=2), encoding="utf-8")
        return self.root


def complete(root: Path) -> FixtureBuilder:
    return (FixtureBuilder(root)
            .add_tensor("model.llm.embed_norm.weight", [8])
            .add_tensor("model.llm.layers.0.attn.q_proj.weight", [8, 4])
            .add_tensor("model.llm.layers.1.attn.q_proj.weight", [8, 4], seed=5)
            .add_slice("model.llm.layers.1.mlp.experts.w13_weight", 3, [4, 8])
            .add_slice("model.llm.layers.1.mlp.experts.w13_weight", 9, [4, 8]))


class FixtureLoadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.addCleanup(self.td.cleanup)

    def test_loads_and_reports_geometry(self) -> None:
        complete(self.root).write()
        fixture = load_fixture(self.root)
        self.assertEqual(len(fixture), 5)
        self.assertEqual(fixture.layers, (0, 1))
        self.assertEqual(fixture.experts, {1: (3, 9)})
        self.assertEqual(fixture.model_id, "thinkingmachines/Inkling-Small")
        self.assertEqual(fixture.payload_bytes, 16 + 64 + 64 + 64 + 64)
        self.assertIn("model.llm.embed_norm.weight", fixture.names())

    def test_reads_payloads_and_verifies_crc(self) -> None:
        complete(self.root).write()
        fixture = load_fixture(self.root)
        blob = fixture.raw("model.llm.layers.0.attn.q_proj.weight")
        self.assertEqual(len(blob), 64)
        self.assertEqual(fixture.verify(), fixture.payload_bytes)

    def test_expert_slices_are_addressed_by_axis0(self) -> None:
        complete(self.root).write()
        fixture = load_fixture(self.root)
        name = "model.llm.layers.1.mlp.experts.w13_weight"
        self.assertTrue(fixture.has(name, 3))
        self.assertTrue(fixture.has(name, 9))
        self.assertFalse(fixture.has(name, 4))
        self.assertNotEqual(fixture.raw(name, 3), fixture.raw(name, 9))
        self.assertEqual(fixture.entry(name, 3).shape, (4, 8))
        self.assertEqual(fixture.entry(name, 3).label, f"{name}[3]")

    def test_layer_entries_are_scoped(self) -> None:
        complete(self.root).write()
        fixture = load_fixture(self.root)
        names = {e.name for e in fixture.layer_entries(0)}
        self.assertEqual(names, {"model.llm.layers.0.attn.q_proj.weight"})
        self.assertEqual(len(fixture.layer_entries(1)), 3)

    def test_source_binding(self) -> None:
        complete(self.root).write()
        fixture = load_fixture(self.root)
        self.assertTrue(fixture.bound_to(CONFIG_SHA, INDEX_SHA))
        self.assertFalse(fixture.bound_to(CONFIG_SHA, "c" * 64))


class FixtureRejectionTest(unittest.TestCase):
    """Every one of these would otherwise produce plausible wrong parity."""

    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.addCleanup(self.td.cleanup)

    def assertRejects(self, needle: str, **overrides) -> None:
        complete(self.root).write(**overrides)
        with self.assertRaises(FixtureError) as ctx:
            load_fixture(self.root)
        self.assertIn(needle, str(ctx.exception))

    def test_missing_manifest(self) -> None:
        with self.assertRaises(FixtureError):
            load_fixture(self.root)

    def test_wrong_format(self) -> None:
        self.assertRejects("not an Inkling parity fixture", format="activations")

    def test_wrong_version(self) -> None:
        self.assertRejects("unsupported fixture version", version=2)

    def test_no_layers(self) -> None:
        self.assertRejects("no valid layer list", layers=[])

    def test_negative_layer(self) -> None:
        self.assertRejects("no valid layer list", layers=[0, -1])

    def test_duplicate_layers(self) -> None:
        self.assertRejects("duplicates", layers=[0, 0, 1])

    def test_expert_selection_outside_layers(self) -> None:
        self.assertRejects("unselected layer", experts={"7": [1]})

    def test_declared_total_must_match(self) -> None:
        self.assertRejects("payload bytes", total_payload_bytes=1)

    def test_path_traversal_is_refused(self) -> None:
        builder = complete(self.root)
        builder.entries[0]["path"] = "../escape.bin"
        with self.assertRaises(FixtureError) as ctx:
            load_fixture(builder.write())
        self.assertIn("plain file name", str(ctx.exception))

    def test_geometry_must_match_byte_count(self) -> None:
        builder = complete(self.root)
        builder.entries[0]["bytes"] = 999
        with self.assertRaises(FixtureError) as ctx:
            load_fixture(builder.write())
        self.assertIn("geometry implies", str(ctx.exception))

    def test_unsupported_dtype(self) -> None:
        builder = complete(self.root)
        builder.entries[0]["dtype"] = "Q8"
        with self.assertRaises(FixtureError) as ctx:
            load_fixture(builder.write())
        self.assertIn("unsupported fixture dtype", str(ctx.exception))

    def test_slice_without_axis0(self) -> None:
        builder = complete(self.root)
        del builder.entries[3]["axis0"]
        with self.assertRaises(FixtureError) as ctx:
            load_fixture(builder.write())
        self.assertIn("no valid axis0", str(ctx.exception))

    def test_tensor_with_axis0(self) -> None:
        builder = complete(self.root)
        builder.entries[0]["axis0"] = 0
        with self.assertRaises(FixtureError) as ctx:
            load_fixture(builder.write())
        self.assertIn("must not carry an axis0", str(ctx.exception))

    def test_duplicate_entry(self) -> None:
        builder = complete(self.root)
        builder.entries.append(dict(builder.entries[0]))
        with self.assertRaises(FixtureError) as ctx:
            load_fixture(builder.write(total_payload_bytes=None))
        self.assertIn("duplicate fixture entry", str(ctx.exception))

    def test_short_payload_file(self) -> None:
        builder = FixtureBuilder(self.root).add_tensor(
            "model.llm.embed_norm.weight", [8], truncate=True)
        with self.assertRaises(FixtureError) as ctx:
            load_fixture(builder.write())
        self.assertIn("manifest says", str(ctx.exception))

    def test_crc_corruption_is_caught_on_read(self) -> None:
        builder = FixtureBuilder(self.root).add_tensor(
            "model.llm.embed_norm.weight", [8], corrupt_crc=True)
        fixture = load_fixture(builder.write())
        with self.assertRaises(FixtureError) as ctx:
            fixture.raw("model.llm.embed_norm.weight")
        self.assertIn("failed CRC verification", str(ctx.exception))
        with self.assertRaises(FixtureError):
            fixture.verify()

    def test_missing_entry_is_named(self) -> None:
        fixture = load_fixture(complete(self.root).write())
        with self.assertRaises(FixtureError) as ctx:
            fixture.raw("model.llm.nonexistent.weight")
        self.assertIn("does not contain", str(ctx.exception))


class FixtureCoverageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.addCleanup(self.td.cleanup)
        self.fixture = load_fixture(complete(self.root).write())

    def test_require_layers_accepts_covered(self) -> None:
        self.fixture.require_layers([0, 1])

    def test_require_layers_refuses_uncovered(self) -> None:
        with self.assertRaises(FixtureError) as ctx:
            self.fixture.require_layers([0, 5])
        self.assertIn("[5] are absent", str(ctx.exception))

    def test_require_experts(self) -> None:
        self.fixture.require_experts(1, [3])
        with self.assertRaises(FixtureError) as ctx:
            self.fixture.require_experts(1, [3, 4])
        self.assertIn("[4] are absent", str(ctx.exception))

    def test_require_experts_checks_layer_first(self) -> None:
        with self.assertRaises(FixtureError) as ctx:
            self.fixture.require_experts(9, [1])
        self.assertIn("are absent", str(ctx.exception))


class DecodeTest(unittest.TestCase):
    """The widening must be exact: it sits underneath every parity number."""

    def test_f32_roundtrip(self) -> None:
        import struct
        # Exactly representable in float32, so the comparison is about the
        # decode and not about double rounding.
        values = [0.0, -0.0, 1.0, -2.5, 0.5, 1024.0, -65536.0]
        raw = struct.pack("<7f", *values)
        self.assertEqual(list(decode_f32(raw, "F32")), values)

    def test_bf16_places_bits_in_the_high_half(self) -> None:
        # 0x3F80 -> 1.0, 0xC000 -> -2.0, 0x0000 -> 0.0
        raw = bytes([0x80, 0x3F, 0x00, 0xC0, 0x00, 0x00])
        self.assertEqual(list(decode_f32(raw, "BF16")), [1.0, -2.0, 0.0])

    def test_f16_normal_subnormal_and_specials(self) -> None:
        import struct
        # 1.0, -2.0, smallest subnormal, +inf
        raw = struct.pack("<4H", 0x3C00, 0xC000, 0x0001, 0x7C00)
        got = list(decode_f32(raw, "F16"))
        self.assertEqual(got[0], 1.0)
        self.assertEqual(got[1], -2.0)
        self.assertAlmostEqual(got[2], 5.960464477539063e-08, places=14)
        self.assertEqual(got[3], float("inf"))

    def test_rejects_ragged_payload(self) -> None:
        with self.assertRaises(FixtureError):
            decode_f32(b"\x00\x00\x00", "BF16")

    def test_rejects_unknown_dtype(self) -> None:
        with self.assertRaises(FixtureError):
            decode_f32(b"\x00\x00", "Q4")

    def test_values_reads_through_the_fixture(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        complete(root).write()
        fixture = load_fixture(root)
        values = fixture.values("model.llm.layers.0.attn.q_proj.weight")
        self.assertEqual(len(values), 32)

    def test_matches_torch_when_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        import struct
        raw = struct.pack("<8H", 0x3F80, 0xC000, 0x0000, 0x3FC0,
                          0x4049, 0xBF00, 0x7F80, 0x0080)
        reference = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
        got = decode_f32(raw, "BF16")
        self.assertEqual(list(got), reference.tolist())

        half = struct.pack("<6H", 0x3C00, 0xC000, 0x0001, 0x7BFF, 0x0400, 0x03FF)
        reference = torch.frombuffer(bytearray(half), dtype=torch.float16).float()
        self.assertEqual(list(decode_f32(half, "F16")), reference.tolist())


class StateDictTest(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.addCleanup(self.td.cleanup)
        self.fixture = load_fixture(complete(self.root).write())

    def test_keys_are_module_relative(self) -> None:
        entries = self.fixture.layer_state_dict_entries(1)
        self.assertEqual(sorted(entries), ["attn.q_proj.weight"])
        self.assertEqual(entries["attn.q_proj.weight"].shape, (8, 4))

    def test_expert_slices_are_excluded_from_the_state_dict(self) -> None:
        for key in self.fixture.layer_state_dict_entries(1):
            self.assertNotIn("experts", key)

    def test_expert_slices_are_addressable(self) -> None:
        slices = self.fixture.expert_slices(1, "w13_weight")
        self.assertEqual(sorted(slices), [3, 9])
        self.assertEqual(slices[3].shape, (4, 8))

    def test_missing_expert_tensor_is_named(self) -> None:
        with self.assertRaises(FixtureError) as ctx:
            self.fixture.expert_slices(1, "w2_weight")
        self.assertIn("no w2_weight slices", str(ctx.exception))

    def test_uncovered_layer_is_refused(self) -> None:
        with self.assertRaises(FixtureError):
            self.fixture.layer_state_dict_entries(4)

    def test_layer_without_dense_tensors_is_refused(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        builder = (FixtureBuilder(root)
                   .add_tensor("model.llm.layers.0.attn.q_proj.weight", [8, 4])
                   .add_slice("model.llm.layers.1.mlp.experts.w13_weight", 3, [4, 8]))
        fixture = load_fixture(builder.write())
        with self.assertRaises(FixtureError) as ctx:
            fixture.layer_state_dict_entries(1)
        self.assertIn("no dense tensors for layer 1", str(ctx.exception))


class FixtureCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.addCleanup(self.td.cleanup)

    def test_cli_reports_and_verifies(self) -> None:
        complete(self.root).write()
        self.assertEqual(main(["--fixture", str(self.root), "--verify", "--json"]), 0)

    def test_cli_fails_on_bad_fixture(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--fixture", str(self.root)])


if __name__ == "__main__":
    unittest.main()
