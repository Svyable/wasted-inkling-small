#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import http.server
import json
import socketserver
import struct
import tempfile
import threading
import unittest
import zlib
from pathlib import Path

from inkling_remote_fixture import (
    HttpRangeSource,
    RemoteFixtureError,
    _required_layer_names,
    build_remote_plan,
    extract_remote_fixture,
)


def safetensors(tensors: dict[str, tuple[str, list[int], bytes]]) -> bytes:
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    raw = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(raw)) + raw + payload


class RangeHandler(http.server.BaseHTTPRequestHandler):
    root: Path
    ignore_range = False
    requests: list[tuple[str, str | None]] = []

    def log_message(self, *_args):
        pass

    def do_GET(self):
        type(self).requests.append((self.path, self.headers.get("Range")))
        parts = self.path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "resolve":
            self.send_error(404)
            return
        path = type(self).root / parts[2]
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        range_header = self.headers.get("Range")
        if range_header and not type(self).ignore_range:
            unit, value = range_header.split("=", 1)
            if unit != "bytes" or "," in value:
                self.send_error(416)
                return
            start_text, end_text = value.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start or end >= len(data):
                self.send_error(416)
                return
            body = data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Server:
    def __init__(self, root: Path, *, ignore_range: bool = False):
        handler = type("FixtureRangeHandler", (RangeHandler,), {})
        handler.root = root
        handler.ignore_range = ignore_range
        handler.requests = []
        self.handler = handler
        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        host, port = self.httpd.server_address
        return f"http://{host}:{port}", self.handler

    def __exit__(self, *_args):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()


def make_release(root: Path):
    config = {
        "text_config": {
            "num_hidden_layers": 3,
            "n_routed_experts": 4,
            "dense_mlp_idx": 1,
        }
    }
    (root / "config.json").write_text(json.dumps(config))

    tensors_a = {
        "model.llm.embed_norm.weight": ("F32", [2], struct.pack("<2f", 1, 2)),
        "model.llm.layers.0.attn_norm.weight": ("F32", [2], struct.pack("<2f", 3, 4)),
        "model.llm.layers.1.attn_norm.weight": ("F32", [2], struct.pack("<2f", 5, 6)),
        "model.llm.layers.1.mlp.experts.w13_weight": (
            "F32", [4, 2, 2], struct.pack("<16f", *range(100, 116))
        ),
    }
    tensors_b = {
        "model.llm.norm.weight": ("F32", [2], struct.pack("<2f", 7, 8)),
        "model.llm.layers.0.mlp.w13_dn.weight": (
            "F32", [2, 2], struct.pack("<4f", 9, 10, 11, 12)
        ),
        "model.llm.layers.1.mlp.gate.weight": (
            "F32", [5, 2], struct.pack("<10f", *range(20, 30))
        ),
        "model.llm.layers.1.mlp.experts.w2_weight": (
            "F32", [4, 2, 2], struct.pack("<16f", *range(200, 216))
        ),
    }
    for name in _required_layer_names(0, sparse=False):
        tensors_a.setdefault(name, ("F32", [1], struct.pack("<f", 0.25)))
    for name in _required_layer_names(1, sparse=True):
        tensors_b.setdefault(name, ("F32", [1], struct.pack("<f", 0.5)))

    shards = {
        "model-00001-of-00002.safetensors": safetensors(tensors_a),
        "model-00002-of-00002.safetensors": safetensors(tensors_b),
    }
    for name, data in shards.items():
        (root / name).write_bytes(data)
    weight_map = {name: "model-00001-of-00002.safetensors" for name in tensors_a}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in tensors_b})
    index = {
        "metadata": {"total_size": sum(len(value) for value in shards.values())},
        "weight_map": weight_map,
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    return config, index


class RemoteFixtureTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.addCleanup(self.td.cleanup)
        self.config, self.index = make_release(self.root)

    def source(self, base: str) -> HttpRangeSource:
        return HttpRangeSource(base, "abcdef0", retries=1)

    def test_plan_reads_only_metadata_and_headers(self):
        with Server(self.root) as (base, handler):
            source = self.source(base)
            plan = build_remote_plan(
                source,
                model_id="test/Inkling-Small",
                layers=[0, 1],
                experts={1: [1, 3]},
                max_total_bytes=1 << 20,
            )
        labels = {(entry.name, entry.axis0) for entry in plan.entries}
        self.assertIn(("model.llm.layers.0.mlp.w13_dn.weight", None), labels)
        self.assertIn(("model.llm.layers.1.mlp.experts.w13_weight", 1), labels)
        self.assertIn(("model.llm.layers.1.mlp.experts.w13_weight", 3), labels)
        self.assertNotIn(("model.llm.layers.1.mlp.experts.w13_weight", None), labels)
        self.assertEqual(len(plan.entries), 2 + 17 + 19 + 4)
        self.assertEqual(plan.payload_bytes, sum(entry.nbytes for entry in plan.entries))
        shard_requests = [
            item for item in handler.requests if item[0].endswith(".safetensors")
        ]
        self.assertTrue(shard_requests)
        self.assertTrue(all(header is not None for _, header in shard_requests))
        self.assertEqual(source.bytes_read, plan.metadata_bytes_read)

    def test_extracts_exact_expert_slices_and_hash_binds_source(self):
        out = self.root / "out"
        with Server(self.root) as (base, _handler):
            source = self.source(base)
            plan = build_remote_plan(
                source,
                model_id="test/Inkling-Small",
                layers=[1],
                experts={1: [2]},
                max_total_bytes=1 << 20,
            )
            manifest = extract_remote_fixture(source, plan, out, chunk_bytes=5)
        self.assertEqual(manifest["source"]["revision"], "abcdef0")
        self.assertEqual(
            manifest["source"]["config_sha256"],
            hashlib.sha256((self.root / "config.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            manifest["reader_payload_bytes"], manifest["total_payload_bytes"]
        )
        entries = {(item["name"], item.get("axis0")): item for item in manifest["entries"]}
        w13 = entries[("model.llm.layers.1.mlp.experts.w13_weight", 2)]
        blob = (out / w13["path"]).read_bytes()
        self.assertEqual(
            struct.unpack("<4f", blob), (108.0, 109.0, 110.0, 111.0)
        )
        self.assertEqual(zlib.crc32(blob) & 0xFFFFFFFF, w13["crc32"])
        self.assertTrue((out / "fixture.json").exists())

    def test_refuses_server_that_ignores_range(self):
        with Server(self.root, ignore_range=True) as (base, _handler):
            with self.assertRaisesRegex(RemoteFixtureError, "ignored Range"):
                build_remote_plan(
                    self.source(base), model_id="test/Inkling-Small", layers=[0]
                )

    def test_refuses_mutable_revision_and_unsafe_index_shard(self):
        with self.assertRaisesRegex(RemoteFixtureError, "immutable"):
            HttpRangeSource("https://example.invalid/model", "main")
        index_path = self.root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        first = next(iter(index["weight_map"]))
        index["weight_map"][first] = "../escape.safetensors"
        index_path.write_text(json.dumps(index))
        with Server(self.root) as (base, _handler):
            with self.assertRaisesRegex(RemoteFixtureError, "unsafe shard"):
                build_remote_plan(
                    self.source(base), model_id="test/Inkling-Small", layers=[0]
                )

    def test_limits_fail_before_payload_download(self):
        with Server(self.root) as (base, handler):
            with self.assertRaisesRegex(RemoteFixtureError, "total byte limit"):
                build_remote_plan(
                    self.source(base),
                    model_id="test/Inkling-Small",
                    layers=[0, 1],
                    experts={1: [0, 1, 2, 3]},
                    max_total_bytes=32,
                )
        for path, header in handler.requests:
            if path.endswith(".safetensors"):
                self.assertTrue(
                    header == "bytes=0-7" or (header and header.startswith("bytes=8-")),
                    header,
                )


if __name__ == "__main__":
    unittest.main()
