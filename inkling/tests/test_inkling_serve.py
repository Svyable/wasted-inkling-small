#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the chat endpoint.

The ctypes binding to the private runtime is already exercised by
`test_inkling_private_c.py`, so these drive the parts this file actually
adds: chat-template rendering, sampling, request validation, the wire
format, and — most importantly — the provenance gating that stops synthetic
output being served as if it were the model's.

No torch, no libwaste, no socket to the outside: the runtime is a stub and
the server is bound to loopback on an ephemeral port.
"""

import ctypes as C
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from inkling_serve import (
    ATTESTATION,
    DEFAULT_TEMPLATE,
    InklingError,
    Session,
    Tokenizer,
    load_template,
    main,
    make_handler,
    render,
    sample,
    stage_provenance,
)


class StubRuntime:
    """Deterministic logits: token id `n` is favoured at position `n`, so a
    greedy decode produces a predictable ramp and the tests can assert on
    content rather than on 'it did not crash'."""

    def __init__(self, vocab=256, context=4096):
        self.vocab = vocab
        self.context = context
        self.steps = []
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.steps.clear()

    def step(self, token, position):
        """Returns the same type the real Runtime does — a ctypes float
        array. An earlier stub returned a plain list, which is why a
        memoryview bug in sample() reached a live server."""
        self.steps.append((token, position))
        buf = (C.c_float * self.vocab)()
        buf[(position + 1) % self.vocab] = 10.0
        return buf

    def close(self):
        pass


def session(vocab=256):
    return Session(StubRuntime(vocab), Tokenizer(vocab), DEFAULT_TEMPLATE)


class TemplateTest(unittest.TestCase):
    def test_renders_roles_in_order_and_ends_with_the_generation_prompt(self):
        out = render([{"role": "system", "content": "S"},
                      {"role": "user", "content": "U"}], DEFAULT_TEMPLATE)
        self.assertLess(out.index("S"), out.index("U"))
        self.assertTrue(out.endswith(DEFAULT_TEMPLATE["generation_prompt"]))

    def test_rejects_an_unknown_role(self):
        with self.assertRaises(InklingError):
            render([{"role": "tool", "content": "x"}], DEFAULT_TEMPLATE)

    def test_rejects_non_string_content(self):
        for bad in (None, 5, ["a"], {"a": 1}):
            with self.assertRaises(InklingError):
                render([{"role": "user", "content": bad}], DEFAULT_TEMPLATE)

    def test_absent_assets_give_the_placeholder_template(self):
        tmpl, kind = load_template(None)
        self.assertEqual(kind, "placeholder")
        self.assertEqual(tmpl, DEFAULT_TEMPLATE)

    def test_a_supplied_template_is_labelled_as_supplied(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "inkling_chat_template.json"
            p.write_text(json.dumps(dict(DEFAULT_TEMPLATE, user="U:{content}")))
            tmpl, kind = load_template(Path(tmp))
        self.assertEqual(kind, "supplied")
        self.assertEqual(tmpl["user"], "U:{content}")

    def test_an_incomplete_supplied_template_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "inkling_chat_template.json"
            p.write_text(json.dumps({"user": "{content}"}))
            with self.assertRaises(InklingError):
                load_template(Path(tmp))


class SampleTest(unittest.TestCase):
    def test_sampling_accepts_a_ctypes_buffer(self):
        """The type the real runtime hands back."""
        import random
        buf = (C.c_float * 4)(1.0, 5.0, 2.0, 0.0)
        self.assertEqual(sample(buf, 4, 0.0, 1.0, random.Random(1)), 1)
        self.assertIn(sample(buf, 4, 1.0, 1.0, random.Random(1)), range(4))

    def test_zero_temperature_is_argmax(self):
        import random
        logits = [1.0, 5.0, 2.0, 0.0]
        self.assertEqual(sample(logits, 4, 0.0, 1.0, random.Random(1)), 1)

    def test_a_dominant_logit_wins_at_temperature(self):
        import random
        logits = [0.0, 0.0, 40.0, 0.0]
        for seed in range(8):
            self.assertEqual(sample(logits, 4, 1.0, 1.0, random.Random(seed)), 2)

    def test_top_p_excludes_the_tail(self):
        import random
        logits = [10.0, 9.9, -40.0, -40.0]
        for seed in range(16):
            self.assertIn(sample(logits, 4, 1.0, 0.9, random.Random(seed)), (0, 1))

    def test_sampling_is_deterministic_for_a_seed(self):
        import random
        logits = [1.0, 1.1, 0.9, 1.05]
        a = [sample(logits, 4, 1.0, 1.0, random.Random(7)) for _ in range(5)]
        b = [sample(logits, 4, 1.0, 1.0, random.Random(7)) for _ in range(5)]
        self.assertEqual(a, b)


class ProvenanceTest(unittest.TestCase):
    """The gate that stops synthetic text being served as the model's."""

    def test_a_stage_without_an_attestation_is_synthetic(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(stage_provenance(Path(tmp)), "synthetic")

    def test_an_attested_stage_is_official(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ATTESTATION).write_text("{}")
            self.assertEqual(stage_provenance(Path(tmp)), "official")

    def test_serving_a_synthetic_stage_without_the_flag_is_refused(self):
        import io
        import tempfile
        from contextlib import redirect_stderr, redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                rc = main(["--stage", tmp])
        self.assertEqual(rc, 2)
        self.assertIn("synthetic", err.getvalue())


class CliOptionTest(unittest.TestCase):
    """--tokenizer and --template name separate artifacts. Conflating them
    meant a run that had a chat template but no tiktoken assets was refused,
    which is exactly the state this project is in until G4."""

    def test_template_can_be_supplied_without_tokenizer_assets(self):
        import io
        import tempfile
        from contextlib import redirect_stderr, redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "inkling_chat_template.json").write_text(
                json.dumps(DEFAULT_TEMPLATE))
            # No stage, so this still exits 2 — but on the stage, not on the
            # template, which is the distinction being asserted.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                rc = main(["--stage", tmp, "--template", tmp,
                           "--i-know-the-weights-are-synthetic"])
        self.assertEqual(rc, 2)
        self.assertNotIn("tokenizer assets", err.getvalue())


class ContextLimitTest(unittest.TestCase):
    """Found by pointing the server at a real stage: overrunning the K/V
    rings surfaced as an opaque 'step failed at position 16' from deep in the
    C. It has to be refused at the edge, where the caller can act on it."""

    def test_a_prompt_plus_generation_over_capacity_is_refused(self):
        s = Session(StubRuntime(context=16), Tokenizer(256), DEFAULT_TEMPLATE)
        with self.assertRaises(InklingError) as ctx:
            list(s.generate([{"role": "user", "content": "hello there"}],
                            max_tokens=8, temperature=0.0, top_p=1.0, seed=1))
        self.assertIn("context exceeded", str(ctx.exception))

    def test_it_is_refused_before_the_runtime_is_touched(self):
        rt = StubRuntime(context=8)
        s = Session(rt, Tokenizer(256), DEFAULT_TEMPLATE)
        with self.assertRaises(InklingError):
            list(s.generate([{"role": "user", "content": "a longer prompt here"}],
                            max_tokens=4, temperature=0.0, top_p=1.0, seed=1))
        self.assertEqual(rt.steps, [], "stepped before refusing")

    def test_a_request_inside_capacity_runs(self):
        s = Session(StubRuntime(context=4096), Tokenizer(256), DEFAULT_TEMPLATE)
        ids = list(s.generate([{"role": "user", "content": "hi"}], 4, 0.0, 1.0, 1))
        self.assertEqual(len(ids), 4)


class GenerationTest(unittest.TestCase):
    def test_prefill_then_decode_visits_every_position_once(self):
        s = session()
        ids = list(s.generate([{"role": "user", "content": "hello"}],
                              max_tokens=4, temperature=0.0, top_p=1.0, seed=1))
        self.assertEqual(len(ids), 4)
        positions = [p for _t, p in s.runtime.steps]
        self.assertEqual(positions, list(range(len(positions))))

    def test_each_generation_resets_the_runtime(self):
        s = session()
        for _ in range(3):
            list(s.generate([{"role": "user", "content": "x"}], 2, 0.0, 1.0, 1))
        self.assertEqual(s.runtime.resets, 3)

    def test_an_empty_prompt_is_refused(self):
        s = Session(StubRuntime(), Tokenizer(256), dict(DEFAULT_TEMPLATE,
                    user="", system="", assistant="", generation_prompt=""))
        with self.assertRaises(InklingError):
            list(s.generate([{"role": "user", "content": ""}], 2, 0.0, 1.0, 1))


class HttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = session()
        handler = make_handler(cls.session, "weights=synthetic tokenizer=fallback",
                               "inkling-test")
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def post(self, body, path="/v1/chat/completions"):
        req = urllib.request.Request(
            self.url(path), data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        return urllib.request.urlopen(req, timeout=30)

    def test_healthz(self):
        with urllib.request.urlopen(self.url("/healthz"), timeout=10) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(json.load(r)["status"], "ok")

    def test_models_lists_the_model_with_its_provenance(self):
        with urllib.request.urlopen(self.url("/v1/models"), timeout=10) as r:
            data = json.load(r)
        self.assertEqual(data["data"][0]["id"], "inkling-test")
        self.assertIn("synthetic", data["data"][0]["provenance"])
        self.assertEqual(data["data"][0]["context_length"], 4096)

    def test_chat_completion_shape(self):
        with self.post({"messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 5, "temperature": 0}) as r:
            self.assertEqual(r.status, 200)
            self.assertIn("synthetic", r.headers["x-waste-provenance"])
            data = json.load(r)
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        self.assertIsInstance(data["choices"][0]["message"]["content"], str)
        self.assertEqual(data["usage"]["completion_tokens"], 5)

    def test_every_response_carries_the_provenance_header(self):
        for path in ("/healthz", "/v1/models"):
            with urllib.request.urlopen(self.url(path), timeout=10) as r:
                self.assertIn("synthetic", r.headers["x-waste-provenance"])

    def test_streaming_emits_sse_frames_and_terminates(self):
        with self.post({"messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 3, "temperature": 0,
                        "stream": True}) as r:
            body = r.read().decode()
        self.assertIn("data: ", body)
        self.assertIn("[DONE]", body)
        frames = [l for l in body.splitlines() if l.startswith("data: ")]
        self.assertGreaterEqual(len(frames), 3)

    def test_bad_requests_are_refused_with_a_reason(self):
        cases = [
            ({}, "messages"),
            ({"messages": []}, "messages"),
            ({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, "max_tokens"),
            ({"messages": [{"role": "user", "content": "x"}], "max_tokens": 99999}, "max_tokens"),
            ({"messages": [{"role": "user", "content": "x"}], "temperature": 9}, "temperature"),
            ({"messages": [{"role": "user", "content": "x"}], "top_p": 0}, "top_p"),
            ({"messages": [{"role": "bogus", "content": "x"}]}, "role"),
        ]
        for body, expect in cases:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self.post(body)
            self.assertEqual(ctx.exception.code, 400, body)
            self.assertIn(expect, json.load(ctx.exception)["error"]["message"])

    def test_unknown_route_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.url("/v1/nope"), timeout=10)
        self.assertEqual(ctx.exception.code, 404)

    def test_malformed_json_is_400_not_500(self):
        req = urllib.request.Request(
            self.url("/v1/chat/completions"), data=b"{not json",
            headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
