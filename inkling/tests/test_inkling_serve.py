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
    DEFAULT_TEMPLATE,
    InklingError,
    Session,
    Tokenizer,
    collect_generation,
    load_template,
    main,
    make_handler,
    open_runtime,
    render,
    sample,
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


class SequenceRuntime(StubRuntime):
    """Favour a prescribed token at each decode position."""

    def __init__(self, sequence, vocab=256, context=4096):
        super().__init__(vocab=vocab, context=context)
        self.sequence = sequence

    def step(self, token, position):
        self.steps.append((token, position))
        buf = (C.c_float * self.vocab)()
        buf[self.sequence[min(position, len(self.sequence) - 1)]] = 10.0
        return buf


class OneTokenByteTokenizer(Tokenizer):
    def encode(self, _text):
        return [0]


class EosByteTokenizer(OneTokenByteTokenizer):
    @property
    def eos(self):
        return 7


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
    """Only the C runtime's official-stage gate can label weights official."""

    @staticmethod
    def runtime_factory(mode, calls):
        class GateRuntime:
            def __init__(self, _lib, _stage, *, ctx, require_official):
                calls.append((ctx, require_official))
                if mode == "corrupt":
                    raise InklingError("corrupt stage", status=-3)
                if mode == "synthetic" and require_official:
                    raise InklingError("not official", status=-5)
        return GateRuntime

    def test_verified_runtime_is_official(self):
        calls = []
        _runtime, kind = open_runtime(
            Path("libwaste.so"), Path("stage"), 2048, False,
            self.runtime_factory("official", calls))
        self.assertEqual(kind, "official")
        self.assertEqual(calls, [(2048, True)])

    def test_synthetic_runtime_requires_the_explicit_override(self):
        calls = []
        with self.assertRaisesRegex(InklingError, "must be treated as synthetic"):
            open_runtime(
                Path("libwaste.so"), Path("stage"), 4096, False,
                self.runtime_factory("synthetic", calls))
        self.assertEqual(calls, [(4096, True)])

    def test_synthetic_override_reopens_without_the_official_requirement(self):
        calls = []
        _runtime, kind = open_runtime(
            Path("libwaste.so"), Path("stage"), 4096, True,
            self.runtime_factory("synthetic", calls))
        self.assertEqual(kind, "synthetic")
        self.assertEqual(calls, [(4096, True), (4096, False)])

    def test_bogus_attestation_file_cannot_claim_official_weights(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "official-weights.json").write_text("{}")
            with self.assertRaises(InklingError):
                open_runtime(
                    Path("libwaste.so"), Path(tmp), 4096, False,
                    self.runtime_factory("synthetic", []))

    def test_corrupt_stage_never_falls_back_to_synthetic(self):
        calls = []
        with self.assertRaisesRegex(InklingError, "corrupt stage"):
            open_runtime(
                Path("libwaste.so"), Path("stage"), 4096, True,
                self.runtime_factory("corrupt", calls))
        self.assertEqual(calls, [(4096, True)])


class CliOptionTest(unittest.TestCase):
    """Tokenizer assets and the chat template remain independent options."""

    def test_template_can_be_supplied_without_tokenizer_assets(self):
        import io
        import tempfile
        from contextlib import redirect_stderr, redirect_stdout
        from unittest.mock import patch

        class MainRuntime:
            lib = object()
            vocab = 256
            context = 4096

            def close(self):
                pass

        class OneShotServer:
            def __init__(self, *_args):
                pass

            def serve_forever(self):
                pass

            def server_close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "inkling_chat_template.json").write_text(
                json.dumps(DEFAULT_TEMPLATE))
            with (patch("inkling_serve.find_library", return_value=Path("libwaste.so")),
                  patch("inkling_serve.open_runtime",
                        return_value=(MainRuntime(), "synthetic")),
                  patch("inkling_serve.ThreadingHTTPServer", OneShotServer),
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err):
                rc = main(["--stage", tmp, "--template", tmp,
                           "--i-know-the-weights-are-synthetic"])
        self.assertEqual(rc, 0)
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

    def test_generator_reports_length_when_the_budget_is_exhausted(self):
        ids, reason = collect_generation(
            session().generate([{"role": "user", "content": "x"}],
                               2, 0.0, 1.0, 1))
        self.assertEqual(len(ids), 2)
        self.assertEqual(reason, "length")

    def test_generator_reports_stop_on_eos(self):
        tok = EosByteTokenizer(256)
        ids, reason = collect_generation(
            Session(SequenceRuntime([7]), tok, DEFAULT_TEMPLATE).generate(
                [{"role": "user", "content": "x"}], 2, 0.0, 1.0, 1))
        self.assertEqual(ids, [])
        self.assertEqual(reason, "stop")


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
        self.assertEqual(data["choices"][0]["finish_reason"], "length")
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
        payloads = [json.loads(line[6:]) for line in frames
                    if line != "data: [DONE]"]
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"], "length")

    def test_streaming_preserves_utf8_across_token_boundaries(self):
        s = Session(SequenceRuntime([0xE2, 0x82, 0xAC]),
                    OneTokenByteTokenizer(256), DEFAULT_TEMPLATE)
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(s, "weights=synthetic", "utf8-test"))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                data=json.dumps({
                    "messages": [{"role": "user", "content": "x"}],
                    "max_tokens": 3, "temperature": 0, "stream": True,
                }).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as response:
                body = response.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()
        payloads = [json.loads(line[6:]) for line in body.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"]
        content = "".join(
            item["choices"][0]["delta"].get("content", "") for item in payloads)
        self.assertEqual(content, "€")
        self.assertNotIn("\ufffd", content)

    def test_bad_requests_are_refused_with_a_reason(self):
        cases = [
            ({}, "messages"),
            ({"messages": []}, "messages"),
            ({"messages": ["not-an-object"]}, "message"),
            ({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, "max_tokens"),
            ({"messages": [{"role": "user", "content": "x"}], "max_tokens": 99999}, "max_tokens"),
            ({"messages": [{"role": "user", "content": "x"}], "max_tokens": True}, "max_tokens"),
            ({"messages": [{"role": "user", "content": "x"}], "temperature": 9}, "temperature"),
            ({"messages": [{"role": "user", "content": "x"}], "temperature": None}, "temperature"),
            ({"messages": [{"role": "user", "content": "x"}], "temperature": "0.5"}, "temperature"),
            ({"messages": [{"role": "user", "content": "x"}], "temperature": float("nan")}, "temperature"),
            ({"messages": [{"role": "user", "content": "x"}], "top_p": 0}, "top_p"),
            ({"messages": [{"role": "user", "content": "x"}], "top_p": {}}, "top_p"),
            ({"messages": [{"role": "user", "content": "x"}], "top_p": float("inf")}, "top_p"),
            ({"messages": [{"role": "user", "content": "x"}], "top_p": 10 ** 400}, "top_p"),
            ({"messages": [{"role": "user", "content": "x"}], "seed": "1"}, "seed"),
            ({"messages": [{"role": "user", "content": "x"}], "seed": True}, "seed"),
            ({"messages": [{"role": "user", "content": "x"}], "stream": "false"}, "stream"),
            ({"messages": [{"role": "user", "content": "x"}], "stream": 0}, "stream"),
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
