#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
inkling_serve.py — an OpenAI-compatible chat endpoint over the staged Inkling
runtime.

WHAT THIS IS, EXACTLY
---------------------
Public `waste_open` still returns WASTE_E_UNSUPPORTED for an Inkling
container, and nothing here changes that. This server drives the
**converter-private** runtime — `waste_inkling_private_open/step` — which is
the same path the parity harnesses use. It exists so the API surface is real,
testable and reviewable *before* the loader dispatch of ROADMAP G6 lands,
rather than after.

Upstream's `serve/` is architecture-agnostic: it reaches the engine only
through `waste_open`, `waste_tokenize` and the step API. So when G6 lands,
this file is not the chat server — `serve/` is, unchanged, and this becomes
redundant. That is the intended outcome, and it is why this deliberately
implements the same wire format rather than a nicer one.

THE MODEL IS SYNTHETIC UNTIL THE CHECKPOINT SAYS OTHERWISE
----------------------------------------------------------
A staged directory built from synthetic weights produces synthetic text. This
server says so, in the startup banner, in `/v1/models`, and in a
`x-waste-provenance` header on every response. It refuses to hide it:
`--i-know-the-weights-are-synthetic` is required to start against a stage that
does not pass the C runtime's verified official Inkling-Small profile gate,
and the flag is deliberately tedious to type.

TOKENIZER
---------
The release ships tiktoken ranks, a special-token map and a chat template.
Where those are present (`--tokenizer DIR`), they are used through libwaste's
own `waste_tok_*`. Where they are not, a byte-level fallback stands in so the
plumbing is exercisable — and every response is labelled `tokenizer=fallback`,
because a byte tokenizer against real weights would produce fluent-looking
nonsense, which is the single most misleading failure this project could ship.

    python3 tools/inkling_serve.py --stage STAGE_DIR [--tokenizer DIR]
                                   [--host 127.0.0.1] [--port 8000]

Endpoints: GET /v1/models, POST /v1/chat/completions (streaming and not),
GET /healthz. Stdlib only.
"""

from __future__ import annotations

import argparse
import codecs
import ctypes as C
import json
import math
import os
import random
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ------------------------------------------------------------ C bindings ----

PRIVATE_OK = 0
PRIVATE_E_UNSUPPORTED = -5
PRIVATE_STATUS = {
    0: "ok", -1: "argument", -2: "io", -3: "format",
    -4: "out of memory", -5: "unsupported", -6: "runtime",
}


class PrivateOptions(C.Structure):
    _fields_ = [("context_capacity", C.c_int),
                ("verify_crc", C.c_int),
                ("require_official_small", C.c_int)]


class InklingError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def _soext() -> str:
    return "dll" if os.name == "nt" else ("dylib" if sys.platform == "darwin" else "so")


def find_library(explicit: str | None = None) -> Path:
    """Most specific first, exactly as serve/engine.py does."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("WASTE_LIB"):
        candidates.append(Path(os.environ["WASTE_LIB"]))
    here = Path(__file__).resolve().parent
    for base in (here.parent, here.parent.parent, Path.cwd()):
        candidates.append(base / f"libwaste.{_soext()}")
    candidates.append(Path(f"libwaste.{_soext()}"))
    for path in candidates:
        if path.name and (path.is_file() or path == candidates[-1]):
            try:
                C.CDLL(str(path))
                return path
            except OSError:
                continue
    raise InklingError(
        f"libwaste.{_soext()} not found — build it with "
        f"`make libwaste.{_soext()}` in an applied WASTE tree, or set WASTE_LIB"
    )


class Runtime:
    """The staged private runtime: open, step, reset, close."""

    def __init__(self, lib_path: Path, stage: Path, ctx: int = 4096,
                 verify_crc: bool = True, require_official: bool = False):
        self.lib = C.CDLL(str(lib_path))
        self._bind()
        opts = PrivateOptions(ctx, 1 if verify_crc else 0,
                              1 if require_official else 0)
        handle = C.c_void_p()
        detail = C.create_string_buffer(512)
        st = self.lib.waste_inkling_private_open(
            str(stage).encode(), C.byref(opts), C.byref(handle),
            detail, len(detail))
        if st != PRIVATE_OK:
            raise InklingError(
                f"cannot open stage {stage}: {PRIVATE_STATUS.get(st, st)}"
                + (f" — {detail.value.decode(errors='replace')}" if detail.value else ""),
                status=st)
        self.handle = handle
        cfg = self.lib.waste_inkling_private_config(handle)
        if not cfg:
            raise InklingError("runtime returned no config")
        # unpadded_vocab is the only field this server needs; read it through
        # the accessor rather than declaring the whole config struct, because
        # a struct declared here would be a second copy of a C layout and this
        # repository has already been bitten once by exactly that.
        self.vocab = int(self.lib.waste_inkling_config_unpadded_vocab(cfg))
        if self.vocab <= 0:
            raise InklingError(f"implausible vocabulary size {self.vocab}")
        self.context = ctx
        self._logits = (C.c_float * self.vocab)()
        self._lock = threading.Lock()

    def _bind(self):
        L = self.lib
        L.waste_inkling_private_open.restype = C.c_int
        L.waste_inkling_private_open.argtypes = [
            C.c_char_p, C.POINTER(PrivateOptions), C.POINTER(C.c_void_p),
            C.c_char_p, C.c_size_t]
        L.waste_inkling_private_step.restype = C.c_int
        L.waste_inkling_private_step.argtypes = [
            C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_float), C.c_size_t]
        L.waste_inkling_private_reset.restype = None
        L.waste_inkling_private_reset.argtypes = [C.c_void_p]
        L.waste_inkling_private_close.restype = None
        L.waste_inkling_private_close.argtypes = [C.c_void_p]
        L.waste_inkling_private_config.restype = C.c_void_p
        L.waste_inkling_private_config.argtypes = [C.c_void_p]
        L.waste_inkling_private_error.restype = C.c_char_p
        L.waste_inkling_private_error.argtypes = [C.c_void_p]
        L.waste_inkling_config_unpadded_vocab.restype = C.c_int
        L.waste_inkling_config_unpadded_vocab.argtypes = [C.c_void_p]

    def error(self) -> str:
        msg = self.lib.waste_inkling_private_error(self.handle)
        return msg.decode(errors="replace") if msg else ""

    def step(self, token: int, position: int):
        """Returns the logits buffer itself — a `c_float * vocab`, which
        indexes and len()s like a sequence. NOT a memoryview: wrapping a
        ctypes float array in one gives format '<f', which raises
        NotImplementedError on the first subscript. That shipped once,
        because the test stub returned a plain list and hid it."""
        st = self.lib.waste_inkling_private_step(
            self.handle, token, position, self._logits, self.vocab)
        if st != PRIVATE_OK:
            raise InklingError(
                f"step failed at position {position}: "
                f"{PRIVATE_STATUS.get(st, st)} — {self.error()}")
        return self._logits

    def reset(self):
        self.lib.waste_inkling_private_reset(self.handle)

    def close(self):
        if getattr(self, "handle", None):
            self.lib.waste_inkling_private_close(self.handle)
            self.handle = None


# ------------------------------------------------------------- tokenizer ----


class Tokenizer:
    """Either the official assets through libwaste, or a byte fallback.

    The fallback exists so the server is runnable and testable with no
    checkpoint. It is never silent: `kind` reaches /v1/models and the
    provenance header, because a byte tokenizer driving real weights would
    emit fluent-looking nonsense."""

    kind = "fallback"

    def __init__(self, vocab: int):
        self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        return [b % self.vocab for b in text.encode("utf-8")]

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(i % 256 for i in ids)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    @property
    def eos(self) -> int:
        return -1


class OfficialTokenizer(Tokenizer):
    kind = "official"

    def __init__(self, lib: C.CDLL, directory: Path, vocab: int):
        super().__init__(vocab)
        lib.waste_tok_open.restype = C.c_void_p
        lib.waste_tok_open.argtypes = [C.c_char_p]
        lib.waste_tok_encode.restype = C.c_int
        lib.waste_tok_encode.argtypes = [C.c_void_p, C.c_char_p,
                                         C.POINTER(C.c_int32), C.c_int]
        lib.waste_tok_decode.restype = C.c_int
        lib.waste_tok_decode.argtypes = [C.c_void_p, C.POINTER(C.c_int32),
                                         C.c_int, C.c_char_p, C.c_int]
        lib.waste_tok_eos.restype = C.c_int
        lib.waste_tok_eos.argtypes = [C.c_void_p]
        self.lib = lib
        self.tok = lib.waste_tok_open(str(directory).encode())
        if not self.tok:
            raise InklingError(f"no tokenizer assets in {directory}")

    def encode(self, text: str) -> list[int]:
        cap = len(text.encode()) + 8
        buf = (C.c_int32 * cap)()
        n = self.lib.waste_tok_encode(self.tok, text.encode(), buf, cap)
        if n < 0:
            raise InklingError("tokenizer refused the input")
        return list(buf[:n])

    def decode_bytes(self, ids: list[int]) -> bytes:
        if not ids:
            return b""
        arr = (C.c_int32 * len(ids))(*ids)
        cap = 4 * len(ids) + 64
        buf = C.create_string_buffer(cap)
        n = self.lib.waste_tok_decode(self.tok, arr, len(ids), buf, cap)
        return buf.raw[:max(n, 0)]

    @property
    def eos(self) -> int:
        return int(self.lib.waste_tok_eos(self.tok))


# --------------------------------------------------------- chat template ----

DEFAULT_TEMPLATE = {
    "system": "<|system|>\n{content}\n",
    "user": "<|user|>\n{content}\n",
    "assistant": "<|assistant|>\n{content}\n",
    "generation_prompt": "<|assistant|>\n",
}


def load_template(directory: Path | None) -> tuple[dict, str]:
    """The release ships a chat template. Until G4 verifies ours renders it
    byte-exactly, a locally supplied one is used and labelled as such — a
    plausible-looking template is exactly the kind of unverified detail this
    project refuses to present as official."""
    if directory:
        path = Path(directory) / "inkling_chat_template.json"
        if path.is_file():
            data = json.loads(path.read_text())
            missing = set(DEFAULT_TEMPLATE) - set(data)
            if missing:
                raise InklingError(
                    f"chat template {path} is missing {sorted(missing)}")
            return data, "supplied"
    return DEFAULT_TEMPLATE, "placeholder"


def render(messages: list[dict], template: dict) -> str:
    out = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            raise InklingError(f"unsupported role {role!r}")
        content = m.get("content")
        if not isinstance(content, str):
            raise InklingError("message content must be a string")
        out.append(template[role].format(content=content))
    out.append(template["generation_prompt"])
    return "".join(out)


# ---------------------------------------------------------------- sample ----


def sample(logits, n: int, temperature: float, top_p: float,
           rng: random.Random) -> int:
    if temperature <= 0.0:
        best, best_v = 0, -math.inf
        for i in range(n):
            if logits[i] > best_v:
                best, best_v = i, logits[i]
        return best
    scaled = [logits[i] / temperature for i in range(n)]
    top = max(scaled)
    exps = [math.exp(v - top) for v in scaled]
    total = sum(exps)
    probs = sorted(((e / total, i) for i, e in enumerate(exps)), reverse=True)
    if 0.0 < top_p < 1.0:
        acc, cut = 0.0, len(probs)
        for k, (p, _i) in enumerate(probs):
            acc += p
            if acc >= top_p:
                cut = k + 1
                break
        probs = probs[:cut]
    mass = sum(p for p, _ in probs)
    r = rng.random() * mass
    acc = 0.0
    for p, i in probs:
        acc += p
        if acc >= r:
            return i
    return probs[-1][1]


def collect_generation(gen) -> tuple[list[int], str]:
    """Consume a generation without losing the generator's finish reason."""
    ids = []
    while True:
        try:
            ids.append(next(gen))
        except StopIteration as done:
            reason = done.value if done.value in ("stop", "length") else "stop"
            return ids, reason


def request_number(req: dict, name: str, default: float) -> float:
    """Read a finite JSON number. Booleans and numeric strings are not numbers."""
    value = req.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InklingError(f"{name} must be a number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise InklingError(f"{name} must be finite") from exc
    if not math.isfinite(value):
        raise InklingError(f"{name} must be finite")
    return value


# ---------------------------------------------------------------- server ----


class Session:
    """One generation. The runtime holds a single K/V state, so generations
    are serialized — correctness before concurrency, and the private runtime
    was never built for the latter."""

    def __init__(self, runtime: Runtime, tokenizer: Tokenizer, template: dict):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.template = template
        self.lock = threading.Lock()

    def generate(self, messages, max_tokens, temperature, top_p, seed):
        prompt = render(messages, self.template)
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise InklingError("prompt encoded to zero tokens")
        # The runtime's K/V rings are sized at open. Stepping past them is a
        # runtime error deep in the C, which surfaced here as an opaque
        # "step failed at position N" the first time this server was pointed
        # at a real stage. Refuse it at the edge, where the caller can act.
        want = len(ids) + max_tokens
        if want > self.runtime.context:
            raise InklingError(
                f"context exceeded: prompt is {len(ids)} tokens and max_tokens "
                f"is {max_tokens}, but this runtime was opened with a context "
                f"capacity of {self.runtime.context}")
        rng = random.Random(seed)
        with self.lock:
            self.runtime.reset()
            pos = 0
            for tok in ids[:-1]:
                self.runtime.step(tok, pos)
                pos += 1
            nxt = ids[-1]
            for _ in range(max_tokens):
                logits = self.runtime.step(nxt, pos)
                pos += 1
                nxt = sample(logits, self.runtime.vocab, temperature, top_p, rng)
                if nxt == self.tokenizer.eos:
                    return "stop"
                yield nxt
            return "length"


def make_handler(session: Session, provenance: str, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "waste-inkling-preview"

        def log_message(self, fmt, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code, payload, ctype="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("x-waste-provenance", provenance)
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code, message, kind="invalid_request_error"):
            self._send(code, {"error": {"message": message, "type": kind}})

        def do_GET(self):
            if self.path == "/healthz":
                return self._send(200, {"status": "ok", "provenance": provenance})
            if self.path in ("/v1/models", "/models"):
                return self._send(200, {"object": "list", "data": [{
                    "id": model_name, "object": "model",
                    "owned_by": "waste-inkling-preview",
                    "provenance": provenance,
                    "context_length": session.runtime.context,
                }]})
            return self._error(404, f"no route {self.path}", "not_found")

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                return self._error(404, f"no route {self.path}", "not_found")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._error(400, "bad Content-Length")
            if length <= 0 or length > 4 << 20:
                return self._error(400, "request body must be 1..4 MiB")
            try:
                req = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                return self._error(400, f"malformed JSON: {exc}")
            if not isinstance(req, dict):
                return self._error(400, "body must be a JSON object")

            messages = req.get("messages")
            if not isinstance(messages, list) or not messages:
                return self._error(400, "messages must be a non-empty array")
            if not all(isinstance(message, dict) for message in messages):
                return self._error(400, "each message must be an object")
            max_tokens = req.get("max_tokens", 128)
            if (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
                    or not 1 <= max_tokens <= 4096):
                return self._error(400, "max_tokens must be 1..4096")
            try:
                temperature = request_number(req, "temperature", 1.0)
                top_p = request_number(req, "top_p", 1.0)
            except InklingError as exc:
                return self._error(400, str(exc))
            if not 0.0 <= temperature <= 4.0:
                return self._error(400, "temperature must be 0..4")
            if not 0.0 < top_p <= 1.0:
                return self._error(400, "top_p must be in (0, 1]")
            seed = req.get("seed")
            if (seed is not None
                    and (isinstance(seed, bool) or not isinstance(seed, int))):
                return self._error(400, "seed must be an integer")
            stream = req.get("stream", False)
            if not isinstance(stream, bool):
                return self._error(400, "stream must be a boolean")

            try:
                gen = session.generate(messages, max_tokens, temperature, top_p, seed)
                if stream:
                    return self._stream(gen)
                ids, finish_reason = collect_generation(gen)
            except InklingError as exc:
                return self._error(400, str(exc))

            text = session.tokenizer.decode(ids)
            self._send(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "provenance": provenance,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }],
                "usage": {"completion_tokens": len(ids)},
            })

        def _stream(self, gen):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("x-waste-provenance", provenance)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            ident = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

            def chunk(delta, finish=None):
                payload = json.dumps({
                    "id": ident, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model_name,
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish}],
                }).encode()
                frame = b"data: " + payload + b"\n\n"
                self.wfile.write(b"%x\r\n" % len(frame) + frame + b"\r\n")
                self.wfile.flush()

            try:
                chunk({"role": "assistant"})
                while True:
                    try:
                        tok = next(gen)
                    except StopIteration as done:
                        finish = (done.value
                                  if done.value in ("stop", "length") else "stop")
                        break
                    content = decoder.decode(
                        session.tokenizer.decode_bytes([tok]), final=False)
                    if content:
                        chunk({"content": content})
                content = decoder.decode(b"", final=True)
                if content:
                    chunk({"content": content})
                chunk({}, finish=finish)
                done = b"data: [DONE]\n\n"
                self.wfile.write(b"%x\r\n" % len(done) + done + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except InklingError as exc:
                try:
                    chunk({"content": f"\n[error: {exc}]"}, finish="stop")
                    self.wfile.write(b"0\r\n\r\n")
                except OSError:
                    pass

    return Handler


def open_runtime(lib_path: Path, stage: Path, ctx: int,
                 allow_synthetic: bool, runtime_cls=Runtime):
    """Let the C runtime prove official provenance before we claim it.

    The first open requires the verified official Inkling-Small stage flag and
    geometry. Only that specific fail-closed refusal may fall back to a
    synthetic open, and only after the caller supplies the explicit override.
    Corrupt, incomplete, or unreadable stages never take the fallback path.
    """
    try:
        return (runtime_cls(lib_path, stage, ctx=ctx, require_official=True),
                "official")
    except InklingError as exc:
        if exc.status != PRIVATE_E_UNSUPPORTED:
            raise
        if not allow_synthetic:
            raise InklingError(
                f"refusing to serve {stage}: the runtime stage did not pass "
                "the verified official Inkling-Small profile gate, so the "
                "weights must be treated as synthetic. Pass "
                "--i-know-the-weights-are-synthetic to serve it anyway.",
                status=exc.status) from exc
    return (runtime_cls(lib_path, stage, ctx=ctx, require_official=False),
            "synthetic")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", type=Path, required=True,
                    help="directory holding runtime-stage.bin and its artifacts")
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="directory with the official tiktoken assets")
    ap.add_argument("--template", type=Path, default=None,
                    help="directory holding inkling_chat_template.json; "
                         "separate from --tokenizer because the two are "
                         "separate artifacts and a run may have either")
    ap.add_argument("--lib", default=None, help="path to libwaste")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--model-name", default="inkling-small-preview")
    ap.add_argument("--i-know-the-weights-are-synthetic", action="store_true",
                    dest="allow_synthetic")
    args = ap.parse_args(argv)

    try:
        lib_path = find_library(args.lib)
        runtime, weights_kind = open_runtime(
            lib_path, args.stage, args.context, args.allow_synthetic)
        if args.tokenizer:
            tokenizer = OfficialTokenizer(runtime.lib, args.tokenizer, runtime.vocab)
        else:
            tokenizer = Tokenizer(runtime.vocab)
        template, template_kind = load_template(args.template or args.tokenizer)
    except InklingError as exc:
        print(f"inkling_serve: {exc}", file=sys.stderr)
        return 2

    provenance = (f"weights={weights_kind} tokenizer={tokenizer.kind} "
                  f"template={template_kind}")
    session = Session(runtime, tokenizer, template)
    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(session, provenance, args.model_name))

    print(f"inkling_serve: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  library      {lib_path}")
    print(f"  stage        {args.stage}")
    print(f"  vocabulary   {runtime.vocab}")
    print(f"  provenance   {provenance}")
    if "weights=synthetic" in provenance:
        print("  NOTE: synthetic weights. The text is not this model's output.")
    if tokenizer.kind == "fallback":
        print("  NOTE: byte-level fallback tokenizer, not the official one.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
