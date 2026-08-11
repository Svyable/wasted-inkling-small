#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate the checked-in Inkling final head on official head weights.

The head is the last three operations of a decode step: the final RMS
normalization, the logits-width completion, and the vocabulary projection.
This gate runs ``waste_inkling_final_head_profile(..., BF16_REFERENCE)`` from
unmodified ``inkling/src`` against the official ``InklingRMSNorm`` and a
bounded set of official unembedding rows, on one supplied hidden-state vector.

What this gate does and does not claim
-------------------------------------

It claims that the checked-in completion semantics of the head match the
official stack for the hidden state it was given, on the recorded reference
profile. It does **not** claim final-model logits: the hidden state is an input
here, not the output of a proven 42-layer decoder. The report says so in
``claims``, and the runner refuses to omit the provenance of that vector.

Everything kernel-sensitive goes through the native BF16 matrix backend, the
same contract the promoted layer profile uses -- the checked-in C refuses the
scalar resident path under BF16_REFERENCE, so a green result cannot have been
manufactured by widening a BF16 table into a double-accumulating dot product.

The vocabulary projection is bounded on purpose. The full table is gigabytes;
a fixture carries selected rows as axis-0 slices, exactly the way sparse-layer
evidence carries selected experts, and the head evaluates the selection in
selection order.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from diagnose_inkling_native_bf16_backend import (
    MatvecCallback,
    native_bfloat16_linear,
)
from inkling_fixture import FixtureError, load_fixture, UNEMBED_NAME
from inkling_fixture_reference import (
    FixtureReferenceError,
    verify_config_binding,
)
from inkling_layer_parity import (
    FP,
    Config,
    LayerParityError,
    MatrixBackend,
    Trace as TraceStruct,
    TraceCollector,
    _build_config,
)
from inkling_release_config import build_transformers_text_config
from measure_inkling_dense_bf16_profile import metrics

BF16_REFERENCE = 1
MAT_UNEMBED = 15
FINAL_NORM_NAME = "model.llm.norm.weight"
POINTS = ("final_norm", "final_norm_scaled", "logits")

# The head needs the model translation unit, which layer parity does not build.
HEAD_SOURCES = ("inkling_model.c", "inkling_layer.c", "inkling_attention.c",
                "inkling_config.c", "inkling.c", "inkling_wexp.c")


class FinalHeadEvidenceError(RuntimeError):
    """The final-head gate cannot proceed without guessing."""


class ModelWeights(ctypes.Structure):
    """Mirror of waste_inkling_model_weights.

    ``layer`` stays an opaque pointer: the head never reaches a decoder layer,
    and declaring the layer weight struct here would be a second transcription
    to keep in sync for no gain.
    """

    _fields_ = [
        ("embedding", FP),
        ("embed_norm", FP),
        ("final_norm", FP),
        ("unembedding", FP),
        ("unembedding_rows", ctypes.c_int),
        ("layer", ctypes.c_void_p),
        ("embedding_get", ctypes.c_void_p),
        ("embedding_ctx", ctypes.c_void_p),
        ("unembedding_get", ctypes.c_void_p),
        ("unembedding_ctx", ctypes.c_void_p),
    ]


def build_head_library(out: Path) -> tuple[Path, dict[str, Any]]:
    """Compile unmodified inkling/src and record what was compiled."""
    source_root = Path(__file__).resolve().parents[2] / "inkling" / "src"
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        raise FinalHeadEvidenceError("no C compiler available")
    command = [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared",
               "-fPIC", f"-I{source_root}"]
    command += [str(source_root / name) for name in HEAD_SOURCES]
    command += ["-o", str(out), "-lm"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise FinalHeadEvidenceError(f"library build failed:\n{result.stderr}")
    digests = {
        f"{name.removesuffix('.c')}_sha256":
            hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in HEAD_SOURCES
    }
    return out, {
        "production_source_unchanged": True,
        "compiled_source_root": str(source_root),
        "compiled_units": list(HEAD_SOURCES),
        "numeric_profile": "WASTE_INKLING_NUMERIC_BF16_REFERENCE",
        "numeric_profile_value": BF16_REFERENCE,
        "source_rewriting": False,
        "unembedding_storage": "matrix-backend-only",
        **digests,
    }


def configure_head_library(lib: ctypes.CDLL) -> None:
    lib.waste_inkling_config_build.restype = ctypes.c_int
    lib.waste_inkling_final_head_profile.restype = ctypes.c_int
    lib.waste_inkling_final_head_profile.argtypes = [
        ctypes.POINTER(Config), ctypes.POINTER(ModelWeights),
        ctypes.POINTER(MatrixBackend), FP,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        FP, ctypes.c_size_t, FP, FP, ctypes.c_int, ctypes.POINTER(TraceStruct),
    ]


class UnembedBackend:
    """Serve the selected official unembedding rows through the C callback."""

    def __init__(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix.detach().to(torch.bfloat16).cpu().contiguous()
        self.calls: list[dict[str, Any]] = []
        self.error: str | None = None
        self.callback = MatvecCallback(self._call)
        self.backend = MatrixBackend(
            ctypes.cast(self.callback, ctypes.c_void_p), None)

    def _call(self, _ctx, layer, kind, index, x, out, rows, cols) -> int:
        try:
            if (layer, kind, index) != (-1, MAT_UNEMBED, 0):
                raise FinalHeadEvidenceError(
                    f"unexpected backend request layer={layer} kind={kind} "
                    f"index={index}"
                )
            if tuple(self.matrix.shape) != (rows, cols):
                raise FinalHeadEvidenceError(
                    f"selected rows {tuple(self.matrix.shape)} != {(rows, cols)}"
                )
            vector = torch.tensor([float(x[i]) for i in range(cols)],
                                  dtype=torch.float32)
            values = native_bfloat16_linear(self.matrix, vector)
            if values.numel() != rows:
                raise FinalHeadEvidenceError(
                    f"backend produced {values.numel()} logits; expected {rows}"
                )
            for index_, value in enumerate(values.tolist()):
                out[index_] = float(value)
            self.calls.append({"rows": rows, "cols": cols,
                               "mode": "native-bf16-linear"})
            return 0
        except Exception as exc:  # surfaced through the C return code
            self.error = str(exc)
            return -1


def _import_official_norm() -> Any:
    try:
        from transformers.models.inkling.modeling_inkling import InklingRMSNorm
    except (ImportError, AttributeError) as exc:
        raise FinalHeadEvidenceError(
            "Transformers with official InklingRMSNorm support is required "
            "(tested API: transformers 5.14.x)"
        ) from exc
    return InklingRMSNorm


def official_head(
    text_config: Any,
    norm_weight: torch.Tensor,
    hidden_state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Run the official normalization, width division, and bounded lm_head."""
    InklingRMSNorm = _import_official_norm()
    hidden = int(text_config.hidden_size)
    if tuple(hidden_state.shape) != (hidden,):
        raise FinalHeadEvidenceError(
            f"hidden state has shape {tuple(hidden_state.shape)}; expected ({hidden},)"
        )
    if norm_weight.shape != (hidden,):
        raise FinalHeadEvidenceError(
            f"official final norm has shape {tuple(norm_weight.shape)}; "
            f"expected ({hidden},)"
        )
    width = float(text_config.logits_mup_width_multiplier)
    if not width > 0.0:
        raise FinalHeadEvidenceError(f"invalid logits width multiplier {width}")

    module = InklingRMSNorm(hidden, eps=float(text_config.rms_norm_eps))
    with torch.no_grad():
        module.weight.copy_(norm_weight.to(dtype))
    module = module.to(dtype).eval()
    with torch.no_grad():
        normalized = module(hidden_state.to(dtype).reshape(1, 1, hidden))
        scaled = normalized / width
        logits = torch.nn.functional.linear(
            scaled.reshape(1, hidden), matrix.to(dtype))
    return {
        "final_norm": normalized.reshape(-1).float().cpu().contiguous(),
        "final_norm_scaled": scaled.reshape(-1).float().cpu().contiguous(),
        "logits": logits.reshape(-1).float().cpu().contiguous(),
    }


def candidate_head(
    lib: ctypes.CDLL,
    cfg: dict[str, Any],
    norm_weight: torch.Tensor,
    hidden_state: torch.Tensor,
    rows: list[int],
    backend: UnembedBackend,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run the checked-in head under BF16_REFERENCE over the same selection."""
    config = _build_config(lib, cfg)
    hidden = int(cfg["hidden"])

    weights = ModelWeights()
    norm = (ctypes.c_float * hidden)(*norm_weight.float().tolist())
    weights.final_norm = ctypes.cast(norm, FP)
    state = (ctypes.c_float * hidden)(*hidden_state.float().tolist())
    normalized = (ctypes.c_float * hidden)()
    row_scratch = (ctypes.c_float * hidden)()
    logits = (ctypes.c_float * len(rows))()
    selection = (ctypes.c_int * len(rows))(*rows)

    # The trace is how the pre-division vector stays observable: the C
    # completes the width division in place.
    collector = TraceCollector()
    rc = lib.waste_inkling_final_head_profile(
        ctypes.byref(config), ctypes.byref(weights),
        ctypes.byref(backend.backend), ctypes.cast(state, FP),
        selection, len(rows), ctypes.cast(logits, FP),
        ctypes.c_size_t(len(rows)), ctypes.cast(normalized, FP),
        ctypes.cast(row_scratch, FP), BF16_REFERENCE,
        ctypes.byref(collector.c_trace),
    )
    if rc:
        detail = f": {backend.error}" if backend.error else ""
        raise FinalHeadEvidenceError(f"checked final head refused the call{detail}")
    if backend.error:
        raise FinalHeadEvidenceError(f"unembedding backend failed: {backend.error}")

    traced: dict[str, torch.Tensor] = {}
    for point in POINTS:
        values = collector.values.get(f"token.0.model.{point}")
        if values is None:
            raise FinalHeadEvidenceError(f"checked head emitted no {point} trace")
        traced[point] = torch.tensor(values, dtype=torch.float32)
    scaled = torch.tensor(list(normalized), dtype=torch.float32)
    if not torch.equal(scaled, traced["final_norm_scaled"]):
        raise FinalHeadEvidenceError(
            "traced scaled vector disagrees with the buffer the projection read"
        )
    if not torch.equal(torch.tensor(list(logits), dtype=torch.float32),
                       traced["logits"]):
        raise FinalHeadEvidenceError("traced logits disagree with the output buffer")
    return traced, {"backend_calls": list(backend.calls)}


def _fixture_tensor(fixture: Any, name: str, axis0: int | None = None) -> torch.Tensor:
    entry = fixture.entry(name, axis0)
    dtype = {"BF16": torch.bfloat16, "F16": torch.float16,
             "F32": torch.float32}.get(entry.dtype)
    if dtype is None:
        raise FinalHeadEvidenceError(f"unsupported fixture dtype {entry.dtype!r}")
    payload = bytearray(fixture.raw(entry.name, entry.axis0))
    return torch.frombuffer(payload, dtype=dtype).reshape(entry.shape).clone()


def selected_rows_matrix(fixture: Any, rows: list[int]) -> torch.Tensor:
    fixture.require_vocab_rows(rows)
    values = [_fixture_tensor(fixture, UNEMBED_NAME, row).reshape(-1)
              for row in rows]
    return torch.stack(values).contiguous()


def tensor_sha256(value: torch.Tensor, dtype: torch.dtype) -> str:
    raw = value.detach().to(dtype).cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture = load_fixture(args.fixture)
    config_json = verify_config_binding(fixture, args.model_config)
    text_config = build_transformers_text_config(config_json)
    cfg = json.loads(Path(args.c_config).read_text())

    hidden = int(cfg["hidden"])
    if hidden != int(text_config.hidden_size):
        raise FinalHeadEvidenceError(
            f"C config hidden {hidden} disagrees with release "
            f"{text_config.hidden_size}"
        )
    values = json.loads(Path(args.hidden_state).read_text())
    if (isinstance(values, list) and values
            and all(isinstance(item, list) for item in values)):
        # Accept an inputs.json-style matrix and take one named position.
        if not 0 <= args.position < len(values):
            raise FinalHeadEvidenceError(
                f"position {args.position} is outside the {len(values)} supplied rows"
            )
        values = values[args.position]
    if not (isinstance(values, list) and len(values) == hidden
            and all(isinstance(item, (int, float)) for item in values)):
        raise FinalHeadEvidenceError(
            f"hidden state must be {hidden} numbers"
        )
    hidden_state = torch.tensor(values, dtype=torch.float32)

    rows = (sorted(set(int(row) for row in args.vocab_rows.split(",") if row))
            if args.vocab_rows else list(fixture.vocab_rows))
    if not rows:
        raise FinalHeadEvidenceError("fixture carries no vocabulary rows")
    matrix = selected_rows_matrix(fixture, rows)
    norm_weight = _fixture_tensor(fixture, FINAL_NORM_NAME)

    reference = official_head(text_config, norm_weight, hidden_state, matrix)

    library, provenance = build_head_library(Path(args.workdir) / "libinkling_head.so")
    lib = ctypes.CDLL(str(library))
    configure_head_library(lib)
    backend = UnembedBackend(matrix)
    candidate, execution = candidate_head(
        lib, cfg, norm_weight, hidden_state, rows, backend)

    comparison: dict[str, Any] = {}
    first_boundary: str | None = None
    for point in POINTS:
        comparison[point] = metrics(reference[point], candidate[point])
        if first_boundary is None and comparison[point]["raw_exact_fraction"] < 1.0:
            first_boundary = point

    classification = ("checked_in_bf16_final_head_exact" if first_boundary is None
                      else f"checked_in_bf16_final_head_diverges_at_{first_boundary}")
    result = {
        "gate": "checked-bf16-final-head",
        "source": provenance,
        "fixture": {
            "root": str(args.fixture),
            "entries": len(fixture),
            "payload_bytes": fixture.payload_bytes,
            "layers": list(fixture.layers),
            "vocab_rows": list(fixture.vocab_rows),
            "config_sha256": fixture.source.get("config_sha256"),
            "index_sha256": fixture.source.get("index_sha256"),
            "revision": fixture.source.get("revision"),
        },
        "selection": {
            "vocab_rows": rows,
            "rows": len(rows),
            "hidden": hidden,
            "unembedding_sha256": tensor_sha256(matrix, torch.bfloat16),
            "final_norm_sha256": tensor_sha256(norm_weight, torch.bfloat16),
        },
        "hidden_state": {
            "origin": args.hidden_state_origin,
            "path": str(args.hidden_state),
            "position": args.position,
            "sha256": tensor_sha256(hidden_state, torch.bfloat16),
        },
        "width_multiplier": float(text_config.logits_mup_width_multiplier),
        "execution": execution,
        "comparison": comparison,
        "claims": {
            "final_norm_and_unembedding_completion": True,
            "final_model_logits": False,
            "public_step_promoted": False,
            "note": ("the hidden state is a supplied bounded vector, not the "
                     "output of a proven full decoder"),
        },
        "decision": {
            "classification": classification,
            "first_boundary": first_boundary,
        },
    }
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--c-config", required=True)
    ap.add_argument("--hidden-state", required=True,
                    help="JSON vector, or matrix plus --position")
    ap.add_argument("--hidden-state-origin", required=True,
                    help="where the vector came from; recorded in the report")
    ap.add_argument("--position", type=int, default=0)
    ap.add_argument("--vocab-rows", default="",
                    help="comma-separated rows; defaults to the whole fixture")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    try:
        result = run(args)
    except (FinalHeadEvidenceError, FixtureError, FixtureReferenceError,
            LayerParityError, OSError, ValueError) as exc:
        ap.error(str(exc))
        return 2
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")
    return 0 if result["decision"]["first_boundary"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
