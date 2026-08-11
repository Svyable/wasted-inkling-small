#!/usr/bin/env python3
"""Offline byte-exact regression for the bounded real Inkling final-head primitive.

This intentionally proves only final RMSNorm, MuP width scaling, and selected
semantic unembedding rows for one already-proven real layer-5 hidden row.  It
makes no claim about the true final layer-41 hidden state or full-model logits.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/inkling/final_head_primitive_real"
PROVENANCE = FIXTURE / "provenance.json"

MODEL_ID = "thinkingmachines/Inkling-Small"
REVISION = "21152b5312c653be115f33a8342759064144e281"
HIDDEN_SIZE = 4096
SEMANTIC_VOCAB = 200058
STORAGE_VOCAB = 201024
EXPECTED_SOURCE_ROW_SHA256 = (
    "b6a560aa44b26561ee0b08dd0b438fa06ef2c29685b5e28e96ab1e6bd7aef88a"
)
EXPECTED_LOGITS_SHA256 = (
    "b30e62a788b9efa30d5673072af07be558a47976fed913fb71f34cd5f929790f"
)
EXPECTED_ROWS = tuple(range(8)) + tuple(range(100025, 100033)) + tuple(
    range(200050, 200058)
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_checked(name: str, provenance: dict) -> bytes:
    record = provenance["files"][name]
    payload = (FIXTURE / name).read_bytes()
    assert len(payload) == int(record["bytes"]), (name, len(payload), record["bytes"])
    assert sha256(payload) == record["sha256"], (name, sha256(payload), record["sha256"])
    return payload


def bf16(raw: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    values = torch.frombuffer(bytearray(raw), dtype=torch.uint16).clone()
    return values.view(torch.bfloat16).reshape(shape)


def raw_bf16(tensor: torch.Tensor) -> bytes:
    assert tensor.dtype == torch.bfloat16
    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .astype("<u2", copy=False)
        .tobytes()
    )


def require_exact(name: str, actual: torch.Tensor, expected: bytes) -> None:
    actual_raw = raw_bf16(actual)
    assert actual_raw == expected, {
        "stage": name,
        "actual_sha256": sha256(actual_raw),
        "expected_sha256": sha256(expected),
    }


def main() -> None:
    provenance = json.loads(PROVENANCE.read_text())
    assert provenance["format"] == "inkling-final-head-primitive-real"
    assert provenance["version"] == 2
    assert provenance["model_id"] == MODEL_ID
    assert provenance["revision"] == REVISION
    assert provenance["source_hidden"]["row_sha256"] == EXPECTED_SOURCE_ROW_SHA256
    assert provenance["source_hidden"]["position"] == 7
    assert "not the final model hidden state" in provenance["source_hidden"]["claim"]

    profile = provenance["profile"]
    assert profile["module_dtype"] == "BF16"
    eps = float(profile["rms_eps"])
    width = float(profile["logits_mup_width_multiplier"])
    assert eps == 1e-6
    assert width == 16.0

    unembed = provenance["source_tensors"]["unembed"]
    assert unembed["dtype"] == "BF16"
    assert unembed["storage_shape"] == [STORAGE_VOCAB, HIDDEN_SIZE]
    assert unembed["storage_vocab_size"] == STORAGE_VOCAB
    assert unembed["semantic_vocab_size"] == SEMANTIC_VOCAB
    assert STORAGE_VOCAB >= SEMANTIC_VOCAB
    assert tuple(unembed["selected_rows"]) == EXPECTED_ROWS
    assert min(EXPECTED_ROWS) >= 0 and max(EXPECTED_ROWS) < SEMANTIC_VOCAB

    norm_meta = provenance["source_tensors"]["final_norm"]
    assert norm_meta["dtype"] == "BF16"
    assert norm_meta["shape"] == [HIDDEN_SIZE]

    hidden_raw = read_checked("input-layer5-pos7.bf16.bin", provenance)
    norm_raw = read_checked("final-norm-weight.bf16.bin", provenance)
    rows_raw = read_checked("selected-unembed-rows.bf16.bin", provenance)
    row_ids_raw = read_checked("selected-row-ids.u32.bin", provenance)
    expected_norm = read_checked("expected-final-norm.bf16.bin", provenance)
    expected_scaled = read_checked("expected-final-norm-scaled.bf16.bin", provenance)
    expected_logits = read_checked("expected-selected-logits.bf16.bin", provenance)

    assert sha256(hidden_raw) == EXPECTED_SOURCE_ROW_SHA256
    assert sha256(expected_logits) == EXPECTED_LOGITS_SHA256
    assert len(hidden_raw) == HIDDEN_SIZE * 2
    assert len(norm_raw) == HIDDEN_SIZE * 2
    assert len(rows_raw) == len(EXPECTED_ROWS) * HIDDEN_SIZE * 2
    assert len(row_ids_raw) == len(EXPECTED_ROWS) * 4
    assert len(expected_norm) == HIDDEN_SIZE * 2
    assert len(expected_scaled) == HIDDEN_SIZE * 2
    assert len(expected_logits) == len(EXPECTED_ROWS) * 2

    decoded_rows = struct.unpack(f"<{len(EXPECTED_ROWS)}I", row_ids_raw)
    assert decoded_rows == EXPECTED_ROWS

    hidden = bf16(hidden_raw, (1, HIDDEN_SIZE))
    norm_weight = bf16(norm_raw, (HIDDEN_SIZE,))
    selected_rows = bf16(rows_raw, (len(EXPECTED_ROWS), HIDDEN_SIZE))

    # Independent replay of the official Inkling RMSNorm semantics: reduce in
    # F32, cast the normalized activation back to the input dtype, then apply
    # the checkpoint BF16 weight.  No Transformers code or network is used.
    with torch.no_grad():
        hidden_f32 = hidden.to(torch.float32)
        variance = hidden_f32.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_f32 * torch.rsqrt(variance + eps)
        final_norm = norm_weight * normalized.to(hidden.dtype)
        scaled = final_norm / width
        logits = F.linear(scaled, selected_rows)

    assert final_norm.dtype == torch.bfloat16
    assert scaled.dtype == torch.bfloat16
    assert logits.dtype == torch.bfloat16
    require_exact("final_norm", final_norm, expected_norm)
    require_exact("final_norm_scaled", scaled, expected_scaled)
    require_exact("selected_logits", logits, expected_logits)

    print(
        "INKLING_FINAL_HEAD_PRIMITIVE_OK "
        + json.dumps(
            {
                "input": "real-proven-layer5-pos7",
                "rows": len(EXPECTED_ROWS),
                "semantic_vocab": SEMANTIC_VOCAB,
                "storage_vocab": STORAGE_VOCAB,
                "logits_sha256": sha256(raw_bf16(logits)),
                "claim": "bounded-final-head-primitive-only",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
