# Roadmap v19 — running the released Inkling-Small open weights

`thinkingmachines/Inkling-Small` shipped on 2026-08-01: Apache-2.0, 276B total
/ 12B active, 531,912,898,740 bytes across 32 BF16 safetensors shards, with the
official tokenizer, tiktoken ranks, special-token map, processor config, and
chat template. The port already recognizes that package exactly and refuses to
run it publicly.

This document is the path from "recognized" to "running," in the order the
gates actually unblock each other. It is written so that each gate has an
owner-free definition of done: a command, an artifact, and a number.

Current state and the evidence behind it: [STATE-OF-THE-PORT.md](STATE-OF-THE-PORT.md).
Structure: [CODEBASE-MAP.md](CODEBASE-MAP.md).

## The shape of the problem

The private C runtime executes tokens today. What it has never done is execute
*these* tokens — every parity result in the repository comes from small
synthetic BF16-rounded weights that validate the binding and execution path,
not the model. Closing that is not a coding problem, it is an evidence problem,
and the harness that produces the evidence has one asymmetry that currently
makes it unrunnable outside a very large machine.

So the ordering is: make parity cheap (G0), prove BF16 correctness (G1),
prove quantized tolerance (G2), measure the conversion (G3), prove the text
interface (G4), and only then extend the public format and dispatch (G5, G6).

---

## G0 — Make parity runnable on one machine ⟵ **the next enhancement**

**Problem.** `tools/inkling_reference.py:110` calls
`AutoModelForCausalLM.from_pretrained(src, …)`. That materializes all 495 GiB
before it can emit a single activation, so the official side of the trace
protocol needs a machine that the C side does not. Meanwhile
`tools/inkling_parity.py` already extracts a **bounded fixture** — selected
layers, explicitly selected axis-0 routed-expert slices, per-tensor and total
byte ceilings, `fixture.json` published last and bound to SHA-256 hashes of
`config.json` and the safetensors index — and nothing consumes it.

### Correction to this gate's first design

The original sketch called for a **layer-scoped partial `runtime-stage.bin`**,
with a header flag letting `inkling_private.c` accept a subset of layers for
tracing. That does not survive contact with the code: layer *N*'s activations
depend on layers *0..N-1*, so a whole-model index holding three of 42 layers
cannot execute anything, flag or no flag. There is nothing to trace.

Layer-level parity is the correct shape, and it needs **no index change and no
C change at all**. Feed the *same* input hidden state to layer *N* on both
sides and compare outputs — which is exactly what `tests/test_inkling_layer_c.py`
already does against PyTorch, only with synthetic weights instead of official
ones. The fixture supplies the weights; the harness supplies the input.

### Status

- ✅ **`tools/inkling_fixture.py`** — the consumer that was missing. Loads and
  CRC-verifies a fixture, addresses expert slices by axis-0 index, decodes
  BF16/F16/F32 to F32 without torch, maps a layer's entries to
  module-relative state-dict keys, and refuses — by name — any layer or expert
  the fixture does not cover. 41 tests, dependency-light, in CI. The BF16 and
  F16 widening is checked bit-for-bit against torch where torch is available.
- ⬜ **Official-side module construction.** Instantiate the official decoder
  layer for one fixture layer, load `layer_state_dict_entries()` into it, and
  drive the existing `register_reference_hooks()`. This is the one piece that
  needs `transformers` with Inkling support present, so it must be written and
  first run on a machine that has it — writing it blind would produce exactly
  the plausible-but-unverified code this port exists to avoid.
- ✅ **`tools/inkling_layer_parity.py`** — the C-side layer harness. Binds one
  layer's weights from a fixture (resolving names through `inkling_plan.py`,
  splitting the provider's row-interleaved gate/up, placing routed expert
  slices at their own indices), runs
  `waste_inkling_layer_step_backend_trace()` over supplied input hidden
  states, and writes the same archive names `inkling_trace.py` uses. Proved by
  running identical weights through this binding and a direct one and
  requiring bit-identical output, plus a probe that compares every ctypes
  struct size against the C compiler's.

```sh
# once, on a machine with the checkpoint mounted (streaming read, ~8 GiB written)
python tools/inkling_parity.py --src /models/Inkling-Small \
  --out /parity/fixture-L0-L2-L5 --layers 0,2,5 \
  --experts "2:4,17,39,88,143,221;5:1,8,22,64,150,201" --max-total-gib 8

# check what you got, on any machine, with no dependencies
python tools/inkling_fixture.py --fixture /parity/fixture-L0-L2-L5 --verify

# then, on a laptop
python tools/inkling_reference.py --fixture /parity/fixture-L0-L2-L5 \
  --layers 0,2,5 --input /parity/inputs.bin --out /parity/python
python tools/inkling_trace.py --fixture /parity/fixture-L0-L2-L5 \
  --layers 0,2,5 --input /parity/inputs.bin --out /parity/waste
python tools/inkling_parity.py --compare-reference /parity/python \
  --compare-candidate /parity/waste --atol 1e-5 --rtol 1e-5 \
  --report /parity/report.json
```

**Done when.** Both archives for layers 0, 2, and 5 are produced on a machine
with < 64 GB RAM and a comparison report exists, whatever it says.

**Cost.** One piece remains — the official-side module construction — and it is
still no C change. This is the highest-leverage item in the
repository: every gate below is blocked on it, and none of them is blocked on
anything else.

---

## G1 — BF16 layerwise and logits parity

**Do.** Run G0's loop and drive the differences to zero, in this order —
earlier points are cheaper to debug and later ones inherit their errors:

1. `embedding_norm`
2. per-layer `q/k/v/r_proj`, then Q/K RMSNorm
3. K and V short convolution (the ring is the classic off-by-one)
4. attention scores, relative bias, `log_tau` scaling
5. attention output and its branch convolution
6. router: **exact int32 index match first**, then normalized weights
7. routed + shared expert contributions, joint normalization
8. per-layer output
9. `final_norm`, `final_norm_scaled`, `logits`

**Watch for.** BF16 semantics, not algebra. Accumulation order, where the
official implementation rounds to BF16 versus keeps FP32, and
`shared_expert_sink` / `norm_after_topk` interacting with the joint
normalization in `waste_inkling_route()`. Expect the first mismatch to be in
routing, because a single index disagreement changes everything downstream and
tolerance-based comparison will not tell you it was an index.

**Done when.** For 8 tokens across a layer set that covers all three shapes —
`0` (dense, since `dense_mlp_idx=2` makes layers 0-1 dense), `2` (sparse,
local), `5` (sparse, global; the globals are 5, 11, 17, 23, 29, 35, 41) — max
abs error ≤ 1e-3 on every activation, routing indices exactly equal, and the
report committed as evidence.

**Then repeat once at full scale** on a machine that can hold the checkpoint,
for all 42 layers and ≥ 32 tokens. G0 makes iteration cheap; it does not
replace the whole-model run.

---

## G2 — Quantized tolerance, measured not asserted

BF16 parity says the arithmetic is right. It says nothing about whether
VQ3R experts and a Q4 trunk still produce a usable model.

**Do.**
- Convert a *sample*: 4 layers' experts at VQ3R and VQ2R, trunk at Q8 and Q4.
- Report per-tensor and per-layer reconstruction error against BF16, and
  logits KL/top-1 agreement over a few hundred tokens.
- Compare against upstream's own operating point — WASTE picked VQ3R for K3
  experts and Q4G for the trunk after `docs/GATES.md` Gate 3, so the question
  is whether Inkling's routed experts behave like Kimi's, not whether 3 bits
  works in general.

**Done when.** A published table of tolerances per format, and an explicit
choice of default (expect VQ3R experts + Q4G trunk + Q8G vocabulary/router,
matching the existing `--trunk-bits` policy) with the measurement behind it.

---

## G3 — Measure the conversion, end to end, once

**Budget, from exact record geometry** (not measurements):

| Item | Size |
| --- | --- |
| Source checkpoint | 531,912,898,740 B ≈ 495.4 GiB |
| One VQ3R routed-expert record | 9,457,664 B |
| One 256-expert layer | 2,421,161,984 B |
| All 40 sparse layers | 96,846,479,360 B ≈ 90.2 GiB |
| Trunk, Q4 bulk + Q8 vocab/router | ≈ 3.642 GiB |
| Trunk, all-Q8 | ≈ 5.629 GiB |
| Canonical resident F32 (norms, conv kernels, biases) | ≈ 9.52 MiB |
| 4K-context K/V + four conv states | ≈ 0.362 GiB |
| One shared WEXP decode workspace | ≈ 105 MiB |
| Per-layer VQ codebooks | ≈ 1.4 MiB |
| **Model-state floor** | **≈ 4.1 GiB** before allocator overhead |

Peak disk during conversion is source + output ≈ 589 GiB. The BF16 expert stage
(~480 GiB) is *not* on this path — use `--quantize-experts` and
`--publish-runtime-vq-stage`, never `--stage-experts`, outside debugging.

**Do.** Run the full pipeline once, wall-clock each phase, then deliberately
kill and resume each phase to prove the sidecars work at scale. Record: elapsed
per phase, peak RSS, bytes written, `libwastevq` on versus off, and
tokens/second for a 2K-context decode on the resulting stage.

**Done when.** `TEST-RESULTS-V19.txt` carries measured numbers with the
hardware named, and a cancel/resume was survived at least once per phase.

---

## G4 — Tokenizer and chat template

The release ships the assets, so this is parity work, not design work.

**Do.** Load the official tiktoken ranks through WASTE's tokenizer; verify
every special token and placeholder id round-trips; render the official chat
template byte-exactly against a Python reference over a corpus of
multi-turn conversations; preserve an explicit raw mode when no template is
selected. Upstream already has `tools/tokdiff.py` and the
"prompt text cannot forge control tokens" check — reuse both rather than
inventing an Inkling-specific harness.

**Done when.** `tokdiff` is clean over the corpus, the control-token forgery
check passes with the Inkling specials, and template rendering matches the
reference exactly.

---

## G5 — Public format extension

Only after G1-G4. The design is already implied by what
`runtime-stage.bin` v3 records, and needs **no new top-level manifest shape**:

- `arch: "inkling"` — already classified;
- `config` — the release config verbatim, multimodal wrapper under `_outer`,
  exactly the K3 convention, so `local_layer_ids`, `dense_mlp_idx`, `d_rel`,
  `rel_extent`, `sconv_kernel_size`, `log_scaling_n_floor`,
  `log_scaling_alpha`, `logits_mup_width_multiplier`, `shared_expert_sink`,
  and `norm_after_topk` arrive without invention;
- `trunk[]` — additional named tensors for the four per-layer short-convolution
  kernels and the relative-position projection banks;
- `layers[].codebook_base`, `expert_quant{…}` — already present and already
  what the private index stores.

**Done when.** A converted Inkling container passes `verify_container.py`, and
the format extension is documented in `docs/FORMAT.md` alongside K3's.

---

## G6 — Public dispatch

Turn the two guards into a branch, in this order:

1. `waste_plan_memory()` → `waste_inkling_plan_decode_memory()` (already tested
   standalone, already field-by-field);
2. loader → Inkling config build + tensor binding via
   `waste_inkling_bind_weights_ex_backend()`;
3. expert cache → the Inkling expert callback, reusing WASTE's `ecache` and its
   optimized VQ kernels rather than the scalar `inkling_wexp.c` path;
4. step / reset / statistics / errors;
5. `manifest.json` published last;
6. `WASTE_E_UNSUPPORTED` **retained** for every variant that has not earned
   promotion — Inkling-Small is not "Inkling."

**Done when.** `waste plan`, `waste run`, `waste chat`, and the
OpenAI-compatible server work against an Inkling container with no
Inkling-specific executable, and the suite is green with the container present.

---

## Deliberately out of scope for v19

- **Multimodal.** The audio (`dmel`, 80 mel bins) and vision (`hmlp`, patch 40)
  configs are recognized and planned for; execution is a separate program of
  work behind text parity.
- **MTP.** `num_nextn_predict_layers: 8` is a speculative-decoding opportunity,
  not a correctness requirement.
- **Native Windows.** The reader's `ReadFile`/`OVERLAPPED` paths have never
  been compiled by MinGW in CI. Add the cross-compile to CI during v19; run
  WSL2 first and claim native Windows only after > 4 GiB offsets, NTFS atomic
  replacement, Unicode paths, and a cancel/resume during a multi-hundred-GB
  conversion have all been exercised.
- **A C-side safetensors reader.** Tempting, and wrong: it would duplicate the
  converter to save a one-time operation. Revisit only if G3 shows conversion
  is the dominant cost of adoption.

## Sequencing

```
G0  fixture-backed reference + partial stage      ← start here, days
 │
 ├─ G1  BF16 parity ──── G2  quantized tolerance ──┐
 │                                                  ├─ G5 format ── G6 dispatch
 └─ G3  conversion measurement                      │
                                                    │
    G4  tokenizer + chat template ──────────────────┘
```

G1 and G3 can run concurrently once G0 lands; G4 is independent of all of them
and is the best parallel track for a second contributor.

## Running record

| Gate | Status | Evidence |
| --- | --- | --- |
| G0 fixture parity | **in progress** — reader and C side landed; official side remains | `inkling_fixture.py`, `inkling_layer_parity.py`, 53 tests |
| G1 BF16 parity | not started | — |
| G2 quantized tolerance | not started | — |
| G3 conversion measurement | not started | — |
| G4 tokenizer / chat | not started | — |
| G5 format extension | not started | — |
| G6 public dispatch | not started | — |

Keep this table honest. It is the only part of the document that will be read
twice.
