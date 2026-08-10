# Roadmap v19 — running the released Inkling-Small open weights

`thinkingmachines/Inkling-Small` shipped on 2026-08-01: Apache-2.0, 276B total
/ 12B active, 531,912,898,740 bytes across 32 BF16 safetensors shards, with the
official tokenizer, tiktoken ranks, special-token map, processor config, and
chat template. The port already recognizes that package exactly and refuses to
run it publicly.

This document is the path from "recognized" to "running," in the order the
gates actually unblock each other. It is written so that each gate has an
owner-free definition of done: a command, an artifact, and a number.

Current handoff: [MVP-READINESS.md](MVP-READINESS.md). Numerical evidence:
[BF16-EVIDENCE.md](BF16-EVIDENCE.md). The dated foundation audit remains in
[STATE-OF-THE-PORT.md](STATE-OF-THE-PORT.md), and repository structure is in
[CODEBASE-MAP.md](CODEBASE-MAP.md).

## The shape of the problem

The private C runtime executes tokens today, and the repository now has bounded
official-weight evidence in addition to synthetic binding tests. The strongest
result is eight-position stateful local-sparse layer-2 exactness on a named
Linux AVX2 reference profile. The remaining problem is no longer making parity
runnable; it is turning profile-bound evidence into a checked-in private BF16
runtime policy, expanding coverage through decoder/logit boundaries, and only
then considering public execution.

So the ordering is: make parity cheap (G0), prove BF16 correctness (G1),
prove quantized tolerance (G2), measure the conversion (G3), prove the text
interface (G4), and only then extend the public format and dispatch (G5, G6).

---

## G0 — Make parity runnable on one machine — **complete**

> **Completion update (2026-08-08).** The bounded remote extractor, official
> fixture-backed reference, C layer runner, deterministic inputs, exact router
> selection, source hashes, and CRC checks are all retained and have run on
> official checkpoint bytes. The historical design notes below explain how
> that foundation was reached.

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
- 🟡 **Official-side module construction — unblocked, and it found something.**
  Earlier notes said this needed "a machine that has transformers." That was
  asserted, not tested, and it was wrong: `pip install transformers` yields
  5.14.1 with the whole Inkling text stack, and `InklingDecoderLayer(config,
  layer_idx)` constructs from config alone with no checkpoint.

  Verified against the official source: the decoder-layer order matches our C
  exactly (norm → attn → attn_sconv → residual → norm → mlp → mlp_sconv →
  residual); `InklingAttention.scaling` is `1/head_dim`, not `1/sqrt(head_dim)`,
  matching our C; the short convolution adds its own input back inside the
  module, as ours does; and every official parameter name matches
  `layer_attention_names(..., "transformers_normalized")` in `inkling_plan.py`.

  Two differences to carry into the comparator: the official router takes
  `topk(..., sorted=False)`, so selected experts must be compared as
  (index, weight) **pairs** rather than positionally — our C returns them in
  descending choice order. And the routed-expert width is now an open question
  (see immediately below), which must be settled before a comparison means
  anything.
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

## ⚠ Blocking question found 2026-08-03: which key holds the routed intermediate?

`transformers` 5.14.1 ships full Inkling support (`InklingDecoderLayer`,
`InklingAttention`, `InklingMoE`, `InklingTopkRouter`, `InklingShortConvolution`).
That makes the official side of G0 buildable with no checkpoint — and the first
thing it surfaced is a config-naming collision this port cannot resolve on its
own.

**The official class and our recorded release config disagree about the routed
expert width.**

| | dense MLP | routed experts |
| --- | --- | --- |
| `InklingTextConfig` field | `intermediate_size` | `moe_intermediate_size` (default **3072**) |
| `tests/data/inkling-small-config.json` | `dense_intermediate_size` = 16384 | `intermediate_size` = **2048** |

Feeding our recorded JSON straight into the official class:

```
intermediate_size      -> 16384   (JSON said 2048)
moe_intermediate_size  -> 3072    (JSON has no such key)
```

The official class evidently accepts `dense_intermediate_size` and folds it
into `intermediate_size`, so **dense agrees at 16384**. The routed width does
not: our JSON's `intermediate_size: 2048` does not land on
`moe_intermediate_size`, which falls back to its 3072 default.

**Why this matters more than a name.** The routed intermediate is load-bearing
for numbers this repository already publishes:

- the VQ3R record size of 9,457,664 bytes is `hidden 4096 x inter 2048` at 3
  b/w plus scales;
- the ~90.2 GiB expert-bank total and the 108 MB minimum expert cache in the
  memory plan promoted to `waste_plan_memory()` derive from it;
- at 3072 instead of 2048 every one of those is 1.5x out.

**What is not in doubt.** `src/inkling_public.c` fails closed here rather than
guessing: fed a config using the official key names it would not find
`dense_intermediate_size`, so `req_int` fails and planning returns
`WASTE_E_FORMAT`. The discipline holds; the number is the open question.

**Do not "fix" this by renaming.** Two readings are consistent with the
evidence — our recorded JSON is missing a key the real release carries, or it
faithfully records a release whose key names differ from the library's and the
library applies a compatibility shim we have only half-observed. Choosing
between them by taste is precisely the plausible-but-unverified move this port
exists to refuse.

**Resolved by one artifact:** the real `config.json` from
`thinkingmachines/Inkling-Small`. A few kilobytes. Until it arrives, treat the
routed intermediate — and everything derived from it — as unconfirmed.

**Reproduce:**

```sh
python3 -c "
import json
from transformers.models.inkling.configuration_inkling import InklingTextConfig
b = json.load(open('inkling/tests/data/inkling-small-config.json'))['text_config']
c = InklingTextConfig(**b)
print(c.intermediate_size, c.moe_intermediate_size, b['intermediate_size'])
"
```

---

## G1 — BF16 layerwise and logits parity

**Current status.** Active, not complete. The retained investigation identifies
the required BF16 boundaries from RMSNorm through the sparse residual. PR #57
composes them through local sparse layer 2 for all eight source-bound positions
on the named Linux AVX2 profile, with exact routed/shared weights, `moe_out`,
`mlp_branch`, and `layer_out`. Production C is unchanged, the earlier
position-zero complete-layer result remains 4/5 across unchanged hosted runs,
and dense/global stateful coverage, decoder continuity, final normalization,
and logits remain open.

**The same-host backend matrix is complete.** Run
[31280408153](https://github.com/Svyable/wasted-inkling-small/actions/runs/31280408153)
cleared the AVX-512 gate and executed all four arms on one host:

| Arm | Exact | First pre-router divergence |
| --- | --- | --- |
| native (AVX512) | no | `q_proj` |
| `ATEN_CPU_CAPABILITY=avx2` | no | `q_proj` |
| `ONEDNN_MAX_CPU_ISA=AVX2` | **yes** | — |
| `torch.backends.mkldnn.enabled=False` | **yes** | — |

Classification `onednn_avx512_isa_path_is_profile_sensitive`, the first of the
branches predeclared before the measurement. Forcing ATen to AVX2 changed
nothing — same exactness, same stage, both layers — so **the ATen vectorized
kernels are excluded**. The seam is oneDNN's AVX-512 path and it is already
open at `q_proj`, the first linear operation in the layer; everything after it
is consequence rather than cause.

This also retires an earlier inference. "AMX is excluded by the
reproduced-failure profiles" was read off CPU flag sets on hosts that happened
not to advertise it. `ONEDNN_MAX_CPU_ISA` is a cap, not a request, so on such a
host the AMX path is never selected and never tested — untested is not
excluded.

**Next.**

1. **Bisect the ISA ladder.** One cap skips every AVX-512 rung at once, so the
   matrix cannot separate the FP32 AVX-512 kernels from the BF16 kernels from
   AMX. `classify_isa_ladder` walks `AVX2 → AVX512_CORE → AVX512_CORE_BF16 →
   AVX512_CORE_AMX → native` and names a rung only when exactness is a clean
   prefix of the ladder. Until it lands on an eligible host, the declared
   reference profile stays at the conservative AVX2 cap.
2. **Promote only the proven arithmetic policy** into a private fail-closed C
   profile.
3. **Rerun layers 0, 2, and 5** without temporary source rewriting before
   extending through logits.

**A parity job whose result depends on runner allocation is not evidence.** The
layer-2 stateful post-reduction check exited on 2026-08-08 with `layer 2
pre-expert regression invalidates MLP diagnosis` and passed on 2026-08-09 with
identical code; nothing changed but the host. Its gated run is now pinned to
the profile the matrix measured exact, and the uncapped native run executes
alongside it — recorded, never gated — so a green board cannot be mistaken for
a closed seam.

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

## G2 — Quantized tolerance, measured not asserted ⟵ **promoted 2026-08-08**

BF16 parity says the arithmetic is right. It says nothing about whether
VQ3R experts and a Q4 trunk still produce a usable model.

**Why it moved up.** [THROUGHPUT.md](THROUGHPUT.md) worked out what a decoded
token costs, and every figure in it — the 9,457,664 B record, the 2.11 GiB
per-token working set, the 90.2 GiB bank, and the budget ladder that puts
Inkling on an 8 GiB machine — is downstream of VQ3R being an acceptable
operating point for *this* model. Upstream chose 3 bits for K3 after Gate 3.
Inkling's routed experts are narrower and its routing sparser, and nobody has
measured them. If this gate says 4 bits, every published figure moves by
~1.33x. It is now the gate that decides whether the headline survives, which
is a better reason to run it than its position in a dependency graph.

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

**The API surface exists already, on the private path.** `tools/inkling_serve.py`
serves `/v1/chat/completions` — streaming and not — over
`waste_inkling_private_open/step`, and has been run end to end against a real
staged runtime: HTTP in, C forward pass, sampled tokens, OpenAI-shaped JSON
out. It is deliberately *not* a second engine. Upstream's `serve/` reaches the
model only through `waste_open`, `waste_tokenize` and the step API, so when
step 2 below lands, `serve/` becomes the chat server unchanged and
`inkling_serve.py` becomes redundant. It exists so the wire format is
reviewable now rather than after.

Every response states its weight and tokenizer provenance. The server labels
weights official only after the C runtime accepts its verified
Inkling-Small profile gate; otherwise it refuses to reopen the stage without
`--i-know-the-weights-are-synthetic`.

Running it for real found three defects the unit tests had not: an unenforced
context limit that surfaced as an opaque `step failed at position 16`, a
`--tokenizer` flag conflating tokenizer assets with the chat template, and a
`memoryview` over a ctypes float array that raises on first subscript — hidden
because the test stub returned a plain list. All three are fixed and
regression-tested, and the stub now returns the type the runtime does.


Turn the two guards into a branch, in this order:

1. ✅ `waste_plan_memory()` → `waste_inkling_plan_decode_memory()`, via
   `src/inkling_public.c`. Promoted ahead of the rest deliberately: a plan is
   geometry, so its failure mode is a wrong byte count rather than a wrong
   token. Everything below stays refused.
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
- **Native Windows.** The cross-compile *and* execution under Wine are now in
  CI, which is what closes the "never compiled" gap. Native Windows still
  waits on > 4 GiB offsets, NTFS atomic replacement, Unicode paths, and a
  cancel/resume during a multi-hundred-GB conversion.
- **A C-side safetensors reader.** Tempting, and wrong: it would duplicate the
  converter to save a one-time operation. Revisit only if G3 shows conversion
  is the dominant cost of adoption.

## Sequencing

```
G0  fixture-backed reference                         ✓
 │
 ├─ G1  BF16 private policy + decoder/logit parity ──┐
 ├─ G2  quantized tolerance ─────────────────────────┤
 └─ G3  conversion measurement                       ├─ G5 format ── G6 dispatch
                                                     │
    G4  tokenizer + chat template ───────────────────┘
```

G1 and G3 can run concurrently once G0 lands; G4 is independent of all of them
and is the best parallel track for a second contributor.

## Running record

| Gate | Status | Evidence |
| --- | --- | --- |
| G0 fixture parity | **done** — bounded official and C sides execute immutable CRC-verified fixtures | `inkling_fixture.py`, `inkling_fixture_reference.py`, `inkling_layer_parity.py`, [REMOTE-FIXTURES.md](REMOTE-FIXTURES.md) |
| G1 BF16 parity | **active** — layer-2 eight-position sparse ladder exact on named AVX2 profile; same-host matrix complete and localizes the seam to oneDNN's AVX-512 path at `q_proj`; ISA rung, production policy, remaining layer classes, decoder continuity, and logits open | [BF16-EVIDENCE.md](BF16-EVIDENCE.md), [INKLING-REFERENCE-PROFILES.md](INKLING-REFERENCE-PROFILES.md), #57, #59 |
| G2 quantized tolerance | **tooling promoted; official quality gate open** | [THROUGHPUT.md](THROUGHPUT.md) §5 |
| — decode cost model | **done** — geometry exact, throughput projected and labelled | [THROUGHPUT.md](THROUGHPUT.md), `inkling_throughput.py`, 38 tests |
| G3 conversion measurement | not started | — |
| G4 tokenizer / chat | not started | — |
| G5 format extension | not started | — |
| G6 public dispatch | **planning promoted**; open/step/serve still refused | `inkling_public.c`, 2 suite checks |

Keep this table honest. It is the only part of the document that will be read
twice.
