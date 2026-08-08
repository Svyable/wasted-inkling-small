# State of the WASTE Inkling port — 2026-08-02

> [!NOTE]
> This is a dated foundation audit and its counts and “next enhancement” wording
> are intentionally preserved as observed on 2026-08-02. For the current
> post-#57 handoff, use [MVP-READINESS.md](MVP-READINESS.md); for the current
> numerical frontier, use [BF16-EVIDENCE.md](BF16-EVIDENCE.md).

An independent audit of what this repository actually contains, what was
re-verified today from a clean clone, and what stands between the current
artifact and running the released `thinkingmachines/Inkling-Small` open weights
through WASTE.

Everything below marked **verified** was executed in this environment on
2026-08-02. Everything marked **unverified** is a claim inherited from an
earlier bundle, or a gate that has no evidence yet. The distinction is the
whole point of the document.

## 1. Headline

The port is **healthy, current, and deliberately inert**.

- Zero rebase debt: upstream `sqliteai/waste` HEAD is *the exact commit* v18
  targets, so there is no drift to absorb before the next enhancement.
- The consolidated patch replays byte-exactly and the full suite is green.
- Nothing about the public engine is broken, because the port does not yet
  touch the public engine — 84 inserted lines across 5 upstream files, plus 62
  new files that no public code path calls.
- The remaining work is not "finish the C runtime." The C runtime executes
  tokens today. The remaining work is **evidence against real weights**, and
  the harness for producing that evidence has one asymmetry that makes it
  impractical to run (§5).

## 2. What was verified today

| Check | Command | Result |
| --- | --- | --- |
| Upstream drift | `git ls-remote https://github.com/sqliteai/waste HEAD` | `6931570` — **identical** to the v18 baseline pin |
| Patch integrity | `sha256sum -c SHA256SUMS` | OK (`8519fad9…6ad41`) |
| Replay | `git am 0018-waste-693157-…patch` onto `6931570` | applied clean, no conflicts, no fuzz |
| Applied tree | `git rev-parse HEAD^{tree}` | `ce6c5272e801c651cc6b71f869a1b0cd7167dab5` — matches `EXPECTED_APPLIED_TREE` |
| Full suite | `PATH=/usr/bin:/bin make check` | **29 passed, 0 failed, 13 skipped** |
| Server suite | (inside `make check`) | **168 checks passed** |
| Inkling seam | `tests/run.sh` | `PASS Inkling architecture seam and scalar runtime primitives` |
| Fail-closed guard | `tests/run.sh` | `PASS Inkling is recognized before Kimi planning or loading` |

Reproduced without adjustment, which means the v18 bundle's recorded evidence
is real and the CI workflow is not the only thing holding it up.

**Update, same day.** Everything inherited has since been re-run in this
environment against the generated tree, and the numbers below have moved as
work landed. Current state: `make check` **31 passed / 0 failed / 13 skipped**,
`make asan` 30/0/14, `make fuzz-asan FUZZ_RUNS=200` (200 cases, 137 rejected,
63 loaded, 0 crashed, 0 hung), a **12-unit** `-Werror` compile natively *and*
cross-compiled for Windows, and — for the first time in this repository — the
complete Inkling Python suite: **152 tests, 0 failures**. The v16 bundle's
99-test claim is confirmed; the rest are the fixture reader and the
layer-parity harness. `dist/waste-inkling-6931570/TEST-RESULTS.txt` is the
authoritative record and is regenerated with the bundle; the figures in §2's
table above are the *original* audit and are left as they were found.

## 3. What exists, by weight

Measured on the applied tree:

| Surface | Size | Status |
| --- | --- | --- |
| Inkling C translation units | 12 TUs including `src/arch.c` and `src/inkling_public.c` | compiled into `libwaste.a` and the shared library |
| Inkling headers | 812 lines | private; not exported through `waste.h` |
| Python converter + parity tooling | 14 files, 5,909 lines | runnable, torch-dependent for the heavy paths |
| Inkling tests (C + Python) | 23 files | 5 dependency-light in CI, 2 C binaries (one runs on Windows), the rest torch |
| Upstream files modified | 5 files, +84 / −7 lines | `Makefile`, `src/model.c`, `src/waste.c`, `tests/run.sh`, `tools/convert.py` |
| New files added | 62 | everything else |

The 84-line number is the important one. This port is 98% additive, which is
why it rebases across upstream releases with two mechanical conflicts, and why
its blast radius on the public engine is currently zero.

## 4. The two fail-closed seams

Both are one classification call against `waste_arch_classify()`
(`src/arch.c`), which lowercases and prefix-matches `"inkling"`:

- `src/waste.c:179` — `waste_plan_memory()` **dispatched** since the planning
  promotion: an Inkling manifest goes to `waste_inkling_plan_memory_json()`
  before a single Kimi dimension is read, so Kimi's formulas still never see
  it. The Inkling planner fails closed rather than defaulting.
- `src/model.c:976` — the loader refuses after the `format_version` check and
  *before* Kimi config parsing or tensor binding, so a plausible set of
  dimensions cannot select the wrong forward path.

This ordering is the correct design and should survive promotion: the guard
becomes a dispatch, not a deletion.

## 5. The gap that actually blocks open-weight execution

There are three real blockers, in order of how much they cost.

### 5.1 The parity harness is asymmetric (cheap to fix, blocks everything)

Patch 15/16 built a two-sided trace protocol, and it is good work: a
CRC-protected activation archive, named records at every significant point in
the layer and the model, a C
side (`tools/inkling_trace.py`) and an official side
(`tools/inkling_reference.py`), and a comparator with tolerance and
routing-index reporting.

But the two sides have wildly different hardware requirements:

- the **C side** runs from a converter-private stage and can be pointed at a
  handful of layers;
- the **official side** calls `AutoModelForCausalLM.from_pretrained(src, …)`
  (`tools/inkling_reference.py:110`), which materializes the complete 532 GB
  checkpoint before it can emit a single activation.

`tools/inkling_parity.py` already knows how to extract a **bounded fixture** —
selected layers, selected axis-0 expert slices, per-tensor and total byte
ceilings, hash-bound to `config.json` and the safetensors index. Nothing
consumes that fixture on the reference side.

So the first enhancement is not more runtime code. It is a fixture-backed
reference path, so BF16 layerwise parity becomes a laptop-scale experiment
instead of a datacenter reservation. This is the single highest-leverage item
in the repository and it is a few hundred lines of Python.

**Status:** `tools/inkling_fixture.py` now closes the consumer half — loading,
CRC verification, axis-0 expert slices, torch-free BF16/F16/F32 decode, and
module-relative state-dict keys, with 41 dependency-light tests. What remains
is the official-side module construction, which needs `transformers` with
Inkling support present and must be written on a machine that has it. See
[ROADMAP-V19.md](ROADMAP-V19.md) G0, including a correction: the layer-scoped
*partial whole-model index* originally sketched there cannot work, because
layer *N* depends on layers *0..N-1*. Layer-level parity is the right shape and
needs no C change.

### 5.2 There is no C-side path from the released checkpoint (expensive, deferred)

`src/inkling_stage_reader.c` reads exactly two converter-private formats:
`IKTN` staged tensors and `IKBF` BF16 expert bank records. `inkling_wexp.c`
reads final `WEXP`/`WCBK`. `inkling_qtensor.c` reads `Q8G`/`Q4G`.

None of them reads safetensors. Every route from the released weights to
executable bytes therefore runs through Python and torch, and the shortest one
still writes ~90 GiB of VQ3R expert banks plus a quantized trunk. That is the
intended design (WASTE converts once, runs many times), but it means
"run the open weights" currently means "budget a full conversion pass first."

A bounded C-side safetensors reader is *not* the right next step — it would
duplicate the converter for a one-time operation. The right step is to make the
Python conversion **measurable and resumable enough to trust**, which it
largely already is, and then measure it (§Roadmap gate G3).

### 5.3 Almost nothing is public yet (correctly)

Planning is public; nothing else is. No `waste_open` dispatch, no tokenizer
integration, no chat template, no serving — all downstream of parity and they
should stay there. The manifest needs no schema extension: the planner reads
only fields format v0 already defines.

The asymmetry is deliberate. A plan is arithmetic over declared geometry, so a
bug yields a wrong byte count the caller sees at once. A forward pass is a
claim about a model whose official weights this code has never read.

## 6. Risks worth naming

- **Evidence dilution.** Four of the bundle's headline numbers are synthetic —
  small BF16-rounded weights validating the *binding and execution path*, not
  model quality. The documents say so, but a reader skimming "29 passed" can
  miss it. Any future README should keep the synthetic/official distinction
  above the fold.
- ~~**The v16 bundle is load-bearing.**~~ **Resolved.** `inkling/` is now the
  source of truth and CI regenerates the patch from it; the v16-v18 bundles are
  frozen provenance. See [CODEBASE-MAP.md](CODEBASE-MAP.md) R1/R2.
- ~~**Patch-shaped review.**~~ **Resolved.** A contributor edits
  `inkling/src/*.c` and sends a diff of that file.
  *(Both are struck through rather than deleted. The original wording — a
  15,824-line patch is not reviewable, and review has to be possible before any
  community can engage with the work — is the reason the refactor happened.)*
- ~~**Windows is entirely unclaimed.**~~ **Partly closed.** All 12 Inkling
  translation units now cross-compile with MinGW under `-Werror`, the full
  `make CC=x86_64-w64-mingw32-gcc-posix` produces `waste.exe` and
  `libwaste.dll`, and `tests/test_inkling_io.c` executes the
  `ReadFile`/`OVERLAPPED` positional-read branch under Wine in CI — the first
  time that code has run anywhere. Still unclaimed: *native* Windows, offsets
  above 4 GiB, NTFS atomic replacement, Unicode paths, and cancel/resume
  during a multi-hundred-GB conversion.

## 7. Verdict

At the time of this audit, the foundation was sound and current; the ceremony
around it was heavier than the code it protected. That was fixed: the C became
readable as C, the patch became a build product, and the differential suite ran
instead of being quoted. Subsequent work did close bounded portions of the
official-weight parity loop; the live scope and next gate are recorded in the
handoff and evidence documents linked above.

- Path to open-weight execution: [ROADMAP-V19.md](ROADMAP-V19.md)
- Structure and refactor: [CODEBASE-MAP.md](CODEBASE-MAP.md)
