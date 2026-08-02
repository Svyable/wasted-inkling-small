# State of the WASTE Inkling port — 2026-08-02

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

Not re-run here (inherited from `TEST-RESULTS-P18.txt`, previously green in
CI): `make asan` (28/0/14), `make fuzz-asan FUZZ_RUNS=200` (200 cases, 0
crashes), the 11-unit `-Werror` strict compile, and the 99 private-runtime
Python tests recorded by v16.

## 3. What exists, by weight

Measured on the applied tree:

| Surface | Size | Status |
| --- | --- | --- |
| Inkling C translation units | 10 units, 2,922 lines (`+ src/arch.c`, 11 TUs total) | compiled into `libwaste.a` and the shared library |
| Inkling headers | 812 lines | private; not exported through `waste.h` |
| Python converter + parity tooling | 14 files, 5,909 lines | runnable, torch-dependent for the heavy paths |
| Inkling tests (C + Python) | 21 files, 4,723 lines | 4 run dependency-light in CI; the rest need torch |
| Upstream files modified | 5 files, +84 / −7 lines | `Makefile`, `src/model.c`, `src/waste.c`, `tests/run.sh`, `tools/convert.py` |
| New files added | 62 | everything else |

The 84-line number is the important one. This port is 98% additive, which is
why it rebases across upstream releases with two mechanical conflicts, and why
its blast radius on the public engine is currently zero.

## 4. The two fail-closed seams

Both are one classification call against `waste_arch_classify()`
(`src/arch.c`), which lowercases and prefix-matches `"inkling"`:

- `src/waste.c:179` — `waste_plan_memory()` refuses an Inkling manifest with
  `WASTE_E_UNSUPPORTED` *before* reading a single architecture dimension, so
  Kimi's memory formulas can never produce a confident wrong floor.
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
reference mode, so BF16 layerwise parity becomes a laptop-scale experiment
instead of a datacenter reservation. This is the single highest-leverage item
in the repository and it is a few hundred lines of Python.

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

### 5.3 Nothing is public yet (correctly)

No manifest schema extension, no `waste_open` dispatch, no tokenizer
integration, no chat template, no serving. All four are downstream of parity
and should stay downstream of parity.

## 6. Risks worth naming

- **Evidence dilution.** Four of the bundle's headline numbers are synthetic —
  small BF16-rounded weights validating the *binding and execution path*, not
  model quality. The documents say so, but a reader skimming "29 passed" can
  miss it. Any future README should keep the synthetic/official distinction
  above the fold.
- **The v16 bundle is load-bearing.** It holds the only copy of the design
  docs, errata, and 99-test evidence, but the sources in it are a *snapshot*,
  not the source of truth — the patch is. Editing `waste-inkling-patch-v16/src`
  changes nothing that CI validates. This is the main structural defect that
  the refactor in [CODEBASE-MAP.md](CODEBASE-MAP.md) exists to remove.
- **Patch-shaped review.** A 15,824-line patch file is not reviewable by the
  open-source community, and "amaze the community" requires review to be
  possible. Same fix.
- **Windows is entirely unclaimed.** The reader has `ReadFile`/`OVERLAPPED`
  paths that have never been compiled by MinGW in CI, let alone run.

## 7. Verdict

The foundation is sound and current; the ceremony around it is heavier than
the code it protects. The next enhancement is not a new subsystem — it is
closing the parity loop on real weights, and restructuring the repository so
the community can read the C instead of a diff.

- Path to open-weight execution: [ROADMAP-V19.md](ROADMAP-V19.md)
- Structure and refactor: [CODEBASE-MAP.md](CODEBASE-MAP.md)
