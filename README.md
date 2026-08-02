# WASTE Inkling-Small Patch Bundle

**Frontier model. Bounded RAM. SSD cardio. Production standards that refuse to
be impressed by adjectives.**

This repository develops Inkling-Small support for
[`sqliteai/waste`](https://github.com/sqliteai/waste). The current integration
artifact is [`waste-inkling-patch-v18/`](waste-inkling-patch-v18/), a single
replay-tested patch for WASTE 0.6.3.

> [!IMPORTANT]
> Public Inkling inference is still disabled. The private conversion/runtime
> path exists and passes synthetic validation, while the public loader returns
> `WASTE_E_UNSUPPORTED` until official-weight activation parity, tokenizer/chat
> parity, and measured resource gates are satisfied.

That is intentional. “Unsupported” is a much better production incident than
“successfully interpreted a 276B model as something else.”

**Synthetic, not official.** Every parity number below comes from small
BF16-rounded synthetic weights that validate the binding and execution path.
The official 532 GB checkpoint has never been executed through this code. Read
those two sentences before any table in this repository.

## Status — audited 2026-08-02

Re-verified from a clean clone on the date above, not quoted from an earlier
bundle:

| Check | Result |
| --- | --- |
| Upstream drift | none — `sqliteai/waste` HEAD is `6931570`, the exact v18 baseline |
| Patch integrity | `sha256sum -c SHA256SUMS` OK |
| Replay onto `6931570` | clean `git am`, tree `ce6c527…` matches the pin |
| `make check` | 29 passed, 0 failed, 13 skipped (server suite: 168 checks) |
| Inkling seam | recognized before Kimi planning or loading |

The port is current, green, and deliberately inert: 62 new files plus **84
inserted lines across 5 upstream files**. Its blast radius on the public engine
is zero because no public code path calls it yet.

Full audit, including what was *not* re-run:
**[docs/STATE-OF-THE-PORT.md](docs/STATE-OF-THE-PORT.md)**.

## Next enhancement — make parity runnable on one machine

The released weights are the blocker, and not for the reason you would expect.
The two-sided trace harness is built, but its official side calls
`AutoModelForCausalLM.from_pretrained()`, which materializes all 495 GiB before
emitting a single activation — while `inkling_parity.py` already extracts
bounded, hash-bound fixtures that nothing consumes.

So the next enhancement is a **fixture-backed reference mode** plus a
layer-scoped partial runtime stage, which turns BF16 layerwise parity from a
datacenter reservation into a laptop experiment. A few hundred lines of Python
that unblock every remaining gate.

Gates, commands, budgets, and the running record:
**[docs/ROADMAP-V19.md](docs/ROADMAP-V19.md)**.

## Where the code is

| Doc | Purpose |
| --- | --- |
| **[docs/CODEBASE-MAP.md](docs/CODEBASE-MAP.md)** | Every translation unit, tool, and test — and the refactor from patch bundles to a reviewable C tree |
| [docs/STATE-OF-THE-PORT.md](docs/STATE-OF-THE-PORT.md) | Verified state, measured sizes, named risks |
| [docs/ROADMAP-V19.md](docs/ROADMAP-V19.md) | G0-G6 path to running the open weights |

Note the structural caveat the map opens with: **the source of truth is the
patch**, not the readable snapshot under `waste-inkling-patch-v16/src`. Editing
that snapshot changes nothing CI validates. Fixing that is what the refactor is
for.

## Current foundation: v18

v18 targets `sqliteai/waste@69315701f634648f7a790915a0a525ed8aabf218`:

- WASTE 0.6.3
- public API version 1
- container format version 0
- cgroup-aware automatic budgeting through `waste_usable_ram()`

It replaces historical patches 1–17 with one consolidated patch. Do not apply
v17 first.

| Artifact | Purpose |
| --- | --- |
| [v18 README](waste-inkling-patch-v18/README.md) | Current comparison, apply instructions, evidence, and remaining gates |
| [v18 handoff](waste-inkling-patch-v18/PATCH18-HANDOFF.md) | The two rebase conflicts and their exact resolutions |
| [v18 validation](waste-inkling-patch-v18/TEST-RESULTS-P18.txt) | Model-free, sanitizer, fuzz, strict C, and Python results |
| [v18 baseline](waste-inkling-patch-v18/BASELINE) | Machine-readable upstream and tree provenance |
| [v18 checksum](waste-inkling-patch-v18/SHA256SUMS) | Patch integrity |
| [v18 CI](.github/workflows/validate-waste-inkling-v18.yml) | Fresh clone, checksum, replay, exact-tree, regression, sanitizer, and fuzz gates |

## What is implemented

- exact recognition of the official `thinkingmachines/Inkling-Small` package;
- bounded/resumable checkpoint inspection, planning, and staging;
- Q8/Q4 trunk artifacts and WEXP/WCBK routed-expert artifacts;
- a private staged runtime with synthetic multi-token token-to-logits tests;
- named activation tracing on both the private C and official reference sides;
- CRC-protected parity archives and tolerance-based comparison reports;
- case-insensitive fail-closed public recognition before Kimi memory planning,
  tensor binding, or loading;
- integration with the WASTE 0.6.3 build, cgroup budget path, test matrix, shared
  library, and server regression suite.

The private runtime is an independent parity oracle. It is not a second public
engine, because rediscovering WASTE's cache, budget, platform, and serving
lessons would be a very expensive way to become nostalgic.

## Apply v18

```sh
git clone https://github.com/sqliteai/waste.git
cd waste
git checkout 69315701f634648f7a790915a0a525ed8aabf218
git am /path/to/waste-inkling-patch-v18/patches/0018-waste-693157-consolidated-inkling-runtime.patch
PATH=/usr/bin:/bin make check
```

Verify the patch first:

```sh
cd /path/to/waste-inkling-patch-v18
sha256sum -c SHA256SUMS
```

The expected applied Git tree is
`ce6c5272e801c651cc6b71f869a1b0cd7167dab5`.

## Current evidence

- WASTE suite: **29 passed, 0 failed, 13 skipped** (re-run 2026-08-02)
- server suite: **168 checks passed** (re-run 2026-08-02)
- ASan + UBSan: **28 passed, 0 failed, 14 skipped**
- sanitizer fuzzing: **200 cases, 0 crashes, 0 hangs**
- strict C: **11 Inkling/architecture translation units**
- dependency-light Python: **17 tests passed**

The v16 bundle records 99 private-runtime Python tests from before the upstream
rebases; nothing in this repository currently re-runs them, which
[the map](docs/CODEBASE-MAP.md) proposes to fix with a deep CI tier. The
official 532 GB checkpoint was not available in this environment, so none of
the counts above is presented as official-weight parity.

## Promotion path

1. Establish BF16 layerwise and logits parity against official weights.
2. Measure Q8/Q4/VQ tolerances, conversion time, memory floor, and throughput.
3. Add official tokenizer and chat-template parity.
4. Design the public manifest extension and loader dispatch.
5. Exercise native Windows, cancellation/resume, concurrency, observability,
   release packaging, and an operator runbook.

The finish line remains deliberately boring: a normal WASTE container through
`waste plan`, `waste run`, `waste chat`, and the OpenAI-compatible server. No
Inkling-only executable, no mystery budget, no “works on my half-terabyte
checkpoint” deployment strategy.

## Provenance

- v18 is the current WASTE 0.6.3 apply target.
- [`waste-inkling-patch-v17/`](waste-inkling-patch-v17/) records the WASTE 0.6.2
  consolidation.
- [`waste-inkling-patch-v16/`](waste-inkling-patch-v16/) retains the historical
  private implementation, tools, tests, and design notes.

Historical bundles are evidence, not a patch-order scavenger hunt.
