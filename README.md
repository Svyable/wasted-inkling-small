# WASTE Inkling-Small Patch Bundle

**Frontier model. Bounded RAM. SSD cardio. Production standards that refuse to
be impressed by adjectives.**

This repository develops Inkling-Small support for
[`sqliteai/waste`](https://github.com/sqliteai/waste).

**Read the code in [`inkling/`](inkling/).** It is the source of truth: eleven
C translation units, fifteen Python tools, and the tests. The integration patch
in [`dist/waste-inkling-6931570/`](dist/waste-inkling-6931570/) is generated
from it by `integration/waste/generate.sh` and verified against a pinned tree
hash — apply that, but never edit it.

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
| Upstream drift | none — `sqliteai/waste` HEAD is `6931570`, the exact baseline |
| Patch generation | reproduces the reviewed v18 tree `ce6c527…` byte-for-byte, and is deterministic across runs |
| `make check` | 29 passed, 0 failed, 13 skipped (server suite: 168 checks) |
| ASan + UBSan | 28 passed, 0 failed, 14 skipped |
| Fuzzing | 200 cases, 0 crashed, 0 hung |
| Strict compile | 11 translation units, `-Werror` |
| Python suite | **140 tests, 0 failures** — run, not quoted |
| Inkling seam | recognized before Kimi planning or loading |

The port is current, green, and deliberately inert: 62 new files plus **84
inserted lines across 5 upstream files**. Its blast radius on the public engine
is zero because no public code path calls it yet.

Full audit: **[docs/STATE-OF-THE-PORT.md](docs/STATE-OF-THE-PORT.md)**.
Evidence: [`dist/waste-inkling-6931570/TEST-RESULTS.txt`](dist/waste-inkling-6931570/TEST-RESULTS.txt).

## Next enhancement — make parity runnable on one machine

The released weights are the blocker, and not for the reason you would expect.
The two-sided trace harness is built, but its official side calls
`AutoModelForCausalLM.from_pretrained()`, which materializes all 495 GiB before
emitting a single activation — while `inkling_parity.py` already extracts
bounded, hash-bound fixtures that nothing consumed.

`inkling/tools/inkling_fixture.py` is now that consumer: dependency-free
loading and CRC verification, axis-0 expert slices, BF16/F16/F32 → F32 decode
checked bit-for-bit against torch, module-relative state-dict keys, and
fail-closed coverage checks that name the missing layer or expert. 41 tests, no
torch, in CI.

What remains for the gate is the official-side module construction, which needs
`transformers` with Inkling support present.

Gates, commands, budgets, and the running record:
**[docs/ROADMAP-V19.md](docs/ROADMAP-V19.md)**.

## Where the code is

| Doc | Purpose |
| --- | --- |
| **[docs/CODEBASE-MAP.md](docs/CODEBASE-MAP.md)** | Every translation unit, tool, and test — and the refactor from patch bundles to a reviewable C tree |
| [docs/STATE-OF-THE-PORT.md](docs/STATE-OF-THE-PORT.md) | Verified state, measured sizes, named risks |
| [docs/ROADMAP-V19.md](docs/ROADMAP-V19.md) | G0-G6 path to running the open weights |

## Layout

```
inkling/                     source of truth — C, tools, tests, design notes
integration/waste/           upstream pin, five overlay diffs, generate + verify
dist/waste-inkling-6931570/  the generated, checksummed bundle to apply
docs/                        audit, map, roadmap
waste-inkling-patch-v16..18/ frozen provenance, kept for the audit trail
```

To change the port: edit `inkling/`, run `integration/waste/verify.sh`, and
regenerate the bundle. CI regenerates it independently and fails if the
committed one is stale, so `dist/` cannot ship code nobody reviewed.

## Current foundation

`sqliteai/waste@69315701f634648f7a790915a0a525ed8aabf218`:

- WASTE 0.6.3
- public API version 1
- container format version 0
- cgroup-aware automatic budgeting through `waste_usable_ram()`

| Artifact | Purpose |
| --- | --- |
| [bundle README](dist/waste-inkling-6931570/README.md) | Apply instructions and what changed |
| [bundle validation](dist/waste-inkling-6931570/TEST-RESULTS.txt) | Generation, tree, sanitizer, fuzz, strict C, and 140 Python tests |
| [bundle baseline](dist/waste-inkling-6931570/BASELINE) | Machine-readable upstream and tree provenance |
| [bundle checksum](dist/waste-inkling-6931570/SHA256SUMS) | Patch integrity, over deterministic bytes |
| [CI](.github/workflows/validate-waste-inkling.yml) | Generate, tree hash, strict compile, regression, stale-bundle check, torch differential suite, sanitizers, fuzzing |

The historical bundles remain applicable as-is; `waste-inkling-patch-v18`
replaced patches 1-17 and is superseded by the generated bundle above.

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

## Apply

```sh
git clone https://github.com/sqliteai/waste.git
cd waste
git checkout 69315701f634648f7a790915a0a525ed8aabf218
git am /path/to/dist/waste-inkling-6931570/patches/0001-Add-the-Inkling-Small-runtime-foundation-to-WASTE.patch
PATH=/usr/bin:/bin make check
```

Verify the patch first:

```sh
cd /path/to/dist/waste-inkling-6931570
sha256sum -c SHA256SUMS
```

The expected applied Git tree is
`e372f1ef2b92c4bcc94f5c2474d6597d068f5c84`.

Or build and check it from source in one command:

```sh
integration/waste/verify.sh /tmp/waste
```

## Current evidence

All measured on 2026-08-02 in one environment, on the generated tree:

- WASTE suite: **29 passed, 0 failed, 13 skipped**
- server suite: **168 checks passed**
- ASan + UBSan: **28 passed, 0 failed, 14 skipped**
- sanitizer fuzzing: **200 cases, 137 rejected, 63 loaded, 0 crashes, 0 hangs**
- strict C: **11 Inkling/architecture translation units**, `-Werror`
- Python: **140 tests passed** (122 + 18), torch differential suite included
- patch generation: reproduces the reviewed v18 tree byte-for-byte, and is
  deterministic across independent runs

The v16 bundle recorded 99 Python tests and nothing re-ran them. They now run
in CI, and they pass. The official 532 GB checkpoint was not available in this
environment, so none of the counts above is official-weight parity.

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

- [`dist/waste-inkling-6931570/`](dist/waste-inkling-6931570/) is the current
  apply target, generated from `inkling/`.
- [`waste-inkling-patch-v18/`](waste-inkling-patch-v18/) records the hand-authored
  WASTE 0.6.3 consolidation the generator was proved against.
- [`waste-inkling-patch-v17/`](waste-inkling-patch-v17/) records the WASTE 0.6.2
  consolidation.
- [`waste-inkling-patch-v16/`](waste-inkling-patch-v16/) retains the historical
  private implementation, tools, tests, and design notes.

Historical bundles are evidence, not a patch-order scavenger hunt.
