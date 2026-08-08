# WASTE × Inkling-Small

**A 276B frontier model, streamed off disk, inside a laptop's RAM — and a
refusal to claim it works until it has been checked.**

This repository develops Inkling-Small support for
[`sqliteai/waste`](https://github.com/sqliteai/waste), the embeddable C11 MoE
engine that keeps routed experts on disk behind a bounded cache.

> [!IMPORTANT]
> **Public Inkling inference is disabled and returns `WASTE_E_UNSUPPORTED`.**
> The private conversion and runtime path exists and passes synthetic
> validation; the public loader refuses until official-weight parity,
> tokenizer parity, and measured resource gates are satisfied.
>
> That is intentional. "Unsupported" is a much better production incident than
> "successfully interpreted a 276B model as something else."

**No official weight has ever been executed by this code.** Every parity
number here comes from small synthetic BF16-rounded weights that validate the
binding and execution path, not the model. Read that sentence before any table
below.

---

## Why this model, on this engine

WASTE was built for Kimi K3 and demonstrated at ~0.5 tok/s on a 64 GB
workstation. Inkling-Small changes the arithmetic, and the reason is geometry
rather than tuning:

| | K3 | Inkling-Small | K3 / Inkling |
| --- | ---: | ---: | ---: |
| bytes read per decoded token | 17.01 GiB | **2.11 GiB** | **8.0×** |
| RAM for the cache resolver's first rung | 50.6 GiB | **7.1 GiB** | 7.1× |
| RAM for its maximum | 89.5 GiB | **11.9 GiB** | 7.5× |

WASTE's budget resolver steps in whole multiples of one token's working set,
because upstream measured that a demand-only cache below that hits *exactly
zero*. That quantum is 17 GiB on K3 and 2.11 GiB here. So **an 8 GiB machine
clears a full working set and a 16 GiB machine reaches the largest cache the
engine will ever ask for** — with the cache in its best regime rather than its
worst.

The geometry is exact: arithmetic over the released config and the WEXP record
layout, checked against a record `inkling_vq.py` actually writes.

## What a token costs — measured, projected, and labelled

**[docs/THROUGHPUT.md](docs/THROUGHPUT.md)** is the full account. Headlines:

- The expert cache is **measured**, not modelled — `inkling_cache_sim.c` links
  upstream's real `ecache.c` and runs the shipping LFRU. It reproduces
  upstream's published curve conservatively (mean −9.9pp) and independently
  reproduces its sharpest finding: **a cache below one token's 2.11 GiB
  working set hits exactly zero.** At the 16 GiB-laptop budget it measures
  **34.7%**.
- Throughput is a **projection**, and says so in every line it prints. It is
  calibrated against upstream's measured K3 decode and must reproduce it
  before being applied here — 0.61 tok/s against a measured 0.56–0.63.
- **The largest available win is a defect in this port, not a missing
  feature.** `inkling_wexp.c` expands every expert to **100.7 MB of F32**
  before multiplying it; upstream's `vq_matvec` never expands one. Measured
  **4.5–5.9× in isolation and 18.3× less DRAM traffic** — 60.0 GiB/token down
  to 3.28. Tempered by upstream's own VQ4P result, where a 3.88× isolation
  win became 1.09–1.18× in the engine for reasons they could not account for.

Two expectations of mine that the measurements overturned, corrected in place
rather than quietly edited: batching is worth *less* on Inkling than on K3, and
the step is compute-bound at the operating point rather than I/O-bound.

## Status — audited 2026-08-08

Re-verified from a clean clone on the date above, not quoted from an earlier
bundle:

| Check | Result |
| --- | --- |
| Upstream | rebased to `d9b919a` (0.6.6), 21 commits on from the previous pin |
| Patch generation | applied tree `62a9240f…`; the committed bundle `git am`s onto the pinned upstream and lands on it |
| `make check` | 33 passed, 0 failed, 13 skipped (server suite: 168 checks) |
| ASan + UBSan | 30 passed, 0 failed, 14 skipped |
| Fuzzing | 200 cases, 0 crashed, 0 hung |
| Strict compile | 12 translation units, `-Werror`, native + MinGW |
| Python suite | **275 tests, 0 failures** — run, not quoted |
| Inkling seam | plans; refuses to load; refuses a mislabelled container |
| Windows | 12 units cross-compile; both C tests pass under Wine |

62 new files plus **84 inserted lines across 5 upstream files**. Blast radius
on the public engine is zero because no public code path calls it yet.

Full audit: **[docs/STATE-OF-THE-PORT.md](docs/STATE-OF-THE-PORT.md)**.
Evidence: [`TEST-RESULTS.txt`](dist/waste-inkling-6931570/TEST-RESULTS.txt).

## Talking to it

`tools/inkling_serve.py` serves an OpenAI-compatible chat endpoint over the
staged runtime — and has been run end to end, not mocked: HTTP in, C forward
pass, sampled tokens, JSON out, SSE streaming included.

```sh
python3 tools/inkling_serve.py --stage STAGE_DIR --i-know-the-weights-are-synthetic
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

Every response carries `x-waste-provenance`, and today it reads
`weights=synthetic tokenizer=fallback` — so the text is meaningless, and the
server says so rather than letting you find out. It refuses to start against
an unattested stage without that deliberately tedious flag.

This drives the **converter-private** path so the wire format is reviewable
now. Upstream's `serve/` reaches the engine only through `waste_open`, so when
G6 lands, `serve/` becomes the chat server unchanged and this becomes
redundant. That is the intended outcome.

| Capability | Status |
| --- | --- |
| `waste_plan_memory()` / `waste plan` | **public** — geometry from a manifest |
| `/v1/chat/completions` over the staged runtime | **preview** — real, labelled, synthetic weights |
| `waste_open`, `waste run`, `waste chat`, serving | **refused** — `WASTE_E_UNSUPPORTED` |

Planning was promoted and inference was not because they are different kinds
of claim. A plan is arithmetic over declared dimensions: the worst a bug
produces is a wrong byte count the caller sees immediately, and the formula is
tested field by field. A forward pass is a claim about a model whose weights
this code has never read.

The planner fails closed. A Kimi container relabelled `inkling`, or an Inkling
config missing one field, is refused rather than defaulted into a plausible
floor — both are in the suite.

## What we found in upstream

Two things worth sending back, both found by measuring rather than reading:

- **`tools/diskbench.c` never bypassed the page cache on Linux** — found here
  independently, and **four days after upstream had already fixed it**
  (`def83ef`, reported by fab2s, LEARNED §49). The patch prepared here was
  withdrawn rather than sent: duplicate, and worse in two ways. See
  [THROUGHPUT.md §2](docs/THROUGHPUT.md), which keeps the account because both
  mistakes were reasoning rather than code.
- **`ecache.h` documents a zero budget as "disables caching (every access
  reads)"**, but `waste_ecache_get` has no slot to hand back and returns
  `NULL`, so at a zero budget a caller must read directly rather than call it.
  Still current in 0.6.6.

## Where the code is

**Read [`inkling/`](inkling/).** It is the source of truth: twelve C
translation units, nineteen tools, and the tests. The integration patch in
[`dist/waste-inkling-6931570/`](dist/waste-inkling-6931570/) is generated from
it and verified against a pinned tree hash — apply that, never edit it.

| Doc | Purpose |
| --- | --- |
| **[docs/CODEBASE-MAP.md](docs/CODEBASE-MAP.md)** | Every translation unit, tool, and test |
| [docs/THROUGHPUT.md](docs/THROUGHPUT.md) | What a token costs, what reaches an order of magnitude, and what would falsify it |
| [docs/WASTE-CONSTRAINTS.md](docs/WASTE-CONSTRAINTS.md) | What upstream blocks, what it gives free, and the promotion design |
| [docs/STATE-OF-THE-PORT.md](docs/STATE-OF-THE-PORT.md) | Verified state, measured sizes, named risks |
| [docs/ROADMAP-V19.md](docs/ROADMAP-V19.md) | G0–G6 path to running the open weights |

```
inkling/                     source of truth — C, tools, tests, design notes
integration/waste/           upstream pin, overlay diffs, generate + verify
dist/waste-inkling-6931570/  the generated, checksummed bundle to apply
docs/                        audit, map, roadmap, throughput
waste-inkling-patch-v16..18/ frozen provenance, kept for the audit trail
```

To change the port: edit `inkling/`, run `integration/waste/verify.sh`, and
regenerate. CI regenerates from source, and separately applies the committed
bundle and checks it lands on the same tree — so `dist/` cannot ship code
nobody reviewed.

## Apply

```sh
git clone https://github.com/sqliteai/waste.git
cd waste
git checkout d9b919a791148b571e643d0af666bf19b4d733ab
git am /path/to/dist/waste-inkling-6931570/patches/0001-*.patch
PATH=/usr/bin:/bin make check
```

Verify the patch first with `sha256sum -c SHA256SUMS`. The expected applied
tree is `62a9240fdd169aae8292f88fd225bf51843cd2f6`. Or build and check from
source in one command:

```sh
integration/waste/verify.sh /tmp/waste
```

## What is implemented

- exact recognition of the official `thinkingmachines/Inkling-Small` package;
- bounded/resumable checkpoint inspection, planning, and staging;
- Q8/Q4 trunk artifacts and WEXP/WCBK routed-expert artifacts;
- a private staged runtime with synthetic multi-token token-to-logits tests;
- **the C decoder layer matched against official `transformers` 5.14.1** at
  float32 epsilon — 2.4e-07 dense/local, 6.0e-08 sparse/global — covering
  attention with and without the sliding window, the relative bias, log
  scaling, all four short convolutions, MoE routing, shared experts, and both
  norms. The first genuinely independent oracle this port has had;
- named activation tracing on both the private C and official reference sides;
- case-insensitive fail-closed public recognition before Kimi memory planning,
  tensor binding, or loading;
- integration with the WASTE 0.6.6 build, cgroup budget path, test matrix,
  shared library, and server regression suite.

## Promotion path

1. BF16 layerwise and logits parity against official weights (**G1**).
2. **G2 — quantized tolerance.** Promoted: every geometry figure in this
   repository is downstream of VQ3R being an acceptable operating point for
   *this* model, which nobody has measured. If it says 4 bits, everything
   moves ~1.33×.
3. Conversion time, memory floor, and throughput measured end to end (G3).
4. Official tokenizer and chat-template parity (G4).
5. Public manifest extension and loader dispatch (G5, G6).

The finish line is deliberately boring: a normal WASTE container through
`waste plan`, `waste run`, `waste chat`, and the OpenAI-compatible server. No
Inkling-only executable, no mystery budget, no "works on my half-terabyte
checkpoint" deployment strategy.

## Foundation and provenance

`sqliteai/waste@d9b919a791148b571e643d0af666bf19b4d733ab` — WASTE 0.6.6,
public API version 1, container format version 0, cgroup-aware budgeting.

[`dist/waste-inkling-6931570/`](dist/waste-inkling-6931570/) is the current
apply target, generated from `inkling/`. `waste-inkling-patch-v16..v18/` are
frozen historical bundles, kept as evidence rather than as a patch-order
scavenger hunt.

## Licence

Apache-2.0, matching upstream and the released weights.
