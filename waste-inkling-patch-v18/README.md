# Inkling-Small on WASTE 0.6.3 — v18

v18 keeps the Inkling foundation on the current production-shaped WASTE tree.
It rebases the consolidated private runtime and fail-closed public seam from
WASTE 0.6.2 onto WASTE 0.6.3, including upstream's new cgroup-aware automatic
memory budget.

The result still refuses public Inkling inference until official-weight parity
earns promotion. What changed is the foundation underneath that refusal: a
container now budgets against RAM it can actually use instead of admiring the
host's RAM through the glass.

## Exact target

- [`sqliteai/waste`](https://github.com/sqliteai/waste)
- commit `69315701f634648f7a790915a0a525ed8aabf218`
- WASTE 0.6.3, public API 1, container format 0
- upstream tree `206a73f61902c41366eaa34f432c8e9b9a5a0675`
- expected applied tree `ce6c5272e801c651cc6b71f869a1b0cd7167dab5`

## What moved since v17

| Foundation | v17 | v18 |
| --- | --- | --- |
| WASTE | 0.6.2 | 0.6.3 |
| Upstream commit | `c7cb640` | `6931570` |
| Upstream delta | — | 12 commits / 35 files |
| Default budget ceiling | physical host RAM | `waste_usable_ram()` |
| Cgroup policy test | absent | `test_memory` retained |
| Model-free suite | 28 pass | 29 pass |
| Serve checks | 167 | 168 |

The replay had two conflicts, both resolved as unions:

- the Makefile includes `memory.c`, `test_memory`, and `sweep` alongside every
  Inkling source and `test_inkling`;
- `tools/convert.py` keeps upstream worker-thread caps and the Inkling guard
  that refuses to route the official checkpoint through the Kimi converter.

## Apply reproducibly

```sh
git clone https://github.com/sqliteai/waste.git
cd waste
git checkout 69315701f634648f7a790915a0a525ed8aabf218
git am /path/to/waste-inkling-patch-v18/patches/0018-waste-693157-consolidated-inkling-runtime.patch
PATH=/usr/bin:/bin make check
```

Patch 18 replaces historical patches 1–17. Do not stack it after Patch 17.

Verify it from this bundle directory:

```sh
sha256sum -c SHA256SUMS
```

## Evidence

- current WASTE suite: 29 passed, 0 failed, 13 skipped
- ASan + UBSan: 28 passed, 0 failed, 14 skipped
- sanitizer fuzzing: 200 cases, 0 crashes, 0 hangs
- strict `-Werror` compile: 11 Inkling/architecture translation units
- dependency-light Python selection: 17 passed
- Python source compilation: passed

See [`TEST-RESULTS-P18.txt`](./TEST-RESULTS-P18.txt) for exact commands and
limitations, and [`PATCH18-HANDOFF.md`](./PATCH18-HANDOFF.md) for the merge
decisions.

## Remaining production gates

1. Run the Patch 15/16 trace workflow against official weights and resolve BF16
   semantic mismatches.
2. Establish measured Q8/Q4/VQ tolerances, conversion time, memory floor, and
   throughput on real hardware.
3. Add official tokenizer and chat-template parity.
4. Design the public manifest extension and dispatch only after those gates.
5. Exercise native Windows, cancellation/resume, observability, and release
   packaging.

The memory budget now understands containers. The confidence budget still
requires evidence. This is annoyingly sensible, which is how production wins.
