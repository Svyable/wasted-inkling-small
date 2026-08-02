# Patch 18 handoff — WASTE 0.6.3 production baseline

Patch 18 consolidates the verified Inkling v16 source surface directly onto
[`sqliteai/waste`](https://github.com/sqliteai/waste) at
`69315701f634648f7a790915a0a525ed8aabf218` (WASTE 0.6.3, API 1, format 0).

This is a **replacement for historical patches 1–17**, not a patch to stack
after them. Patch 17 remains useful provenance for the 0.6.2 baseline; Patch 18
is the current apply target.

## Why this rebase matters

Upstream advanced 12 commits from the Patch 17 baseline. The production-relevant
change is cgroup-aware automatic memory budgeting: WASTE now sizes a default
budget against `waste_usable_ram()` rather than host RAM that a container cannot
actually use. Patch 18 preserves that path, its public API, and `test_memory`
while integrating the private Inkling sources and fail-closed architecture seam.

The two replay conflicts were resolved additively:

- the Makefile retains `src/memory.c`, `test_memory`, and `sweep` alongside all
  Inkling translation units and `test_inkling`;
- `tools/convert.py` retains upstream worker thread caps and rejects Inkling
  checkpoints before entering the Kimi conversion path.

## What is integrated

- Source inspection, exact official Small recognition, conversion planning,
  bounded staging, Q8/Q4 trunk conversion, and WEXP/VQ expert conversion.
- Private staged-directory opening and synthetic quantized token-to-logits
  execution.
- Activation archive, official-reference capture, C tracing, and comparison
  tools from patches 15–16.
- All private Inkling C sources in the WASTE 0.6.3 build and shared library.
- `test_inkling` plus the complete current upstream test matrix.
- Case-insensitive fail-closed recognition in memory planning and model loading,
  returning `WASTE_E_UNSUPPORTED` before Kimi-specific logic.

## Apply

```sh
git clone https://github.com/sqliteai/waste.git
cd waste
git checkout 69315701f634648f7a790915a0a525ed8aabf218
git am /path/to/0018-waste-693157-consolidated-inkling-runtime.patch
PATH=/usr/bin:/bin make check
```

## What this does not claim

The public WASTE manifest and runtime still do not dispatch Inkling. Official
532 GB checkpoint activation parity has not run in this environment. Tokenizer,
chat-template, serving, multimodal execution, native Windows testing, and
production performance remain gates.

That boundary is deliberate: the private engine can advance while the public
loader remains pleasantly boring and refuses anything it cannot yet run
correctly. The container budget is now cgroup-aware; the confidence budget is
still reviewer-aware.
