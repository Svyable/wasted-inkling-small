# Patch 17 handoff — current WASTE foundation

Patch 17 consolidates the verified Inkling v16 source surface directly onto
[`sqliteai/waste`](https://github.com/sqliteai/waste) at
`c7cb64022963871506908317a661338f1794f70e` (WASTE 0.6.2, API 1, format 0).

This is a **replacement for historical patches 1–16**, not patch 17 to apply
after them. Those patches remain useful provenance, but their early Makefile
and loader hunks predate the current upstream layout and no longer apply
cleanly. The consolidated patch removes that ambiguity: one pinned base, one
patch, one test story.

## What is integrated

- Source inspection, exact official Small recognition, conversion planning,
  bounded staging, Q8/Q4 trunk conversion, and WEXP/VQ expert conversion.
- Private staged-directory opening and synthetic quantized token-to-logits
  execution.
- Activation archive, official-reference capture, C tracing, and comparison
  tools from patches 15–16.
- All private Inkling C sources in the current upstream build and shared
  library.
- A current `test_inkling` target plus upstream-driver coverage.
- Case-insensitive fail-closed recognition in memory planning and model loading,
  returning `WASTE_E_UNSUPPORTED` before Kimi-specific logic.

## Apply

```sh
git clone https://github.com/sqliteai/waste.git
cd waste
git checkout c7cb64022963871506908317a661338f1794f70e
git am /path/to/0017-waste-c7cb640-consolidated-inkling-runtime.patch
PATH=/usr/bin:/bin make check
```

## What this does not claim

The public WASTE manifest and runtime still do not dispatch Inkling. Official
532 GB checkpoint activation parity has not run in this environment. Tokenizer,
chat-template, serving, multimodal execution, native Windows testing, and
production performance remain gates.

That boundary is deliberate: the private engine can advance quickly while the
public loader remains pleasantly boring and refuses anything it cannot yet run
correctly. Frontier model, adult supervision.
