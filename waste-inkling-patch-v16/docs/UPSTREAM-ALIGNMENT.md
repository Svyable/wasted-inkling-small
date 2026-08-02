# Upstream alignment and production roadmap

Baseline reviewed: [`sqliteai/waste@c7cb640`](https://github.com/sqliteai/waste/commit/c7cb64022963871506908317a661338f1794f70e)
on 2026-08-01. That commit identifies itself as WASTE 0.6.2, public API version
1, container format version 0.

This document compares the Patch 16 Inkling-Small handoff with the current WASTE
foundation. It is an integration plan, not a production-readiness claim.

## Executive decision

Keep the private Inkling runtime as an independent correctness oracle and staging
harness. Build shipping Inkling-Small support through WASTE's existing public
container, loader, RAM planner, expert cache, tokenizer boundary, CLI, and server.

That avoids creating two engines with two formats, two caches, two memory models,
and—eventually—two excitingly different classes of bug. Frontier is good for the
model; one production runtime is enough frontier for the rest of us.

## What already aligns

| Area | Patch 16 bundle | WASTE 0.6.2 | Decision |
| --- | --- | --- | --- |
| Expert storage | Per-layer WEXP banks with gate/up/down in one record | Per-layer WEXP banks, one 4 KiB-aligned `pread` per expert | Reuse the upstream record reader and cache |
| Expert quantization | Residual VQ2R/VQ3R plus WCBK codebooks | Residual VQ2R/VQ3R plus WCBK codebooks | Emit the frozen v0 layout directly |
| Trunk quantization | Q8/Q4 artifacts with FP16 group scales | Q8G/Q4G trunk tensors in `trunk.bin` | Map Inkling tensor names/shapes into the existing trunk index |
| Bounded conversion | Resumable trunk/expert staging and atomic sidecars | Resumable converter and end-to-end pipeline | Keep bounded source reads; adopt upstream final publication |
| Integrity | Record identity, geometry, offsets, optional CRC | Mandatory header validation and optional per-record CRC | Use upstream untrusted-container rules |
| Correctness | Synthetic C/PyTorch parity and named activation archives | Synthetic containers plus PyTorch oracle checks | Make the Inkling archive one upstream oracle path |
| Runtime dependencies | C11 private runtime; Python only for conversion/reference | C11 public runtime; Python only for tools | Preserve the dependency boundary |

## The production gaps

| Seam | Current Inkling state | Production target |
| --- | --- | --- |
| Container | Private `runtime-stage.bin`; deliberately no `manifest.json` | Normal `.waste` directory with format-v0 manifest, trunk, expert banks, codebooks, tokenizer, specials, and chat assets |
| Architecture dispatch | New `arch.c/.h` classifier and private entry points | Fail-closed architecture selection inside the live loader, planner, and model dispatch |
| Memory | Geometry estimates and private allocations | `waste_plan_memory` and `waste_cfg.ram_budget_bytes` cover every Inkling allocation and refuse budgets below the floor |
| Expert I/O | Private bank readers and one shared decode workspace | Existing LFRU cache, page-cache bypass, record validation, usage hotlist, and runtime statistics |
| Public API | Private open/step/reset/trace surface | Standard `waste_open`, `waste_eval`, `waste_generate`, reset/state, introspection, and stats |
| Tokenizer and prompt | Official assets inspected but not executed | Self-contained tokenizer/specials plus strict markup-vs-user-content separation and verified chat rendering |
| CLI and serving | None | Existing `plan`, `run`, `chat`, `eval`, and OpenAI-compatible endpoints without Inkling-specific forks |
| Platform evidence | Python tests and strict compilation; native Windows not claimed | Fresh-clone synthetic suite, ASan/UBSan, parser fuzzing, macOS/Linux/Windows CI, then real-container gates |
| Real-weight evidence | Harness exists; official 532 GB comparison not run | Layerwise BF16 oracle parity first, quantized tolerances second, fixed-prompt token parity last |
| Performance evidence | Format-size estimates | Measured floor, peak RSS, bytes/token, cache hit rate, I/O time, prefill, and decode on named hardware |

## Production gates

Each gate must fail closed. A missing model, compiler, or oracle is a visible
`SKIP`, never a green check that ran nothing.

### P0 — rebase and architecture seam

- Refresh the patch stack onto the pinned upstream commit.
- Add Inkling as a fail-closed architecture; an Inkling manifest must never enter
  Kimi memory formulas or forward dispatch.
- Keep public behavior unchanged for existing Kimi containers.
- Add a tiny architecture-classification test before changing the forward pass.

Exit: upstream `make check`, sanitizer, and fuzz targets remain green with no
Inkling container present.

### P1 — publish a real WASTE container

- Extend `tools/convert.py` or add a model adapter behind it; do not ship a second
  final-format converter.
- Emit format-v0 `manifest.json`, `trunk.bin`, `experts-L*.bin`, `codebooks.bin`,
  tokenizer assets, and special-token metadata.
- Preserve one aligned read per expert and validate every manifest dimension,
  offset, shape, and record identity as untrusted input.
- Generate a few-megabyte synthetic Inkling container in the real format.

Exit: the public parser, record checker, and fuzzer accept valid synthetic
Inkling and reject corrupt or cross-architecture inputs.

### P2 — public loader and hard memory ceiling

- Route `waste_plan_memory` and `waste_open` through an Inkling-specific shape
  contract while reusing the upstream allocator, expert cache, direct-I/O path,
  cancellation, error reporting, and statistics.
- Account for trunk, recurrent/convolution state, KV state, logits, scratch,
  codebooks, expert workspace, and cache under the caller's hard budget.
- Refuse an under-floor budget instead of relying on swap.

Exit: planned floor matches post-open memory, peak RSS stays inside the selected
budget, and sequential/chunked synthetic execution agrees.

### P3 — byte-to-logits parity

- Bind the existing Inkling attention, convolution, dense, router, MoE, norm,
  embedding, and unembedding math behind the live WASTE model dispatch.
- Feed named traces from the public path into the Patch 16 activation archive.
- Compare official BF16 reference activations before quantized activations.
- Set tolerances from measurements; never derive them from optimism.

Exit: selected layers, routing indices, final logits, argmax, and fixed-prompt
token sequences pass the documented oracle thresholds.

### P4 — tokenizer, chat, and state

- Execute the official tokenizer and special-token map inside the self-contained
  container.
- Preserve WASTE's security boundary: template markup may resolve control tokens;
  user, document, and tool content may not.
- Differentially test chat rendering against the released template.
- Verify reset plus save/load continuation across local/global attention and
  short-convolution state.

Exit: `waste run` and `waste chat` produce the oracle prompt ids and continuation,
with prompt-injection regression coverage at the tokenizer boundary.

### P5 — platform and operations

- Add model-free synthetic Inkling checks to upstream CI on macOS arm64, Linux
  arm64/x86_64, and Windows x86_64.
- Run ASan/UBSan and container fuzzing after every parser or allocation change.
- Build an Inkling pipeline matching upstream's download → convert → verify → run
  → oracle report, with resumable and atomic stages.
- Verify interrupted conversion, disk-full cleanup, paths with spaces/Unicode,
  files above 4 GiB, and copied-container CRC checks.

Exit: a fresh clone proves the engine path without real weights, while a machine
with the official checkpoint unlocks explicit real-weight gates.

### P6 — measured production performance

- Measure on named hardware with the same prompt, context, quantization, cache,
  and direct-I/O settings.
- Record peak RSS, container bytes, conversion time, prefill tokens/s, decode
  tokens/s, bytes read/token, cache hit rate, and I/O share.
- Profile before adding GPU work. WASTE's current evidence says many dependent
  matvec dispatches can lose to a bandwidth-saturated CPU; Inkling must earn any
  different conclusion with measurements.

Exit: the documented default fits its target machine without swap and has a
reproducible performance report. "SOTA" means a number with a command next to it.

## Production definition of done

Inkling-Small is production-ready for WASTE only when all of these are true:

1. `waste plan inkling-small.waste` reports a complete, enforced memory plan.
2. `waste run`, `chat`, `eval`, state save/load, and the server use the same public
   API and container as every other supported model.
3. A fresh clone runs meaningful synthetic Inkling checks without downloading
   weights; real-weight checks are explicit and never silently skipped.
4. Official BF16 traces and logits match the reference, and quantized thresholds
   are measured separately.
5. Corrupt manifests, trunks, codebooks, expert records, and tokenizer metadata
   fail with actionable errors.
6. Supported platforms pass CI, sanitizers, and fuzzing; measured RAM never
   exceeds the configured ceiling.
7. The release notes name the hardware, container recipe, accuracy gates, and
   performance commands used for every claim.

## Keeping the baseline honest

Run the checker from this bundle against a local WASTE checkout:

```sh
python tools/check_waste_baseline.py /path/to/sqliteai-waste
python tools/check_waste_baseline.py /path/to/sqliteai-waste --json
```

It verifies the pinned public API/format assumptions and the repository seams
this roadmap intends to reuse. It cannot prove semantic compatibility; a warning
about a newer upstream version is a prompt to repeat this comparison.
