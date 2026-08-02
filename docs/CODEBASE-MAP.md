# Codebase map and refactor plan

Two things in one document, because they are the same problem viewed twice:

1. **The map** — what every Inkling translation unit, tool, and test actually
   does, and how they layer. Read this to find code.
2. **The refactor** — how to move from "a repository of patch bundles" to
   "a repository of C that happens to ship a patch." Read this to change code.

Sizes and relationships were measured on the applied v18 tree
(`ce6c5272e801c651cc6b71f869a1b0cd7167dab5`) on 2026-08-02.

---

# Part I — The map

## 0. Where the code lives

**R1 is done.** The source of truth is now a tree of files, and the patch is a
build product:

```
inkling/src, tools, tests, docs           ← AUTHORITATIVE. Edit these.
integration/waste/overlay/*.diff          ← the five upstream edits, 193 lines
integration/waste/{baseline.env,generate.sh,verify.sh}
dist/waste-inkling-6931570/               ← GENERATED. Never edit.

waste-inkling-patch-v18/                  ← frozen provenance (WASTE 0.6.3)
waste-inkling-patch-v17/                  ← frozen provenance (WASTE 0.6.2)
waste-inkling-patch-v16/                  ← frozen provenance + historical notes
```

`integration/waste/generate.sh` clones the pinned upstream, copies `inkling/`
over it, applies the overlay fragments, and commits; `verify.sh` refuses to
continue unless the resulting tree hash matches `EXPECTED_APPLIED_TREE`.

The extraction was proved by reproducing the reviewed v18 tree
(`ce6c5272e801c651cc6b71f869a1b0cd7167dab5`) byte-for-byte *before* any source
changed. The generator pins the commit date and suppresses the Git-version
signature, so repeated runs match; the gate that matters is the applied tree
hash, which no toolchain difference can move.

The map below describes the generated tree. Paths are given as they appear
there — `src/inkling_layer.c` in the applied tree is `inkling/src/inkling_layer.c`
in this repository.

## 1. Layering

Nothing below points upward. This is the property worth preserving through the
refactor.

```
                    ┌─────────────────────────────────────┐
   public WASTE     │ waste.c   model.c   cli/   serve/    │  ← untouched, except
                    └───────────────┬─────────────────────┘     two guard blocks
                                    │ waste_arch_classify()
                    ┌───────────────┴─────────────────────┐
   seam             │ arch.c / arch.h  (38 lines)          │  ← the whole public
                    └───────────────┬─────────────────────┘     contact surface
                                    │
   ─────────────────────────────────┼──────────────────────── promotion boundary
                                    │
   orchestration    │ inkling_private.c        791 │  staged dir → logits
                    │ inkling_bind.c           135 │  names → structs
                    ├──────────────────────────────┤
   readers          │ inkling_stage_reader.c   392 │  IKTN tensors, IKBF experts
                    │ inkling_wexp.c           369 │  WEXP records + WCBK books
                    │ inkling_qtensor.c        143 │  Q8G / Q4G + direct matvec
                    ├──────────────────────────────┤
   execution        │ inkling_model.c          282 │  embed → 42 layers → logits
                    │ inkling_layer.c          341 │  one decoder layer
                    │ inkling_attention.c      142 │  GQA, local ring / global
                    │ inkling.c                142 │  route, sconv, rel-bias, tau
                    ├──────────────────────────────┤
   contract         │ inkling_config.c         185 │  geometry + memory plan
                    └──────────────────────────────┘
```

## 2. Translation units

### `src/arch.[ch]` — the seam (13 + 25 lines)

`waste_arch_classify()` lowercases and prefix-matches `"inkling"`. Called from
exactly two places, both of which refuse before doing anything architecture-
specific:

- `src/waste.c:179` in `waste_plan_memory()` → `WASTE_E_UNSUPPORTED`
- `src/model.c:976` in the loader, after `format_version`, before config
  parsing or tensor binding → `-3` → `WASTE_E_UNSUPPORTED`

At promotion this file becomes the dispatch table. It does not get deleted.

### `src/inkling_config.[ch]` — the geometry contract (185 + 125)

`waste_inkling_config_build()` takes normalized config arguments plus
`local_layer_ids` and derives each of the 42 layers' exact attention geometry
(local vs global, head counts, head dim, relative extent). It validates
everything up front: divisibility of heads by KV heads, `top_k ≤ n_routed`,
`unpadded_vocab ≤ vocab`, duplicate local-layer ids, kernel bounds.

`waste_inkling_plan_decode_memory()` is the memory contract, broken out field
by field (`kv_bytes`, `conv_bytes`, `token_vector_bytes`, `projection_bytes`,
`attention_score_bytes`, `router_bytes`, `expert_workspace_bytes`,
`dense_workspace_bytes`, `shared_workspace_bytes`, `logits_bytes`) rather than
a flat estimate, so a future allocator can be checked term by term. Every
addition and multiplication is overflow-checked.

**This is the file that `waste_plan_memory()` will eventually call.** It is
already tested standalone (`tests/test_inkling_config_c.py`, dependency-light,
runs in CI).

### `src/inkling.c` — scalar primitives (142)

Four pure functions, separated from the layer so they can be differential
-tested before anything is wired:

- `waste_inkling_route()` — sigmoid + correction bias, top-k selection,
  joint log-sigmoid normalization across selected routed *and all shared*
  experts, scaled by `route_scale * global_scale`. Deterministic ties (lower
  expert id wins).
- `waste_inkling_sconv_step()` — one causal depthwise short-convolution update
  against an in-place `[channels][kernel]` fp32 ring.
- `waste_inkling_relative_bias()` — materializes one query token's learned
  relative-position bias from `[heads][d_rel] × [d_rel][extent]`, zero outside
  `[0, extent)`.
- `waste_inkling_log_tau()` — `1 + alpha·log(max((pos+1)/floor, 1))`.

### `src/inkling_attention.c` (142) and `src/inkling_layer.c` (341)

Attention is grouped-query, one token at a time, with a local ring buffer or a
global linear scan chosen per layer. The layer composes the exact official
order: per-head Q/K RMSNorm → K/V short conv → attention → learned relative
bias → log scaling → output projection → attention-branch short conv → dense
gated-SiLU *or* routed+shared MoE → MLP-branch short conv → residuals.

Both have traced variants (`*_trace`) that emit named F32/int32 records; the
untraced entry points delegate with a null trace, so production pays nothing.

### `src/inkling_model.c` (282)

Whole-model step: row-backed embedding → embedding RMSNorm → 42 layers →
final RMSNorm → divide by `logits_mup_width_multiplier` → independent
row-backed unembedding → truncate to `unpadded_vocab`.

Exposes exact caller-owned storage sizes (`…_state_floats()`,
`…_scratch_floats()`, `…_scratch_ints()`). Nothing in the execution layer
allocates.

### `src/inkling_bind.c` (135)

Binds canonical `inkling.*` tensor names to the runtime structs with exact
shape checks and no duplicates. Three variants, increasingly permissive about
where the bytes live:

| function | embedding/unembedding | matrices |
| --- | --- | --- |
| `bind_weights` | resident F32 | resident F32 |
| `bind_weights_ex` | resident **or** row callback | resident F32 |
| `bind_weights_ex_backend` | resident **or** row callback | may be NULL → quantized backend |

Routed expert matrices are deliberately absent from all three: production
supplies them through the expert callback, which is the same shape as WASTE's
existing expert cache.

### `src/inkling_stage_reader.c` (392) — **the reader**

Reads the converter's two private staging formats. Everything is positional
(`pread` / `ReadFile`+`OVERLAPPED`); nothing is mapped; no allocation happens
per row.

**`IKTN` — one canonical tensor per file.** 64-byte header: magic, version 1,
dtype (1 = BF16, 2 = F16, 3 = F32), ndim, four `uint32` dims, payload bytes,
payload CRC32, canonical-name CRC32, 20 reserved bytes that must be zero.
Payload starts at 64 and the file is padded to 4 KiB. `_open()` validates the
declared shape against the caller's expected shape, recomputes the aligned
stored size and compares it to the real file size, checks the name CRC, and
optionally streams a full payload CRC in 64 KiB blocks. `_row()` serves one
row through a preallocated row buffer; `_read_all()` streams the whole tensor
in 32k-value blocks.

**`IKBF` — one BF16 expert record per (layer, expert).** Fixed record stride,
identity fields (layer, expert) checked against the caller's expectation,
three equal-sized gate/up/down sections at validated offsets, payload CRC.
`_bank_open_with_workspace()` exists so all 40 sparse layers can share one set
of three F32 matrices and one raw record buffer.

Reader invariants that should survive any refactor: *shape is an input, not an
output* (the caller says what it expects and the file must agree); reserved
bytes must be zero; file size must equal the computed size exactly; every
arithmetic step goes through `mul64`/`add64`/`align64`.

Known limits: `_row()` is 2-D only; dtype is limited to BF16/F16/F32; CRC
verification is all-or-nothing per tensor; there is no safetensors path, so
nothing here can read the released checkpoint directly.

### `src/inkling_wexp.c` (369)

The final-format reader: `WEXP` expert records plus per-layer `WCBK`
codebooks, in WASTE's existing on-disk layouts. Validates codebook headers,
absolute codebook ids, record identity, blocked-index geometry
(`[block][vector][row][stage]`), offsets, file size, reserved fields, and
optional payload CRC, then dequantizes one expert into the shared workspace
and hands it back through the same callback the model already uses.

Intentionally scalar. At promotion the public engine should route through
WASTE's optimized LUT/VQ kernels (`src/vq.c`) and keep this one as the
validation oracle.

### `src/inkling_qtensor.c` (143)

Q8G/Q4G artifacts: 96-byte header, row-major payload, FP16 group scales,
independent CRCs for payload and scales, 4 KiB padding. Supports bounded row
reads *and* direct quantized matvec without materializing F32.

### `src/inkling_private.c` (791) — the orchestrator

Parses `runtime-stage.bin` — the converter-private index, never
`manifest.json` — and assembles a runnable model. Three index versions are
accepted:

| version | trunk | experts |
| --- | --- | --- |
| 1 | F32/BF16 staged tensors | `IKBF` BF16 banks |
| 2 | F32/BF16 staged tensors | `WEXP` + `WCBK` |
| 3 | mixed Q8/Q4 quantized trunk | `WEXP` + `WCBK` |

It validates entry counts and sizes, reserved bytes, path safety (relative,
no traversal), exact config/layer geometry, tensor count, duplicate names, and
every shape before opening a descriptor; allocates state and scratch from the
tested contract; runs token-to-logits with deterministic reset; and closes
every fd and allocation on any partial-open failure. Reports quantized
resident bytes and canonical F32 resident bytes separately.

## 3. Python tooling

| Tool | Lines | Role |
| --- | --- | --- |
| `inkling_source.py` | 36 | is this config an Inkling config? |
| `inspect_inkling.py` | 799 | bounded inspection of config, safetensors index/headers, tokenizer, processor, chat assets |
| `inkling_release.py` | 287 | **fail-closed identity**: exact official Small architecture, package (532 GB / 32 shards / tensor inventory), sidecar readiness |
| `inkling_plan.py` | 519 | derive every layer descriptor, validate exact source shapes |
| `inkling_weights.py` | 369 | source tensor adapter |
| `inkling_trunk.py` | 557 | stage resident trunk tensors → `IKTN`, bounded, resumable, deinterleaves fused gate/up |
| `inkling_stage.py` | 381 | stage routed experts → `IKBF`, one expert at a time |
| `inkling_qtrunk.py` | 254 | trunk → Q8G/Q4G |
| `inkling_vq.py` | 927 | official BF16 experts → final `WEXP`/`WCBK` directly, optional `libwastevq` for the encode |
| `inkling_runtime_stage.py` | 748 | verify everything, publish `runtime-stage.bin` atomically |
| `inkling_parity.py` | 416 | bounded fixture extraction + CRC-protected activation archives + comparison |
| `inkling_fixture.py` | 268 | **dependency-free** fixture reader: CRC verification, axis-0 expert slices, BF16/F16/F32 → F32 decode, module-relative state-dict keys, fail-closed coverage checks |
| `inkling_trace.py` | 147 | C side of the trace protocol |
| `inkling_reference.py` | 163 | official Transformers side of the trace protocol |
| `convert_inkling.py` | 306 | the CLI that drives all of the above |

Entry point, in the order a laptop should run it:

```sh
python tools/convert_inkling.py --src MODEL --out STAGE --plan-only
python tools/convert_inkling.py --src MODEL --out STAGE --stage-trunk
python tools/convert_inkling.py --src MODEL --out STAGE --quantize-trunk --trunk-bits 4
python tools/convert_inkling.py --src MODEL --out STAGE --quantize-experts --vq-stages 3
python tools/convert_inkling.py --src MODEL --out STAGE --publish-runtime-vq-stage
```

This path never writes the ~480 GiB BF16 expert stage. `--stage-experts` plus
`--publish-runtime-stage` remains as the BF16 debug route.

## 4. Tests

- **C:** `tests/test_inkling.c` — the seam and scalar primitives; runs in
  `make check` on every platform.
- **Dependency-light Python (CI):** `test_inspect_inkling.py`,
  `test_inkling_plan.py`, `test_inkling_release.py`, `test_inkling_config_c.py`
  — 17 tests, no torch.
- **Torch-dependent (not in CI):** the other 16 files, including the C/PyTorch
  differential tests for router, relative bias, short conv, attention, complete
  layer, complete model, and the staged-directory end-to-end tests. The v16
  bundle records 99 passing; nothing in this repository re-runs them.

That last line is a gap, not a complaint — see the refactor's CI tier below.

---

# Part II — The refactor

## 5. What was wrong, and what is left

Nothing in the C. Four things in the packaging — two now fixed:

1. ~~**The source of truth is a diff.**~~ **Fixed (R1).** `inkling/` is the
   source of truth; the patch is generated from it and the tree hash is the
   proof.
2. **Provenance is duplicated, not derived.** v16-v18 each carry a full copy of
   every Inkling source. They are kept as frozen provenance, so the duplication
   is now bounded rather than growing: the next upstream rebase adds a
   `baseline.env` edit, not another 16,000-line patch.
3. ~~**Only 4 of 20 Python test files run anywhere.**~~ **Fixed (R4, partly).**
   A `differential` CI job installs torch and runs the complete suite. The 99
   tests were confirmed to pass on 2026-08-02 — the first time this repository
   ran rather than quoted them — and are now 140 with the fixture reader.
4. **The public seam is described in prose in five documents** and implemented
   in 38 lines. The prose is where it drifts. Still true; R5 is where it stops
   mattering.

## 6. Layout as built

```
inkling/                     ← THE source of truth; plain C, plain Python
  src/            arch.[ch], inkling*.[ch]        (11 TUs, 3,782 lines)
  tools/          the 15 Python tools
  tests/          test_inkling.c + 21 Python test files + tests/data
  docs/           INKLING.md

integration/waste/
  baseline.env    upstream pin, expected tree, generated-commit identity/date
  overlay/        the ONLY upstream edits, five diffs totalling 193 lines:
                    Makefile.diff          sources + test_inkling
                    src_model.c.diff       the loader guard
                    src_waste.c.diff       the plan guard
                    tests_run.sh.diff      the two Inkling checks
                    tools_convert.py.diff  refuse Inkling before the Kimi path
  tree-extras/    files the bundle places at the upstream root
  generate.sh     clone → checkout pin → copy inkling/ → overlay → commit → patch
  verify.sh       generate.sh → tree hash → -Werror compile → make check

dist/
  waste-inkling-6931570/    GENERATED, committed for consumers:
    patches/0001-…patch, BASELINE, SHA256SUMS, TEST-RESULTS.txt, README.md

docs/             STATE-OF-THE-PORT.md, CODEBASE-MAP.md, ROADMAP-V19.md
```

The insight that made this cheap: measured on v18, the port modifies **5
upstream files by +84/−7 lines** and adds **62 new files**. The 62 are a
directory copy. The 84 lines are five diffs. A patch that large does not need
to be authored by hand — it needs to be generated and its tree hash pinned,
which is exactly what CI already checked.

Two implementation notes worth keeping, because both cost a debugging cycle:

- `git add -A` **silently drops** `tests/data/*.json`, because upstream's
  `.gitignore` lists `data/`. `git am` never consulted `.gitignore`, so the
  generator must use `--force` or the generated tree quietly loses two files.
- `format-patch` embeds **the commit date and the local Git version**. The
  date is pinned by `PATCH_DATE`; the version is suppressed with
  `--no-signature`. Both were found the hard way — the second by CI, on a
  runner with Git 2.54.0 against a committer on 2.43.0, failing a byte
  comparison with nothing actually stale.
- Which is the deeper lesson: **compare trees, not patches.** A patch file is a
  rendering, and renderings have toolchain in them. The applied tree hash is
  content-addressed, is what consumers depend on, and is what the CI staleness
  gate now checks by running `git am` on the committed bundle.

## 7. Migration, in order, each step green before the next

**R1 — Extract without changing bytes. ✅ done.**
The generated patch reproduced tree
`ce6c5272e801c651cc6b71f869a1b0cd7167dab5` byte-for-byte from `inkling/` plus
the overlay, with no source edits, before anything else was allowed to change.

**R2 — Make the bundle generated. ✅ done.**
`dist/waste-inkling-6931570/` is produced by `generate.sh` and checked by CI
against a fresh regeneration, so a stale `dist/` cannot ship unreviewed code.
`waste-inkling-patch-v16` through `-v18` stay exactly where they are as frozen
provenance; deleting them would cost the audit trail and buy nothing.

**R3 — Rebasing becomes a one-line change.** ⬜
The next upstream release is a `baseline.env` edit plus whichever overlay
fragment stopped applying. `generate.sh` names the failing fragment and exits;
that message replaces `PATCH18-HANDOFF.md`-style conflict narration, because
the fragments *are* the conflict resolution, in executable form.

**R4 — Tier the CI. ✅ mostly done.**
- *fast* (every PR): generate → tree hash → `-Werror` compile → `make check` →
  regenerated-bundle comparison → the dependency-light Python tests.
- *deep* (same workflow, separate jobs): `make asan`, `make fuzz-asan`, and the
  torch differential suite — so the 99 tests stopped being folklore.
- *weights* (manual, self-hosted): the parity gates from
  [ROADMAP-V19.md](ROADMAP-V19.md). Still to build.

**R5 — Promote the seam last.**
Only after G1–G4 in the roadmap: `arch.c` grows a dispatch,
`waste_plan_memory()` calls `waste_inkling_plan_decode_memory()`,
`waste_open`/step/reset/stats route to the Inkling model, and the manifest
extension lands. The guard becomes a branch. The `WASTE_E_UNSUPPORTED` return
stays for every variant that has not earned promotion.

## 8. The manifest extension, sketched now so §R5 is not a design meeting

WASTE's manifest already carries `format_version`, `arch`, `tensor_prefix`,
`config`, `expert_quant {stages, vec_dim, entries, index_block,
bits_per_weight}`, `layers [{file, experts, bytes, codebook_base}]`, and
`trunk [{name, fmt, off, shape, group, scale_off, bytes}]`.

Inkling needs **no new top-level shape**. It needs:

- `arch: "inkling"` — already classified, currently a refusal;
- `config` — the release config verbatim with the multimodal wrapper under
  `_outer`, exactly as K3 does it, so `local_layer_ids`, `dense_mlp_idx`,
  `d_rel`, `rel_extent`, `sconv_kernel_size`, `log_scaling_*`,
  `logits_mup_width_multiplier`, `shared_expert_sink`, and `norm_after_topk`
  arrive without invention;
- `trunk` entries for the four per-layer short-convolution kernels and the
  relative-position projection banks, which are simply more named tensors;
- `layers[].codebook_base` — already present, already what
  `runtime-stage.bin` v2/v3 records.

The private index is therefore a *rehearsal* of the public manifest, not a
competing format. That is the strongest argument for promotion when the time
comes, and the reason not to invent a fifth format before then.

## 9. What this buys the open-source reader

A contributor who wants to fix a bug in the short-convolution ring opens
`inkling/src/inkling_layer.c`, runs `make -C build check`, and sends a diff of
that file. Today they would have to regenerate a 15,824-line patch and hope the
tree hash matches. The C is already production-shaped — overflow-checked,
caller-owned allocation, fail-closed readers, no dependencies, sanitizer- and
fuzz-clean. The packaging is what stops people from seeing that.
