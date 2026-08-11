# Inkling-Small porting status

> Canonical re-entry point for active porting work. Update this file when a gate
> is promoted, invalidated, or superseded. Historical investigation remains in
> `docs/`, but this file describes what `main` can claim now.

## Baseline

- Main checkpoint: `thinkingmachines/Inkling-Small`
- Pinned revision: `21152b5312c653be115f33a8342759064144e281`
- Exact BF16 reference profile: Linux CPU with `ONEDNN_MAX_CPU_ISA=AVX2`
- Merged foundation: through PR #68
- Generated WASTE tree: `b428f70a6dcbbd0cec7364fe12739474f0b4851e`
- Generated patch SHA-256: `356d7613fbc5acd7b6fef221617d700e940520bfc748e721a3fc49d4bd8171ff`

## Proven on `main`

### Public WASTE boundary

- `waste plan`, normal container open/binding, and `waste info` work for Inkling.
- Trunk matrices use the normal WASTE backend; row-backed embedding/unembedding
  tables use row callbacks.
- Expert-bank metadata/record geometry is validated at load time.
- Public step, prefill, eval, generation, chat execution, and serving remain
  deliberately refused. Routed expert records are not yet opened publicly.

### Checked BF16 arithmetic

Official-weight gates compile checked-in C unchanged (`source_rewriting=false`).

- dense layer 0: exact across eight source-bound positions;
- local sparse layer 2: position-zero and eight-token stateful ladder exact;
- global sparse layer 5: position-zero and eight-token stateful ladder exact;
- consecutive dense 0 -> dense 1 -> sparse 2: exact across eight positions;
- consecutive local sparse 4 -> global sparse 5: exact across eight positions,
  including independently reproduced routes and a matching BF16 chained trace.

### Frozen final-head primitive

`waste_inkling_final_head_profile()` in `inkling/src/inkling_model.c` is the one
checked-in definition of the head: final RMS normalization, the logits-width
completion, and the vocabulary projection over an explicit row selection. The
public F32 step now calls it instead of carrying its own copy, and the head
shares the layer's `waste_inkling_rmsnorm_profile()` rather than repeating the
normalization policy. That function deliberately stays in `inkling_layer.c`:
the retained evidence transforms rewrite its text in a copied tree, and moving
it left three harnesses matching nothing.

Both profiles are pinned by `inkling/tests/test_inkling_final_head_c.py`
against an independent bit-level Python reference — ctypes only, no torch, so
the cheap gate runs it. Under `BF16_REFERENCE` the projection is
backend-only: the checked-in C refuses the scalar resident path, the width
completion is a division rather than a reciprocal multiply (the two agree at
the release value 16, not in general), and a bounded row selection returns the
same values the full head would for those rows.

This is a head primitive with a validated policy, not a logits claim. It
computes the head of whatever hidden state it is handed; see open gate 1.

Routing and arithmetic are separate contracts: raw WASTE routes are retained;
reference routes are injected only for downstream arithmetic when the official
stack selects a different valid tied top-k subset.

### Evidence hygiene

- semantic-producer freshness fails closed;
- exact fixture entry/byte/request counts are asserted before download;
- bounded fixtures now also carry vocabulary rows as axis-0 slices, and a
  head-only fixture selects no decoder layer at all, so final-head evidence
  costs about 100 KiB instead of hundreds of megabytes;
- route archives and important output hashes are source-controlled;
- expensive official-weight jobs run only after cheap provenance/import/build
  gates;
- generated WASTE bundles are content-addressed and independently re-applied to
  pinned upstream before publication.

## Open gates

1. **True final hidden state and logits.** Representative consecutive decoder
   transitions are proven, and the head primitive that consumes a final hidden
   state is now checked in and validated against official head weights on a
   supplied bounded vector
   (`.github/workflows/evidence-inkling-checked-bf16-final-head.yml`). What
   remains open is the vector itself: no proven 42-layer decoder output feeds
   that head yet, so nothing here is final-model logits.
2. **Public expert-cache execution / step.** Loader/binding is promoted; routed
   expert record opening/cache dispatch and public stepping remain fail-closed.
3. **Tokenizer/chat-template release parity.** Not yet a promoted public gate.
4. **Quantized quality.** Official VQ/Q8/Q4 reconstruction/logit quality and the
   release format choice remain open.
5. **Full conversion/operational measurement.** Full conversion, resume/cancel,
   wall time, peak RSS/disk, and real decode throughput remain open.
6. **Production official-checkpoint converter.** Format/synthetic writing exist;
   production conversion is not promoted.

## Next load-bearing work

### Numerical path

1. ~~Freeze and validate final-norm/unembedding completion semantics on a
   bounded real hidden-state vector.~~ Done: the primitive is checked in, the
   policy is pinned by a dependency-light differential test, and the official
   head weights run through it in a cheap gated workflow. The report records
   `final_model_logits: false` and the hidden state's provenance.
2. Extend chained decoder/localizer evidence toward the true final-layer endpoint
   with evidence-driven routed-expert acquisition.
3. Feed that true final hidden state into the frozen head primitive — the
   runner already accepts an archived vector through `--hidden-state` — and pin
   logits before public token execution.

### Runtime path in parallel

1. Open routed expert records through WASTE's existing expert cache and the
   architecture-specific native backend; no second scalar engine.
2. Bind checked BF16 stepping privately against a public-container load while
   retaining public refusal.
3. Promote normal WASTE step/prefill only after decoder/logit evidence closes.

### Release path in parallel

- tokenizer/chat-template parity;
- quantized-quality measurement and format decision;
- one real full conversion with restart/resume and resource measurements.

## Merge discipline

Keep the active PR stack short. Merge an earned gate before opening multiple
children. Never repair stale evidence by editing hashes: reacquire/regenerate
from the current semantic parent. Preserve merge ancestry for stacked evidence.
