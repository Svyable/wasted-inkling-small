# MVP readiness handoff

## Where the previous work stopped

The public memory planner is live for valid Inkling manifests. Public loading,
step execution, chat, and serving remain deliberately refused. The next hard
gate is official-weight layer parity, not another loader branch.

The repository has bounded checkpoint fixtures, a fixture-backed C decoder
layer runner, and an official `InklingDecoderLayer` harness that avoids
materializing the complete checkpoint or 256-expert bank.

The real release `config.json` confirms the recorded Small geometry: dense
width 16384 and routed-expert width 2048. The apparent 3072 width came from
passing the release schema directly into Transformers 5.14.1, whose field names
have different meanings. See `docs/CONFIG-SCHEMA-RESOLUTION.md`.

PR #13 established an independent synthetic oracle: the C decoder layer and
official Transformers layer compute the same dense/local and sparse/global
functions to float32 epsilon. Official-weight parity is still not established.

## Evidence harness

`mvp/inkling_fixture_reference.py` contains the bounded official layer
implementation. It:

- verifies `config.json` against the SHA-256 recorded in `fixture.json`;
- instantiates the official `InklingDecoderLayer` on the meta device;
- removes the full routed-expert bank before materializing the layer;
- loads provider-raw attention, convolution, dense/shared MLP, and router
  tensors into the official module;
- materializes only expert slices selected by the bounded fixture;
- fails closed, naming any expert selected by the router but absent from the
  fixture;
- emits the existing CRC-protected activation archive format.

Use `mvp/run_inkling_fixture_reference.py` as the entry point. It applies the
release-to-Transformers schema translation before constructing the layer and
canonicalizes routed `(expert_id, weight)` pairs. Invoking the implementation
file directly bypasses that compatibility boundary.

`mvp/compare_inkling_layer_archives.py` canonicalizes router pairs in both
archives before calling the existing activation comparator. This matters
because the official router uses `topk(sorted=False)` while the C side emits a
different slot order.

These tools intentionally remain outside the generated WASTE patch until they
have run against an official fixture. Promotion into `inkling/tools/` should be
the commit that records the run and regenerates the bundle.

## Routed expert coverage is now evidence-driven

The original remote plan used six speculative routed experts per sparse layer.
Before downloading that fixture, the official router was probed over eight
deterministic BF16 positions.

The probe constructs each official sparse layer from a bounded one-expert seed,
replaces the expert bank before forward, captures the exact unsorted expert IDs
and attached weights, and stops before expert computation. The seed expert
values cannot influence routing.

The immutable release selected:

- layer 2: 33 unique experts;
- layer 5: 26 unique experts.

The full per-position choices, weights, input SHA-256, release revision, config
hash, and index hash are committed in
`docs/OFFICIAL-ROUTER-SELECTION.json`. CI reproduces that JSON exactly.

## Acquire the bounded official fixture

A local 532 GB checkpoint mount is no longer required. The remote extractor
reads only release metadata, safetensors headers, selected layer tensors, and
selected expert slices from immutable release commit
`21152b5312c653be115f33a8342759064144e281`.

The evidence-selected fixture plan is committed in
`docs/OFFICIAL-FIXTURE-PLAN.json`:

- 175 entries;
- 3,842,395,658 payload bytes, approximately 3.58 GiB;
- 179,397 metadata bytes and 52 requests to plan;
- 25 of 32 shards touched by header ranges.

Plan first; this downloads no tensor payloads. Use the exact command stored in
`docs/OFFICIAL-FIXTURE-PLAN.json`, which contains the 59 discovered expert IDs.
Then run the same command without `--plan-only` and add:

```sh
--out /parity/fixture-L0-L2-L5
```

Verify every payload and CRC:

```sh
python inkling/tools/inkling_fixture.py \
  --fixture /parity/fixture-L0-L2-L5 --verify
```

The extractor refuses mutable revisions, unsafe index paths, missing tensors,
invalid safetensors geometry, incomplete sparse expert selections, and any
server that ignores HTTP `Range`. See `docs/REMOTE-FIXTURES.md`.

## Run the G0 loop

Generate the same deterministic BF16 input sequence recorded by the router
selection artifact:

```sh
PYTHONPATH=mvp python - <<'PY'
import json
from pathlib import Path
from discover_inkling_router_experts import deterministic_hidden_states

values = deterministic_hidden_states(8, 4096, 19)
Path('/parity/inputs.json').write_text(json.dumps(values.tolist()) + '\n')
PY
```

Then run both sides:

```sh
python -m pip install 'transformers==5.14.1' torch

python mvp/run_inkling_fixture_reference.py \
  --fixture /parity/fixture-L0-L2-L5 \
  --model-config /models/Inkling-Small/config.json \
  --inputs /parity/inputs.json \
  --layers 0,2,5 \
  --dtype bfloat16 \
  --out /parity/python

python inkling/tools/inkling_layer_parity.py \
  --fixture /parity/fixture-L0-L2-L5 \
  --config /parity/normalized-config.json \
  --inputs /parity/inputs.json \
  --layers 0,2,5 \
  --out /parity/waste

python mvp/compare_inkling_layer_archives.py \
  --compare-reference /parity/python \
  --compare-candidate /parity/waste \
  --atol 1e-3 --rtol 1e-3 \
  --report /parity/report.json
```

The committed expert set covers this exact eight-position input sequence.
Different hidden states may route differently and require a separately recorded
selection artifact rather than silent fixture expansion.

## MVP sequence from here

1. Extract and CRC-verify the 3.58 GiB official fixture.
2. Run the official and C archives over the committed deterministic hidden
   states and commit the first comparison report, whatever it says.
3. Require exact routing-index agreement first; then fix the first activation
   mismatch until every recorded activation is within the G1 `1e-3` bound.
4. Promote the evidence tools into `inkling/tools/`, regenerate the distribution
   patch, and update the applied-tree hash.
5. Establish tokenizer and chat-template parity in parallel.
6. Only after BF16 and text-interface evidence is green, promote public loading
   and stepping. Keep unsupported behavior for every unverified variant.

## Definition of an MVP

The first usable MVP is a normal WASTE text flow, not a special parity binary:
`waste plan`, `waste run`, and the OpenAI-compatible server against one verified
Inkling-Small container. Chat can follow once template parity is recorded.
Multimodal execution and speculative decoding remain outside that boundary.
