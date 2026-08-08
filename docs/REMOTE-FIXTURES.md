# Remote bounded parity fixtures

The released Inkling-Small checkpoint is approximately 532 GB across 32
safetensors shards. Layer parity does not require downloading those shards in
full. It requires:

- the release `config.json` and safetensors index;
- the headers of shards containing selected layer tensors;
- complete non-expert tensors for selected layers;
- selected axis-0 routed-expert slices.

`mvp/inkling_remote_fixture.py` extracts exactly that surface with strict HTTP
byte ranges and writes the existing `inkling-parity-fixture` format.

## Safety properties

The extractor fails closed:

- `--revision` must be an immutable 7–64 character lowercase hexadecimal
  commit, never `main`;
- every shard name from the index must be a plain `.safetensors` filename;
- each safetensors header, dtype, shape, offset, and payload geometry is
  validated before a payload request;
- dense layers reject routed-expert selections, while sparse layers require
  them;
- missing global, attention, MLP, router, or shared-expert tensors are named and
  refused;
- per-entry and total payload ceilings are enforced while planning, before
  payload download;
- range responses must be HTTP 206 with the exact `Content-Range`; a proxy or
  server returning HTTP 200 is refused rather than downloading a 13–19 GB
  shard;
- bearer authorization is removed when a request redirects to another host;
- payloads are streamed through temporary files, CRC32-checked in the fixture
  manifest, and the manifest is published last;
- the source repository, immutable revision, config SHA-256, and index SHA-256
  are recorded in `fixture.json`.

The adversarial unit suite uses a local HTTP server and verifies exact expert
slicing, hash binding, byte ceilings, unsafe shard rejection, and refusal of a
server that ignores `Range`.

## Evidence-driven expert selection

The first plan used six speculative expert IDs per sparse layer. Those IDs were
planning placeholders, not routing evidence.

`mvp/discover_inkling_router_experts.py` now runs the official layer far enough
to capture its exact `topk(sorted=False)` choices. It constructs each layer
from a bounded one-expert seed, replaces the routed expert bank before forward,
and aborts immediately after the official router supplies IDs and attached
weights. Seed expert values therefore cannot affect selection.

For eight deterministic BF16 positions (`seed=19`, input SHA-256
`53102b39703fcec1ea4593c4a149e168169d0062769f7bac7944c1afe9831b7f`), the
immutable release selected:

- layer 2: **33 unique routed experts**;
- layer 5: **26 unique routed experts**.

The exact per-position ID/weight pairs and source hashes are committed in
`docs/OFFICIAL-ROUTER-SELECTION.json`. CI reconstructs the one-expert seed and
requires byte-for-byte JSON equality with that artifact.

## Plan without downloading weights

The release upload commit is:

```text
21152b5312c653be115f33a8342759064144e281
```

The authoritative command is recorded in `docs/OFFICIAL-FIXTURE-PLAN.json`.
A plan fetches only config, index, and relevant safetensors headers:

```sh
python mvp/inkling_remote_fixture.py \
  --revision 21152b5312c653be115f33a8342759064144e281 \
  --layers 0,2,5 \
  --experts '2:13,19,25,26,27,30,33,39,60,67,69,81,90,91,92,95,96,112,116,117,126,131,133,140,146,150,152,166,175,217,238,247,254;5:6,21,32,44,45,56,60,63,66,91,105,110,120,145,152,156,163,166,179,180,219,233,236,238,250,252' \
  --max-total-gib 8 \
  --plan-only > /parity/fixture-plan.json
```

### Reproduced release plan

GitHub Actions ran that command against the immutable release and recorded:

| Field | Result |
| --- | ---: |
| Fixture entries | 175 |
| Planned payload | 3,842,395,658 bytes (~3.58 GiB) |
| Metadata and headers read | 179,397 bytes |
| HTTP requests | 52 |
| Shards touched | 25 of 32 |
| Config SHA-256 | `dcb5b1d587bce2f1e6b29833d739a724d05b4bfaa2dc1164fbe679330478ba53` |
| Index SHA-256 | `68b1ab9dade825da1d9d162303f7356167ba2b90fd2c5fdf519e898d45adb0d9` |

The complete machine-readable result is committed as
`docs/OFFICIAL-FIXTURE-PLAN.json`. CI repeats the live plan and compares every
source, selection, geometry, byte-count, request-count, and shard field to that
file. A release or planner change therefore requires an explicit evidence
update.

## Extract the fixture

Run the command recorded in `docs/OFFICIAL-FIXTURE-PLAN.json` without
`--plan-only`, adding the output directory:

```sh
python mvp/inkling_remote_fixture.py \
  --revision 21152b5312c653be115f33a8342759064144e281 \
  --layers 0,2,5 \
  --experts '2:13,19,25,26,27,30,33,39,60,67,69,81,90,91,92,95,96,112,116,117,126,131,133,140,146,150,152,166,175,217,238,247,254;5:6,21,32,44,45,56,60,63,66,91,105,110,120,145,152,156,163,166,179,180,219,233,236,238,250,252' \
  --max-total-gib 8 \
  --out /parity/fixture-L0-L2-L5

python inkling/tools/inkling_fixture.py \
  --fixture /parity/fixture-L0-L2-L5 --verify
```

That expert set covers the committed eight-position input sequence. Different
hidden states may route differently and require a separately recorded selection
artifact; the extractor never silently expands coverage.

## Scope boundary

This is an evidence acquisition tool under `mvp/`. It does not alter the WASTE
runtime, generated patch, public loader, memory planner, or inference dispatch.
Promotion into `inkling/tools/` belongs with the first committed official-weight
comparison report and regenerated bundle.
