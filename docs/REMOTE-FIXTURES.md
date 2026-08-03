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

## Plan without downloading weights

The release upload commit is:

```text
21152b5312c653be115f33a8342759064144e281
```

A plan fetches only config, index, and relevant safetensors headers:

```sh
python mvp/inkling_remote_fixture.py \
  --revision 21152b5312c653be115f33a8342759064144e281 \
  --layers 0,2,5 \
  --experts '2:4,17,39,88,143,221;5:1,8,22,64,150,201' \
  --max-total-gib 8 \
  --plan-only > /parity/fixture-plan.json
```

The plan reports selected entries, exact payload bytes, touched shards,
metadata bytes read, request count, and source hashes. No tensor payload is
requested.

## Extract the fixture

Remove `--plan-only` and provide an output directory:

```sh
python mvp/inkling_remote_fixture.py \
  --revision 21152b5312c653be115f33a8342759064144e281 \
  --layers 0,2,5 \
  --experts '2:4,17,39,88,143,221;5:1,8,22,64,150,201' \
  --max-total-gib 8 \
  --out /parity/fixture-L0-L2-L5

python inkling/tools/inkling_fixture.py \
  --fixture /parity/fixture-L0-L2-L5 --verify
```

If routing later names an expert absent from the fixture, add that ID and
extract again. The expert selection is evidence-driven; it is not silently
expanded.

## Scope boundary

This is an evidence acquisition tool under `mvp/`. It does not alter the WASTE
runtime, generated patch, public loader, memory planner, or inference dispatch.
Promotion into `inkling/tools/` belongs with the first committed official-weight
comparison report and regenerated bundle.
