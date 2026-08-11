# Inkling-Small on WASTE 0.6.6 — generated bundle

This directory is a **build product** generated from `inkling/` and `integration/waste/`. Do not edit the patch by hand.

## Apply

```sh
git clone https://github.com/sqliteai/waste.git
cd waste
git checkout d9b919a791148b571e643d0af666bf19b4d733ab
git am /path/to/dist/waste-inkling-d9b919a/patches/0001-Add-the-Inkling-Small-runtime-foundation-to-WASTE.patch
PATH=/usr/bin:/bin make check
```

Verify the bundle first with `sha256sum -c SHA256SUMS` from this directory.
The applied Git tree must be `7fafa7942ee43abe6196481bd88c5621e40e6843`.

## What this bundle now contains

- the checked-in Inkling numerical/runtime foundation and final-head primitive;
- load-time bound trunk and row-backed vocabulary tensors;
- validated Inkling WEXP expert-bank geometry;
- routed-expert **storage** through WASTE's existing bounded `ecache` and
  native direct-I/O bank opener, including get, hold/release and routing hints;
- strict raw WEXP identity, geometry, codebook and optional CRC validation;
- an explicit zero-cache fallback that remains bounded and reports honest misses;
- public Inkling step/prefill/generation still fail-closed.

The hot-path follow-up is cached native WEXP -> WASTE's existing VQ/SIMD
gate/up/down kernels. This bundle does not add a decompressed-F32 expert engine.

## Regenerate

```sh
integration/waste/verify.sh WORKDIR
integration/waste/generate.sh WORKDIR2 dist/waste-inkling-d9b919a/patches
```

The generator pins commit metadata and emits a content-addressed tree; CI
independently re-applies this committed patch to pinned WASTE 0.6.6.
