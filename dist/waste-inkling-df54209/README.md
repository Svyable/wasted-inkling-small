# Inkling-Small on WASTE 0.6.6 — generated bundle

This directory is a **build product**. It is regenerated from
[`inkling/`](../../inkling) and
[`integration/waste/`](../../integration/waste) by
`integration/waste/generate.sh`; nothing here is edited by hand.

## Apply

```sh
git clone https://github.com/sqliteai/waste.git
cd waste
git checkout df542092e1c17bc8a403ad25d3112d6f06e157ea
git am /path/to/dist/waste-inkling-df54209/patches/0001-Add-the-Inkling-Small-runtime-foundation-to-WASTE.patch
PATH=/usr/bin:/bin make check
```

Verify the bundle first:

```sh
cd /path/to/dist/waste-inkling-df54209
sha256sum -c SHA256SUMS
```

The applied Git tree must be `b74b3e5f4dc30016aa566202345f6bec68709cc5`.

## What the 0.6.6 rebase preserves

- the existing Inkling architecture seam and fail-closed public boundary;
- `waste_cfg.cpu_list` with an explicitly verified 64-bit ABI layout;
- safe `waste_cfg_init()` defaults, including no default CPU pinning;
- Linux disk benchmarking that probes cache bypass and labels fallback;
- an explicit, model-specific GB/token term rather than K3's hidden value;
- WQ_VQ4P format 8 and opt-in expert-parallel execution;
- converter controls for VQ shape and reclaim-safe staging;
- the Inkling guard that refuses the Kimi conversion path.

None of these features is enabled as an Inkling runtime by this bundle.
Public loading, generation, chat, and serving remain unsupported.

## Regenerate and verify from source

```sh
integration/waste/verify.sh
integration/waste/generate.sh WORKDIR dist/waste-inkling-df54209/patches
```

The authoritative invariant is the applied tree hash. The patch SHA-256
is also fixed here because the generator pins commit metadata and omits
the local Git signature.

## Evidence recorded at packaging

See [`TEST-RESULTS.txt`](./TEST-RESULTS.txt). The rebase gate proved:

- 33 passed, 0 failed, 13 skipped in upstream `make check`;
- all 168 OpenAI-compatible server checks passed;
- 12 Inkling translation units compiled with warnings as errors;
- the WASTE 0.6.6 `waste_cfg` layout and defaults matched exactly;
- the corrected diskbench contract compiled and exposed an explicit
  GB/token input;
- the generated patch reproduced the reviewed tree exactly.

Differential, sanitizer, and Windows cross-execution results belong to
the pull-request validation matrix and are not pre-claimed here.
