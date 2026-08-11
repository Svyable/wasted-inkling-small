# Inkling-Small support in WASTE

Status: **the official Inkling-Small release is identified exactly; source
inspection, conversion planning, bounded staging, direct final-format routed
expert quantization, private Q8/Q4/VQ staged-directory opening, synthetic
quantized token-to-logits inference, and integration with WASTE at
`69315701f634648f7a790915a0a525ed8aabf218` are implemented. Official-weight
differential validation, public manifest/loader integration, tokenizer/chat
execution, and serving remain gated.**

## 1. Official release identity

Patch 10 targets the released `thinkingmachines/Inkling-Small` package rather
than an inferred family profile.

| Field | Official Small value | Source of truth |
|---|---:|---|
| license | Apache-2.0 | model card |
| total / active parameters | 276B / 12B | model card |
| BF16 checkpoint index size | 531,912,898,740 bytes | safetensors index |
| BF16 shards | 32 | safetensors index |
| decoder layers | 42 | config |
| hidden size | 4,096 | config |
| configured context | 1,048,576 | config |
| vocabulary / output rows | 201,024 / 200,058 | config |
| dense layers | 0-1 | `dense_mlp_idx=2` |
| routed experts / top-k | 256 / 6 | config |
| shared experts | 2 | config |
| dense / routed intermediate | 16,384 / 2,048 | config |
| global Q/KV heads | 32 / 8 | config |
| local Q/KV heads | 32 / 8 | config |
| head dimension | 128 | config |
| local window | 512 | config |
| global layers | 5, 11, 17, 23, 29, 35, 41 | config-derived |
| relative state / global extent | 16 / 1,024 | config |
| short-convolution kernel | 4 | config |
| logits width multiplier | 16 | config |
| global log-scaling floor / alpha | 128,000 / 0.1 | config |

The release also contains an official tokenizer, tiktoken rank file, special
-token map, processor configuration, and chat template. Those assets remove any
need to guess prompt formatting, but execution through WASTE remains disabled
until tokenizer/template parity tests are integrated.

## 2. Fail-closed release verifier

`tools/inkling_release.py` is the single source of truth for the released Small
profile. It separately checks:

1. **Architecture identity:** exact text, audio, and vision config fields.
2. **Package identity:** total indexed bytes, 32 unique shards, and the complete
   provider-raw text tensor-name inventory.
3. **Sidecar readiness:** chat template, processor config, tokenizer files, and
   special-token map.

A generic Inkling checkpoint can still use the parameterized planner, but it is
never labelled Inkling-Small when any pinned field differs.

## 3. Converter pipeline

### Inspection and planning

The source inspector reads config, safetensors index/header metadata, tokenizer,
processor, and chat assets without loading complete tensors. The conversion
plan derives every layer descriptor and validates exact source shapes.

### Resident trunk staging

`tools/inkling_trunk.py` writes one atomic canonical artifact per resident text
tensor. It:

- copies large embedding and unembedding tables in bounded chunks;
- deinterleaves fused dense and shared gate/up tensors in one source pass;
- canonicalizes Conv1d weights from `[channels,1,kernel]` metadata to
  `[channels,kernel]` without changing payload bytes;
- records source hashes, canonical-name CRC, payload CRC, shape, dtype, and
  aligned stored size;
- resumes verified artifacts without rereading source payloads.

### Routed expert staging

`tools/inkling_stage.py` reads one routed expert at a time and writes one aligned
BF16 gate/up/down record. Each record carries layer/expert identity, exact
geometry, and a payload CRC. Completed banks are source-hash-bound and
resumable.

### Direct final-format routed-expert quantization

`tools/inkling_vq.py` reads one official BF16 routed expert at a time and writes
WASTE's existing `WEXP` plus `WCBK` layouts directly. It does not require or
consume the intermediate BF16 expert stage.

The converter:

- samples normalized 8-value vectors from a bounded set of experts;
- trains separate residual codebooks for gate, up, and down matrices;
- uses deterministic residual Lloyd iterations;
- optionally uses `libwastevq` for the expensive final nearest-centroid encode;
- writes blocked indices in WASTE's existing `[block][vector][row][stage]`
  layout;
- writes FP16 per-output-row scales and the existing payload CRC;
- publishes one source- and training-identity-bound sidecar per layer;
- verifies every record, codebook ID, offset, identity, and CRC before reuse;
- publishes `vq-stage.json`, never `manifest.json`.

For the released Small dimensions at VQ3R, one routed expert record is exactly
9,457,664 bytes. A complete 256-expert layer is 2,421,161,984 bytes, and all 40
sparse layers occupy 96,846,479,360 bytes (about 90.20 GiB), before the trunk
and filesystem metadata. This is exact record geometry, not a measured final
container size.

Example:

```sh
python tools/convert_inkling.py --src MODEL --out FINAL-STAGE \
  --quantize-experts --vq-stages 3 --device cpu
```

The pure-Torch path is correctness-first. A locally built `libwastevq` is used
automatically for final expert encoding when available; official-checkpoint
conversion throughput still needs measurement on Windows and Linux.

### Private runtime-stage publication

`tools/inkling_runtime_stage.py` verifies the complete trunk and all 40 sparse
expert banks, then publishes a fixed little-endian `runtime-stage.bin`. It is a
converter-private index and is not a WASTE manifest.

The recommended laptop-oriented command sequence is:

```sh
python tools/convert_inkling.py --src MODEL --out STAGE --plan-only
python tools/convert_inkling.py --src MODEL --out STAGE --stage-trunk
python tools/convert_inkling.py --src MODEL --out STAGE --quantize-experts
python tools/convert_inkling.py --src MODEL --out STAGE --publish-runtime-vq-stage
```

This path never writes the 480+ GiB BF16 expert stage. The older
`--stage-experts` plus `--publish-runtime-stage` sequence remains available as
a BF16 parity/debug route and produces version 1 of the private index. Final
WEXP/WCBK banks produce version 2.

`manifest.json` remains absent throughout.

## 4. Architecture-private C runtime

The private scalar runtime implements the exact official text order:

1. row-backed token embedding;
2. embedding RMSNorm;
3. 42 parameterized decoder layers;
4. final RMSNorm;
5. division by the logits width multiplier;
6. independent row-backed unembedding;
7. output truncation to the unpadded vocabulary.

Each decoder layer implements:

- per-head Q/K RMSNorm;
- K and V depthwise short convolution;
- local-ring or global-linear grouped-query attention;
- learned hidden-state-conditioned relative bias;
- global context log scaling;
- attention output projection and branch short convolution;
- dense gated-SiLU or sparse routed-plus-shared MLP;
- joint normalization across selected routed experts and all shared experts;
- MLP branch short convolution and residuals.

All K/V caches, four convolution states per layer, and scratch buffers are
caller-owned and overflow-checked.

## 5. Patch 10 staged-directory opener

`src/inkling_private.[ch]` establishes the complete private byte-to-logits
boundary:

- parses only `runtime-stage.bin`, never `manifest.json`;
- validates entry sizes, counts, reserved bytes, safe relative paths, exact
  config/layer geometry, tensor count, duplicate names, and tensor shapes;
- recognizes the exact official Small profile when requested;
- retains embedding and unembedding artifacts as positional row readers;
- converts smaller BF16/F16/F32 trunk tensors to resident F32 in bounded chunks;
- opens every sparse-layer expert bank with one shared expert workspace;
- verifies record identity, geometry, file size, and optional CRC before use;
- allocates model state and scratch from the tested contract;
- runs private token-to-logits inference and supports deterministic reset;
- closes all file descriptors and allocations after any partial-open failure.

The stage reader includes POSIX and Windows positional-read code paths. Native
Windows execution is not yet claimed because MinGW and Windows filesystem tests
have not run.

## 6. Patch 11 final-format expert path

`src/inkling_wexp.[ch]` independently reads the final `WEXP` and per-layer
`WCBK` files. It validates codebook headers, dense absolute codebook IDs,
record identity, blocked-index geometry, offsets, file size, reserved fields,
and optional payload CRC before exposing one dequantized expert through the
same callback used by the private model.

The reader is intentionally scalar and materializes one F32 expert for parity.
The public runtime should ultimately reuse WASTE's optimized LUT/VQ compute
path rather than retain this validation implementation as the hot kernel.

A complete synthetic model test now streams routed experts from final WEXP
records and matches a reference constructed from an independent Python
dequantizer across multiple tokens.


## 6. Patch 12 final-VQ private runtime index

The private runtime index now has two explicitly versioned expert-bank modes:

- version 1: legacy converter-private BF16 expert staging;
- version 2: final WEXP/VQ expert banks plus per-layer WCBK codebooks.

`publish_runtime_vq_stage()` validates `trunk-stage.json`, every complete
`vq-stage.json` layer, codebook IDs, record geometry, file sizes, and CRCs,
then atomically replaces `runtime-stage.bin`. The v2 header records one shared
VQ geometry and each bank entry carries its absolute codebook base. Codebook
paths are deterministic (`codebooks-L{layer}.bin`) and are not guessed from
model branding.

`src/inkling_private.c` accepts both versions. In v2 it opens one WEXP bank and
resident codebook set per sparse layer, while all layers share a single record
buffer and three F32 expert matrices. For official Small VQ3R this removes the
roughly 480 GiB BF16 expert-stage dependency and bounds the active expert
workspace to roughly 105 MiB (three dequantized 2048x4096 matrices plus one
9.0 MiB record), excluding the small per-layer codebooks.

A complete synthetic staged-directory test now executes multiple tokens through
version-2 `runtime-stage.bin` and final WEXP/WCBK banks. Its logits match a
separate C model whose routed expert weights were produced by the independent
Python dequantizer. The version-1 BF16 private runtime remains covered for
backward compatibility.

## 7. Correctness evidence

Current tests cover:

- exact official release/profile/package recognition;
- generic-versus-official labelling;
- required official source tensor inventory;
- parameterized 42-layer Small planning;
- every local/global and dense/sparse descriptor;
- fused-weight transformations;
- bounded trunk and expert staging, CRC, resume, and cleanup;
- canonical Conv1d shape handling;
- C/PyTorch router, relative-bias, short-convolution, attention, complete-layer,
  and complete-model parity;
- resident and callback-backed vocabulary tables;
- strict canonical tensor binding;
- private runtime-index publication and corruption rejection;
- private staged-directory open, multiple token steps, legacy BF16 experts,
  reset, and bit-identical logits versus the already verified direct C model;
- version-2 private index publication and complete final WEXP/WCBK
  staged-directory logits versus independently dequantized routed weights;
- deterministic WEXP/WCBK output, source/training-bound resume, CRC corruption
  rejection, independent Python/C dequantization parity, and complete-model
  logits through the final-format expert callback.

The last comparison uses small synthetic BF16-rounded weights. It validates the
container-private binding and execution path, not official checkpoint quality.

## 8. Remaining gates before public support

### Official BF16 parity

Run the official Small tensors through the private opener and compare against a
trusted Transformers reference at:

- embedding normalization;
- every attention projection and convolution;
- per-layer attention output;
- router indices and normalized weights;
- routed and shared expert contributions;
- every layer output;
- final hidden state and logits;
- fixed-prompt generated token sequences.

### Laptop-oriented quantization

The private parity opener converts non-vocabulary trunk tensors to resident F32.
That is a correctness implementation, not the final laptop layout. Before
public Windows deployment:

- quantize resident attention, dense, router, and shared-expert tensors;
- connect the now-completed final-VQ private index path to the public expert
  cache without re-materializing experts outside the bounded callback workspace;
- measure codebook quality on official Small tensors and document quantized
  tolerances separately from BF16 parity;
- avoid requiring source, BF16 stage, and final container simultaneously;
- benchmark expert read amplification, top-6 cache behavior, and shared-expert
  compute.

### Public WASTE integration

- ✅ define the final Inkling manifest/schema extension — no new top-level
  shape, canonical `inkling.*` names, empty `tensor_prefix`; recorded in
  `docs/INKLING-CONTAINER.md` and written by
  `tools/make_inkling_container.py`;
- ✅ route `waste_plan_memory()` through the tested Inkling contract
  (`inkling_public.c`);
- ✅ route `waste_open` through the Inkling config build and tensor binding
  (`inkling_container.c`) — quantized matrices stay in their stored width and
  reach the arithmetic through `waste_matmul_t`; the two vocabulary tables
  stay on disk and are read a row at a time;
- connect the public expert cache to the Inkling callback — the bank metadata
  is validated and recorded, and no descriptor is opened yet;
- route step, reset, statistics, and errors to the Inkling model; until then
  every execution entry refuses and the suite asserts it;
- publish `manifest.json` last;
- retain fail-closed handling for unsupported variants.

### Tokenizer, chat, and serving

- load the official tiktoken/tokenizer assets;
- verify every official special token and placeholder ID;
- render the official chat template exactly;
- preserve explicit raw mode when no template is selected;
- add OpenAI-compatible text serving after raw token/logit parity.

### Multimodal

Official image and audio metadata is now present, but text parity remains the
priority. Vision and audio support must separately reproduce official patching,
normalization, dMel processing, placeholder expansion, and embedding counts.

## 9. Windows completion checklist

For a Windows laptop, the recommended first execution target is WSL2. Native
MinGW follows after:

- compiling all new sources with the upstream MinGW configuration;
- exercising files and offsets above 4 GiB;
- validating positional reads and NTFS atomic replacement;
- testing paths with spaces and Unicode;
- confirming antivirus does not invalidate throughput measurements;
- testing cancellation and resume during multi-hundred-gigabyte conversion;
- packaging the dependency-free runtime DLL/EXE;
- measuring realistic 2K-4K context memory and throughput.

No Windows-native performance or correctness result is claimed yet.

## Patch 13: quantized trunk artifact boundary

Patch 13 adds converter-private Q8G/Q4G artifacts for canonical resident trunk
matrices. It intentionally does not change `runtime-stage.bin`, `manifest.json`,
or the public loader. The purpose is to freeze and differentially validate the
byte format and C arithmetic before rebinding the full decoder.

The format stores a fixed 96-byte little-endian header, row-major quantized
payload, FP16 group scales, independent CRC32 values for payload and scales,
and 4 KiB file padding. Vocabulary tables and router matrices remain Q8 even
under a Q4 bulk policy. Norm vectors, scalar scales, and routing bias remain in
the canonical BF16/F32 stage.

For a matrix with `P` parameters and group size 128, approximate storage is:

* Q8G: `P + P/64` bytes, about 8.125 bits/weight.
* Q4G: `P/2 + P/64` bytes, about 4.125 bits/weight.

The C reader supports bounded row reads and direct Q8/Q4 matvec without
materializing an F32 matrix. Full-model use remains gated on a versioned runtime
index and end-to-end quantized-logit parity.

## Patch 14: quantized full-model private runtime

Patch 14 connects the Patch 13 Q8/Q4 tensor artifacts to the complete private
text runtime without changing the public WASTE loader or manifest format.
`runtime-stage.bin` version 3 now indexes a mixed trunk:

- Q8 row-backed embedding and unembedding tables;
- Q8 router matrices;
- Q4 or Q8 attention, dense-MLP, and shared-expert projections;
- canonical BF16/F32 norms, routing biases, scalar scales, relative-position
  profile banks, and four depthwise short-convolution kernels per layer;
- final WEXP/VQ routed-expert banks from Patch 11/12.

The decoder's resident-F32 ABI remains unchanged. A separate per-layer matrix
backend is consulted only when a matrix pointer is NULL, so all previous F32
and BF16 private-stage tests continue to exercise the original path. Shared
expert slices use bounded row ranges from their 3-D quantized artifacts.

Quantized payloads and FP16 group scales are loaded once at open for projection
throughput. Vocabulary tables remain on disk and are read one row at a time.
The runtime reports quantized resident bytes and canonical resident-F32 bytes
separately. Version-1 BF16 and version-2 WEXP/F32-trunk indexes remain accepted.

For the released Inkling-Small geometry at group 128, the exact format-size
estimate under the Patch 14 policy is:

- all supported projection matrices Q8: 5.629 GiB;
- Q4 bulk projections with Q8 vocabulary/router: 3.642 GiB;
- canonical resident F32 tensors (norms, relative profiles, convolutions,
  biases, scalar scales): approximately 9.52 MiB;
- 4K-context K/V plus four convolution states: approximately 0.362 GiB;
- largest decoder scratch: approximately 0.64 MiB;
- one shared WEXP decode workspace: approximately 105 MiB;
- per-layer VQ codebooks: approximately 1.4 MiB.

That places the model-state floor near 4.1 GiB before allocator overhead, the
caller output buffer, OS headroom, and any routed-expert cache. These are
configuration-derived format sizes, not measurements from a completed official
checkpoint conversion.

Synthetic end-to-end validation opens a version-3 staged directory, keeps the
vocabulary row-backed, executes resident Q8/Q4 matvecs and final WEXP experts,
and matches a separate C model populated from independently dequantized weights
across multiple tokens. This establishes byte-to-logit correctness for the
quantized storage path; official-weight layerwise parity is still required.

## Patch 15: bounded official-weight parity harness

Patch 15 adds a source-checkpoint fixture extractor and a deterministic activation
archive. It is designed for the released `thinkingmachines/Inkling-Small` BF16
package and refuses to label a source official unless the release verifier passes.

The extractor never copies the complete embedding or unembedding table. It copies
only selected decoder-layer tensors and explicitly selected axis-0 routed-expert
slices, under per-tensor and total byte ceilings. `fixture.json` is published last
and binds the fixture to SHA-256 hashes of `config.json` and the safetensors index.

Activation archives store named F32 or int32 tensors in independent CRC-protected
records. The comparison command reports missing entries, shape/dtype differences,
maximum absolute and relative floating-point errors, and exact routing-index
mismatches. This creates the interchange needed for official Python-versus-C
layerwise validation without requiring either implementation to parse the other's
internal memory layout.

Example extraction after the official BF16 checkpoint is present locally:

```powershell
python tools/inkling_parity.py `
  --src D:\models\Inkling-Small `
  --out D:\models\Inkling-Small-parity-L0-L2 `
  --layers 0,1,2 `
  --experts "2:4,17,39,88,143,221" `
  --max-total-gib 8
```

Example activation comparison:

```powershell
python tools/inkling_parity.py `
  --compare-reference D:\parity\python `
  --compare-candidate D:\parity\waste `
  --atol 1e-5 --rtol 1e-5 `
  --report D:\parity\report.json
```

A comparison mismatch exits with status 2. Patch 15 supplies the extraction and
interchange protocol; running the official reference and resolving any differences
remains a required completion gate.

## Patch 16 — executable activation tracing

Patch 16 turns the parity archive protocol into an executable two-sided trace.
The verified C layer/model APIs now have optional traced variants.  They emit
named F32 or int32 records after embedding normalization, layer normalization,
Q/K/V/R projection, K/V short convolution, attention, attention-branch short
convolution, post-attention normalization, dense MLP or sparse routing/MoE,
MLP-branch short convolution, complete layer output, final normalization, width
scaling, and logits.  Existing untraced entry points delegate with a null trace,
so production inference pays no callback cost.

`tools/inkling_trace.py` runs the converter-private staged C runtime and writes
these callbacks through the CRC-protected activation archive.  The output names
include token position and layer number, allowing multi-token cache and
convolution-state comparisons.

`tools/inkling_reference.py` is the official Transformers side.  It loads the
released model through `AutoModelForCausalLM`, registers hooks on the official
Inkling modules, performs incremental one-token decoding with cache, and writes
the same archive names.  Loading the full 532 GB checkpoint may require a
provider-supported or high-memory system; this patch does not claim that such a
run occurred in the development environment.

The expected real-weight workflow is now:

1. Produce a private quantized runtime stage.
2. Run `inkling_reference.py` for a fixed token sequence and selected layers.
3. Build/load a private Inkling shared library and run `inkling_trace.py` on the
   same tokens.
4. Compare both archives with `inkling_parity.py --compare-reference ...`.
5. Resolve BF16 semantic differences before setting Q8/Q4/VQ tolerances.

## Patch 18 — WASTE 0.6.3 production baseline

Patch 18 rebases the verified v16 source surface directly onto WASTE
`69315701f634648f7a790915a0a525ed8aabf218` (WASTE 0.6.3, API 1, format 0).
It is a consolidated replacement for historical patches 1–17, not a patch to
stack after them. The rebase retains upstream's cgroup-aware
`waste_usable_ram()` budget path, `test_memory`, sweep harness, and converter
worker limits alongside the Inkling sources and fail-closed dispatch seam.

The current WASTE build now compiles every private Inkling runtime source into
`libwaste`, builds the scalar `test_inkling` target, and runs that target in the
upstream test driver. The public Kimi loader remains fail-closed: Inkling is
recognized case-insensitively and returns `WASTE_E_UNSUPPORTED` before Kimi
memory planning or tensor binding. End-to-end tests cover `waste plan`,
`waste info`, and the direct internal loader path.

This is an upstream-compatible foundation, not a public-support claim. Public
dispatch remains behind the official-weight activation-parity gate above. The
model may be frontier; the loader is still required to know when it is out of
its depth.
