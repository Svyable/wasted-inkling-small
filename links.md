# Inkling-Small port source pack

This is the working source index for porting Thinking Machines Lab's
Inkling-Small into WASTE. It separates conversion and behavior authorities from
implementation references, runtime comparison targets, and secondary reading.

## Authority and revision policy

Use sources in this order when they disagree:

1. the official Inkling-Small checkpoint at the immutable revision recorded by
   this repository;
2. official checkpoint metadata, tokenizer assets, processor configuration, and
   chat template at that same revision;
3. the pinned WASTE source and generated integration bundle;
4. official framework implementations as differential-test references;
5. secondary and community material for orientation only.

Mutable `main` links below are for discovery. Reproducible evidence must use
immutable revisions and verify the recorded hashes. The current repository
binds its official model evidence to
[`thinkingmachines/Inkling-Small@21152b5312c653be115f33a8342759064144e281`](https://huggingface.co/thinkingmachines/Inkling-Small/tree/21152b5312c653be115f33a8342759064144e281)
and its WASTE integration to
[`sqliteai/waste@d9b919a791148b571e643d0af666bf19b4d733ab`](https://github.com/sqliteai/waste/tree/d9b919a791148b571e643d0af666bf19b4d733ab).
See [BF16 evidence](docs/BF16-EVIDENCE.md),
[reference profiles](docs/INKLING-REFERENCE-PROFILES.md), and the
[Linux AVX2 reference profile](docs/OFFICIAL-REFERENCE-PROFILE-LINUX-AVX2.json)
before making an exact-parity claim.

The official checkpoint is the conversion and behavior authority. Do not use a
runtime recipe, quantization, mirror, news article, or generated fixture as a
substitute for the official BF16 tensors and official repository assets.

## Start here

- [WASTE upstream](https://github.com/sqliteai/waste)
- [Inkling-Small official model card](https://huggingface.co/thinkingmachines/Inkling-Small)
- [Official file browser](https://huggingface.co/thinkingmachines/Inkling-Small/tree/main)
- [Official revision history](https://huggingface.co/thinkingmachines/Inkling-Small/commits/main)
- [Official Inkling model collection](https://huggingface.co/collections/thinkingmachines/inkling)
- [Thinking Machines Inkling announcement](https://thinkingmachines.ai/news/introducing-inkling/)
- [Hugging Face Inkling announcement](https://huggingface.co/blog/thinkingmachines-inkling)

The official repository currently exposes the principal checkpoint as 32
Safetensors shards, plus the model index, model configuration, chat template,
processor and tokenizer assets, evaluation results, and a separate MTP
checkpoint. Lock all automated acquisition to an immutable revision.

## Verified architecture notes

The official model card and configuration describe Inkling-Small as an
Apache-2.0, open-weight, multimodal autoregressive model with 276B total and 12B
active parameters. Its text backbone is a 42-layer sparse MoE decoder. Each
token uses 6 of 256 routed experts plus 2 shared experts, 32 attention heads and
8 KV heads, 512-token sliding-window local attention with periodic global
layers, and short convolution with kernel size 4.

Inkling uses learned **relative-attention features**, not RoPE. The official
configuration exposes `d_rel` and `rel_extent`, and the official architecture
description explicitly distinguishes relative attention from RoPE. Port and
test the relative projection and distance-dependent attention terms; do not add
a RoPE path based on a generic decoder assumption.

- [Official architecture section](https://huggingface.co/thinkingmachines/Inkling-Small#3-model-properties)
- [Official `config.json`](https://huggingface.co/thinkingmachines/Inkling-Small/blob/main/config.json)
- [NVIDIA NeMo Inkling coverage](https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/thinkingmachines/inkling)
- [vLLM Inkling-Small recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-Small)

## WASTE internals

Read this group before changing code. The intended boundary is to generalize
model metadata, conversion, the forward path, and prompt/multimodal adaptation
while preserving the container, bounded expert cache, and platform I/O
machinery wherever its contracts still fit.

- [README](https://github.com/sqliteai/waste/blob/main/README.md)
- [Repository map and contribution guidance](https://github.com/sqliteai/waste/blob/main/CLAUDE.md)
- [Container format](https://github.com/sqliteai/waste/blob/main/docs/FORMAT.md)
- [Execution engine](https://github.com/sqliteai/waste/blob/main/docs/ENGINE.md)
- [Measured performance work](https://github.com/sqliteai/waste/blob/main/docs/EFFICIENCY.md)
- [Prior experiments and constraints](https://github.com/sqliteai/waste/blob/main/docs/LEARNED.md)
- [Kimi implementation notes](https://github.com/sqliteai/waste/blob/main/docs/K3.md)
- [C public API](https://github.com/sqliteai/waste/blob/main/src/waste.h)
- [Forward-path implementation](https://github.com/sqliteai/waste/blob/main/src/model.c)
- [Bounded expert cache](https://github.com/sqliteai/waste/blob/main/src/ecache.c)
- [Model/container open and parsing](https://github.com/sqliteai/waste/blob/main/src/container.c)
- [Platform and direct-I/O layer](https://github.com/sqliteai/waste/blob/main/src/platform.h)
- [SIMD CPU kernels](https://github.com/sqliteai/waste/tree/main/src)
- [CLI](https://github.com/sqliteai/waste/tree/main/cli)
- [Converters and validation scripts](https://github.com/sqliteai/waste/tree/main/tools)
- [Core tests](https://github.com/sqliteai/waste/tree/main/tests)
- [Server package](https://github.com/sqliteai/waste/tree/main/serve)
- [Server reference](https://github.com/sqliteai/waste/blob/main/docs/SERVE.md)
- [Server tests](https://github.com/sqliteai/waste/tree/main/tests/serve)

The reusable WASTE shape is a resident trunk plus disk-resident expert banks,
bounded user-space caching, direct I/O, and aligned expert records intended to
make selected experts predictable positional reads.

## Official checkpoint inputs

Parse these files directly and lock them to the same immutable model revision.
The `main` URLs are convenient browsers, not reproducibility pins.

- [Model configuration](https://huggingface.co/thinkingmachines/Inkling-Small/blob/main/config.json)
- [Chat template](https://huggingface.co/thinkingmachines/Inkling-Small/blob/main/chat_template.jinja)
- [Processor configuration](https://huggingface.co/thinkingmachines/Inkling-Small/blob/main/processor_config.json)
- [Special-token map](https://huggingface.co/thinkingmachines/Inkling-Small/blob/main/special_tokens_map.json)
- [Weight index](https://huggingface.co/thinkingmachines/Inkling-Small/blob/main/model.safetensors.index.json)
- [Weight shards](https://huggingface.co/thinkingmachines/Inkling-Small/tree/main)
- [MTP checkpoint](https://huggingface.co/thinkingmachines/Inkling-Small/blob/main/mtp.safetensors)
- [Tokenizer directory](https://huggingface.co/thinkingmachines/Inkling-Small/tree/main/tiktoken)
- [Evaluation prompts and results](https://huggingface.co/thinkingmachines/Inkling-Small/tree/main/.eval_results)

Start with a text-only BF16 conversion of the principal checkpoint. Treat the
separate 4.46 GB `mtp.safetensors` file as a later, optional artifact unless MTP
or speculative decoding is explicitly in scope.

## Architecture implementation references

These projects are useful for locating mature Inkling implementations and
cross-checking tensor semantics. They do not outrank the pinned official
checkpoint.

- [SGLang repository](https://github.com/sgl-project/sglang)
- [vLLM repository](https://github.com/vllm-project/vllm)
- [Transformers repository](https://github.com/huggingface/transformers)
- [Transformers documentation](https://huggingface.co/docs/transformers/index)

## Port coverage

| System component | Text MVP | Full Inkling-Small |
| --- | :---: | :---: |
| Tokenizer and special tokens | Yes | Yes |
| Jinja chat template and tool-call formatting | Yes | Yes |
| 42-layer decoder, RMSNorm/residual path, relative attention, and logits | Yes | Yes |
| Hybrid local/global attention, KV state, and short convolution | Yes | Yes |
| Router, top-6 selection, and 256 routed experts | Yes | Yes |
| Two shared experts kept resident | Yes | Yes |
| BF16 loading and first container conversion | Yes | Yes |
| WASTE expert quantization and container packing | Yes | Yes |
| Image patch encoder and image placeholder expansion | No | Yes |
| Audio processing/tokenization and placeholder expansion | No | Yes |
| MTP/speculative decoding | No | Optional |

The official model accepts text, image, and audio and generates text. Finish
reproducible text logits and text chat behavior before adding image, audio, or
MTP paths.

## Conversion and reference validation

- [Hugging Face Hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [`snapshot_download`](https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.snapshot_download)
- [Hugging Face Xet storage](https://huggingface.co/docs/hub/xet)
- [Safetensors repository and format](https://github.com/huggingface/safetensors)
- [Safetensors documentation](https://huggingface.co/docs/safetensors/index)
- [PyTorch serialization guidance](https://pytorch.org/docs/stable/notes/serialization.html)
- [`torch.testing.assert_close`](https://pytorch.org/docs/stable/testing.html)
- [Transformers `AutoProcessor`](https://huggingface.co/docs/transformers/main_classes/processors)
- [Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation)

Use BF16 as the first converter and numerical reference. Add NVFP4 ingestion
only after named intermediate boundaries and final logits agree, so numerical
format is not a second uncontrolled debugging axis.

## Existing runtime baselines

Use these for differential checks of prompt rendering, token IDs, selected
routes, intermediate values where exposed, generated output, and GPU
performance. They are comparison targets, not conversion authorities.

- [Official vLLM quickstart](https://huggingface.co/thinkingmachines/Inkling-Small#vllm)
- [Official SGLang quickstart](https://huggingface.co/thinkingmachines/Inkling-Small#sglang)
- [Official Transformers quickstart](https://huggingface.co/thinkingmachines/Inkling-Small#transformers)
- [vLLM recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-Small)
- [Hugging Face local-app integration](https://huggingface.co/thinkingmachines/Inkling-Small)
- [Unsloth Inkling-Small GGUF search](https://huggingface.co/models?search=Inkling-Small-GGUF)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Multimodal phase

Do not begin this phase until text logits, tokenizer behavior, and text chat are
reproducible.

- [Official processor configuration](https://huggingface.co/thinkingmachines/Inkling-Small/blob/main/processor_config.json)
- [Transformers multimodal chat templates](https://huggingface.co/docs/transformers/chat_templating_multimodal)
- [Pillow](https://pillow.readthedocs.io/)
- [WAV format reference](https://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html)
- [torchaudio](https://pytorch.org/audio/stable/index.html)
- [FFmpeg](https://ffmpeg.org/documentation.html)
- [WASTE image implementation](https://github.com/sqliteai/waste/blob/main/src/image.c)
- [WASTE vision implementation](https://github.com/sqliteai/waste/blob/main/src/vision.c)

For full coverage, include the official model-card ranges in validation: image
dimensions from 40 px through 4096 px, and WAV audio sampled at 16 kHz, with the
recommended audio duration kept below roughly two minutes.

## Serving and Streamlit

- [WASTE HTTP server](https://github.com/sqliteai/waste/tree/main/serve)
- [WASTE server documentation](https://github.com/sqliteai/waste/blob/main/docs/SERVE.md)
- [Native engine ctypes bridge](https://github.com/sqliteai/waste/blob/main/serve/engine.py)
- [HTTP server implementation](https://github.com/sqliteai/waste/blob/main/serve/server.py)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [OpenAI chat-completions API](https://platform.openai.com/docs/api-reference/chat)
- [Streamlit chat widgets](https://docs.streamlit.io/develop/api-reference/chat)
- [Streamlit session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)

The intended deployment shape is:

`Streamlit -> localhost WASTE OpenAI-compatible server -> libwaste -> Inkling-Small.waste on internal NVMe`

Keep the model in the long-lived server process rather than in the Streamlit
script, which reruns during interaction. This repository's current public
Inkling loader, generation, chat, and serving paths remain fail-closed until the
documented promotion gates pass.

## Hardware and I/O

- [WASTE platform support](https://github.com/sqliteai/waste#platforms)
- [WASTE efficiency report](https://github.com/sqliteai/waste/blob/main/docs/EFFICIENCY.md)
- [Linux `O_DIRECT`](https://man7.org/linux/man-pages/man2/open.2.html)
- [macOS `fcntl`](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)
- [Windows file buffering and direct I/O](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)
- [MinGW-w64](https://www.mingw-w64.org/)
- [Apple Metal](https://developer.apple.com/metal/)
- [Apple Accelerate](https://developer.apple.com/documentation/accelerate)

The first port target is CPU/NVMe execution. WASTE's mainline path uses SIMD CPU
kernels, bounded DRAM caching, and NVMe-backed expert banks; a GPU backend is
not the default WASTE execution route.

## Methodical execution order

1. **Pin revisions.** Record WASTE and Inkling-Small SHAs; hash and retain the
   configuration, weight index, processor configuration, tokenizer assets, and
   chat template used by each evidence run.
2. **Map every tensor.** Generate an explicit converter map for resident,
   local/global attention, relative projection, convolution, router, shared
   expert, routed expert, vision, and audio tensors.
3. **Build a tiny synthetic model.** Exercise local and global attention,
   relative attention, short convolution, routing, shared experts, residuals,
   and cache lifecycle without the real checkpoint.
4. **Convert BF16 first.** Convert a text-only official BF16 container and keep
   a Python reference harness for embeddings, layer outputs, expert IDs and
   weights, final hidden state, and logits.
5. **Design expert placement.** Keep each router and shared experts resident;
   pack every routed expert as a complete aligned record so each selected top-6
   payload can be fetched predictably.
6. **Prove prompt fidelity.** Differential-test rendered Jinja prompts, special
   tokens, tool-call boundaries, and token IDs against the official processor.
7. **Promote CLI/server only after gates.** Add ordinary `waste run`, `waste
   chat`, and `/v1/chat/completions` behavior only after numerical and prompt
   gates pass; preserve fail-closed public behavior until then.
8. **Benchmark the intended system.** Measure container size, resident trunk,
   expert bytes per token, RAM floor, cache-hit curve, cold/warm decode, prefill,
   and 4K/32K/128K context behavior.
9. **Add multimodal paths.** Port image preprocessing and the hierarchical patch
   encoder, then audio tokenization; compare intermediate embeddings before
   relying on generated text.
10. **Evaluate optional MTP.** Decide whether the separate MTP artifact merits
    speculative decoding only after the ordinary converter/runtime is correct
    and debuggable.

## Additional official and ecosystem reading

- [Thinking Machines Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Hugging Face Inkling announcement](https://huggingface.co/blog/thinkingmachines-inkling)
- [NVIDIA NeMo Inkling model coverage](https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/thinkingmachines/inkling)

## Secondary and community reading

The links in this section may be useful for orientation, deployment anecdotes,
or discovering questions to test. They are not authorities for tensor mapping,
architecture, tokenizer/chat behavior, numerical parity, licensing, or release
claims.

- [MarkTechPost: Inkling-Small release overview](https://www.marktechpost.com/2026/08/02/thinking-machines-lab-releases-inkling-small-276b-open-weights-multimodal-moe-model/)
- [YouTube: Inkling model overview](https://www.youtube.com/watch?v=EOK_haQSMIA)
- [DataCamp: running Inkling locally](https://www.datacamp.com/tutorial/how-to-run-thinking-machines-inkling-locally)
- [ExplainX: Inkling-Small overview](https://explainx.ai/blog/inkling-small-thinking-machines-open-weights-july-2026)
- [Eigent: Inkling-Small overview](https://www.eigent.ai/blog/thinking-machines-inkling-small-open-weights-model)
- [Reddit r/LocalLLaMA: Inkling-Small thread](https://www.reddit.com/r/LocalLLaMA/comments/1vb16gj/inklingsmall_by_thinkingmachines/)
- [Reddit r/LocalLLaMA: Inkling release thread](https://www.reddit.com/r/LocalLLaMA/comments/1uxdv34/thinking_machines_releases_first_openweight_model/)
- [Kie.ai: Inkling architecture overview](https://kie.ai/blog/what-is-inkling)
- [VentureBeat: Inkling-Small release coverage](https://venturebeat.com/technology/thinking-machines-debuts-inkling-small-open-source-ai-model-nearing-performance-of-predecessor-at-about-1-4-size)
- [Sebastian Raschka: Inkling architecture and benchmark notes](https://sebastianraschka.com/blog/2026/inkling-architecture-benchmark-notes.html)

## Maintenance rules

- Add immutable revision links beside mutable discovery links when a source is
  used by executable evidence.
- Record artifact hashes and reference-profile metadata in the evidence files,
  not only in prose or PR comments.
- Prefer primary sources and implementation code over summaries.
- Label runtime recipes and quantized community artifacts as comparison targets.
- Keep public Inkling execution fail-closed while any promotion gate is open.
