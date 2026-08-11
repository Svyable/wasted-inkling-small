# The public Inkling container

What a WASTE container holds when its `arch` is Inkling, and why each part of
it is where it is. This document exists because [G6 step 2](ROADMAP-V19.md#g6--public-dispatch)
made the loader read one: before that, the tensor names were an implementation
detail of a converter-private index, and now they are a contract.

The executable version of this document is
[`inkling/tools/make_inkling_container.py`](../inkling/tools/make_inkling_container.py),
which writes a complete container in a few hundred kilobytes, and
[`inkling/tests/test_inkling_container.c`](../inkling/tests/test_inkling_container.c),
which loads what it writes. Where prose and those two disagree, they are right.

## 1. No new top-level shape

Format v0 already carries everything Inkling needs:

```
manifest.json      format_version, arch, tensor_prefix, config,
                   expert_quant, trunk[], layers[]
trunk.bin          every named tensor at its declared offset and width
codebooks.bin      the VQ codebooks the expert banks index into
experts-L<N>.bin   one bank per sparse layer
```

`arch` must classify as Inkling — `waste_arch_classify()` lowercases and
prefix-matches `"inkling"`, so `inkling`, `Inkling-Small` and
`InklingForConditionalGeneration` all reach the same branch.

`tensor_prefix` **must be empty**. Kimi uses it because its tensors keep their
Hugging Face names under a `language_model.` prefix; Inkling's are canonical
already, and a prefix would describe tensors the binding will not find. A
container that sets one is refused rather than searched twice.

## 2. `config`

The release's text config, flattened, with the multimodal wrapper kept under
`_outer` — exactly the K3 convention. A raw release config with a nested
`text_config` is also accepted, and the loader and the planner resolve it
through the same function so a container cannot be planned as one model and
opened as another.

Required, with no default standing in for a missing one:

| key | meaning |
| --- | --- |
| `num_hidden_layers`, `hidden_size`, `vocab_size` | the obvious three |
| `model_max_length` | context ceiling; a larger request is refused at plan and at load |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | global-attention geometry |
| `sliding_window_size` | local-attention window, and the local layers' relative extent |
| `d_rel`, `rel_extent` | learned relative-position projection |
| `sconv_kernel_size` | short-convolution kernel width |
| `dense_intermediate_size` | dense MLP width |
| `intermediate_size` | **routed** expert width — see [CONFIG-SCHEMA-RESOLUTION.md](CONFIG-SCHEMA-RESOLUTION.md) before assuming this means what Transformers means by it |
| `n_routed_experts`, `num_experts_per_tok`, `n_shared_experts` | the MoE |
| `rms_norm_eps`, `route_scale`, `logits_mup_width_multiplier` | must be positive |

Optional, with the default stated: `unpadded_vocab_size` (defaults to
`vocab_size`), `dense_mlp_idx` (0), `local_layer_ids` (empty — every layer
global), `log_scaling_n_floor` (0), `log_scaling_alpha` (0), `eos_token_id`
(0, meaning "not stated").

Validation belongs to `waste_inkling_config_build()`: divisibility of heads by
KV heads, `top_k ≤ n_routed`, `unpadded_vocab ≤ vocab`, duplicate local-layer
ids, kernel bounds. The manifest reader does not second-guess it.

## 3. `trunk[]` — canonical names

Names are the private runtime index's, verbatim. That is not tidiness deferred:
`inkling_bind.c` binds one name set, and reusing it means the private index and
the public manifest are the same contract written twice rather than two
contracts to keep in step.

```
inkling.embed                          [vocab][hidden]
inkling.embed_norm                     [hidden]
inkling.final_norm                     [hidden]
inkling.unembed                        [unpadded_vocab][hidden]

inkling.layer.<L>.input_norm           [hidden]
inkling.layer.<L>.post_attention_norm  [hidden]
inkling.layer.<L>.q                    [heads*head_dim][hidden]
inkling.layer.<L>.k                    [kv_heads*head_dim][hidden]
inkling.layer.<L>.v                    [kv_heads*head_dim][hidden]
inkling.layer.<L>.r                    [heads*d_rel][hidden]
inkling.layer.<L>.o                    [hidden][heads*head_dim]
inkling.layer.<L>.q_norm               [head_dim]
inkling.layer.<L>.k_norm               [head_dim]
inkling.layer.<L>.rel_proj             [d_rel][relative_extent]
inkling.layer.<L>.k_sconv              [kv_heads*head_dim][sconv_kernel_size]
inkling.layer.<L>.v_sconv              [kv_heads*head_dim][sconv_kernel_size]
inkling.layer.<L>.attn_sconv           [hidden][sconv_kernel_size]
inkling.layer.<L>.mlp_sconv            [hidden][sconv_kernel_size]

# dense layers, L < dense_mlp_idx
inkling.layer.<L>.mlp.gate             [dense_intermediate][hidden]
inkling.layer.<L>.mlp.up               [dense_intermediate][hidden]
inkling.layer.<L>.mlp.down             [hidden][dense_intermediate]
inkling.layer.<L>.mlp.global_scale     [1]

# sparse layers, L >= dense_mlp_idx
inkling.layer.<L>.router.weight            [n_routed + n_shared][hidden]
inkling.layer.<L>.router.correction_bias   [n_routed]
inkling.layer.<L>.router.global_scale      [1]
inkling.layer.<L>.shared.gate              [n_shared][moe_inter][hidden]
inkling.layer.<L>.shared.up                [n_shared][moe_inter][hidden]
inkling.layer.<L>.shared.down              [n_shared][hidden][moe_inter]
```

`relative_extent` is `sliding_window_size` on a local layer and `rel_extent` on
a global one. Every tensor must appear exactly once; a duplicate name is an
error, not a last-wins.

Routed expert matrices are deliberately absent. They live in the banks and
arrive through the expert callback, which is the same shape as WASTE's existing
expert cache.

### Which tensors may be quantized

Not a style question — the binder dereferences some of these directly and hands
others to a matrix backend, and only the second kind can stay packed.

| may be `fmt` 2/3 (Q8G/Q4G) | must be `fmt` 0 (F32) |
| --- | --- |
| `q`, `k`, `v`, `r`, `o` | every norm: `embed_norm`, `final_norm`, `input_norm`, `post_attention_norm`, `q_norm`, `k_norm` |
| `mlp.gate`, `mlp.up`, `mlp.down` | `rel_proj` |
| `router.weight` | the four `*_sconv` kernels |
| `shared.gate`, `shared.up`, `shared.down` | `mlp.global_scale`, `router.global_scale`, `router.correction_bias` |
| `embed`, `unembed` | |

The four short-convolution kernels and `rel_proj` are two-dimensional and are
still in the right-hand column: they are not matmuls, and nothing consults a
backend for them. A container that quantizes one fails the binding by name.

### The two vocabulary tables

`inkling.embed` and `inkling.unembed` are the only tensors of which a single
row is read per token, and Inkling has two of them where Kimi has one — the
unembedding is independent, not tied.

When they are quantized, the loader leaves them on disk and preads a row, and
the planner leaves them out of the resident floor. When they are F32 the loader
makes them resident, because there is no row unpacker for F32 on disk, and the
planner counts them. One function, `waste_arch_row_backed()`, is why those two
decisions cannot drift apart, and `tests/test_inkling_container.c` asserts the
planned trunk equals the resident set to the byte.

A real container quantizes both. At Inkling-Small's geometry, Q8G, that is
1.6 GiB the resident floor does not carry.

## 4. `expert_quant`

`{stages, vec_dim, entries, index_block, bits_per_weight}`, read under the same
bounds the Kimi loader applies: 1–8 stages, `vec_dim` 1–64, 1–256 entries, and
`index_bits` 8 unless it is the 6/4/64 packing. The expected operating point is
VQ3R — three stages, `vec_dim` 8, 256 entries — pending
[G2](ROADMAP-V19.md#g2--quantized-tolerance-measured-not-asserted).

## 5. `layers[]`

An array, one entry per **sparse** layer, in ascending layer order:

```json
{"file": "experts-L2.bin", "layer": 2, "experts": 256,
 "bytes": 2421161984, "codebook_base": 0}
```

`layer` is optional and checked when present. `file` must name a file inside
the container — no separators, no `..`. `bytes` must divide by `experts`; the
quotient is the record stride, and it is what the planner derives the minimum
expert cache from.

An array rather than the object-keyed-by-layer shape the Kimi path accepts,
because a short array is short and a sparse object is merely quiet.

## 6. What the loader does today, and what it does not

Promoted:

- geometry, from `config`, through the reader the planner shares;
- `trunk.bin`, through upstream's own `load_trunk` — one format, one reader;
- binding, through `waste_inkling_bind_weights_ex_backend()`, with quantized
  matrices left in their stored width and reached through `waste_matmul_t`;
- bank metadata, validated and recorded, no descriptor opened;
- `waste plan` and `waste info`.

Still refused, and the refusal is asserted rather than assumed:
`waste_model_step`, `waste_model_prefill`, `waste_eval`, `waste_generate`. The
expert cache is step 3 and execution is step 4. Loading decides where bytes
live; it does not decide what a token is, and the second claim is the one this
port has not yet earned.
