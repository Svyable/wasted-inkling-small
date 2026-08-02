# What upstream WASTE blocks, and what it gives us for free

The three existing documents look inward: what this port has, what it proves,
how it is built. This one looks the other way. WASTE is a **Kimi** engine that
happens to be well factored, and the question that decides the promotion design
is: which parts of it can carry Inkling, and which parts are shaped like K3 all
the way down?

Read on the applied tree at `69315701f634648f7a790915a0a525ed8aabf218`
(WASTE 0.6.3). Line numbers are that tree's.

The short answer: **the container format and the I/O machinery are
architecture-neutral and reusable as-is. The model struct, the forward
dispatch, the memory formula, and the tokenizer's pre-tokenizer are Kimi-shaped
and are the real work.** That split is why the promotion boundary belongs where
`arch.c` already puts it.

---

## 1. What is holding us back

### 1.1 `waste_model` has nowhere to put a KV cache — the biggest one

`src/model.h:117-127` is the whole state of a loaded model:

```c
float *S[WASTE_MAX_LAYERS];        /* KDA recurrent state              */
float *conv[WASTE_MAX_LAYERS];     /* KDA short-conv rings (3 x C x K-1) */
float *latcache[WASTE_MAX_LAYERS]; /* [kv_cap][kv_lora + qk_rope]      */
int n_kv[WASTE_MAX_LAYERS], kv_cap;
```

There is no K/V cache in WASTE, because neither Kimi attention needs one:
KDA keeps an O(1) recurrent state, and MLA caches the *latent* and absorbs
`kv_b_proj` into the query and output rather than expanding per-head K and V
(`src/model.c:2027`, `mla_layer`).

Inkling needs exactly the thing that is absent: per-layer K and V at
`[tokens][kv_heads][head_dim]`, grouped-query, with a **sliding window on 35 of
42 layers and full context on the other 7**. It also needs **four** short
convolution rings per layer — K, V, the attention branch, and the MLP branch —
where `conv[]` is sized for three.

This is not a field to add. Local layers cap at `sliding_window` while global
layers grow to `ctx_tokens`, so the per-layer allocation is heterogeneous in a
way `kv_cap` does not express. The port already models this correctly in
`waste_inkling_plan_decode_memory()`, which sizes each layer from
`l->is_local ? min(ctx, sliding_window) : ctx`.

**Consequence:** Inkling cannot be a third branch inside `waste_model` without
either bloating that struct with a second state discipline or lying about what
`kv_cap` means. It wants to be a sibling model behind a dispatch.

### 1.2 The forward path is a two-way switch with no third slot

`src/model.c:3239`:

```c
if (c->kda_layer[L]) kda_layer(m, L, norm, resid);
else                 mla_layer(m, L, norm, resid, pos);
```

Every layer is KDA or MLA. There is no architecture tag on the layer, only a
boolean array `kda_layer[WASTE_MAX_LAYERS]`. Inkling's local/global GQA is a
third kind, and its *sparse/dense* split (`dense_mlp_idx=2`) is a second,
orthogonal axis that `first_dense` half-expresses.

### 1.3 Tensors are found by hardcoded Kimi names, on every step

`src/model.c:3227` and throughout:

```c
snprintf(b, sizeof b, "%smodel.layers.%d.input_layernorm.weight", c->prefix, L);
waste_rmsnorm(norm, m->x, waste_find(m, b)->data, hid, c->eps);
```

`waste_find` is a name lookup per tensor per layer per token. Two problems:
the names are K3's, and the lookup is on the hot path. The port already does
this better — `inkling_bind.c` binds every canonical name to a struct field
**once at load**, with exact shape checks and duplicate rejection, and the
decode loop touches pointers. Promotion should not inherit `snprintf` in the
inner loop.

### 1.4 `waste_config` is a Kimi feature bag

`src/model.h:47-77` carries `kv_lora`, `q_lora`, `qk_nope`, `qk_rope`,
`v_head`, `kda_heads`, `kda_dim`, `conv_k`, `latent_dim`, `latent_norm`,
`attn_res_block`, `full_rank_gate`, `gate_lower_bound`, `mla_output_gate`,
`act_situ`, `situ_beta`, `situ_linear_beta`.

Inkling needs `d_rel`, `rel_extent`, `sliding_window`, `log_scaling_n_floor`,
`log_scaling_alpha`, `logits_mup_width_multiplier`, `shared_expert_sink`,
`norm_after_topk`, and per-layer local/global geometry. The union of the two is
a struct where most fields are zero for any given model and no reader can tell
which. `waste_inkling_config` already exists and is validated up front; keep
them separate.

### 1.5 `waste_plan_memory()` computes an MLA floor

`src/waste.c:270-271, 288, 333` read `kv_lora_rank` and `qk_rope_head_dim`
straight out of the manifest and size the cache as
`(kv_lora + qk_rope) * 4` per token. Fed an Inkling manifest this produces a
confident, wrong, *plausible* number — which is precisely why the guard at
`src/waste.c:179` refuses before reading a single dimension.

This one is already solved: `waste_inkling_plan_decode_memory()` is written,
field-by-field, overflow-checked, and tested dependency-light in CI. It is
waiting for a caller.

### 1.6 The tokenizer's pre-tokenizer is hand-transliterated and not selectable

This is the one that is easy to miss and sits directly on the path to
`waste chat`.

`src/tokenizer.c:331`, `next_piece()`, implements a specific tiktoken
pre-tokenization pattern **by hand, in C, over decoded codepoints** — a
`\p{Han}+` branch, the `(?i:'s|'t|'re|'ve|'m|'ll|'d)` contractions, a letter
run with an optional leading non-letter, `\p{N}{1,3}`, and a
`" ?[^\s\p{L}\p{N}]+[\r\n]*"` branch. The header says why
(`src/tokenizer.h:9-13`): rather than pull in a regex engine, the classes are
implemented directly.

Everything else in the tokenizer is reusable — the rank-file loader accepts
`tokenizer.model` or `tiktoken.model` (`src/tokenizer.c:176`), `specials.json`
is read separately (`:109`), and `waste_tok_set_eos()` lets the container state
its own EOS. Inkling ships all of those assets.

**But the pattern is a compile-time constant.** If Inkling's pre-tokenization
regex differs from Kimi's in any branch, G4 is not "load the assets" — it is
"hand-write a second `next_piece` and select it per container," with a
correctness bar of byte-exact agreement against Python tiktoken over a corpus.

**This is uncosted work on the critical path, and it is cheap to resolve:**
read the pattern string out of the released tokenizer assets and diff it
against the branches above. One afternoon, no GPU, no checkpoint conversion —
and it either removes a gate or names a real task. See §4.

---

## 2. What WASTE gives us for free

Substantially more than the blocker list suggests, and this is the reason not
to build a second engine.

| Component | Why it carries over unchanged |
| --- | --- |
| **Container format v0** (`src/waste_format.h`) | `trunk[]` entries are `name/fmt/off/shape/group/scale_off/bytes`; `layers[]` carry `file/experts/bytes/codebook_base`; `expert_quant` is parameterized (`stages`, `vec_dim`, `entries`, `index_block`). Inkling's short-conv kernels and relative-position banks are *just more named tensors*. **No format change is needed** — see [CODEBASE-MAP.md](CODEBASE-MAP.md) §8. |
| **Expert cache** (`src/ecache.[ch]`) | Keyed by `layer<<16 \| expert` with a `fetch(user, layer, expert, dst)` callback and a fixed `rec_bytes`. Nothing in it knows what an expert *is*. Inkling's expert callback is already that shape. LFRU policy, pinning, read-ahead, and the O_DIRECT discipline all apply. |
| **VQ decode + SIMD** (`src/vq.c`, `simd_avx2.c`, `simd_avx512.c`, `kda_neon.c`) | VQ3R/VQ2R dequant and the quantized matvec kernels are format-level, not architecture-level. The scalar `inkling_wexp.c` is a validation oracle and should **not** become the hot path. |
| **cgroup-aware budgeting** (`src/memory.c`, `waste_usable_ram()`) | Pure host/container policy. |
| **Serving** (`serve/`, 168 checks) | Sits above tokens and logits. |
| **Tokenizer BPE core, specials, EOS override** | Reusable; only the pre-tokenizer branch is model-specific (§1.6). |
| **Session save/load, stats, error discipline** | Generic. |

The port modifies **5 upstream files by +84/−7 lines**. That number is small
because the reusable surface is large.

---

## 3. What is left, end to end

Grouped by who owns it. "Ours" is this repository; "upstream-shaped" means it
changes WASTE proper and needs the same care upstream would apply.

**Ours, no dependencies (can start today)**
- G0's C-side layer harness: bind fixture weights, run
  `waste_inkling_layer_forward_trace()`, emit the archive.
- Tokenizer pattern comparison (§4.1) — decides whether G4 is free or is a
  hand-written pre-tokenizer.

**Ours, needs a machine with `transformers`**
- G0's official side: build one official decoder layer from the fixture and
  drive the existing hooks.

**Ours, needs the checkpoint**
- G1 BF16 parity, G2 quantized tolerances, G3 conversion measurement.

**Upstream-shaped, and blocked on parity**
- A `waste_arch_kind` on the layer, or a sibling model type, resolving §1.1-1.2.
- Load-time binding replacing per-step `waste_find` (§1.3).
- `waste_plan_memory()` dispatching to the Inkling planner (§1.5) — the
  smallest of these and the natural first promotion, because the callee is
  already written and tested.
- Expert-cache wiring to the Inkling callback (§2, reusing `ecache`).
- A second `next_piece` if §4.1 says so.

**Not scoped, deliberately**
- Multimodal (audio `dmel`, vision `hmlp`), MTP speculative decoding, native
  Windows. See [ROADMAP-V19.md](ROADMAP-V19.md) "out of scope".

---

## 4. Highest-value next steps

Ranked by *information gained per unit of work*, which is the right metric for
a project whose blocker is evidence rather than code.

### 4.1 Diff the tokenizer pre-tokenization pattern — do this first

**Why it wins:** it is hours, needs no checkpoint, no GPU, and no torch, and it
resolves a gate that is currently uncosted in both directions. Either
`next_piece()` already matches Inkling's pattern — in which case G4 collapses to
asset loading and the tokenizer stops being a risk — or it does not, and there
is a hand-written Unicode pre-tokenizer on the critical path to `waste chat`
that nobody has budgeted. Both answers change the plan; not knowing is the
worst state.

**Definition of done:** the released pattern string, quoted, next to a
branch-by-branch verdict against `src/tokenizer.c:331`, and a line in the
roadmap costing G4 accordingly.

### 4.2 Finish G0's C side

**Why:** it is buildable and testable *here*, with synthetic fixtures, using
machinery that already exists — `inkling_bind.c` binds, the layer already has
traced variants, and `test_inkling_layer_c.py` already drives the C layer
through ctypes. When the official side lands, parity is one command, not a
project.

### 4.3 Promote `waste_plan_memory()` — the smallest real promotion

**Why:** it converts a refusal into a working feature with a tested callee and
no state-machine risk. `waste_plan_memory()` currently returns
`WASTE_E_UNSUPPORTED` for Inkling; `waste_inkling_plan_decode_memory()` is
written, field-by-field, overflow-checked, and already in CI. Wiring them is a
contained change that makes `waste plan` answer honestly for an Inkling
container — useful on its own, and it exercises the dispatch seam before any
forward path depends on it.

**Caveat, stated plainly:** this is the one item here that promotes public
behaviour ahead of official-weight parity. It is defensible because a memory
*plan* is a statement about geometry, not about numerics — it cannot produce a
wrong token, only a wrong byte count, and its formula is tested. If the project
prefers the strict reading of "nothing public before G1," defer it; it is
step 3 of 3 for exactly that reason.

### 4.4 Then, and only then, the parity gates

G1 → G2 → G3 in order, on a machine with the checkpoint. Everything above
exists to make those cheap and to make their results trustworthy.

---

## 5. The design conclusion

`arch.c` already draws the line in the right place, and both guards refuse
*before* any architecture-specific work. Promotion should keep that shape:

- **not** a third branch in `waste_model` — §1.1 and §1.4 make the union
  unreadable and the state discipline ambiguous;
- **a sibling model type behind the existing classification**, reusing
  `ecache`, `vq`, `memory`, the container format, the tokenizer core, and the
  server, and owning only what is genuinely different: config, state, layer
  dispatch, and binding.

That is roughly what `inkling_private.c` already is. The promotion work is not
writing it — it is connecting it, and deleting the parts that duplicate what
WASTE does better.
