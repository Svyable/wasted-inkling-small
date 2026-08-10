/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/arch.h"
#include "../src/inkling.h"
#include "../src/inkling_config.h"
#include "../src/inkling_attention.h"
#include "../src/inkling_layer.h"

static int closef(float a, float b)
{
    return fabsf(a - b) < 1e-6f;
}

int main(void)
{
    int fail = 0;
    fail += waste_arch_classify(NULL) != WASTE_ARCH_LEGACY;
    fail += waste_arch_classify("") != WASTE_ARCH_LEGACY;
    fail += waste_arch_classify("kimi-k3") != WASTE_ARCH_LEGACY;
    fail += waste_arch_classify("inkling-small") != WASTE_ARCH_INKLING;
    fail += waste_arch_classify("inkling") != WASTE_ARCH_INKLING;
    fail += waste_arch_classify("InklingForConditionalGeneration") !=
            WASTE_ARCH_INKLING;
    fail += waste_arch_classify("INKLING-SMALL") != WASTE_ARCH_INKLING;
    fail += waste_arch_classify("inking") != WASTE_ARCH_LEGACY;

    {
        const float logits[] = { 2.0f, -1.0f, 0.5f, 1.0f };
        const float bias[] = { 0.0f, 0.0f, 0.0f };
        int idx[2]; float rw[2], sw[1];
        fail += waste_inkling_route(logits, bias, 3, 1, 2, 1.0f, 1.0f,
                                    idx, rw, sw) != 0;
        fail += idx[0] != 0 || idx[1] != 2;
        fail += !closef(rw[0] + rw[1] + sw[0], 1.0f);
    }

    {
        /* The legacy route API must remain byte-identical to the explicit F32
         * profile.  BF16_REFERENCE is a separate internal contract. */
        const float logits[] = { 2.0f, -1.0f, 0.5f, 1.0f };
        const float bias[] = { 0.0f, 0.0f, 0.0f };
        int legacy_i[2], f32_i[2];
        float legacy_r[2], legacy_s[1], f32_r[2], f32_s[1];
        fail += waste_inkling_route(logits, bias, 3, 1, 2, 1.0f, 1.0f,
                                    legacy_i, legacy_r, legacy_s) != 0;
        fail += waste_inkling_route_profile(
                    logits, bias, 3, 1, 2, 1.0f, 1.0f,
                    f32_i, f32_r, f32_s, WASTE_INKLING_NUMERIC_F32) != 0;
        fail += memcmp(legacy_i, f32_i, sizeof legacy_i) != 0;
        fail += memcmp(legacy_r, f32_r, sizeof legacy_r) != 0;
        fail += memcmp(legacy_s, f32_s, sizeof legacy_s) != 0;
        fail += waste_inkling_route_profile(
                    logits, bias, 3, 1, 2, 1.0f, 1.0f,
                    f32_i, f32_r, f32_s,
                    (waste_inkling_numeric_profile)99) == 0;
    }

    {
        /* These two sigmoid scores are distinct in F32 but complete to the
         * same BF16 choice value.  The BF16 profile must therefore use the
         * portable lower-ID tie rule rather than inherit the F32 ordering. */
        const float logits[] = { 0.0f, 0.0001f };
        const float bias[] = { 0.0f, 0.0f };
        int f32_i[1], bf16_i[1];
        float f32_r[1], bf16_r[1];
        fail += waste_inkling_route_profile(
                    logits, bias, 2, 0, 1, 1.0f, 1.0f,
                    f32_i, f32_r, NULL, WASTE_INKLING_NUMERIC_F32) != 0;
        fail += waste_inkling_route_profile(
                    logits, bias, 2, 0, 1, 1.0f, 1.0f,
                    bf16_i, bf16_r, NULL,
                    WASTE_INKLING_NUMERIC_BF16_REFERENCE) != 0;
        fail += f32_i[0] != 1;
        fail += bf16_i[0] != 0;
        fail += bf16_r[0] != waste_inkling_bf16_round(bf16_r[0]);
        fail += bf16_r[0] != 1.0f;
    }

    {
        const float logits[] = { 2.0f, -1.0f, 0.5f, 1.0f };
        const float bias[] = { 0.0f, 0.0f, 0.0f };
        int idx[2]; float rw[2], sw[1];
        fail += waste_inkling_route_profile(
                    logits, bias, 3, 1, 2, 8.0f, 0.75f,
                    idx, rw, sw, WASTE_INKLING_NUMERIC_BF16_REFERENCE) != 0;
        fail += rw[0] != waste_inkling_bf16_round(rw[0]);
        fail += rw[1] != waste_inkling_bf16_round(rw[1]);
        fail += sw[0] != waste_inkling_bf16_round(sw[0]);
    }

    {
        /* Nearest-even BF16 completion is a shared primitive, not an ad-hoc
         * cast hidden in one evidence harness. */
        fail += waste_inkling_bf16_round(1.00390625f) != 1.0f;
        fail += waste_inkling_bf16_round(1.01171875f) != 1.015625f;
    }

    {
        const int local[] = { 0, 1 };
        waste_inkling_config_args a;
        waste_inkling_config c;
        waste_inkling_memory m;
        memset(&a, 0, sizeof a);
        a.n_layers = 3; a.hidden = 16; a.vocab = 64;
        a.unpadded_vocab = 60; a.max_context = 1024;
        a.global_heads = 4; a.global_kv_heads = 2; a.global_head_dim = 4;
        a.local_heads = 4; a.local_kv_heads = 2; a.local_head_dim = 4;
        a.sliding_window = 8; a.d_rel = 2; a.rel_extent = 8;
        a.conv_kernel = 4; a.dense_layers = 1; a.dense_intermediate = 32;
        a.moe_intermediate = 8; a.n_routed_experts = 4; a.top_k = 2;
        a.n_shared_experts = 2; a.rms_eps = 1e-6f; a.route_scale = 8.0f;
        a.logits_width_multiplier = 2.0f; a.log_scaling_n_floor = 128000;
        a.log_scaling_alpha = 0.1f; a.local_layer_ids = local; a.n_local_layers = 2;
        fail += waste_inkling_config_build(&c, &a) != 0;
        fail += !c.layer[0].is_local || !c.layer[1].is_local || c.layer[2].is_local;
        fail += c.layer[2].relative_extent != 8;
        fail += waste_inkling_plan_decode_memory(&c, 16, &m) != 0;
        fail += m.kv_bytes != 2048 || m.conv_bytes != 2304 || m.state_bytes != 4352;
        fail += m.decode_scratch_bytes != 3344;
        fail += waste_inkling_layer_scratch_floats(&c, 0, 8) == 0;
        fail += waste_inkling_layer_scratch_ints(&c) != 2;
        {
            const int bad_local[] = { 0, 0 };
            a.local_layer_ids = bad_local;
            fail += waste_inkling_config_build(&c, &a) == 0;
        }
    }

    {
        float kcache[2] = {0}, vcache[2] = {0}, scores[2] = {0}, out[1] = {0};
        const float one[1] = {1.0f}, zero[1] = {0.0f};
        waste_inkling_attention_state a;
        fail += waste_inkling_attention_init(&a, 0, 1, 1, 1, 1, 1, 2,
                                              1e-6f, kcache, vcache) != 0;
        fail += waste_inkling_attention_step(&a, one, one, one, one, one,
                                              zero, zero, 0, 0, 0.0f,
                                              scores, out) != 0;
        fail += !closef(out[0], 1.0f);
    }

    {
        /* Released Inkling config profile: verifies the generic builder is
         * not accidentally specialized to the unverified Small candidate. */
        int local[66], nlocal = 0;
        waste_inkling_config_args a;
        waste_inkling_config c;
        waste_inkling_memory m;
        for (int i = 0; i < 66; i++) if ((i + 1) % 6) local[nlocal++] = i;
        memset(&a, 0, sizeof a);
        a.n_layers = 66; a.hidden = 6144; a.vocab = 201024;
        a.unpadded_vocab = 200058; a.max_context = 1048576;
        a.global_heads = 64; a.global_kv_heads = 8; a.global_head_dim = 128;
        a.local_heads = 64; a.local_kv_heads = 16; a.local_head_dim = 128;
        a.sliding_window = 512; a.d_rel = 16; a.rel_extent = 1024;
        a.conv_kernel = 4; a.dense_layers = 2; a.dense_intermediate = 24576;
        a.moe_intermediate = 3072; a.n_routed_experts = 256; a.top_k = 6;
        a.n_shared_experts = 2; a.rms_eps = 1e-6f; a.route_scale = 8.0f;
        a.logits_width_multiplier = 24.0f; a.log_scaling_n_floor = 128000;
        a.log_scaling_alpha = 0.1f; a.local_layer_ids = local;
        a.n_local_layers = nlocal;
        fail += waste_inkling_config_build(&c, &a) != 0;
        fail += c.layer[5].is_local || !c.layer[4].is_local;
        fail += c.layer[4].num_kv_heads != 16 || c.layer[5].num_kv_heads != 8;
        fail += waste_inkling_plan_decode_memory(&c, 4096, &m) != 0;
        fail += m.state_bytes == 0 || m.decode_scratch_bytes == 0;
    }

    if (fail) {
        fprintf(stderr, "INKLING FAIL %d\n", fail);
        return 1;
    }
    puts("INKLING OK");
    return 0;
}
