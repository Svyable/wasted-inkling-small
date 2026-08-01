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
