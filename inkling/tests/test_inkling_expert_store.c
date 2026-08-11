/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 *
 * Storage-only routed expert contract.  This deliberately does not execute an
 * expert: it proves that Inkling's native WEXP bytes survive the same bounded
 * cache path WASTE already uses, including identity/CRC rejection and the
 * explicit zero-cache fallback.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/crc32.h"
#include "../src/ecache.h"
#include "../src/inkling_expert_store.h"
#include "../src/platform.h"
#include "../src/waste_format.h"

static int fails;

static void check(int ok, const char *what)
{
    if (!ok) {
        fprintf(stderr, "INKLING EXPERT STORE: %s\n", what);
        fails++;
    }
}

static void make_record(uint8_t *record, int layer, int expert, int codebook,
                        int hidden, int moe_inter)
{
    memset(record, 0, WASTE_ALIGN);
    waste_expert_hdr *h = (waste_expert_hdr *)record;
    h->magic = WASTE_MAGIC_EXPERT;
    h->layer = (uint16_t)layer;
    h->expert_id = (uint16_t)expert;
    h->fmt = WQ_VQ3R;
    h->codebook_id = (uint16_t)codebook;
    h->rec_4k_blocks = 1;
    h->gate_off = (uint32_t)sizeof *h;
    h->up_off = 64;
    h->down_off = 80;
    h->chan_corr_off = 96;

    const size_t scale_rows = (size_t)2 * (size_t)moe_inter + (size_t)hidden;
    const size_t payload_end = (size_t)h->chan_corr_off +
                               scale_rows * sizeof(uint16_t);
    for (size_t i = sizeof *h; i < payload_end; i++)
        record[i] = (uint8_t)(17u + (unsigned)expert * 29u + (unsigned)i);
    h->crc32 = waste_crc32(record + sizeof *h, payload_end - sizeof *h);
}

int main(void)
{
    enum { LAYER = 1, EXPERTS = 2, CODEBOOK = 7, HIDDEN = 2, MOE_INTER = 2 };
    static const char path[] = "test-inkling-expert-store.tmp";
    uint8_t *records = (uint8_t *)malloc((size_t)EXPERTS * WASTE_ALIGN);
    uint8_t *bad = (uint8_t *)malloc(WASTE_ALIGN);
    waste_model m;
    FILE *f = NULL;

    check(records && bad, "out of memory");
    if (!records || !bad) goto done;
    for (int e = 0; e < EXPERTS; e++)
        make_record(records + (size_t)e * WASTE_ALIGN,
                    LAYER, e, CODEBOOK, HIDDEN, MOE_INTER);

    f = fopen(path, "wb");
    check(f != NULL, "cannot create bank fixture");
    if (!f) goto done;
    check(fwrite(records, WASTE_ALIGN, EXPERTS, f) == EXPERTS,
          "cannot write bank fixture");
    check(fclose(f) == 0, "cannot close bank fixture");
    f = NULL;

    memset(&m, 0, sizeof m);
    for (int L = 0; L < WASTE_MAX_LAYERS; L++) m.bank[L].fd = -1;
    m.inkling = &m;                 /* non-NULL architecture sentinel */
    m.cfg.n_layers = 2;
    m.cfg.first_dense = LAYER;
    m.cfg.hidden = HIDDEN;
    m.cfg.moe_inter = MOE_INTER;
    m.stages = 3;
    m.verify = 1;
    m.bank[LAYER].n_experts = EXPERTS;
    m.bank[LAYER].cb_base = CODEBOOK;
    m.bank[LAYER].rec_bytes = WASTE_ALIGN;
    check(pthread_mutex_init(&m.fetch_mu, NULL) == 0, "mutex init failed");
    m.bank[LAYER].fd = waste_open_stream(path, 0);
    check(m.bank[LAYER].fd >= 0, "cannot open bank fixture");
    check(waste_ecache_init(&m.cache, (size_t)2 * WASTE_ALIGN,
                            WASTE_ALIGN, 0) == 0,
          "cache init failed");
    m.miss_buf = (uint8_t *)waste_dio_alloc(WASTE_ALIGN);
    check(m.miss_buf != NULL, "miss buffer allocation failed");
    if (m.bank[LAYER].fd < 0 || !m.miss_buf) goto cleanup_model;

    const uint8_t *first = waste_inkling_expert_record_get(&m, LAYER, 0);
    check(first != NULL, "first expert fetch failed");
    check(m.expert_reads == 1, "first cache miss did not perform one read");
    check(m.cache.misses == 1 && m.cache.hits == 0,
          "first expert fetch was not one cache miss");
    check(first && waste_inkling_expert_record_validate(&m, LAYER, 0, first),
          "cached record did not validate");

    const uint8_t *again = waste_inkling_expert_record_get(&m, LAYER, 0);
    check(again == first, "cache hit did not return the resident record");
    check(m.expert_reads == 1, "cache hit performed disk I/O");
    check(m.cache.misses == 1 && m.cache.hits == 1,
          "second expert fetch was not a cache hit");

    const uint8_t *held = waste_inkling_expert_record_hold(&m, LAYER, 1);
    check(held != NULL && m.cache.n_held == 1,
          "expert hold did not pin a second record");
    check(m.expert_reads == 2 && m.cache.misses == 2,
          "held expert was not fetched exactly once");
    waste_inkling_expert_release(&m);
    check(m.cache.n_held == 0, "expert release left a cache record pinned");

    check(waste_inkling_expert_record_get(&m, LAYER, EXPERTS) == NULL,
          "expert index past the bank was served");
    check(waste_inkling_expert_record_get(&m, -1, 0) == NULL,
          "negative layer was served");

    if (first) {
        memcpy(bad, first, WASTE_ALIGN);
        bad[sizeof(waste_expert_hdr)] ^= 1u;
        check(!waste_inkling_expert_record_validate(&m, LAYER, 0, bad),
              "payload corruption passed CRC validation");
        memcpy(bad, first, WASTE_ALIGN);
        ((waste_expert_hdr *)bad)->expert_id = 1;
        check(!waste_inkling_expert_record_validate(&m, LAYER, 0, bad),
              "record identity mismatch was accepted");
    }

    /* A zero-byte cache is an explicit supported mode, not a half-created
     * cache. The same record is read on every demand through the aligned miss
     * buffer, which keeps memory bounded and statistics honest. */
    waste_ecache_free(&m.cache);
    check(waste_ecache_init(&m.cache, 0, WASTE_ALIGN, 0) == 0,
          "zero-cache init failed");
    const uint64_t reads_before = m.expert_reads;
    check(waste_inkling_expert_record_get(&m, LAYER, 0) != NULL &&
          waste_inkling_expert_record_get(&m, LAYER, 0) != NULL,
          "zero-cache fallback fetch failed");
    check(m.expert_reads == reads_before + 2,
          "zero-cache fallback unexpectedly retained the record");
    check(m.cache.hits == 0 && m.cache.misses == 2,
          "zero-cache statistics are not two misses");

cleanup_model:
    waste_ecache_free(&m.cache);
    waste_dio_free(m.miss_buf);
    if (m.bank[LAYER].fd >= 0) close(m.bank[LAYER].fd);
    pthread_mutex_destroy(&m.fetch_mu);
done:
    if (f) fclose(f);
    remove(path);
    free(bad);
    free(records);
    if (!fails) puts("INKLING_EXPERT_STORE_OK");
    return fails ? 1 : 0;
}
