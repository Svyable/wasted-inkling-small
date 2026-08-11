/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#include "inkling_expert_store.h"

#include "crc32.h"
#include "ecache.h"
#include "platform.h"
#include "waste_format.h"

#include <limits.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int safe_bank_file(const char *name)
{
    return name && name[0] && !strchr(name, '/') && !strchr(name, '\\') &&
           strcmp(name, ".") && strcmp(name, "..");
}

static int read_all_at(int fd, uint8_t *dst, size_t n, int64_t off)
{
    while (n) {
        const int64_t got = waste_pread(fd, dst, n, off);
        if (got <= 0 || (uint64_t)got > (uint64_t)n) return -1;
        dst += (size_t)got;
        n -= (size_t)got;
        off += got;
    }
    return 0;
}

int waste_inkling_expert_record_validate(const waste_model *m,
                                          int layer, int expert,
                                          const uint8_t *record)
{
    if (!m || !record || layer < 0 || layer >= m->cfg.n_layers) return 0;
    const waste_bank *b = &m->bank[layer];
    if (expert < 0 || expert >= b->n_experts || b->rec_bytes <= 0) return 0;
    const size_t rec_bytes = (size_t)b->rec_bytes;
    if (rec_bytes < sizeof(waste_expert_hdr)) return 0;

    const waste_expert_hdr *h = (const waste_expert_hdr *)record;
    const int expected_fmt = m->stages == 3 ? WQ_VQ3R
                           : m->stages == 2 ? WQ_VQ2R : -1;
    if (expected_fmt < 0 || h->magic != WASTE_MAGIC_EXPERT ||
        h->layer != (uint16_t)layer || h->expert_id != (uint16_t)expert ||
        h->fmt != (uint8_t)expected_fmt || h->flags != 0 ||
        h->codebook_id != (uint16_t)b->cb_base || h->lowrank_id != 0 ||
        h->reserved0 != 0 || h->reserved1[0] != 0 || h->reserved1[1] != 0 ||
        h->rec_4k_blocks == 0 ||
        (size_t)h->rec_4k_blocks * WASTE_ALIGN != rec_bytes)
        return 0;

    if (!(h->gate_off == sizeof *h && h->gate_off < h->up_off &&
          h->up_off < h->down_off && h->down_off < h->chan_corr_off &&
          h->chan_corr_off <= rec_bytes))
        return 0;

    if (m->cfg.moe_inter <= 0 || m->cfg.hidden <= 0) return 0;
    const uint64_t scale_rows = (uint64_t)2 * (uint64_t)m->cfg.moe_inter +
                                (uint64_t)m->cfg.hidden;
    if (scale_rows > (uint64_t)SIZE_MAX / sizeof(uint16_t)) return 0;
    const size_t scale_bytes = (size_t)scale_rows * sizeof(uint16_t);
    if ((size_t)h->chan_corr_off > rec_bytes ||
        scale_bytes > rec_bytes - (size_t)h->chan_corr_off)
        return 0;
    const size_t payload_end = (size_t)h->chan_corr_off + scale_bytes;

    if (m->verify) {
        const size_t payload_off = sizeof *h;
        if (payload_end < payload_off ||
            waste_crc32(record + payload_off, payload_end - payload_off) != h->crc32)
            return 0;
    }
    return 1;
}

static int store_fetch(void *user, int layer, int expert, uint8_t *dst)
{
    waste_model *m = (waste_model *)user;
    if (!m || !dst || layer < 0 || layer >= m->cfg.n_layers) return -1;
    const waste_bank *b = &m->bank[layer];
    if (expert < 0 || expert >= b->n_experts || b->fd < 0 || b->rec_bytes <= 0)
        return -1;
    if ((uint64_t)expert > (uint64_t)INT64_MAX / (uint64_t)b->rec_bytes)
        return -1;
    const int64_t off = (int64_t)expert * (int64_t)b->rec_bytes;
    if (read_all_at(b->fd, dst, (size_t)b->rec_bytes, off) ||
        !waste_inkling_expert_record_validate(m, layer, expert, dst))
        return -1;

    pthread_mutex_lock(&m->fetch_mu);
    m->expert_reads++;
    pthread_mutex_unlock(&m->fetch_mu);
    return 0;
}

static void close_sparse_banks(waste_model *m)
{
    if (!m) return;
    for (int L = m->cfg.first_dense; L < m->cfg.n_layers; L++) {
        if (m->bank[L].fd >= 0) {
            close(m->bank[L].fd);
            m->bank[L].fd = -1;
        }
    }
}

int waste_inkling_expert_store_open(waste_model *m, const char *dir,
                                    const js_doc *d,
                                    const waste_load_opts *opt)
{
    if (!m || !dir || !d || !m->inkling || m->cfg.n_layers <= 0 ||
        m->cfg.first_dense < 0 || m->cfg.first_dense >= m->cfg.n_layers ||
        m->miss_buf || m->cache.slot)
        return -1;

    static const waste_load_opts defaults = { .direct_io = 1 };
    if (!opt) opt = &defaults;
    const int layers = js_get(d, 0, "layers");
    const int n = js_size(d, layers);
    const int want = m->cfg.n_layers - m->cfg.first_dense;
    if (n != want || (n > 0 && js_at(d, layers, 0) < 0)) return -1;

    size_t max_rec = 0;
    m->direct_io = m->want_direct;
    for (int i = 0; i < n; i++) {
        const int L = m->cfg.first_dense + i;
        const int e = js_at(d, layers, i);
        char file[128], path[768];
        js_str(d, js_get(d, e, "file"), file, sizeof file);
        const int path_n = snprintf(path, sizeof path, "%s/%s", dir, file);
        if (!safe_bank_file(file) || m->bank[L].n_experts <= 0 ||
            m->bank[L].rec_bytes <= 0 || path_n < 0 ||
            (size_t)path_n >= sizeof path) {
            close_sparse_banks(m);
            return -1;
        }
        m->bank[L].fd = waste_expert_bank_open(
            path, (size_t)m->bank[L].rec_bytes, m->want_direct, &m->direct_io);
        if (m->bank[L].fd < 0) {
            close_sparse_banks(m);
            return -1;
        }
        const int64_t actual = waste_file_size(m->bank[L].fd);
        const uint64_t expected = (uint64_t)m->bank[L].rec_bytes *
                                  (uint64_t)m->bank[L].n_experts;
        if (actual < 0 || (uint64_t)actual != expected) {
            close_sparse_banks(m);
            return -1;
        }
        if ((size_t)m->bank[L].rec_bytes > max_rec)
            max_rec = (size_t)m->bank[L].rec_bytes;
    }

    if (!max_rec || waste_ecache_init(&m->cache, opt->cache_bytes, max_rec,
                                      opt->policy)) {
        close_sparse_banks(m);
        return -1;
    }
    m->miss_buf = (uint8_t *)waste_dio_alloc(max_rec);
    if (!m->miss_buf) {
        waste_ecache_free(&m->cache);
        close_sparse_banks(m);
        return -1;
    }
    return 0;
}

const uint8_t *waste_inkling_expert_record_get(waste_model *m,
                                                int layer, int expert)
{
    if (!m || !m->inkling || !m->miss_buf) return NULL;
    if (m->cache.n_slots > 0)
        return waste_ecache_get(&m->cache, layer, expert, store_fetch, m);

    if (layer < 0 || layer >= m->cfg.n_layers || expert < 0 ||
        expert >= m->bank[layer].n_experts)
        return NULL;
    m->cache.misses++;
    m->cache.bytes_read += (size_t)m->bank[layer].rec_bytes;
    return store_fetch(m, layer, expert, m->miss_buf) == 0 ? m->miss_buf : NULL;
}

void waste_inkling_expert_hint(waste_model *m, int layer,
                               const int *experts, int n)
{
    if (m && m->inkling && m->cache.n_slots > 0 && experts && n > 0)
        waste_ecache_hint(&m->cache, layer, experts, n);
}

const uint8_t *waste_inkling_expert_record_hold(waste_model *m,
                                                 int layer, int expert)
{
    if (!m || !m->inkling || m->cache.n_slots <= 0) return NULL;
    return waste_ecache_hold(&m->cache, layer, expert, store_fetch, m);
}

void waste_inkling_expert_release(waste_model *m)
{
    if (m && m->inkling && m->cache.n_slots > 0)
        waste_ecache_release(&m->cache);
}
