# Fixes that belong to `sqliteai/waste`

These are bugs in upstream, not in the Inkling port. They are kept here as
standalone patches against the pinned baseline so they can be sent upstream
without the port attached — nobody should have to take 15,000 lines of Inkling
to get a benchmark that measures the right thing.

Baseline: `sqliteai/waste@69315701f634648f7a790915a0a525ed8aabf218` (0.6.3).

| Patch | What it fixes |
| --- | --- |
| [`0001-diskbench-bypass-the-page-cache-on-Linux.patch`](0001-diskbench-bypass-the-page-cache-on-Linux.patch) | `tools/diskbench.c` measured the page cache on Linux and reported it as "cache bypassed" |

---

## 1. `diskbench.c` never bypassed the page cache on Linux

`tools/diskbench.c` describes its third measurement as "random record reads,
cache-bypassed ← **the number that sets tok/s**". On Linux it was neither.

```c
static void nocache(int fd) {
#ifdef __APPLE__
    fcntl(fd, F_NOCACHE, 1);
    fcntl(fd, F_RDAHEAD, 0);
#endif
}
```

There is no `#else`, and no `open()` in the file passes `O_DIRECT`. So on
Linux, whenever the benchmark file fits in RAM — which is the default, since
`file_gb` defaults to 8 — the random-read loop reads back what it just wrote,
from the page cache, and the program prints the result under the label
"cache bypassed".

This is precisely the failure `docs/GATES.md` Gate 5 was written to prevent:

> on a 64 GB machine the kernel was quietly holding all 17 GB of Kimi-Linear,
> so the measured I/O cost was fiction. K3's ~816 GB gets no such help.

The engine does the right thing — `ecache.c` opens with `O_DIRECT` on Linux
and `F_NOCACHE` on macOS. Only the benchmark that calibrates expectations
against it was affected.

### Measured

Same host, same 8 GiB file, same 9,457,664 B record, `-O2`:

| | unpatched | with `O_DIRECT` | inflation |
| --- | ---: | ---: | ---: |
| seq read | 5.66 GiB/s | 2.39 GiB/s | 2.4× |
| rand 1 thread | 5.67 GiB/s | 2.08 GiB/s | 2.7× |
| rand 2 threads | 8.41 GiB/s | 2.41 GiB/s | 3.5× |
| rand 4 threads | **14.49 GiB/s** | **3.00 GiB/s** | **4.8×** |

The unpatched run reports 14.49 GiB/s — higher than the 12.89 GiB/s
`docs/EFFICIENCY.md` §2 measured on an M5 Pro internal SSD — on a cloud VM
whose real cache-bypassed random-read rate is 3.00 GiB/s.

Note the shape as well as the level: the unpatched numbers scale strongly with
thread count (5.67 → 14.49) because they measure `memcpy` from page cache,
which scales with cores. The real device is nearly saturated at queue depth 1,
which is the observation `EFFICIENCY.md` §2 draws its read-ahead conclusion
from.

### What the patch does

1. Reads open `O_DIRECT` on Linux, falling back to buffered plus
   `POSIX_FADV_DONTNEED` where the filesystem refuses it (tmpfs, some overlay
   and network filesystems).
2. **A refused `O_DIRECT` is reported**, in the result line and on stderr,
   rather than silently downgraded. A benchmark that quietly falls back is
   worse than one that fails: it produces a plausible number that is wrong by
   the size of RAM.
3. The trailing `tok/s` column takes a `gb_per_token` argument instead of
   hardcoding K3's 12.5, so the tool can be pointed at another model.
4. `nocache()` gets a `(void)fd` on the non-Apple path, so the file compiles
   clean under `-Wall -Wextra -Werror`.

macOS behaviour is unchanged: `F_NOCACHE` and `F_RDAHEAD` still apply, and
`O_DIRECT` does not exist there so the `#ifdef` compiles it out.

### Reproduce

```sh
cc -O2 -o diskbench tools/diskbench.c -lpthread          # before
./diskbench /var/tmp/.bench 8 9.021484375 4

# after, at Inkling-Small's record and per-token read
./diskbench /var/tmp/.bench 8 9.021484375 4 2.114
```

Use a path on a real block device. On a machine with more RAM than `file_gb`
the unpatched numbers are the page cache; the patched ones are the device.

### Apply

```sh
git checkout 69315701f634648f7a790915a0a525ed8aabf218
git am upstream/0001-diskbench-bypass-the-page-cache-on-Linux.patch
```

---

## Not a patch, but worth reporting: `waste_ecache_get` at a zero budget

`ecache.h` documents the budget argument as:

> `budget_bytes` 0 disables caching (every access reads).

`waste_ecache_init` honours that by leaving `n_slots` at 0, but
`waste_ecache_get` then has no slot to fill or return and answers `NULL`,
which its own contract gives as the failure signal. So "caching disabled" is
not a mode a caller can drive through `get` — at a zero budget the caller must
read directly instead.

The engine does not hit this, because its own budget resolver never lands
there. It is a documentation/API mismatch rather than a defect, and it cost an
afternoon to rediscover from the outside, which is the only reason it is
written down here.
