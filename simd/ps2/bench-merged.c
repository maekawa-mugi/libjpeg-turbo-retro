/* Exhaustive-ish per-variant correctness and same-ELF warm-cache timing.
 * Uses identical IJG 16.16 equations, 2h1v/2h2v, four RGBX orderings.
 * SPDX-License-Identifier: Zlib
 */
#include "bench-harness.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_W 129
#define CAP_IN (MAX_W + 64)
#define CAP_OUT (4 * MAX_W + 64)
#define GROUP 1
#define BENCH_REPEATS 192

#define MERGED_VARIANT(v) { #v, { \
  { ps2_bench_##v##_h2v1_rgbx, ps2_bench_##v##_h2v1_bgrx, \
    ps2_bench_##v##_h2v1_xbgr, ps2_bench_##v##_h2v1_xrgb }, \
  { ps2_bench_##v##_h2v2_rgbx, ps2_bench_##v##_h2v2_bgrx, \
    ps2_bench_##v##_h2v2_xbgr, ps2_bench_##v##_h2v2_xrgb } \
} }

static void portable_h2v1_rgbx(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);
static void portable_h2v1_bgrx(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);
static void portable_h2v1_xbgr(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);
static void portable_h2v1_xrgb(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);
static void portable_h2v2_rgbx(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);
static void portable_h2v2_bgrx(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);
static void portable_h2v2_xbgr(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);
static void portable_h2v2_xrgb(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);

static const struct {
  const char *name;
  ps2_merged_fn fn[2][4];
} variants[] = {
  { "portable_c", {
    { portable_h2v1_rgbx, portable_h2v1_bgrx,
      portable_h2v1_xbgr, portable_h2v1_xrgb },
    { portable_h2v2_rgbx, portable_h2v2_bgrx,
      portable_h2v2_xbgr, portable_h2v2_xrgb }
  }},
  MERGED_VARIANT(scalar), MERGED_VARIANT(pmul4),
  MERGED_VARIANT(pmul8), MERGED_VARIANT(addpack),
  MERGED_VARIANT(vector), MERGED_VARIANT(table)
};

static const struct {
  const char *name;
  int red, green, blue, alpha;
} layouts[] = {
  { "RGBX", EXT_RGBX_RED, EXT_RGBX_GREEN, EXT_RGBX_BLUE, 3 },
  { "BGRX", EXT_BGRX_RED, EXT_BGRX_GREEN, EXT_BGRX_BLUE, 3 },
  { "XBGR", EXT_XBGR_RED, EXT_XBGR_GREEN, EXT_XBGR_BLUE, 0 },
  { "XRGB", EXT_XRGB_RED, EXT_XRGB_GREEN, EXT_XRGB_BLUE, 0 }
};

static JSAMPLE inmem[3][4][CAP_IN] __attribute__((aligned(16)));
static JSAMPLE outmem[2][CAP_OUT] __attribute__((aligned(16)));
static JSAMPROW inrows[3][4];
static JSAMPARRAY image[3];
static JSAMPROW outrows[2];


static JSAMPLE clamp(int v);

/* A true scalar C baseline.  The existing "scalar" variant only has a
 * scalar chroma matrix; it still uses the MMI pack4 kernel.
 */
static void
portable_merged(JDIMENSION width, JSAMPIMAGE src, JDIMENSION row,
                JSAMPARRAY dst, unsigned vertical, unsigned layout)
{
  unsigned k, yrow;
  for (yrow = 0; yrow < vertical; yrow++)
    for (k = 0; k < width; k++) {
      int y = src[0][row * vertical + yrow][k];
      int cb = (int)src[1][row][k / 2] - 128;
      int cr = (int)src[2][row][k / 2] - 128;
      JSAMPLE *p = dst[yrow] + 4 * k;
      p[layouts[layout].red] = clamp(y + ((91881 * cr + 32768) >> 16));
      p[layouts[layout].green] =
        clamp(y + ((-22554 * cb - 46802 * cr + 32768) >> 16));
      p[layouts[layout].blue] = clamp(y + ((116130 * cb + 32768) >> 16));
      p[layouts[layout].alpha] = 255;
    }
}

#define PORTABLE_MERGED(name, vv, li) \
static void name(JDIMENSION w, JSAMPIMAGE s, JDIMENSION r, JSAMPARRAY d) \
{ portable_merged(w, s, r, d, vv, li); }
PORTABLE_MERGED(portable_h2v1_rgbx, 1, 0)
PORTABLE_MERGED(portable_h2v1_bgrx, 1, 1)
PORTABLE_MERGED(portable_h2v1_xbgr, 1, 2)
PORTABLE_MERGED(portable_h2v1_xrgb, 1, 3)
PORTABLE_MERGED(portable_h2v2_rgbx, 2, 0)
PORTABLE_MERGED(portable_h2v2_bgrx, 2, 1)
PORTABLE_MERGED(portable_h2v2_xbgr, 2, 2)
PORTABLE_MERGED(portable_h2v2_xrgb, 2, 3)

static JSAMPLE *
aligned_offset(JSAMPLE *p, unsigned offset)
{
  return (JSAMPLE *)((((uintptr_t)p + 15u) & ~(uintptr_t)15u) + offset);
}

static JSAMPLE
clamp(int v)
{
  return (JSAMPLE)(v < 0 ? 0 : v > 255 ? 255 : v);
}

static void
prepare(unsigned width, unsigned offset, unsigned seed)
{
  unsigned plane, row, col;
  (void)width;
  for (plane = 0; plane < 3; plane++) {
    image[plane] = inrows[plane];
    for (row = 0; row < 4; row++) {
      inrows[plane][row] = aligned_offset(inmem[plane][row],
                                         (offset + plane * 5u + row * 7u) & 15u);
      for (col = 0; col < MAX_W; col++) {
        unsigned value = (seed * 31u + col * 19u + plane * 71u +
                          row * 53u + ((col * col) >> 2)) & 255u;
        if ((col % 11u) == 0 && plane != 0)
          value = 128;
        if (seed && (col % 13u) == 0)
          value = plane == 1 ? 0 : 255;
        inrows[plane][row][col] = (JSAMPLE)value;
      }
    }
  }
  for (row = 0; row < 2; row++) {
    outrows[row] = aligned_offset(outmem[row],
                                  (offset + row * 3u) & 15u);
    memset(outrows[row], 0xa5, 4 * MAX_W + 16);
  }
}

static int
verify(unsigned layout, int vertical, unsigned width)
{
  unsigned row, x, channel;
  unsigned active = vertical == 2 ? 2u : 1u;
  for (row = 0; row < 2; row++) {
    if (row < active) {
      for (x = 0; x < width; x++) {
        int cb = (int)image[1][GROUP][x / 2] - 128;
        int cr = (int)image[2][GROUP][x / 2] - 128;
        int y = image[0][GROUP * vertical + row][x];
        JSAMPLE expected[4] = { 0, 0, 0, 0 };
        expected[layouts[layout].red] =
          clamp(y + ((91881 * cr + 32768) >> 16));
        expected[layouts[layout].green] =
          clamp(y + ((-22554 * cb - 46802 * cr + 32768) >> 16));
        expected[layouts[layout].blue] =
          clamp(y + ((116130 * cb + 32768) >> 16));
        expected[layouts[layout].alpha] = 255;
        for (channel = 0; channel < 4; channel++)
          if (outrows[row][4 * x + channel] != expected[channel])
            return 1;
      }
    }
    for (x = row < active ? 4 * width : 0; x < 4 * width + 16; x++)
      if (outrows[row][x] != 0xa5)
        return 1;
  }
  return 0;
}

typedef struct {
  ps2_merged_fn fn;
  JDIMENSION width;
  unsigned vertical;
} call_t;

static void
call_merged(void *ptr)
{
  call_t *ctx = (call_t *)ptr;
  ctx->fn(ctx->width, image, GROUP, outrows);
}

static uint32_t
digest_merged(void *ptr)
{
  const call_t *ctx = (const call_t *)ptr;
  uint32_t hash = 2166136261u;
  unsigned row;
  for (row = 0; row < ctx->vertical; row++)
    hash = ps2_bench_fnv(outrows[row], 4u * ctx->width, hash);
  return hash;
}

int
ps2_bench_run_merged(void)
{
  static const unsigned widths[] = {
    1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17,
    31, 32, 33, 63, 64, 65, 127, 128, 129
  };
  static const unsigned time_widths[] = { 128, 129 };
  unsigned v, layout, vert, i, off, seed, wi;
  int failures = 0;
  int ok[sizeof(variants) / sizeof(variants[0])];

  for (v = 0; v < sizeof(variants) / sizeof(variants[0]); v++) {
    ok[v] = 1;
    for (layout = 0; layout < 4; layout++)
      for (vert = 1; vert <= 2; vert++)
        for (off = 0; off <= 1; off++)
          for (seed = 0; seed < 2; seed++)
            for (i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
              prepare(widths[i], off ? 7u : 0u, seed);
              variants[v].fn[vert - 1][layout](widths[i], image,
                                                GROUP, outrows);
              if (verify(layout, (int)vert, widths[i])) {
                printf("FAIL,merged,%s,%s,h2v%u,width=%u,off=%u,seed=%u\n",
                       variants[v].name, layouts[layout].name,
                       vert, widths[i], off, seed);
                ok[v] = 0;
                failures++;
                /* Continue with other variants without flooding console. */
                goto next_merged_variant;
              }
            }
next_merged_variant:
    if (ok[v])
      printf("PASS,merged,%s,correctness,640_cases\n", variants[v].name);
  }

  /* Each workload is measured in rotating contender order.  No variant
   * gets a fixed first/last advantage; all output digests must match.
   */
  for (layout = 0; layout < 4; layout++)
    for (vert = 1; vert <= 2; vert++)
      for (off = 0; off < 2; off++)
        for (wi = 0; wi < 2; wi++) {
          ps2_bench_variant entries[sizeof(variants) / sizeof(variants[0])];
          call_t contexts[sizeof(variants) / sizeof(variants[0])];
          unsigned n = 0;
          char workload[40];
          prepare(time_widths[wi], off ? 1u : 0u, 1);
          for (v = 0; v < sizeof(variants) / sizeof(variants[0]); v++) {
            if (!ok[v])
              continue;
            contexts[n].fn = variants[v].fn[vert - 1][layout];
            contexts[n].width = time_widths[wi];
            contexts[n].vertical = vert;
            entries[n].name = variants[v].name;
            entries[n].run = call_merged;
            entries[n].context = &contexts[n];
            entries[n].digest = digest_merged;
            entries[n].reset = NULL;
            n++;
          }
          snprintf(workload, sizeof(workload), "h2v%u_%s",
                   vert, layouts[layout].name);
          if (n < 2 ||
              ps2_bench_compare("merged", workload, time_widths[wi],
                                (int)off, entries, n, BENCH_REPEATS))
            failures++;
        }
  return failures ? 1 : 0;
}
