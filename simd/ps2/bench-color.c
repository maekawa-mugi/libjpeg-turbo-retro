/* Both YCbCr kernel implementations are linked into the same EE ELF.
 * Validate all seven outputs and benchmark the four-byte conversions.
 * SPDX-License-Identifier: Zlib
 */
#include "bench-harness.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define WIDTH_MAX 129
#define ROWS 4
#define SRC_CAP (WIDTH_MAX + 64)
#define DST_CAP (4 * WIDTH_MAX + 64)
#define COLOR_VARIANT(v) { #v, { \
  ps2_bench_##v##_rgb, ps2_bench_##v##_extrgb, \
  ps2_bench_##v##_extbgr, ps2_bench_##v##_rgbx, \
  ps2_bench_##v##_bgrx, ps2_bench_##v##_xbgr, \
  ps2_bench_##v##_xrgb \
} }

static void portable_rgb(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);
static void portable_extrgb(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);
static void portable_extbgr(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);
static void portable_rgbx(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);
static void portable_bgrx(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);
static void portable_xbgr(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);
static void portable_xrgb(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);

static const struct {
  const char *name;
  ps2_color_fn fn[7];
} variants[] = {
  { "portable_c", { portable_rgb, portable_extrgb, portable_extbgr,
                    portable_rgbx, portable_bgrx, portable_xbgr,
                    portable_xrgb }},
  COLOR_VARIANT(scalar), COLOR_VARIANT(pmul4), COLOR_VARIANT(pmul8),
  COLOR_VARIANT(regpack), COLOR_VARIANT(table)
};

static const struct {
  const char *name;
  unsigned pixel_size;
  int red, green, blue, alpha;
} layouts[] = {
  { "RGB", RGB_PIXELSIZE, RGB_RED, RGB_GREEN, RGB_BLUE, -1 },
  { "EXT_RGB", EXT_RGB_PIXELSIZE, EXT_RGB_RED, EXT_RGB_GREEN,
    EXT_RGB_BLUE, -1 },
  { "EXT_BGR", EXT_BGR_PIXELSIZE, EXT_BGR_RED, EXT_BGR_GREEN,
    EXT_BGR_BLUE, -1 },
  { "RGBX", EXT_RGBX_PIXELSIZE, EXT_RGBX_RED, EXT_RGBX_GREEN,
    EXT_RGBX_BLUE, 3 },
  { "BGRX", EXT_BGRX_PIXELSIZE, EXT_BGRX_RED, EXT_BGRX_GREEN,
    EXT_BGRX_BLUE, 3 },
  { "XBGR", EXT_XBGR_PIXELSIZE, EXT_XBGR_RED, EXT_XBGR_GREEN,
    EXT_XBGR_BLUE, 0 },
  { "XRGB", EXT_XRGB_PIXELSIZE, EXT_XRGB_RED, EXT_XRGB_GREEN,
    EXT_XRGB_BLUE, 0 }
};

static JSAMPLE srcmem[3][ROWS][SRC_CAP] __attribute__((aligned(16)));
static JSAMPLE dstmem[2][DST_CAP] __attribute__((aligned(16)));
static JSAMPROW rows[3][ROWS];
static JSAMPARRAY image[3];
static JSAMPROW output[2];

static JSAMPLE
clip_color(int v)
{
  return (JSAMPLE)(v < 0 ? 0 : v > 255 ? 255 : v);
}


static void
portable_color(JDIMENSION width, JSAMPIMAGE src, JDIMENSION row,
               JSAMPARRAY dst, int num_rows, unsigned layout)
{
  int yrow;
  unsigned x;
  for (yrow = 0; yrow < num_rows; yrow++)
    for (x = 0; x < width; x++) {
      int y = src[0][row + yrow][x];
      int cb = (int)src[1][row + yrow][x] - 128;
      int cr = (int)src[2][row + yrow][x] - 128;
      JSAMPLE *out = dst[yrow] + x * layouts[layout].pixel_size;
      out[layouts[layout].red] = clip_color(y + ((91881 * cr + 32768) >> 16));
      out[layouts[layout].green] =
        clip_color(y + ((-22554 * cb - 46802 * cr + 32768) >> 16));
      out[layouts[layout].blue] =
        clip_color(y + ((116130 * cb + 32768) >> 16));
      if (layouts[layout].alpha >= 0)
        out[layouts[layout].alpha] = 255;
    }
}
#define PORTABLE_COLOR(name, layout) \
static void name(JDIMENSION w, JSAMPIMAGE s, JDIMENSION r, \
                 JSAMPARRAY d, int count) \
{ portable_color(w, s, r, d, count, layout); }
PORTABLE_COLOR(portable_rgb, 0)
PORTABLE_COLOR(portable_extrgb, 1)
PORTABLE_COLOR(portable_extbgr, 2)
PORTABLE_COLOR(portable_rgbx, 3)
PORTABLE_COLOR(portable_bgrx, 4)
PORTABLE_COLOR(portable_xbgr, 5)
PORTABLE_COLOR(portable_xrgb, 6)

static JSAMPLE *
color_ptr(JSAMPLE *p, unsigned offset)
{
  return (JSAMPLE *)((((uintptr_t)p + 15u) & ~(uintptr_t)15u) + offset);
}

static void
setup_color(unsigned offset, unsigned seed)
{
  unsigned plane, row, col;
  for (plane = 0; plane < 3; plane++) {
    image[plane] = rows[plane];
    for (row = 0; row < ROWS; row++) {
      rows[plane][row] = color_ptr(srcmem[plane][row],
                                  (offset + plane * 5u + row * 7u) & 15u);
      for (col = 0; col < WIDTH_MAX; col++) {
        unsigned val = (seed * 23u + plane * 91u + row * 47u +
                        col * 17u + ((col * col) >> 3)) & 255u;
        if (col % 13u == 0 && plane != 0)
          val = 128;
        if (seed && col % 11u == 0)
          val = (plane == 2) ? 0 : 255;
        rows[plane][row][col] = (JSAMPLE)val;
      }
    }
  }
  for (row = 0; row < 2; row++) {
    output[row] = color_ptr(dstmem[row], (offset + row * 3u) & 15u);
    memset(output[row], 0xa5, 4 * WIDTH_MAX + 16);
  }
}

static int
check_color(unsigned layout, unsigned width)
{
  unsigned row, x, c;
  for (row = 0; row < 2; row++) {
    for (x = 0; x < width; x++) {
      int y = rows[0][row + 1][x];
      int cb = (int)rows[1][row + 1][x] - 128;
      int cr = (int)rows[2][row + 1][x] - 128;
      JSAMPLE expected[4] = { 0, 0, 0, 0 };
      expected[layouts[layout].red] =
        clip_color(y + ((91881 * cr + 32768) >> 16));
      expected[layouts[layout].green] =
        clip_color(y + ((-22554 * cb - 46802 * cr + 32768) >> 16));
      expected[layouts[layout].blue] =
        clip_color(y + ((116130 * cb + 32768) >> 16));
      if (layouts[layout].alpha >= 0)
        expected[layouts[layout].alpha] = 255;
      for (c = 0; c < layouts[layout].pixel_size; c++)
        if (output[row][x * layouts[layout].pixel_size + c] != expected[c])
          return 1;
    }
    for (x = width * layouts[layout].pixel_size;
         x < width * layouts[layout].pixel_size + 16; x++)
      if (output[row][x] != 0xa5)
        return 1;
  }
  return 0;
}

typedef struct {
  ps2_color_fn fn;
  JDIMENSION width;
  unsigned layout;
} color_ctx;

static void
call_color(void *ptr)
{
  color_ctx *ctx = (color_ctx *)ptr;
  ctx->fn(ctx->width, image, 1, output, 2);
}

static uint32_t
digest_color(void *ptr)
{
  const color_ctx *ctx = (const color_ctx *)ptr;
  uint32_t hash = 2166136261u;
  unsigned row;
  for (row = 0; row < 2; row++)
    hash = ps2_bench_fnv(output[row],
                         ctx->width * layouts[ctx->layout].pixel_size, hash);
  return hash;
}

int
ps2_bench_run_color(void)
{
  static const unsigned widths[] = {
    1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17,
    31, 32, 33, 63, 64, 65, 127, 128, 129
  };
  static const unsigned timed[] = { 128, 129 };
  unsigned v, li, wi, off, seed;
  int valid[sizeof(variants) / sizeof(variants[0])] = { 1, 1, 1, 1, 1, 1 };
  int failures = 0;

  for (v = 0; v < sizeof(variants) / sizeof(variants[0]); v++) {
    for (li = 0; li < 7; li++)
      for (off = 0; off < 2; off++)
        for (seed = 0; seed < 2; seed++)
          for (wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
            setup_color(off ? 7u : 0u, seed);
            variants[v].fn[li](widths[wi], image, 1, output, 2);
            if (check_color(li, widths[wi])) {
              printf("FAIL,color,%s,%s,width=%u,off=%u,seed=%u\n",
                     variants[v].name, layouts[li].name,
                     widths[wi], off, seed);
              valid[v] = 0;
              failures++;
              goto next_color;
            }
          }
next_color:
    if (valid[v])
      printf("PASS,color,%s,correctness,560_cases\n", variants[v].name);
  }

  for (li = 3; li < 7; li++)
    for (off = 0; off < 2; off++)
      for (wi = 0; wi < 2; wi++) {
        ps2_bench_variant entries[sizeof(variants) / sizeof(variants[0])];
        color_ctx contexts[sizeof(variants) / sizeof(variants[0])];
        unsigned n = 0;
        setup_color(off ? 1u : 0u, 1);
        for (v = 0; v < sizeof(variants) / sizeof(variants[0]); v++) {
          if (!valid[v])
            continue;
          contexts[n].fn = variants[v].fn[li];
          contexts[n].width = timed[wi];
          contexts[n].layout = li;
          entries[n].name = variants[v].name;
          entries[n].run = call_color;
          entries[n].context = &contexts[n];
          entries[n].digest = digest_color;
          entries[n].reset = NULL;
          n++;
        }
        if (n < 2 ||
            ps2_bench_compare("color", layouts[li].name, timed[wi],
                              (int)off, entries, n, 192))
          failures++;
      }
  return failures ? 1 : 0;
}
