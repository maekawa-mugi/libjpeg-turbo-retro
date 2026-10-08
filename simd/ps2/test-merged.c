/*
 * PS2 EE MMI merged upsampling reference test.
 * Compare the four 4-byte output layouts against the IJG 16.16 arithmetic
 * used by the generic merged color converter in src/jdmerge.c.
 *
 * This test calls both 2h1v and 2h2v kernels directly, regardless of the
 * PS2_EXPERIMENTAL_MERGED dispatch switch.  No full JPEG stream is decoded.
 *
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define WIDTH 65
#define SRC_BYTES 128
#define DST_BYTES (4 * WIDTH + 32)
#define GROUP 1

typedef void (*merged_fn)(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);

typedef struct {
  const char *name;
  merged_fn h2v1, h2v2;
  int red, green, blue, alpha;
} layout;

static const layout layouts[] = {
  { "RGBX", jsimd_h2v1_extrgbx_merged_upsample_ps2mmi,
    jsimd_h2v2_extrgbx_merged_upsample_ps2mmi,
    EXT_RGBX_RED, EXT_RGBX_GREEN, EXT_RGBX_BLUE, 3 },
  { "BGRX", jsimd_h2v1_extbgrx_merged_upsample_ps2mmi,
    jsimd_h2v2_extbgrx_merged_upsample_ps2mmi,
    EXT_BGRX_RED, EXT_BGRX_GREEN, EXT_BGRX_BLUE, 3 },
  { "XBGR", jsimd_h2v1_extxbgr_merged_upsample_ps2mmi,
    jsimd_h2v2_extxbgr_merged_upsample_ps2mmi,
    EXT_XBGR_RED, EXT_XBGR_GREEN, EXT_XBGR_BLUE, 0 },
  { "XRGB", jsimd_h2v1_extxrgb_merged_upsample_ps2mmi,
    jsimd_h2v2_extxrgb_merged_upsample_ps2mmi,
    EXT_XRGB_RED, EXT_XRGB_GREEN, EXT_XRGB_BLUE, 0 }
};

static JSAMPLE source_mem[3][4][SRC_BYTES];
static JSAMPROW source_rows[3][4];
static JSAMPARRAY source_image[3];
static JSAMPLE output_mem[2][DST_BYTES];
static JSAMPROW output_rows[2];

static JSAMPROW
sample_ptr(JSAMPLE *p, unsigned offset)
{
  return (JSAMPROW)((((uintptr_t)p + 15) & ~(uintptr_t)15) + offset);
}

static JSAMPLE
clamp_byte(int x)
{
  return (JSAMPLE)(x < 0 ? 0 : x > 255 ? 255 : x);
}

static void
expected_color(const layout *l, JSAMPLE *dst, int y, int cb, int cr)
{
  int r, g, b;
  cb -= 128;
  cr -= 128;
  r = y + ((91881 * cr + 32768) >> 16);
  g = y + ((-22554 * cb - 46802 * cr + 32768) >> 16);
  b = y + ((116130 * cb + 32768) >> 16);
  dst[l->red] = clamp_byte(r);
  dst[l->green] = clamp_byte(g);
  dst[l->blue] = clamp_byte(b);
  dst[l->alpha] = 255;
}

static int
run_case(const layout *l, unsigned width, int vertical,
         unsigned src_offset, unsigned dst_pattern, unsigned phase)
{
  JSAMPARRAY dst = output_rows;
  unsigned plane, row, col;
  int nrows = vertical == 2 ? 2 : 1;

  for (plane = 0; plane < 3; plane++) {
    source_image[plane] = source_rows[plane];
    for (row = 0; row < 4; row++) {
      source_rows[plane][row] = sample_ptr(source_mem[plane][row], src_offset);
      for (col = 0; col < SRC_BYTES - 16; col++) {
        unsigned v = (phase * 67 + plane * 71 + row * 29 +
                      col * (plane * 17 + row * 3 + 43)) & 255;
        /* Include neutral Cb/Cr, pure black, and pure white. */
        if (col % 11 == 0)
          v = plane == 0 ? 0 : 128;
        if (col % 13 == 0)
          v = 255;
        source_rows[plane][row][col] = (JSAMPLE)v;
      }
    }
  }

  for (row = 0; row < 2; row++) {
    output_rows[row] = sample_ptr(output_mem[row],
                                  (dst_pattern >> row) & 1);
    memset(output_rows[row], 0xa5, 4 * WIDTH + 16);
  }

  if (vertical == 2)
    l->h2v2(width, source_image, GROUP, dst);
  else
    l->h2v1(width, source_image, GROUP, dst);

  for (row = 0; row < 2; row++) {
    if ((int)row < nrows) {
      for (col = 0; col < width; col++) {
        JSAMPLE expected[4];
        unsigned k;
        unsigned yrow = GROUP * (unsigned)vertical + row;
        expected_color(l, expected, source_image[0][yrow][col],
                       source_image[1][GROUP][col / 2],
                       source_image[2][GROUP][col / 2]);
        for (k = 0; k < 4; k++) {
          if (output_rows[row][4 * col + k] != expected[k]) {
            printf("FAIL merged %s width=%u v=%d src=%u dst=%u phase=%u "
                   "row=%u col=%u channel=%u: %u != %u\n",
                   l->name, width, vertical, src_offset, dst_pattern, phase,
                   row, col, k, (unsigned)output_rows[row][4 * col + k],
                   (unsigned)expected[k]);
            return 1;
          }
        }
      }
    }
    /* The h2v1 converter must not write a second output row. */
    for (col = row < (unsigned)nrows ? 4 * width : 0;
         col < 4 * width + 16; col++) {
      if (output_rows[row][col] != 0xa5) {
        printf("OVERWRITE merged %s width=%u v=%d src=%u dst=%u "
               "row=%u byte=%u\n", l->name, width, vertical,
               src_offset, dst_pattern, row, col);
        return 1;
      }
    }
  }
  return 0;
}

int main(void)
{
  static const unsigned widths[] =
    { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65 };
  unsigned li, wi, src_offset, dst_pattern, phase, count = 0;
  int vertical;

  for (li = 0; li < sizeof(layouts) / sizeof(layouts[0]); li++)
    for (vertical = 1; vertical <= 2; vertical++)
      for (src_offset = 0; src_offset <= 1; src_offset++)
        for (dst_pattern = 0; dst_pattern < 4; dst_pattern++)
          for (phase = 0; phase <= 1; phase++)
            for (wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
              if (run_case(&layouts[li], widths[wi], vertical, src_offset,
                           dst_pattern, phase))
                return 1;
              count++;
            }
  printf("PS2 MMI merged upsampling: PASS (%u cases)\n", count);
  return 0;
}
