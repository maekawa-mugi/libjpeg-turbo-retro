/*
 * PS2 MMI YCbCr-to-RGB reference test for all seven output layouts.
 *
 * The four-byte formats test R5900 MMI clamp/pack, and the three-byte
 * formats exercise the scalar tail for the same matrix.  The reference
 * uses the IJG 16.16 coefficients and explicit per-component clipping.
 *
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ROWS 4
#define WIDTH 65
#define SRC_CAP 128
#define DST_CAP 320

typedef void (*converter_fn)(JDIMENSION, JSAMPIMAGE, JDIMENSION,
                             JSAMPARRAY, int);

typedef struct {
  const char *name;
  converter_fn func;
  int step;
  int r, g, b, alpha;
} converter_case;

static const converter_case cases[] = {
  { "RGB", jsimd_ycc_rgb_convert_ps2mmi,
    RGB_PIXELSIZE, RGB_RED, RGB_GREEN, RGB_BLUE, -1 },
  { "EXT_RGB", jsimd_ycc_extrgb_convert_ps2mmi,
    EXT_RGB_PIXELSIZE, EXT_RGB_RED, EXT_RGB_GREEN, EXT_RGB_BLUE, -1 },
  { "EXT_BGR", jsimd_ycc_extbgr_convert_ps2mmi,
    EXT_BGR_PIXELSIZE, EXT_BGR_RED, EXT_BGR_GREEN, EXT_BGR_BLUE, -1 },
  { "EXT_RGBX", jsimd_ycc_extrgbx_convert_ps2mmi,
    EXT_RGBX_PIXELSIZE, EXT_RGBX_RED, EXT_RGBX_GREEN, EXT_RGBX_BLUE, 3 },
  { "EXT_BGRX", jsimd_ycc_extbgrx_convert_ps2mmi,
    EXT_BGRX_PIXELSIZE, EXT_BGRX_RED, EXT_BGRX_GREEN, EXT_BGRX_BLUE, 3 },
  { "EXT_XBGR", jsimd_ycc_extxbgr_convert_ps2mmi,
    EXT_XBGR_PIXELSIZE, EXT_XBGR_RED, EXT_XBGR_GREEN, EXT_XBGR_BLUE, 0 },
  { "EXT_XRGB", jsimd_ycc_extxrgb_convert_ps2mmi,
    EXT_XRGB_PIXELSIZE, EXT_XRGB_RED, EXT_XRGB_GREEN, EXT_XRGB_BLUE, 0 }
};

static JSAMPLE samples[3][ROWS][SRC_CAP];
static JSAMPROW input_rows[3][ROWS];
static JSAMPARRAY input_image[3];
static JSAMPLE output_mem[2][DST_CAP];
static JSAMPROW output_rows[2];

static JSAMPROW
align_pointer(JSAMPLE *ptr, unsigned offset)
{
  return (JSAMPROW)((((uintptr_t)ptr + 15) & ~(uintptr_t)15) + offset);
}

static JSAMPLE
limit(int x)
{
  return (JSAMPLE)(x < 0 ? 0 : (x > 255 ? 255 : x));
}

static int
run_test(const converter_case *cc, unsigned width, unsigned offset,
         unsigned phase)
{
  unsigned plane, row, col;
  for (plane = 0; plane < 3; plane++) {
    input_image[plane] = input_rows[plane];
    for (row = 0; row < ROWS; row++) {
      input_rows[plane][row] = align_pointer(samples[plane][row], offset);
      for (col = 0; col < width; col++) {
        unsigned v = (17 * col + 71 * row + 107 * plane +
                      phase * (col * col + 31 * plane)) & 255U;
        /* Repeatedly hit signed endpoint and neutral chroma cases. */
        if (col % 7 == 0)
          v = (phase & 1U) ? 0 : 255;
        if (plane > 0 && col % 11 == 0)
          v = 128;
        input_rows[plane][row][col] = (JSAMPLE)v;
      }
    }
  }

  for (row = 0; row < 2; row++) {
    output_rows[row] = align_pointer(output_mem[row], offset);
    memset(output_rows[row], 0xa5, WIDTH * 4 + 16);
  }
  cc->func(width, input_image, 1, output_rows, 2);

  for (row = 0; row < 2; row++) {
    for (col = 0; col < width; col++) {
      int y = input_image[0][row + 1][col];
      int cb = (int)input_image[1][row + 1][col] - 128;
      int cr = (int)input_image[2][row + 1][col] - 128;
      JSAMPLE expected[4] = { 0, 0, 0, 0 };
      unsigned i;
      expected[cc->r] = limit(y + ((91881 * cr + 32768) >> 16));
      expected[cc->g] =
        limit(y + ((-22554 * cb - 46802 * cr + 32768) >> 16));
      expected[cc->b] = limit(y + ((116130 * cb + 32768) >> 16));
      if (cc->alpha >= 0)
        expected[cc->alpha] = 255;

      for (i = 0; i < (unsigned)cc->step; i++) {
        if (output_rows[row][cc->step * col + i] != expected[i]) {
          printf("FAIL color %s width=%u offset=%u phase=%u "
                 "row=%u pixel=%u channel=%u: %u != %u\n",
                 cc->name, width, offset, phase, row, col, i,
                 (unsigned)output_rows[row][cc->step * col + i],
                 (unsigned)expected[i]);
          return 1;
        }
      }
    }
    for (col = width * cc->step; col < width * cc->step + 16; col++) {
      if (output_rows[row][col] != 0xa5) {
        printf("FAIL color output guard %s width=%u row=%u byte=%u\n",
               cc->name, width, row, col);
        return 1;
      }
    }
  }
  return 0;
}

int
main(void)
{
  static const unsigned widths[] =
    { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 31, 32, 33, 64, 65 };
  unsigned ci, wi, offset, phase, checks = 0;

  for (ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++)
    for (wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++)
      for (offset = 0; offset < 2; offset++)
        for (phase = 0; phase < 2; phase++) {
          if (run_test(&cases[ci], widths[wi], offset, phase))
            return 1;
          checks++;
        }

  printf("PS2 MMI YCbCr to RGB: PASS (%u cases)\n", checks);
  return 0;
}
