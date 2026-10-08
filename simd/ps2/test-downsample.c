/*
 * R5900 MMI h2v1/h2v2 downsampling correctness test.
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ROWS 2
#define CAP 320
static JSAMPLE source_storage[ROWS][CAP];
static JSAMPLE dest_storage[ROWS][CAP];
static JSAMPROW source[ROWS];
static JSAMPROW dest[ROWS];

static JSAMPROW
offset_ptr(JSAMPLE *p, int offset)
{
  return (JSAMPROW)((((uintptr_t)p + 15) & ~(uintptr_t)15) + offset);
}

static unsigned
pixel(int row, unsigned col, unsigned seed)
{
  return (seed + 57U * (unsigned)row + col * 19U +
          ((unsigned)row + 3U) * (col * col + 7U)) & 255U;
}

static int
run_case(int image_width, int mode, int offset, unsigned seed)
{
  JDIMENSION out_cols = (JDIMENSION)((image_width + 15) / 16 * 8);
  JDIMENSION input_cols = out_cols * 2;
  int vfactor = mode == 1 ? 2 : 1;
  int row, col;

  for (row = 0; row < ROWS; row++) {
    source[row] = offset_ptr(source_storage[row], offset);
    dest[row] = offset_ptr(dest_storage[row], offset);
    memset(source[row], 0xc3, CAP - 32);
    memset(dest[row], 0xa5, CAP - 32);
    for (col = 0; col < image_width; col++)
      source[row][col] = (JSAMPLE)pixel(row, (unsigned)col, seed);
  }

  if (mode == 1)
    jsimd_h2v1_downsample_ps2mmi((JDIMENSION)image_width, ROWS, vfactor,
                                out_cols / 8, source, dest);
  else
    jsimd_h2v2_downsample_ps2mmi((JDIMENSION)image_width, ROWS, vfactor,
                                out_cols / 8, source, dest);

  for (row = 0; row < ROWS; row++) {
    for (col = image_width; col < (int)input_cols; col++) {
      if (source[row][col] !=
          (JSAMPLE)pixel(row, (unsigned)(image_width - 1), seed)) {
        printf("FAIL pad width=%d mode=%d off=%d seed=%u\n",
               image_width, mode, offset, seed);
        return 1;
      }
    }
    for (col = (int)input_cols; col < (int)input_cols + 16; col++) {
      if (source[row][col] != 0xc3) {
        puts("FAIL source guard");
        return 1;
      }
    }
  }

  for (row = 0; row < ROWS; row++) {
    if (row < vfactor) {
      for (col = 0; col < (int)out_cols; col++) {
        int sr = mode == 1 ? row : 0;
        unsigned p0 = (unsigned)(col * 2);
        unsigned p1 = p0 + 1;
        unsigned x0 = p0 < (unsigned)image_width ?
          pixel(sr, p0, seed) : pixel(sr, image_width - 1, seed);
        unsigned x1 = p1 < (unsigned)image_width ?
          pixel(sr, p1, seed) : pixel(sr, image_width - 1, seed);
        unsigned sum = x0 + x1 + (unsigned)(mode == 1 ? (col & 1) : 0);
        if (mode == 2) {
          unsigned y0 = p0 < (unsigned)image_width ?
            pixel(1, p0, seed) : pixel(1, image_width - 1, seed);
          unsigned y1 = p1 < (unsigned)image_width ?
            pixel(1, p1, seed) : pixel(1, image_width - 1, seed);
          sum = x0 + x1 + y0 + y1 + 1U + (unsigned)(col & 1);
        }
        if (dest[row][col] != (JSAMPLE)(sum >> (mode == 1 ? 1 : 2))) {
          printf("FAIL width=%d mode=%d off=%d seed=%u row=%d col=%d\n",
                 image_width, mode, offset, seed, row, col);
          return 1;
        }
      }
    }
    for (col = row < vfactor ? (int)out_cols : 0;
         col < (int)out_cols + 16; col++) {
      if (dest[row][col] != 0xa5) {
        puts("FAIL output guard");
        return 1;
      }
    }
  }
  return 0;
}

int
main(void)
{
  static const int widths[] =
    { 1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 127, 128, 129 };
  unsigned i, seed;
  int mode, offset;
  unsigned cases = 0;

  for (mode = 1; mode <= 2; mode++)
    for (offset = 0; offset <= 1; offset++)
      for (seed = 0; seed <= 1; seed++)
        for (i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
          if (run_case(widths[i], mode, offset, seed ? 173U : 0U))
            return 1;
          cases++;
        }

  printf("PS2 MMI downsampling: PASS (%u cases)\n", cases);
  return 0;
}
