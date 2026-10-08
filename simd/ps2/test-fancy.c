/*
 * Exact-pixel standalone test for PS2 EE MMI fancy upsampling.
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define NROWS 4
#define MAX_WIDTH 65
#define CAPACITY (2 * MAX_WIDTH + 64)

static JSAMPLE input_mem[NROWS + 2][CAPACITY];
static JSAMPLE output_mem[NROWS][CAPACITY];
static JSAMPROW input_rows[NROWS + 2];
static JSAMPROW output_rows[NROWS];

static JSAMPROW
aligned_ptr(JSAMPLE *src, unsigned extra)
{
  return (JSAMPROW)((((uintptr_t)src + 15) & ~(uintptr_t)15) + extra);
}

static int
verify(unsigned width, int vertical, unsigned extra, unsigned seed)
{
  JSAMPARRAY input = input_rows + 1; /* input[-1] and input[+2] are valid */
  JSAMPARRAY output = output_rows;
  int r, col, nrows = vertical == 2 ? 2 : NROWS;

  for (r = 0; r < NROWS + 2; r++) {
    input_rows[r] = aligned_ptr(input_mem[r], extra);
    for (col = 0; col < MAX_WIDTH; col++)
      input_rows[r][col] =
        (JSAMPLE)((seed + r * 71 + col * 43 + col * r * 13) & 255);
  }
  for (r = 0; r < NROWS; r++) {
    output_rows[r] = aligned_ptr(output_mem[r], extra);
    memset(output_rows[r], 0xa5, 2 * MAX_WIDTH + 16);
  }

  if (vertical == 2)
    jsimd_h2v2_fancy_upsample_ps2mmi(NROWS, width, input, &output);
  else
    jsimd_h2v1_fancy_upsample_ps2mmi(NROWS, width, input, &output);

  for (r = 0; r < NROWS; r++) {
    int input_row = vertical == 2 ? r / 2 : r;
    int far_row = input_row + ((vertical == 2 && (r & 1)) ? 1 : -1);
    if (vertical == 1)
      far_row = input_row;
    for (col = 0; col < (int)width; col++) {
      unsigned left = col > 0 ? col - 1 : col;
      unsigned right = col + 1 < width ? col + 1 : col;
      unsigned a = (unsigned)input[input_row][col];
      unsigned expected_l, expected_r;
      if (vertical == 2) {
        unsigned t = 3 * a + input[far_row][col];
        unsigned tl = 3 * input[input_row][left] + input[far_row][left];
        unsigned tr = 3 * input[input_row][right] + input[far_row][right];
        expected_l = (3 * t + tl + 8) >> 4;
        expected_r = (3 * t + tr + 7) >> 4;
      } else {
        expected_l = (3 * a + input[input_row][left] + 1) >> 2;
        expected_r = (3 * a + input[input_row][right] + 2) >> 2;
      }
      if (output[r][col * 2] != (JSAMPLE)expected_l ||
          output[r][col * 2 + 1] != (JSAMPLE)expected_r) {
        printf("FAIL fancy: width=%u v=%d offset=%u seed=%u row=%d col=%d\n",
               width, vertical, extra, seed, r, col);
        return 1;
      }
    }
    for (col = 2 * width; col < 2 * width + 16; col++) {
      if (output[r][col] != 0xa5) {
        printf("OVERWRITE fancy: width=%u v=%d offset=%u row=%d col=%d\n",
               width, vertical, extra, r, col);
        return 1;
      }
    }
  }

  (void)nrows;
  return 0;
}

int
main(void)
{
  static const unsigned widths[] =
    { 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 47, 64, 65 };
  unsigned i, extra, seed;
  int v;
  unsigned checks = 0;

  for (v = 1; v <= 2; v++)
    for (extra = 0; extra < 2; extra++)
      for (seed = 0; seed < 2; seed++)
        for (i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
          if (verify(widths[i], v, extra, seed ? 191 : 0))
            return 1;
          checks++;
        }

  printf("PS2 MMI fancy upsampling: PASS (%u cases)\n", checks);
  return 0;
}
