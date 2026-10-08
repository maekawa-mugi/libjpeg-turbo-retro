/*
 * PS2 EE MMI plain upsampling self-test.
 * Build with -DWITH_PS2_MMI_TESTS=ON; run the resulting ELF on a PS2.
 *
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_WIDTH 65
#define ROW_CAPACITY (2 * MAX_WIDTH + 64)
#define ROWS 4

static JSAMPLE input_storage[ROWS][ROW_CAPACITY];
static JSAMPLE output_storage[ROWS][ROW_CAPACITY];
static JSAMPROW input_rows[ROWS];
static JSAMPROW output_rows[ROWS];

/* Return a 16-byte aligned address plus the requested optional offset. */
static JSAMPROW
sample_ptr(JSAMPLE *p, int offset)
{
  return (JSAMPROW)((((uintptr_t)p + 15) & ~(uintptr_t)15) + offset);
}

static int
test_case(int width, int vertical, int source_offset, int output_pattern)
{
  JSAMPARRAY out = output_rows;
  int r, i;
  int source_rows = vertical == 2 ? ROWS / 2 : ROWS;

  for (r = 0; r < ROWS; r++) {
    input_rows[r] = sample_ptr(input_storage[r], source_offset);
    output_rows[r] = sample_ptr(output_storage[r],
                                (output_pattern >> (r & 1)) & 1);
    memset(input_rows[r], 0, (MAX_WIDTH + 1) / 2 + 16);
    memset(output_rows[r], 0xa5, MAX_WIDTH + 16);
  }
  for (r = 0; r < source_rows; r++)
    for (i = 0; i < (width + 1) / 2; i++)
      input_rows[r][i] = (JSAMPLE)((r * 37 + i * 29 + 7) & 255);

  if (vertical == 2)
    jsimd_h2v2_upsample_ps2mmi(ROWS, (JDIMENSION)width, input_rows, &out);
  else
    jsimd_h2v1_upsample_ps2mmi(ROWS, (JDIMENSION)width, input_rows, &out);

  for (r = 0; r < ROWS; r++) {
    int src_row = vertical == 2 ? r / 2 : r;
    for (i = 0; i < width; i++) {
      JSAMPLE want = input_rows[src_row][i / 2];
      if (output_rows[r][i] != want) {
        printf("MMI FAIL width=%d v=%d src_offset=%d dst_pattern=%d row=%d col=%d: %d != %d\n",
               width, vertical, source_offset, output_pattern, r, i,
               (int)output_rows[r][i], (int)want);
        return 1;
      }
    }
    for (i = width; i < width + 16; i++) {
      if (output_rows[r][i] != 0xa5) {
        printf("MMI overwrite width=%d v=%d src_offset=%d dst_pattern=%d row=%d col=%d\n",
               width, vertical, source_offset, output_pattern, r, i);
        return 1;
      }
    }
  }
  return 0;
}

int
main(void)
{
  static const int widths[] = { 1, 2, 7, 15, 16, 23, 24, 25, 31, 32, 33,
      40, 41, 47, 63, 64, 65 };
  unsigned i;
  int v, source_offset, output_pattern;

  for (v = 1; v <= 2; v++) {
    for (source_offset = 0; source_offset < 2; source_offset++)
      for (output_pattern = 0; output_pattern < 4; output_pattern++)
        for (i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
          if (test_case(widths[i], v, source_offset, output_pattern))
            return 1;
        }
  }
  puts("PS2 MMI upsampling: PASS (272 cases)");
  return 0;
}
