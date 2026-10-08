/*
 * Compare the PS2 MMI islow IDCT against IJG reference implementation.
 * Does not require a complete JPEG stream or initialized memory manager.
 *
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ROW_BYTES 48

static JCOEF coef_storage[64 + 16] __attribute__((aligned(16)));
static ISLOW_MULT_TYPE quant[64];
static JSAMPLE reference_mem[8][ROW_BYTES];
static JSAMPLE ps2_mem[8][ROW_BYTES];
static JSAMPROW reference_rows[8], ps2_rows[8];
static JSAMPLE range_table[5 * 256 + 128];

static JSAMPLE
expected_range(int x)
{
  unsigned int index = (unsigned int)x & 1023;
  if (index < 128)
    return (JSAMPLE)(index + 128);
  if (index < 512)
    return 255;
  if (index < 896)
    return 0;
  return (JSAMPLE)(index - 896);
}

static int
run_case(unsigned iteration, unsigned seed, int shifted)
{
  struct jpeg_decompress_struct cinfo;
  jpeg_component_info component;
  JSAMPLE *post = range_table + 256 + 128;
  unsigned state = seed + iteration * 7919U;
  int i, row;
  JCOEFPTR coef = coef_storage + (shifted ? 1 : 0);
  unsigned output_col = shifted ? 7 : 0;

  memset(&cinfo, 0, sizeof(cinfo));
  memset(&component, 0, sizeof(component));
  cinfo.data_precision = 8;
  cinfo.sample_range_limit = range_table + 256;
  component.dct_table = quant;

  for (i = 0; i < 1024; i++)
    post[i] = expected_range(i);

  for (i = 0; i < 64; i++) {
    state = state * 1664525U + 1013904223U;
    quant[i] = (ISLOW_MULT_TYPE)(1 + (state % 9U));
    coef[i] = 0;
  }

  /* DC only (including signed values), sparse AC and dense AC cases. */
  if (iteration % 5 == 0) {
    coef[0] = (JCOEF)((int)(state % 1024U) - 512);
  } else {
    for (i = 0; i < 64; i++) {
      state = state * 1664525U + 1013904223U;
      if (iteration % 5 == 1 && i != 0 && (state & 7U) != 0)
        continue;
      coef[i] = (JCOEF)((int)((state >> 12) % 129U) - 64);
    }
  }
  /* The second pass uses an actually unaligned (but 2-byte aligned)
   * coefficient pointer, forcing the scalar zero-AC detector. */
  for (row = 0; row < 8; row++) {
    reference_rows[row] = reference_mem[row];
    ps2_rows[row] = ps2_mem[row];
    memset(reference_rows[row], 0xc9, ROW_BYTES);
    memset(ps2_rows[row], 0xc9, ROW_BYTES);
  }
  _jpeg_idct_islow(&cinfo, &component, coef, reference_rows, output_col);
  jsimd_idct_islow_ps2mmi(quant, coef, ps2_rows, output_col);

  for (row = 0; row < 8; row++)
    for (i = 0; i < ROW_BYTES; i++)
      if (reference_rows[row][i] != ps2_rows[row][i]) {
        printf("FAIL IDCT iteration=%u seed=%u col=%u row=%d byte=%d: %u != %u\n",
               iteration, seed, output_col, row, i,
               reference_rows[row][i], ps2_rows[row][i]);
        return 1;
      }
  return 0;
}

int
main(void)
{
  unsigned n, seed;
  int offset;
  unsigned count = 0;

  for (offset = 0; offset <= 1; offset++)
    for (seed = 0; seed <= 1; seed++)
      for (n = 0; n < 128; n++) {
        if (run_case(n, seed ? 91871U : 0U, offset))
          return 1;
        count++;
      }
  printf("PS2 MMI islow IDCT: PASS (%u reference comparisons)\n", count);
  return 0;
}
