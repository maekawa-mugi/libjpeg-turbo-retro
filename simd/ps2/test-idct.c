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
static ISLOW_MULT_TYPE quant_storage[64 + 16] __attribute__((aligned(16)));
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
run_case(unsigned iteration, unsigned seed, int align_case)
{
  struct jpeg_decompress_struct cinfo;
  jpeg_component_info component;
  JSAMPLE *post = range_table + 256 + 128;
  unsigned state = seed + iteration * 7919U;
  int i, row;
  JCOEFPTR coef = coef_storage + ((align_case & 1) ? 1 : 0);
  ISLOW_MULT_TYPE *quant = quant_storage + ((align_case & 2) ? 1 : 0);
  unsigned output_col = (align_case & 1) ? 7 : 0;

  memset(&cinfo, 0, sizeof(cinfo));
  memset(&component, 0, sizeof(component));
  cinfo.data_precision = 8;
  cinfo.sample_range_limit = range_table + 256;
  component.dct_table = quant;

  for (i = 0; i < 1024; i++)
    post[i] = expected_range(i);

  for (i = 0; i < 64; i++) {
    state = state * 1664525U + 1013904223U;
    quant[i] = (ISLOW_MULT_TYPE)(1 + (state % 31U));
    coef[i] = 0;
  }

  /*
   * Exercise DC-only, single-AC, sparse, dense and high-frequency blocks.
   * Keep intermediate values in the original 32-bit fixed-point range.
   */
  if (iteration % 8 == 0) {
    coef[0] = (JCOEF)((int)(state % 1024U) - 512);
  } else if (iteration % 8 == 1 || iteration % 8 == 2) {
    coef[0] = (JCOEF)((int)(state % 257U) - 128);
    coef[iteration % 8 == 1 ? 1 : 63] =
      (JCOEF)((int)((state >> 8) % 129U) - 64);
  } else {
    for (i = 0; i < 64; i++) {
      state = state * 1664525U + 1013904223U;
      if (iteration % 8 == 3 && i != 0 && (state & 7U) != 0)
        continue;
      if (iteration % 8 == 4 && (i & 1) == 0)
        continue;
      coef[i] = (JCOEF)((int)((state >> 12) % 129U) - 64);
    }
  }
  /*
   * Additional signed-range stress: a single odd-frequency coefficient
   * with a larger quantization multiplier.  This exercises the guarded
   * middle-stage multiply path without generating arbitrarily large
   * coefficient blocks that could overflow the IJG 32-bit reference.
   */
  if (iteration % 16 == 7) {
    for (i = 0; i < 64; i++) {
      coef[i] = 0;
      quant[i] = 1;
    }
    coef[9] = (JCOEF)((iteration & 16U) ? 240 : -240);
    quant[9] = 16;
  }

  /* Separate alignment variations exercise both the PMULTH dequantizer
   * and the reference scalar multiplication fallback. */
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
        printf("FAIL IDCT iteration=%u seed=%u col=%u align=%d row=%d byte=%d: %u != %u\n",
               iteration, seed, output_col, align_case, row, i,
               reference_rows[row][i], ps2_rows[row][i]);
        return 1;
      }
  return 0;
}

int
main(void)
{
  unsigned n, seed;
  int align_case;
  unsigned count = 0;

  for (align_case = 0; align_case < 4; align_case++)
    for (seed = 0; seed <= 1; seed++)
      for (n = 0; n < 256; n++) {
        if (run_case(n, seed ? 91871U : 0U, align_case))
          return 1;
        count++;
      }
  printf("PS2 MMI islow IDCT: PASS (%u reference comparisons)\n", count);
  return 0;
}
