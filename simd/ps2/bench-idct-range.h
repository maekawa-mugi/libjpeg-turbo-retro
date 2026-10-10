/* The exact shared 8-bit IDCT range buffer expected by both IJG kernels.
 * The integer kernel indexes base + CENTERJSAMPLE; the float kernel
 * indexes base directly.  Populate BOTH overlapping views.
 * Expects 5 * 256 + 128 JSAMPLE entries, just as jdmaster.c allocates.
 * SPDX-License-Identifier: IJG
 */
#ifndef PS2_BENCH_IDCT_RANGE_H
#define PS2_BENCH_IDCT_RANGE_H
static inline void
ps2_bench_init_idct_range(JSAMPLE *table)
{
  unsigned i;
  for (i = 0; i < 1024; i++) {
    unsigned value = i < 128 ? i + 128 :
                     i < 512 ? 255 : i < 896 ? 0 : i - 896;
    table[256 + 128 + i] = (JSAMPLE)value;
  }
  /* The float kernel's unshifted base[0..127], which the integer
   * post-IDCT view never reads, must contain 0..127. */
  for (i = 0; i < 128; i++)
    table[256 + i] = (JSAMPLE)i;
}
#endif
