/* Host-only exhaustive range-lookup regression.
 * Checks all 1024 wrapped integer IDCT samples and both overlapping
 * sample_range_limit layouts consumed by JPEG islow and float IDCT.
 * SPDX-License-Identifier: Zlib
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef uint8_t JSAMPLE;
#include "idct-range-lut.h"

static unsigned
idct_clip(unsigned i)
{
  return i < 128 ? i + 128 : i < 512 ? 255 : i < 896 ? 0 : i - 896;
}

int main(void)
{
  JSAMPLE table[5 * 256 + 128];
  JSAMPLE *base = table + 256;
  unsigned i;
  memset(table, 0xc9, sizeof(table));
  for (i = 0; i < 1024; ++i)
    base[128 + i] = (JSAMPLE)idct_clip(i);
  for (i = 0; i < 128; ++i)
    base[i] = (JSAMPLE)i;
  for (i = 0; i < 1024; ++i) {
    unsigned wanted = idct_clip(i);
    if (ps2_idct_lut[i] != wanted || base[128 + i] != wanted) {
      printf("FAIL,idct,wrapped_lut,index=%u\n", i);
      return 1;
    }
    if (i < 256 && base[i] != i) {
      printf("FAIL,idct,float_identity,index=%u\n", i);
      return 1;
    }
    if (i >= 256 && i < 512 && base[i] != 255) {
      printf("FAIL,idct,float_upper_clamp,index=%u\n", i);
      return 1;
    }
    if (i >= 512 && base[i] != 0) {
      printf("FAIL,idct,float_lower_clamp,index=%u\n", i);
      return 1;
    }
  }
  puts("PASS,idct,range_lut_and_float_base,1024_cases");
  return 0;
}
