/*
 * Host-side exhaustive proof of the split IJG 16.16 YCbCr coefficient math.
 * This does NOT execute any R5900 MMI instruction.  Run separately from the
 * PS2 correctness suite to catch sign and rounding regressions quickly.
 *
 * cc -O2 -std=c99 -Wall -Wextra -Werror -fsanitize=undefined \
 *    simd/ps2/test-color-math-host.c -o /tmp/test-color-math
 * /tmp/test-color-math
 *
 * SPDX-License-Identifier: Zlib
 */
#include <stdio.h>

static int
clip(int x)
{
  return x < 0 ? 0 : x > 255 ? 255 : x;
}

int
main(void)
{
  unsigned cases = 0;
  int cbi, cri, y;

  for (cbi = 0; cbi <= 255; cbi++) {
    for (cri = 0; cri <= 255; cri++) {
      int cb = cbi - 128, cr = cri - 128;
      int ref_r = (91881 * cr + 32768) >> 16;
      int ref_g = (-22554 * cb - 46802 * cr + 32768) >> 16;
      int ref_b = (116130 * cb + 32768) >> 16;
      int p0 = 26345 * cr, p1 = -14942 * cb;
      int p2 = -22554 * cb, p3 = 18734 * cr;
      int new_r = cr + ((p0 + 32768) >> 16);
      int new_g = -cr + ((p2 + p3 + 32768) >> 16);
      int new_b = 2 * cb + ((p1 + 32768) >> 16);

      if (ref_r != new_r || ref_g != new_g || ref_b != new_b) {
        printf("FAIL offsets cb=%d cr=%d\n", cbi, cri);
        return 1;
      }
      for (y = 0; y <= 255; y++) {
        if (clip(y + ref_r) != clip(y + new_r) ||
            clip(y + ref_g) != clip(y + new_g) ||
            clip(y + ref_b) != clip(y + new_b)) {
          printf("FAIL RGB cb=%d cr=%d y=%d\n", cbi, cri, y);
          return 1;
        }
      }
      cases++;
    }
  }
  printf("PS2 MMI chroma math (host): PASS (%u Cb/Cr pairs, 256 Y each)\n",
         cases);
  return 0;
}
