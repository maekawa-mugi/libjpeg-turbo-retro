/* Portable, exhaustive correctness test for the optional 3-KiB
 * YCbCr chroma table.  Does not execute R5900 instructions.
 * Verify every pair of input chroma bytes against IJG 16.16 math, keeping
 * green's two terms together before the signed rounding shift.
 * SPDX-License-Identifier: Zlib
 */
#include <stdint.h>
#include <stdio.h>
#define INLINE inline
#include "color-table-mmi.h"

static int
clip(int x)
{
  return x < 0 ? 0 : x > 255 ? 255 : x;
}

int
main(void)
{
  unsigned cb, cr, y, tests = 0;
  static const unsigned ys[] = { 0, 1, 127, 128, 254, 255 };
  for (cb = 0; cb < 256; cb++) {
    for (cr = 0; cr < 256; cr++) {
      int c = (int)cb - 128, r = (int)cr - 128;
      int dr = (91881 * r + 32768) >> 16;
      int dg = (-22554 * c - 46802 * r + 32768) >> 16;
      int db = (116130 * c + 32768) >> 16;
      int tr, tg, tb;
      ps2_table_chroma_offsets(cb, cr, &tr, &tg, &tb);
      if (tr != dr || tg != dg || tb != db) {
        printf("FAIL table offsets Cb=%u Cr=%u: (%d,%d,%d) != (%d,%d,%d)\n",
               cb, cr, tr, tg, tb, dr, dg, db);
        return 1;
      }
      for (y = 0; y < sizeof(ys) / sizeof(ys[0]); y++) {
        int l = (int)ys[y];
        if (clip(l + tr) != clip(l + dr) ||
            clip(l + tg) != clip(l + dg) ||
            clip(l + tb) != clip(l + db)) {
          printf("FAIL table output Cb=%u Cr=%u Y=%d\n", cb, cr, l);
          return 1;
        }
      }
      tests++;
    }
  }
  printf("PS2 chroma lookup table (host): PASS (%u pairs)\n", tests);
  return 0;
}
