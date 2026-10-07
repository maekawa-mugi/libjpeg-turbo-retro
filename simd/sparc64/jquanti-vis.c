/*
 * Integer sample conversion and quantization (SPARC VIS 1)
 *
 * This file intentionally uses VIS1 only.  VIS2 bshuffle and later
 * instructions are not required.
 */

#include "../jsimdint.h"
#include <visintrin.h>


HIDDEN void
jsimd_convsamp_vis(JSAMPARRAY sample_data, JDIMENSION start_col,
                   DCTELEM *workspace)
{
  const __v2hi widen_exact = { 256, 256 };
  const __v4hi center = { CENTERJSAMPLE, CENTERJSAMPLE,
                          CENTERJSAMPLE, CENTERJSAMPLE };
  int row;

  for (row = 0; row < DCTSIZE; row++) {
    JSAMPROW src = sample_data[row] + start_col;
    DCTELEM *dst = workspace + row * DCTSIZE;

    if (((JUINTPTR)src & 3) == 0 && ((JUINTPTR)dst & 7) == 0) {
      __v4qi s0 = *(const __v4qi *)(const void *)(src + 0);
      __v4qi s1 = *(const __v4qi *)(const void *)(src + 4);
      __v4hi d0 = __vis_fmul8x16au(s0, widen_exact);
      __v4hi d1 = __vis_fmul8x16au(s1, widen_exact);

      d0 = __vis_fpsub16(d0, center);
      d1 = __vis_fpsub16(d1, center);

      *(__v4hi *)(void *)(dst + 0) = d0;
      *(__v4hi *)(void *)(dst + 4) = d1;
    } else {
      int col;

      for (col = 0; col < DCTSIZE; col++)
        dst[col] = (DCTELEM)(GETJSAMPLE(src[col]) - CENTERJSAMPLE);
    }
  }
}
