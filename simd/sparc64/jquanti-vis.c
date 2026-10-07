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


typedef union {
  __v4hi v;
  int16_t s[4];
  uint16_t u[4];
} vis_quant4;


/*
 * VIS1 has no unsigned 16x16 high-half multiply and no per-lane variable
 * shift.  Keep those two operations in integer registers, but vectorize sign
 * handling and correction addition four coefficients at a time.  This is
 * deliberately exact rather than approximating the reciprocal arithmetic.
 */
HIDDEN void
jsimd_quantize_vis(JCOEFPTR coef_block, DCTELEM *divisors,
                   DCTELEM *workspace)
{
  UDCTELEM *recip = (UDCTELEM *)divisors;
  UDCTELEM *corr = (UDCTELEM *)divisors + DCTSIZE2;
  DCTELEM *shift = divisors + 3 * DCTSIZE2;
  const __v4hi zero = { 0, 0, 0, 0 };
  int i;

  for (i = 0; i < DCTSIZE2; i += 4) {
    vis_quant4 in, neg, absv, cv, sum;
    int lane;

    in.v = *(__v4hi *)(void *)(workspace + i);
    neg.v = __vis_fpsub16(zero, in.v);

    /*
     * Do not depend on the architecture-specific compare-mask bit ordering.
     * The negation is still four-lane VIS; scalar sign selection is exact and
     * folds into the scalar multiply/variable-shift glue below.
     */
    for (lane = 0; lane < 4; lane++)
      absv.u[lane] = in.s[lane] < 0 ? neg.u[lane] : in.u[lane];

    cv.u[0] = corr[i + 0];
    cv.u[1] = corr[i + 1];
    cv.u[2] = corr[i + 2];
    cv.u[3] = corr[i + 3];
    sum.v = __vis_fpadd16(absv.v, cv.v);

    for (lane = 0; lane < 4; lane++) {
      uint32_t product = (uint32_t)sum.u[lane] * recip[i + lane];
      int16_t q = (int16_t)(product >> (16 + shift[i + lane]));

      coef_block[i + lane] =
        in.s[lane] < 0 ? (JCOEF)-q : (JCOEF)q;
    }
  }
}
