/*
 * PS2 EE 8-bit accurate integer IDCT reference backend.
 *
 * Algorithm adapted from src/jidctint.c in this repository.
 * Copyright (C) 1991-1998, Thomas G. Lane.
 * Modifications developed 2002-2018 by Guido Vollbeding.
 * libjpeg-turbo modifications copyright its respective authors.
 * For conditions of distribution and use, see README.ijg.
 *
 * This deliberately retains IJG's arithmetic and table-equivalent range
 * limiting.  It is the correctness baseline for a future 128-bit MMI
 * butterfly implementation, not yet a SIMD-speed implementation.
 *
 * Enable only with WITH_PS2_EXPERIMENTAL_IDCT=ON.
 */
#include "../jsimdint.h"
#include <stdint.h>
#include <string.h>

#define CONST_BITS 13
#define PASS1_BITS 2
#define MULTIPLY(var, constant) MULTIPLY16C16(var, constant)
#define DEQUANTIZE(coef, quantval) (((ISLOW_MULT_TYPE)(coef)) * (quantval))

/* Fixed-point constants match the 8-bit JDCT_ISLOW implementation. */
#define FIX_0_298631336  ((JLONG)2446)
#define FIX_0_390180644  ((JLONG)3196)
#define FIX_0_541196100  ((JLONG)4433)
#define FIX_0_765366865  ((JLONG)6270)
#define FIX_0_899976223  ((JLONG)7373)
#define FIX_1_175875602  ((JLONG)9633)
#define FIX_1_501321110  ((JLONG)12299)
#define FIX_1_847759065  ((JLONG)15137)
#define FIX_1_961570560  ((JLONG)16069)
#define FIX_2_053119869  ((JLONG)16819)
#define FIX_2_562915447  ((JLONG)20995)
#define FIX_3_072711026  ((JLONG)25172)

/* Reproduce jdmaster.c's post-IDCT range table (with & RANGE_MASK).
 * Do not replace with simple 0..255 clipping: malformed JPEG coefficients
 * can wrap through the IJG range table and must behave identically. */
static JSAMPLE
ps2_idct_range(int sample)
{
  const unsigned int i = (unsigned int)sample & RANGE_MASK;
  if (i < 128)
    return (JSAMPLE)(i + 128);
  if (i < 512)
    return (JSAMPLE)255;
  if (i < 896)
    return (JSAMPLE)0;
  return (JSAMPLE)(i - 896);
}

static int
ps2_dc_only(const JCOEF *coeff)
{
  int i;
  for (i = 1; i < DCTSIZE2; i++)
    if (coeff[i])
      return 0;
  return 1;
}

HIDDEN void
jsimd_idct_islow_ps2ref(void *dct_table, JCOEFPTR coef_block,
                         JSAMPARRAY output_buf, JDIMENSION output_col)
{
  JLONG tmp0, tmp1, tmp2, tmp3;
  JLONG tmp10, tmp11, tmp12, tmp13;
  JLONG z1, z2, z3, z4, z5;
  JCOEFPTR inptr;
  ISLOW_MULT_TYPE *quantptr;
  int *wsptr;
  JSAMPROW outptr;
  int ctr;
  int workspace[DCTSIZE2];      /* buffers data between passes */
  SHIFT_TEMPS

  /* Common, fully DC-only block: no butterflies and no row padding
   * writes.  Copy eight bytes per row, even for unaligned output_col. */
  if (ps2_dc_only(coef_block)) {
    const JLONG dc = (JLONG)coef_block[0] *
                     ((const ISLOW_MULT_TYPE *)dct_table)[0];
    const JSAMPLE pixel = ps2_idct_range((int)DESCALE(dc, 3));
    uint64_t eight = (uint64_t)pixel * UINT64_C(0x0101010101010101);
    int row;
    for (row = 0; row < DCTSIZE; row++)
      memcpy(output_buf[row] + output_col, &eight, DCTSIZE);
    return;
  }

  /* Pass 1: process columns from input, store into work array. */
  /* Note results are scaled up by sqrt(8) compared to a true IDCT; */
  /* furthermore, we scale the results by 2**PASS1_BITS. */

  inptr = coef_block;
  quantptr = (ISLOW_MULT_TYPE *)dct_table;
  wsptr = workspace;
  for (ctr = DCTSIZE; ctr > 0; ctr--) {
    /* Due to quantization, we will usually find that many of the input
     * coefficients are zero, especially the AC terms.  We can exploit this
     * by short-circuiting the IDCT calculation for any column in which all
     * the AC terms are zero.  In that case each output is equal to the
     * DC coefficient (with scale factor as needed).
     * With typical images and quantization tables, half or more of the
     * column DCT calculations can be simplified this way.
     */

    if (inptr[DCTSIZE * 1] == 0 && inptr[DCTSIZE * 2] == 0 &&
        inptr[DCTSIZE * 3] == 0 && inptr[DCTSIZE * 4] == 0 &&
        inptr[DCTSIZE * 5] == 0 && inptr[DCTSIZE * 6] == 0 &&
        inptr[DCTSIZE * 7] == 0) {
      /* AC terms all zero */
      int dcval = LEFT_SHIFT(DEQUANTIZE(inptr[DCTSIZE * 0],
                             quantptr[DCTSIZE * 0]), PASS1_BITS);

      wsptr[DCTSIZE * 0] = dcval;
      wsptr[DCTSIZE * 1] = dcval;
      wsptr[DCTSIZE * 2] = dcval;
      wsptr[DCTSIZE * 3] = dcval;
      wsptr[DCTSIZE * 4] = dcval;
      wsptr[DCTSIZE * 5] = dcval;
      wsptr[DCTSIZE * 6] = dcval;
      wsptr[DCTSIZE * 7] = dcval;

      inptr++;                  /* advance pointers to next column */
      quantptr++;
      wsptr++;
      continue;
    }

    /* Even part: reverse the even part of the forward DCT. */
    /* The rotator is sqrt(2)*c(-6). */

    z2 = DEQUANTIZE(inptr[DCTSIZE * 2], quantptr[DCTSIZE * 2]);
    z3 = DEQUANTIZE(inptr[DCTSIZE * 6], quantptr[DCTSIZE * 6]);

    z1 = MULTIPLY(z2 + z3, FIX_0_541196100);
    tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065);
    tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865);

    z2 = DEQUANTIZE(inptr[DCTSIZE * 0], quantptr[DCTSIZE * 0]);
    z3 = DEQUANTIZE(inptr[DCTSIZE * 4], quantptr[DCTSIZE * 4]);

    tmp0 = LEFT_SHIFT(z2 + z3, CONST_BITS);
    tmp1 = LEFT_SHIFT(z2 - z3, CONST_BITS);

    tmp10 = tmp0 + tmp3;
    tmp13 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp12 = tmp1 - tmp2;

    /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */

    tmp0 = DEQUANTIZE(inptr[DCTSIZE * 7], quantptr[DCTSIZE * 7]);
    tmp1 = DEQUANTIZE(inptr[DCTSIZE * 5], quantptr[DCTSIZE * 5]);
    tmp2 = DEQUANTIZE(inptr[DCTSIZE * 3], quantptr[DCTSIZE * 3]);
    tmp3 = DEQUANTIZE(inptr[DCTSIZE * 1], quantptr[DCTSIZE * 1]);

    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    z4 = tmp1 + tmp3;
    z5 = MULTIPLY(z3 + z4, FIX_1_175875602); /* sqrt(2) * c3 */

    tmp0 = MULTIPLY(tmp0, FIX_0_298631336); /* sqrt(2) * (-c1+c3+c5-c7) */
    tmp1 = MULTIPLY(tmp1, FIX_2_053119869); /* sqrt(2) * ( c1+c3-c5+c7) */
    tmp2 = MULTIPLY(tmp2, FIX_3_072711026); /* sqrt(2) * ( c1+c3+c5-c7) */
    tmp3 = MULTIPLY(tmp3, FIX_1_501321110); /* sqrt(2) * ( c1+c3-c5-c7) */
    z1 = MULTIPLY(z1, -FIX_0_899976223); /* sqrt(2) * ( c7-c3) */
    z2 = MULTIPLY(z2, -FIX_2_562915447); /* sqrt(2) * (-c1-c3) */
    z3 = MULTIPLY(z3, -FIX_1_961570560); /* sqrt(2) * (-c3-c5) */
    z4 = MULTIPLY(z4, -FIX_0_390180644); /* sqrt(2) * ( c5-c3) */

    z3 += z5;
    z4 += z5;

    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;

    /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

    wsptr[DCTSIZE * 0] = (int)DESCALE(tmp10 + tmp3, CONST_BITS - PASS1_BITS);
    wsptr[DCTSIZE * 7] = (int)DESCALE(tmp10 - tmp3, CONST_BITS - PASS1_BITS);
    wsptr[DCTSIZE * 1] = (int)DESCALE(tmp11 + tmp2, CONST_BITS - PASS1_BITS);
    wsptr[DCTSIZE * 6] = (int)DESCALE(tmp11 - tmp2, CONST_BITS - PASS1_BITS);
    wsptr[DCTSIZE * 2] = (int)DESCALE(tmp12 + tmp1, CONST_BITS - PASS1_BITS);
    wsptr[DCTSIZE * 5] = (int)DESCALE(tmp12 - tmp1, CONST_BITS - PASS1_BITS);
    wsptr[DCTSIZE * 3] = (int)DESCALE(tmp13 + tmp0, CONST_BITS - PASS1_BITS);
    wsptr[DCTSIZE * 4] = (int)DESCALE(tmp13 - tmp0, CONST_BITS - PASS1_BITS);

    inptr++;                    /* advance pointers to next column */
    quantptr++;
    wsptr++;
  }

  /* Pass 2: process rows from work array, store into output array. */
  /* Note that we must descale the results by a factor of 8 == 2**3, */
  /* and also undo the PASS1_BITS scaling. */

  wsptr = workspace;
  for (ctr = 0; ctr < DCTSIZE; ctr++) {
    outptr = output_buf[ctr] + output_col;
    /* Rows of zeroes can be exploited in the same way as we did with columns.
     * However, the column calculation has created many nonzero AC terms, so
     * the simplification applies less often (typically 5% to 10% of the time).
     * On machines with very fast multiplication, it's possible that the
     * test takes more time than it's worth.  In that case this section
     * may be commented out.
     */

#ifndef NO_ZERO_ROW_TEST
    if (wsptr[1] == 0 && wsptr[2] == 0 && wsptr[3] == 0 && wsptr[4] == 0 &&
        wsptr[5] == 0 && wsptr[6] == 0 && wsptr[7] == 0) {
      /* AC terms all zero */
      JSAMPLE dcval = ps2_idct_range((int)DESCALE((JLONG)wsptr[0],
                                                PASS1_BITS + 3));

      outptr[0] = dcval;
      outptr[1] = dcval;
      outptr[2] = dcval;
      outptr[3] = dcval;
      outptr[4] = dcval;
      outptr[5] = dcval;
      outptr[6] = dcval;
      outptr[7] = dcval;

      wsptr += DCTSIZE;         /* advance pointer to next row */
      continue;
    }
#endif

    /* Even part: reverse the even part of the forward DCT. */
    /* The rotator is sqrt(2)*c(-6). */

    z2 = (JLONG)wsptr[2];
    z3 = (JLONG)wsptr[6];

    z1 = MULTIPLY(z2 + z3, FIX_0_541196100);
    tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065);
    tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865);

    tmp0 = LEFT_SHIFT((JLONG)wsptr[0] + (JLONG)wsptr[4], CONST_BITS);
    tmp1 = LEFT_SHIFT((JLONG)wsptr[0] - (JLONG)wsptr[4], CONST_BITS);

    tmp10 = tmp0 + tmp3;
    tmp13 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp12 = tmp1 - tmp2;

    /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */

    tmp0 = (JLONG)wsptr[7];
    tmp1 = (JLONG)wsptr[5];
    tmp2 = (JLONG)wsptr[3];
    tmp3 = (JLONG)wsptr[1];

    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    z4 = tmp1 + tmp3;
    z5 = MULTIPLY(z3 + z4, FIX_1_175875602); /* sqrt(2) * c3 */

    tmp0 = MULTIPLY(tmp0, FIX_0_298631336); /* sqrt(2) * (-c1+c3+c5-c7) */
    tmp1 = MULTIPLY(tmp1, FIX_2_053119869); /* sqrt(2) * ( c1+c3-c5+c7) */
    tmp2 = MULTIPLY(tmp2, FIX_3_072711026); /* sqrt(2) * ( c1+c3+c5-c7) */
    tmp3 = MULTIPLY(tmp3, FIX_1_501321110); /* sqrt(2) * ( c1+c3-c5-c7) */
    z1 = MULTIPLY(z1, -FIX_0_899976223); /* sqrt(2) * ( c7-c3) */
    z2 = MULTIPLY(z2, -FIX_2_562915447); /* sqrt(2) * (-c1-c3) */
    z3 = MULTIPLY(z3, -FIX_1_961570560); /* sqrt(2) * (-c3-c5) */
    z4 = MULTIPLY(z4, -FIX_0_390180644); /* sqrt(2) * ( c5-c3) */

    z3 += z5;
    z4 += z5;

    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;

    /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

    outptr[0] = ps2_idct_range((int)DESCALE(tmp10 + tmp3,
                                         CONST_BITS + PASS1_BITS + 3));
    outptr[7] = ps2_idct_range((int)DESCALE(tmp10 - tmp3,
                                         CONST_BITS + PASS1_BITS + 3));
    outptr[1] = ps2_idct_range((int)DESCALE(tmp11 + tmp2,
                                         CONST_BITS + PASS1_BITS + 3));
    outptr[6] = ps2_idct_range((int)DESCALE(tmp11 - tmp2,
                                         CONST_BITS + PASS1_BITS + 3));
    outptr[2] = ps2_idct_range((int)DESCALE(tmp12 + tmp1,
                                         CONST_BITS + PASS1_BITS + 3));
    outptr[5] = ps2_idct_range((int)DESCALE(tmp12 - tmp1,
                                         CONST_BITS + PASS1_BITS + 3));
    outptr[3] = ps2_idct_range((int)DESCALE(tmp13 + tmp0,
                                         CONST_BITS + PASS1_BITS + 3));
    outptr[4] = ps2_idct_range((int)DESCALE(tmp13 - tmp0,
                                         CONST_BITS + PASS1_BITS + 3));

    wsptr += DCTSIZE;           /* advance pointer to next row */
  }
}

