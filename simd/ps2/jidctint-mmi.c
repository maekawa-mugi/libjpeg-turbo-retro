/*
 * PlayStation 2 Emotion Engine partial MMI 8x8 accurate integer IDCT.
 *
 * The full-coefficient 8x8 IDCT is based directly on the IJG/Loeffler
 * integer algorithm in src/jidctint.c, originally:
 * Copyright (C) 1991-1998, Thomas G. Lane.
 * Modifications developed 2002-2018 by Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2015, 2020, 2022, 2026, D. R. Commander.
 * See README.ijg and the original jidctint.c for terms and attribution.  The MMI fast path detects DC-only blocks with 128-bit
 * registers, and bypasses both scalar passes in that common case.
 *
 * This implementation is intentionally conservative: the complete IDCT
 * retains the reference integer operations and exact post-IDCT wrapping,
 * rather than using inaccurate floating point approximations.
 *
 * SPDX-License-Identifier: IJG
 */
#include "../jsimdint.h"
#include <stdint.h>
#include <string.h>

#define CONST_BITS 13
#define PASS1_BITS 2
#define FIX_0_298631336  ((JLONG)2446)          /* FIX(0.298631336) */
#define FIX_0_390180644  ((JLONG)3196)          /* FIX(0.390180644) */
#define FIX_0_541196100  ((JLONG)4433)          /* FIX(0.541196100) */
#define FIX_0_765366865  ((JLONG)6270)          /* FIX(0.765366865) */
#define FIX_0_899976223  ((JLONG)7373)          /* FIX(0.899976223) */
#define FIX_1_175875602  ((JLONG)9633)          /* FIX(1.175875602) */
#define FIX_1_501321110  ((JLONG)12299)         /* FIX(1.501321110) */
#define FIX_1_847759065  ((JLONG)15137)         /* FIX(1.847759065) */
#define FIX_1_961570560  ((JLONG)16069)         /* FIX(1.961570560) */
#define FIX_2_053119869  ((JLONG)16819)         /* FIX(2.053119869) */
#define FIX_2_562915447  ((JLONG)20995)         /* FIX(2.562915447) */
#define FIX_3_072711026  ((JLONG)25172)         /* FIX(3.072711026) */

#define MULTIPLY(var, constant) MULTIPLY16C16(var, constant)

/*
 * IJG's post-IDCT range-limit table maps 0..1023 as follows:
 * 0..127 -> 128..255, 128..511 -> 255,
 * 512..895 -> 0, 896..1023 -> 0..127.
 */
static INLINE JSAMPLE
ps2_idct_range(int x)
{
  unsigned int index = (unsigned int)x & RANGE_MASK;
  if (index < 128)
    return (JSAMPLE)(index + 128);
  if (index < 512)
    return 255;
  if (index < 896)
    return 0;
  return (JSAMPLE)(index - 896);
}

/* Detect nonzero AC coefficients in eight 128-bit chunks, masking DC. */
static int
ps2_dc_only(const JCOEF *coefficients)
{
  static const unsigned short mask[8] __attribute__((aligned(16))) =
    { 0, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff };
  unsigned int reduced[4] __attribute__((aligned(16)));
  int i;

  if (((uintptr_t)coefficients & 15) != 0) {
    for (i = 1; i < DCTSIZE2; i++) {
      if (coefficients[i] != 0)
        return 0;
    }
    return 1;
  }

  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $8, 0(%0)\n\t"
    "lq $9, 0(%1)\n\t"
    "pand $8, $8, $9\n\t"
    "lq $9, 16(%0)\n\t"
    "por $8, $8, $9\n\t"
    "lq $9, 32(%0)\n\t"
    "por $8, $8, $9\n\t"
    "lq $9, 48(%0)\n\t"
    "por $8, $8, $9\n\t"
    "lq $9, 64(%0)\n\t"
    "por $8, $8, $9\n\t"
    "lq $9, 80(%0)\n\t"
    "por $8, $8, $9\n\t"
    "lq $9, 96(%0)\n\t"
    "por $8, $8, $9\n\t"
    "lq $9, 112(%0)\n\t"
    "por $8, $8, $9\n\t"
    "sq $8, 0(%2)\n\t"
    ".set pop\n\t"
    :
    : "r" (coefficients), "r" (mask), "r" (reduced)
    : "$8", "$9", "memory");

  return (reduced[0] | reduced[1] | reduced[2] | reduced[3]) == 0;
}


/*
 * The last Loeffler butterfly has four independent pairs:
 *   {tmp10,tmp11,tmp12,tmp13} +/- {tmp3,tmp2,tmp1,tmp0}.
 * Evaluate four 32-bit signed lanes using R5900 MMI.  Add the same
 * rounding bias as IJG DESCALE before the arithmetic right shift.
 * All operands are explicitly 16-byte-aligned local arrays.
 */
static INLINE void
ps2_butterfly_pass1(const JLONG a[4], const JLONG b[4],
        JLONG plus[4], JLONG minus[4])
{
  static const JLONG round_bias[4] __attribute__((aligned(16))) =
    { 1 << (11 - 1), 1 << (11 - 1),
      1 << (11 - 1), 1 << (11 - 1) };

  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $8, 0(%0)\n\t"
    "lq $9, 0(%1)\n\t"
    "lq $10, 0(%2)\n\t"
    "paddw $11, $8, $9\n\t"
    "psubw $12, $8, $9\n\t"
    "paddw $11, $11, $10\n\t"
    "paddw $12, $12, $10\n\t"
    "psraw $11, $11, 11\n\t"
    "psraw $12, $12, 11\n\t"
    "sq $11, 0(%3)\n\t"
    "sq $12, 0(%4)\n\t"
    ".set pop\n\t"
    :
    : "r" (a), "r" (b), "r" (round_bias), "r" (plus), "r" (minus)
    : "$8", "$9", "$10", "$11", "$12", "memory");
}


/*
 * The last Loeffler butterfly has four independent pairs:
 *   {tmp10,tmp11,tmp12,tmp13} +/- {tmp3,tmp2,tmp1,tmp0}.
 * Evaluate four 32-bit signed lanes using R5900 MMI.  Add the same
 * rounding bias as IJG DESCALE before the arithmetic right shift.
 * All operands are explicitly 16-byte-aligned local arrays.
 */
static INLINE void
ps2_butterfly_pass2(const JLONG a[4], const JLONG b[4],
        JLONG plus[4], JLONG minus[4])
{
  static const JLONG round_bias[4] __attribute__((aligned(16))) =
    { 1 << (18 - 1), 1 << (18 - 1),
      1 << (18 - 1), 1 << (18 - 1) };

  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $8, 0(%0)\n\t"
    "lq $9, 0(%1)\n\t"
    "lq $10, 0(%2)\n\t"
    "paddw $11, $8, $9\n\t"
    "psubw $12, $8, $9\n\t"
    "paddw $11, $11, $10\n\t"
    "paddw $12, $12, $10\n\t"
    "psraw $11, $11, 18\n\t"
    "psraw $12, $12, 18\n\t"
    "sq $11, 0(%3)\n\t"
    "sq $12, 0(%4)\n\t"
    ".set pop\n\t"
    :
    : "r" (a), "r" (b), "r" (round_bias), "r" (plus), "r" (minus)
    : "$8", "$9", "$10", "$11", "$12", "memory");
}


/*
 * Eight simultaneous 16x16 -> 32-bit dequantization products.
 * PMULTH writes the low/high accumulators in the PS2-specific lane order:
 *   LO: {0,1,4,5}, HI: {2,3,6,7}.
 * PCPYLD/PCPYUD put the products back into natural 0..7 order.
 *
 * Keep this in a separate non-inlined function so the R5900 HI/LO
 * accumulators are not live across other compiler-generated arithmetic.
 * The caller guarantees 16-byte alignment and a 16-bit quant table.
 */
static __attribute__((noinline)) void
ps2_dequant8_mmi(const JCOEF *coef, const ISLOW_MULT_TYPE *quant, JLONG *dest)
{
  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $8, 0(%0)\n\t"
    "lq $9, 0(%1)\n\t"
    "pmulth $10, $8, $9\n\t"
    "pmflo $11\n\t"
    "pmfhi $12\n\t"
    "pcpyld $13, $12, $11\n\t"
    "pcpyud $14, $12, $11\n\t"
    "sq $13, 0(%2)\n\t"
    "sq $14, 16(%2)\n\t"
    ".set pop\n\t"
    :
    : "r" (coef), "r" (quant), "r" (dest)
    : "$8", "$9", "$10", "$11", "$12", "$13", "$14", "memory");
}

/* Full block dequantization.  The MMI path is restricted to an aligned,
 * native short multiplier table, matching the 8-bit WITH_SIMD build.
 * An unaligned coefficient buffer or a non-short quant table falls
 * back to reference C multiplication.
 */
static void
ps2_dequant_block(const JCOEF *coef, const ISLOW_MULT_TYPE *quant,
                  JLONG products[DCTSIZE2])
{
  int i;
  if (sizeof(ISLOW_MULT_TYPE) == 2 &&
      (((uintptr_t)coef | (uintptr_t)quant | (uintptr_t)products) & 15) == 0) {
    for (i = 0; i < DCTSIZE2; i += 8)
      ps2_dequant8_mmi(coef + i, quant + i, products + i);
  } else {
    for (i = 0; i < DCTSIZE2; i++)
      products[i] = (JLONG)((ISLOW_MULT_TYPE)coef[i]) * quant[i];
  }
}

/*
 * IJG reference LL&M transform (unaltered fixed-point constants, shifts,
 * and 8-bit limiting, except for the MMI DC-only early exit).
 */
HIDDEN void
jsimd_idct_islow_ps2mmi(void *dct_table,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
  JLONG tmp0, tmp1, tmp2, tmp3;
  JLONG tmp10, tmp11, tmp12, tmp13;
  JLONG z1, z2, z3, z4, z5;
  JCOEFPTR inptr;
  JLONG dequant[DCTSIZE2] __attribute__((aligned(16)));
  int *wsptr;
  JSAMPROW outptr;
  /* Range mapping below is the exact IJG post-IDCT 10-bit wrap table. */
  int ctr;
  int workspace[DCTSIZE2];      /* buffers data between passes */
  SHIFT_TEMPS

  /*
   * A DC-only block is very common at high compression ratios.  Use MMI
   * to test all 63 AC coefficients in parallel, then bypass both DCT passes
   * and emit the identical eight rows.  The scalar fallback handles
   * non-16-byte-aligned coefficient buffers.
   */
  if (ps2_dc_only(coef_block)) {
    ISLOW_MULT_TYPE *q = (ISLOW_MULT_TYPE *)dct_table;
    JLONG dc = LEFT_SHIFT((JLONG)coef_block[0] * q[0], PASS1_BITS);
    JSAMPLE value =
      ps2_idct_range((int)DESCALE(dc, PASS1_BITS + 3) & RANGE_MASK);
    int row;
    for (row = 0; row < DCTSIZE; row++)
      memset(output_buf[row] + output_col, value, DCTSIZE);
    return;
  }

  /* Pass 1: process columns from input, store into work array. */
  /* Note results are scaled up by sqrt(8) compared to a true IDCT; */
  /* furthermore, we scale the results by 2**PASS1_BITS. */

  inptr = coef_block;
  ps2_dequant_block(coef_block, (ISLOW_MULT_TYPE *)dct_table, dequant);
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
      int dcval = LEFT_SHIFT(dequant[DCTSIZE * 0 + (int)(inptr - coef_block)], PASS1_BITS);

      wsptr[DCTSIZE * 0] = dcval;
      wsptr[DCTSIZE * 1] = dcval;
      wsptr[DCTSIZE * 2] = dcval;
      wsptr[DCTSIZE * 3] = dcval;
      wsptr[DCTSIZE * 4] = dcval;
      wsptr[DCTSIZE * 5] = dcval;
      wsptr[DCTSIZE * 6] = dcval;
      wsptr[DCTSIZE * 7] = dcval;

      inptr++;                  /* advance pointers to next column */
        wsptr++;
      continue;
    }

    /* Even part: reverse the even part of the forward DCT. */
    /* The rotator is sqrt(2)*c(-6). */

    z2 = dequant[DCTSIZE * 2 + (int)(inptr - coef_block)];
    z3 = dequant[DCTSIZE * 6 + (int)(inptr - coef_block)];

    z1 = MULTIPLY(z2 + z3, FIX_0_541196100);
    tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065);
    tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865);

    z2 = dequant[DCTSIZE * 0 + (int)(inptr - coef_block)];
    z3 = dequant[DCTSIZE * 4 + (int)(inptr - coef_block)];

    tmp0 = LEFT_SHIFT(z2 + z3, CONST_BITS);
    tmp1 = LEFT_SHIFT(z2 - z3, CONST_BITS);

    tmp10 = tmp0 + tmp3;
    tmp13 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp12 = tmp1 - tmp2;

    /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */

    tmp0 = dequant[DCTSIZE * 7 + (int)(inptr - coef_block)];
    tmp1 = dequant[DCTSIZE * 5 + (int)(inptr - coef_block)];
    tmp2 = dequant[DCTSIZE * 3 + (int)(inptr - coef_block)];
    tmp3 = dequant[DCTSIZE * 1 + (int)(inptr - coef_block)];

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

    {
      JLONG aa[4] __attribute__((aligned(16))) =
        { tmp10, tmp11, tmp12, tmp13 };
      JLONG bb[4] __attribute__((aligned(16))) =
        { tmp3, tmp2, tmp1, tmp0 };
      JLONG sum[4] __attribute__((aligned(16)));
      JLONG diff[4] __attribute__((aligned(16)));
      int k;

      ps2_butterfly_pass1(aa, bb, sum, diff);
      for (k = 0; k < 4; k++) {
        wsptr[DCTSIZE * k] = (int)sum[k];
        wsptr[DCTSIZE * (7 - k)] = (int)diff[k];
      }
    }

    inptr++;                    /* advance pointers to next column */
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
                                                PASS1_BITS + 3) & RANGE_MASK);

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

    {
      JLONG aa[4] __attribute__((aligned(16))) =
        { tmp10, tmp11, tmp12, tmp13 };
      JLONG bb[4] __attribute__((aligned(16))) =
        { tmp3, tmp2, tmp1, tmp0 };
      JLONG sum[4] __attribute__((aligned(16)));
      JLONG diff[4] __attribute__((aligned(16)));
      int k;

      ps2_butterfly_pass2(aa, bb, sum, diff);
      for (k = 0; k < 4; k++) {
        outptr[k] = ps2_idct_range((int)sum[k] & RANGE_MASK);
        outptr[7 - k] = ps2_idct_range((int)diff[k] & RANGE_MASK);
      }
    }

    wsptr += DCTSIZE;           /* advance pointer to next row */
  }
}
