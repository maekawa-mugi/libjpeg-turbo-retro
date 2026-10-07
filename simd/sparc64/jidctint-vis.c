/*
 * Accurate integer inverse DCT (SPARC VIS 1)
 *
 * The column pass processes four coefficients at once using exact VIS1
 * signed 16x16->32 products.  The row pass remains scalar because VIS1 has no
 * inexpensive arbitrary 16-bit transpose.  Rounding points match jidctint.c.
 */

#include "../jsimdint.h"
#include <stdint.h>
#include <visintrin.h>


#define CONST_BITS  13
#define PASS1_BITS  2
#define DESCALE_P1  (CONST_BITS - PASS1_BITS)

#define F_0_298  2446
#define F_0_390  3196
#define F_0_541  4433
#define F_0_765  6270
#define F_0_899  7373
#define F_1_175  9633
#define F_1_501  12299
#define F_1_847  15137
#define F_1_961  16069
#define F_2_053  16819
#define F_2_562  20995
#define F_3_072  25172


typedef union {
  __v2hi h;
  __v4qi b;
} vis_halfword_bytes;

typedef union {
  __v4hi v;
  int16_t lane[4];
  struct {
    __v2hi hi;
    __v2hi lo;
  } p;
} vis_4h;

typedef struct {
  __v2si hi;
  __v2si lo;
} vis_4w;


static inline __v2si
vis_mul16x16_2(__v2hi a, __v2hi b)
{
  vis_halfword_bytes ab;
  __v2si hi, lo;

  ab.h = a;
  hi = __vis_fmuld8sux16(ab.b, b);
  lo = __vis_fmuld8ulx16(ab.b, b);
  return __vis_fpadd32(hi, lo);
}


static inline vis_4w
vis_mul16_const(__v4hi a, int16_t c)
{
  vis_4h in;
  vis_4w out;
  const __v2hi k = { c, c };

  in.v = a;
  out.hi = vis_mul16x16_2(in.p.hi, k);
  out.lo = vis_mul16x16_2(in.p.lo, k);
  return out;
}


static inline vis_4w
vis_addw(vis_4w a, vis_4w b)
{
  vis_4w r;

  r.hi = __vis_fpadd32(a.hi, b.hi);
  r.lo = __vis_fpadd32(a.lo, b.lo);
  return r;
}


static inline vis_4w
vis_subw(vis_4w a, vis_4w b)
{
  vis_4w r;

  r.hi = __vis_fpsub32(a.hi, b.hi);
  r.lo = __vis_fpsub32(a.lo, b.lo);
  return r;
}


static inline __v4hi
vis_packw(vis_4w a, int32_t bias)
{
  vis_4h out;
  const __v2si b = { bias, bias };

  a.hi = __vis_fpadd32(a.hi, b);
  a.lo = __vis_fpadd32(a.lo, b);
  out.p.hi = __vis_fpackfix(a.hi);
  out.p.lo = __vis_fpackfix(a.lo);
  return out.v;
}


static inline __v4hi
vis_dequant4(const JCOEF *coef, const ISLOW_MULT_TYPE *quant, int row, int col)
{
  vis_4h out;
  int i;

  for (i = 0; i < 4; i++)
    out.lane[i] = (DCTELEM)
      ((ISLOW_MULT_TYPE)coef[row * DCTSIZE + col + i] *
       quant[row * DCTSIZE + col + i]);

  return out.v;
}


static inline void
vis_store4(DCTELEM *workspace, int row, int col, __v4hi value)
{
  vis_4h x;
  int i;

  x.v = value;
  for (i = 0; i < 4; i++)
    workspace[row * DCTSIZE + col + i] = x.lane[i];
}


static inline int32_t
descale32(int32_t x, int n)
{
  return (x + (1 << (n - 1))) >> n;
}


static inline JSAMPLE
idct_clamp(int x)
{
  x += CENTERJSAMPLE;
  if (x < 0)
    return 0;
  if (x > MAXJSAMPLE)
    return MAXJSAMPLE;
  return (JSAMPLE)x;
}


HIDDEN void
jsimd_idct_islow_vis(void *dct_table, JCOEFPTR coef_block,
                     JSAMPARRAY output_buf, JDIMENSION output_col)
{
  ISLOW_MULT_TYPE *quant = (ISLOW_MULT_TYPE *)dct_table;
  DCTELEM workspace[DCTSIZE2];
  int col, row;

  /*
   * Column pass.  fpackfix() computes (x << scale) >> 16.  The column
   * outputs need >> 11, so GSR.scale = 5 after explicit +1024 rounding.
   */
  __builtin_vis_write_gsr(5 << 3);

  for (col = 0; col < DCTSIZE; col += 4) {
    __v4hi r0 = vis_dequant4(coef_block, quant, 0, col);
    __v4hi r1 = vis_dequant4(coef_block, quant, 1, col);
    __v4hi r2 = vis_dequant4(coef_block, quant, 2, col);
    __v4hi r3 = vis_dequant4(coef_block, quant, 3, col);
    __v4hi r4 = vis_dequant4(coef_block, quant, 4, col);
    __v4hi r5 = vis_dequant4(coef_block, quant, 5, col);
    __v4hi r6 = vis_dequant4(coef_block, quant, 6, col);
    __v4hi r7 = vis_dequant4(coef_block, quant, 7, col);

    __v4hi z2v = r2;
    __v4hi z3v = r6;
    vis_4w z1w =
      vis_mul16_const(__vis_fpadd16(z2v, z3v), F_0_541);
    vis_4w tmp2w =
      vis_addw(z1w, vis_mul16_const(z3v, -F_1_847));
    vis_4w tmp3w =
      vis_addw(z1w, vis_mul16_const(z2v, F_0_765));

    vis_4w tmp0w =
      vis_mul16_const(__vis_fpadd16(r0, r4), 1 << CONST_BITS);
    vis_4w tmp1w =
      vis_mul16_const(__vis_fpsub16(r0, r4), 1 << CONST_BITS);

    vis_4w tmp10w = vis_addw(tmp0w, tmp3w);
    vis_4w tmp13w = vis_subw(tmp0w, tmp3w);
    vis_4w tmp11w = vis_addw(tmp1w, tmp2w);
    vis_4w tmp12w = vis_subw(tmp1w, tmp2w);

    __v4hi z1v = __vis_fpadd16(r7, r1);
    z2v = __vis_fpadd16(r5, r3);
    z3v = __vis_fpadd16(r7, r3);
    __v4hi z4v = __vis_fpadd16(r5, r1);
    vis_4w z5w =
      vis_mul16_const(__vis_fpadd16(z3v, z4v), F_1_175);

    tmp0w = vis_mul16_const(r7, F_0_298);
    tmp1w = vis_mul16_const(r5, F_2_053);
    tmp2w = vis_mul16_const(r3, F_3_072);
    tmp3w = vis_mul16_const(r1, F_1_501);
    z1w = vis_mul16_const(z1v, -F_0_899);
    vis_4w z2w = vis_mul16_const(z2v, -F_2_562);
    vis_4w z3w = vis_addw(vis_mul16_const(z3v, -F_1_961), z5w);
    vis_4w z4w = vis_addw(vis_mul16_const(z4v, -F_0_390), z5w);

    tmp0w = vis_addw(vis_addw(tmp0w, z1w), z3w);
    tmp1w = vis_addw(vis_addw(tmp1w, z2w), z4w);
    tmp2w = vis_addw(vis_addw(tmp2w, z2w), z3w);
    tmp3w = vis_addw(vis_addw(tmp3w, z1w), z4w);

    vis_store4(workspace, 0, col,
               vis_packw(vis_addw(tmp10w, tmp3w), 1 << (DESCALE_P1 - 1)));
    vis_store4(workspace, 7, col,
               vis_packw(vis_subw(tmp10w, tmp3w), 1 << (DESCALE_P1 - 1)));
    vis_store4(workspace, 1, col,
               vis_packw(vis_addw(tmp11w, tmp2w), 1 << (DESCALE_P1 - 1)));
    vis_store4(workspace, 6, col,
               vis_packw(vis_subw(tmp11w, tmp2w), 1 << (DESCALE_P1 - 1)));
    vis_store4(workspace, 2, col,
               vis_packw(vis_addw(tmp12w, tmp1w), 1 << (DESCALE_P1 - 1)));
    vis_store4(workspace, 5, col,
               vis_packw(vis_subw(tmp12w, tmp1w), 1 << (DESCALE_P1 - 1)));
    vis_store4(workspace, 3, col,
               vis_packw(vis_addw(tmp13w, tmp0w), 1 << (DESCALE_P1 - 1)));
    vis_store4(workspace, 4, col,
               vis_packw(vis_subw(tmp13w, tmp0w), 1 << (DESCALE_P1 - 1)));
  }

  /* Exact scalar row pass. */
  for (row = 0; row < DCTSIZE; row++) {
    DCTELEM *w = workspace + row * DCTSIZE;
    JSAMPROW out = output_buf[row] + output_col;
    int32_t tmp0, tmp1, tmp2, tmp3;
    int32_t tmp10, tmp11, tmp12, tmp13;
    int32_t z1, z2, z3, z4, z5;
    int shift = CONST_BITS + PASS1_BITS + 3;

    if (w[1] == 0 && w[2] == 0 && w[3] == 0 && w[4] == 0 &&
        w[5] == 0 && w[6] == 0 && w[7] == 0) {
      JSAMPLE dc = idct_clamp(descale32(w[0], PASS1_BITS + 3));

      out[0] = dc; out[1] = dc; out[2] = dc; out[3] = dc;
      out[4] = dc; out[5] = dc; out[6] = dc; out[7] = dc;
      continue;
    }

    z2 = w[2];
    z3 = w[6];

    z1 = (z2 + z3) * F_0_541;
    tmp2 = z1 + z3 * -F_1_847;
    tmp3 = z1 + z2 * F_0_765;

    tmp0 = (w[0] + w[4]) << CONST_BITS;
    tmp1 = (w[0] - w[4]) << CONST_BITS;

    tmp10 = tmp0 + tmp3;
    tmp13 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp12 = tmp1 - tmp2;

    tmp0 = w[7];
    tmp1 = w[5];
    tmp2 = w[3];
    tmp3 = w[1];

    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    z4 = tmp1 + tmp3;
    z5 = (z3 + z4) * F_1_175;

    tmp0 *= F_0_298;
    tmp1 *= F_2_053;
    tmp2 *= F_3_072;
    tmp3 *= F_1_501;
    z1 *= -F_0_899;
    z2 *= -F_2_562;
    z3 = z3 * -F_1_961 + z5;
    z4 = z4 * -F_0_390 + z5;

    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;

    out[0] = idct_clamp(descale32(tmp10 + tmp3, shift));
    out[7] = idct_clamp(descale32(tmp10 - tmp3, shift));
    out[1] = idct_clamp(descale32(tmp11 + tmp2, shift));
    out[6] = idct_clamp(descale32(tmp11 - tmp2, shift));
    out[2] = idct_clamp(descale32(tmp12 + tmp1, shift));
    out[5] = idct_clamp(descale32(tmp12 - tmp1, shift));
    out[3] = idct_clamp(descale32(tmp13 + tmp0, shift));
    out[4] = idct_clamp(descale32(tmp13 - tmp0, shift));
  }
}
