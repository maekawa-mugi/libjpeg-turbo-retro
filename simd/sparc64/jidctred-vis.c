/*
 * Reduced integer inverse DCT (SPARC VIS 1)
 *
 * 4x4 processes the first four source columns in parallel with VIS1 and
 * handles the sparse tail scalarly.  2x2 processes the first two useful
 * columns as a VIS pair.  This avoids paying a full transpose cost for tiny
 * output blocks.
 */

#include "../jsimdint.h"
#include <stdint.h>
#include <visintrin.h>


#define CONST_BITS  13
#define PASS1_BITS  2

#define F_0_211  1730
#define F_0_509  4176
#define F_0_601  4926
#define F_0_720  5906
#define F_0_765  6270
#define F_0_850  6967
#define F_0_899  7373
#define F_1_061  8697
#define F_1_272  10426
#define F_1_451  11893
#define F_1_847  15137
#define F_2_172  17799
#define F_2_562  20995
#define F_3_624  29692


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
vis_mul16_const4(__v4hi a, int16_t c)
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
vis_addw4(vis_4w a, vis_4w b)
{
  vis_4w r;

  r.hi = __vis_fpadd32(a.hi, b.hi);
  r.lo = __vis_fpadd32(a.lo, b.lo);
  return r;
}


static inline vis_4w
vis_subw4(vis_4w a, vis_4w b)
{
  vis_4w r;

  r.hi = __vis_fpsub32(a.hi, b.hi);
  r.lo = __vis_fpsub32(a.lo, b.lo);
  return r;
}


static inline __v4hi
vis_packw4(vis_4w a, int32_t bias)
{
  vis_4h out;
  const __v2si b = { bias, bias };

  out.p.hi = __vis_fpackfix(__vis_fpadd32(a.hi, b));
  out.p.lo = __vis_fpackfix(__vis_fpadd32(a.lo, b));
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


static void
idct4_column_scalar(const JCOEF *coef, const ISLOW_MULT_TYPE *quant,
                    DCTELEM *workspace, int col)
{
  int32_t r0, r1, r2, r3, r5, r6, r7;
  int32_t tmp0, tmp2, tmp10, tmp12;
  int32_t a, b;

  r0 = coef[0 * DCTSIZE + col] * quant[0 * DCTSIZE + col];
  r1 = coef[1 * DCTSIZE + col] * quant[1 * DCTSIZE + col];
  r2 = coef[2 * DCTSIZE + col] * quant[2 * DCTSIZE + col];
  r3 = coef[3 * DCTSIZE + col] * quant[3 * DCTSIZE + col];
  r5 = coef[5 * DCTSIZE + col] * quant[5 * DCTSIZE + col];
  r6 = coef[6 * DCTSIZE + col] * quant[6 * DCTSIZE + col];
  r7 = coef[7 * DCTSIZE + col] * quant[7 * DCTSIZE + col];

  tmp0 = r0 << (CONST_BITS + 1);
  tmp2 = r2 * F_1_847 - r6 * F_0_765;
  tmp10 = tmp0 + tmp2;
  tmp12 = tmp0 - tmp2;

  a = r7 * -F_0_211 + r5 * F_1_451 +
      r3 * -F_2_172 + r1 * F_1_061;
  b = r7 * -F_0_509 + r5 * -F_0_601 +
      r3 * F_0_899 + r1 * F_2_562;

  workspace[0 * DCTSIZE + col] =
    (DCTELEM)descale32(tmp10 + b, CONST_BITS - PASS1_BITS + 1);
  workspace[3 * DCTSIZE + col] =
    (DCTELEM)descale32(tmp10 - b, CONST_BITS - PASS1_BITS + 1);
  workspace[1 * DCTSIZE + col] =
    (DCTELEM)descale32(tmp12 + a, CONST_BITS - PASS1_BITS + 1);
  workspace[2 * DCTSIZE + col] =
    (DCTELEM)descale32(tmp12 - a, CONST_BITS - PASS1_BITS + 1);
}


HIDDEN void
jsimd_idct_4x4_vis(void *dct_table, JCOEFPTR coef_block,
                   JSAMPARRAY output_buf, JDIMENSION output_col)
{
  ISLOW_MULT_TYPE *quant = (ISLOW_MULT_TYPE *)dct_table;
  DCTELEM workspace[DCTSIZE * 4];
  int row;

  /* First four source columns in parallel.  Final column descale is >> 12. */
  __builtin_vis_write_gsr(4 << 3);
  {
    __v4hi r0 = vis_dequant4(coef_block, quant, 0, 0);
    __v4hi r1 = vis_dequant4(coef_block, quant, 1, 0);
    __v4hi r2 = vis_dequant4(coef_block, quant, 2, 0);
    __v4hi r3 = vis_dequant4(coef_block, quant, 3, 0);
    __v4hi r5 = vis_dequant4(coef_block, quant, 5, 0);
    __v4hi r6 = vis_dequant4(coef_block, quant, 6, 0);
    __v4hi r7 = vis_dequant4(coef_block, quant, 7, 0);

    vis_4w tmp0 = vis_mul16_const4(r0, 1 << (CONST_BITS + 1));
    vis_4w tmp2 =
      vis_addw4(vis_mul16_const4(r2, F_1_847),
                vis_mul16_const4(r6, -F_0_765));
    vis_4w tmp10 = vis_addw4(tmp0, tmp2);
    vis_4w tmp12 = vis_subw4(tmp0, tmp2);

    vis_4w a = vis_addw4(
      vis_addw4(vis_mul16_const4(r7, -F_0_211),
                vis_mul16_const4(r5, F_1_451)),
      vis_addw4(vis_mul16_const4(r3, -F_2_172),
                vis_mul16_const4(r1, F_1_061)));
    vis_4w b = vis_addw4(
      vis_addw4(vis_mul16_const4(r7, -F_0_509),
                vis_mul16_const4(r5, -F_0_601)),
      vis_addw4(vis_mul16_const4(r3, F_0_899),
                vis_mul16_const4(r1, F_2_562)));

    vis_store4(workspace, 0, 0, vis_packw4(vis_addw4(tmp10, b), 2048));
    vis_store4(workspace, 3, 0, vis_packw4(vis_subw4(tmp10, b), 2048));
    vis_store4(workspace, 1, 0, vis_packw4(vis_addw4(tmp12, a), 2048));
    vis_store4(workspace, 2, 0, vis_packw4(vis_subw4(tmp12, a), 2048));
  }

  /* Column 4 is unused.  Finish 5, 6, 7 without VIS shuffle overhead. */
  workspace[0 * DCTSIZE + 4] = workspace[1 * DCTSIZE + 4] =
  workspace[2 * DCTSIZE + 4] = workspace[3 * DCTSIZE + 4] = 0;
  idct4_column_scalar(coef_block, quant, workspace, 5);
  idct4_column_scalar(coef_block, quant, workspace, 6);
  idct4_column_scalar(coef_block, quant, workspace, 7);

  for (row = 0; row < 4; row++) {
    DCTELEM *w = workspace + row * DCTSIZE;
    JSAMPROW out = output_buf[row] + output_col;
    int32_t tmp0, tmp2, tmp10, tmp12, a, b;
    int shift = CONST_BITS + PASS1_BITS + 3 + 1;

    if (w[1] == 0 && w[2] == 0 && w[3] == 0 &&
        w[5] == 0 && w[6] == 0 && w[7] == 0) {
      JSAMPLE dc = idct_clamp(descale32(w[0], PASS1_BITS + 3));
      out[0] = dc; out[1] = dc; out[2] = dc; out[3] = dc;
      continue;
    }

    tmp0 = ((int32_t)w[0]) << (CONST_BITS + 1);
    tmp2 = (int32_t)w[2] * F_1_847 - (int32_t)w[6] * F_0_765;
    tmp10 = tmp0 + tmp2;
    tmp12 = tmp0 - tmp2;

    a = (int32_t)w[7] * -F_0_211 + (int32_t)w[5] * F_1_451 +
        (int32_t)w[3] * -F_2_172 + (int32_t)w[1] * F_1_061;
    b = (int32_t)w[7] * -F_0_509 + (int32_t)w[5] * -F_0_601 +
        (int32_t)w[3] * F_0_899 + (int32_t)w[1] * F_2_562;

    out[0] = idct_clamp(descale32(tmp10 + b, shift));
    out[3] = idct_clamp(descale32(tmp10 - b, shift));
    out[1] = idct_clamp(descale32(tmp12 + a, shift));
    out[2] = idct_clamp(descale32(tmp12 - a, shift));
  }
}


static void
idct2_column_scalar(const JCOEF *coef, const ISLOW_MULT_TYPE *quant,
                    DCTELEM *workspace, int col)
{
  int32_t r0, r1, r3, r5, r7;
  int32_t tmp10, a;

  r0 = coef[0 * DCTSIZE + col] * quant[0 * DCTSIZE + col];
  r1 = coef[1 * DCTSIZE + col] * quant[1 * DCTSIZE + col];
  r3 = coef[3 * DCTSIZE + col] * quant[3 * DCTSIZE + col];
  r5 = coef[5 * DCTSIZE + col] * quant[5 * DCTSIZE + col];
  r7 = coef[7 * DCTSIZE + col] * quant[7 * DCTSIZE + col];

  tmp10 = r0 << (CONST_BITS + 2);
  a = r7 * -F_0_720 + r5 * F_0_850 +
      r3 * -F_1_272 + r1 * F_3_624;

  workspace[0 * DCTSIZE + col] =
    (DCTELEM)descale32(tmp10 + a, CONST_BITS - PASS1_BITS + 2);
  workspace[1 * DCTSIZE + col] =
    (DCTELEM)descale32(tmp10 - a, CONST_BITS - PASS1_BITS + 2);
}


HIDDEN void
jsimd_idct_2x2_vis(void *dct_table, JCOEFPTR coef_block,
                   JSAMPARRAY output_buf, JDIMENSION output_col)
{
  ISLOW_MULT_TYPE *quant = (ISLOW_MULT_TYPE *)dct_table;
  DCTELEM workspace[DCTSIZE * 2];
  int row, col;

  /*
   * Reduced 2x2 uses only columns 0,1,3,5,7.  Keep the arithmetic exact and
   * skip the vector packing cost for the three isolated tail columns.
   */
  for (col = 0; col < DCTSIZE; col++)
    workspace[col] = workspace[DCTSIZE + col] = 0;

  idct2_column_scalar(coef_block, quant, workspace, 0);
  idct2_column_scalar(coef_block, quant, workspace, 1);
  idct2_column_scalar(coef_block, quant, workspace, 3);
  idct2_column_scalar(coef_block, quant, workspace, 5);
  idct2_column_scalar(coef_block, quant, workspace, 7);

  for (row = 0; row < 2; row++) {
    DCTELEM *w = workspace + row * DCTSIZE;
    JSAMPROW out = output_buf[row] + output_col;
    int32_t tmp10, a;
    int shift = CONST_BITS + PASS1_BITS + 3 + 2;

    if (w[1] == 0 && w[3] == 0 && w[5] == 0 && w[7] == 0) {
      JSAMPLE dc = idct_clamp(descale32(w[0], PASS1_BITS + 3));
      out[0] = dc;
      out[1] = dc;
      continue;
    }

    tmp10 = ((int32_t)w[0]) << (CONST_BITS + 2);
    a = (int32_t)w[7] * -F_0_720 + (int32_t)w[5] * F_0_850 +
        (int32_t)w[3] * -F_1_272 + (int32_t)w[1] * F_3_624;

    out[0] = idct_clamp(descale32(tmp10 + a, shift));
    out[1] = idct_clamp(descale32(tmp10 - a, shift));
  }
}
