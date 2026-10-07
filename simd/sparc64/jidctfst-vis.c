/*
 * Fast integer inverse DCT (SPARC VIS 1)
 *
 * Pass 1 processes four columns in parallel.  VIS1 has no cheap arbitrary
 * 16-bit transpose, so pass 2 remains scalar.  This preserves IJG ifast
 * arithmetic exactly while moving the multiply-heavy half of the transform
 * into VIS.
 */

#include "../jsimdint.h"
#include <stdint.h>
#include <visintrin.h>


#define PASS1_BITS  2
#define FIX_1_082392200  277
#define FIX_1_414213562  362
#define FIX_1_847759065  473
#define FIX_2_613125930  669


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


static inline __v4hi
vis_mul_shift8(__v4hi a, int16_t c)
{
  vis_4h in, out;
  const __v2hi k = { c, c };
#ifdef USE_ACCURATE_ROUNDING
  const __v2si bias = { 128, 128 };
  __v2si p0, p1;
#endif

  in.v = a;
#ifdef USE_ACCURATE_ROUNDING
  p0 = __vis_fpadd32(vis_mul16x16_2(in.p.hi, k), bias);
  p1 = __vis_fpadd32(vis_mul16x16_2(in.p.lo, k), bias);
  out.p.hi = __vis_fpackfix(p0);
  out.p.lo = __vis_fpackfix(p1);
#else
  out.p.hi = __vis_fpackfix(vis_mul16x16_2(in.p.hi, k));
  out.p.lo = __vis_fpackfix(vis_mul16x16_2(in.p.lo, k));
#endif
  return out.v;
}


static inline __v4hi
vis_dequant4(const JCOEF *coef, const IFAST_MULT_TYPE *quant, int row, int col)
{
  vis_4h out;
  int i;

  for (i = 0; i < 4; i++)
    out.lane[i] = (DCTELEM)
      ((IFAST_MULT_TYPE)coef[row * DCTSIZE + col + i] *
       quant[row * DCTSIZE + col + i]);

  return out.v;
}


static inline void
vis_store4_workspace(DCTELEM *workspace, int row, int col, __v4hi value)
{
  vis_4h x;
  int i;

  x.v = value;
  for (i = 0; i < 4; i++)
    workspace[row * DCTSIZE + col + i] = x.lane[i];
}


static inline DCTELEM
scalar_mul_shift8(DCTELEM x, int c)
{
  int32_t p = (int32_t)x * c;

#ifdef USE_ACCURATE_ROUNDING
  p += 128;
#endif
  return (DCTELEM)(p >> 8);
}


static inline int
idct_descale(DCTELEM x, int n)
{
#ifdef USE_ACCURATE_ROUNDING
  return ((int)x + (1 << (n - 1))) >> n;
#else
  return (int)x >> n;
#endif
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
jsimd_idct_ifast_vis(void *dct_table, JCOEFPTR coef_block,
                     JSAMPARRAY output_buf, JDIMENSION output_col)
{
  IFAST_MULT_TYPE *quant = (IFAST_MULT_TYPE *)dct_table;
  DCTELEM workspace[DCTSIZE2];
  int col, row;

  /* fpackfix computes (x << scale) >> 16.  scale=8 gives x >> 8. */
  __builtin_vis_write_gsr(8 << 3);

  for (col = 0; col < DCTSIZE; col += 4) {
    __v4hi r0 = vis_dequant4(coef_block, quant, 0, col);
    __v4hi r1 = vis_dequant4(coef_block, quant, 1, col);
    __v4hi r2 = vis_dequant4(coef_block, quant, 2, col);
    __v4hi r3 = vis_dequant4(coef_block, quant, 3, col);
    __v4hi r4 = vis_dequant4(coef_block, quant, 4, col);
    __v4hi r5 = vis_dequant4(coef_block, quant, 5, col);
    __v4hi r6 = vis_dequant4(coef_block, quant, 6, col);
    __v4hi r7 = vis_dequant4(coef_block, quant, 7, col);
    const __v4hi zero = { 0, 0, 0, 0 };

    if (!(__vis_fcmpne16(r1, zero) | __vis_fcmpne16(r2, zero) |
          __vis_fcmpne16(r3, zero) | __vis_fcmpne16(r4, zero) |
          __vis_fcmpne16(r5, zero) | __vis_fcmpne16(r6, zero) |
          __vis_fcmpne16(r7, zero))) {
      vis_store4_workspace(workspace, 0, col, r0);
      vis_store4_workspace(workspace, 1, col, r0);
      vis_store4_workspace(workspace, 2, col, r0);
      vis_store4_workspace(workspace, 3, col, r0);
      vis_store4_workspace(workspace, 4, col, r0);
      vis_store4_workspace(workspace, 5, col, r0);
      vis_store4_workspace(workspace, 6, col, r0);
      vis_store4_workspace(workspace, 7, col, r0);
      continue;
    }

    __v4hi tmp0 = r0;
    __v4hi tmp1 = r2;
    __v4hi tmp2 = r4;
    __v4hi tmp3 = r6;
    __v4hi tmp10 = __vis_fpadd16(tmp0, tmp2);
    __v4hi tmp11 = __vis_fpsub16(tmp0, tmp2);
    __v4hi tmp13 = __vis_fpadd16(tmp1, tmp3);
    __v4hi tmp12 =
      __vis_fpsub16(vis_mul_shift8(__vis_fpsub16(tmp1, tmp3),
                                   FIX_1_414213562), tmp13);

    tmp0 = __vis_fpadd16(tmp10, tmp13);
    tmp3 = __vis_fpsub16(tmp10, tmp13);
    tmp1 = __vis_fpadd16(tmp11, tmp12);
    tmp2 = __vis_fpsub16(tmp11, tmp12);

    __v4hi tmp4 = r1;
    __v4hi tmp5 = r3;
    __v4hi tmp6 = r5;
    __v4hi tmp7 = r7;

    __v4hi z13 = __vis_fpadd16(tmp6, tmp5);
    __v4hi z10 = __vis_fpsub16(tmp6, tmp5);
    __v4hi z11 = __vis_fpadd16(tmp4, tmp7);
    __v4hi z12 = __vis_fpsub16(tmp4, tmp7);

    tmp7 = __vis_fpadd16(z11, z13);
    tmp11 = vis_mul_shift8(__vis_fpsub16(z11, z13), FIX_1_414213562);

    __v4hi z5 =
      vis_mul_shift8(__vis_fpadd16(z10, z12), FIX_1_847759065);
    tmp10 = __vis_fpsub16(vis_mul_shift8(z12, FIX_1_082392200), z5);
    tmp12 = __vis_fpadd16(vis_mul_shift8(z10, -FIX_2_613125930), z5);

    tmp6 = __vis_fpsub16(tmp12, tmp7);
    tmp5 = __vis_fpsub16(tmp11, tmp6);
    tmp4 = __vis_fpadd16(tmp10, tmp5);

    vis_store4_workspace(workspace, 0, col, __vis_fpadd16(tmp0, tmp7));
    vis_store4_workspace(workspace, 7, col, __vis_fpsub16(tmp0, tmp7));
    vis_store4_workspace(workspace, 1, col, __vis_fpadd16(tmp1, tmp6));
    vis_store4_workspace(workspace, 6, col, __vis_fpsub16(tmp1, tmp6));
    vis_store4_workspace(workspace, 2, col, __vis_fpadd16(tmp2, tmp5));
    vis_store4_workspace(workspace, 5, col, __vis_fpsub16(tmp2, tmp5));
    vis_store4_workspace(workspace, 4, col, __vis_fpadd16(tmp3, tmp4));
    vis_store4_workspace(workspace, 3, col, __vis_fpsub16(tmp3, tmp4));
  }

  /* Scalar row pass.  Keeping the exact IJG operation order here avoids the
   * transpose/shuffle overhead that VIS2 bshuffle would normally eliminate.
   */
  for (row = 0; row < DCTSIZE; row++) {
    DCTELEM *w = workspace + row * DCTSIZE;
    JSAMPROW out = output_buf[row] + output_col;
    DCTELEM tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    DCTELEM tmp10, tmp11, tmp12, tmp13;
    DCTELEM z5, z10, z11, z12, z13;
    int v;

    if (w[1] == 0 && w[2] == 0 && w[3] == 0 && w[4] == 0 &&
        w[5] == 0 && w[6] == 0 && w[7] == 0) {
      JSAMPLE dc = idct_clamp(idct_descale(w[0], PASS1_BITS + 3));

      out[0] = dc; out[1] = dc; out[2] = dc; out[3] = dc;
      out[4] = dc; out[5] = dc; out[6] = dc; out[7] = dc;
      continue;
    }

    tmp10 = (DCTELEM)(w[0] + w[4]);
    tmp11 = (DCTELEM)(w[0] - w[4]);
    tmp13 = (DCTELEM)(w[2] + w[6]);
    tmp12 = (DCTELEM)
      (scalar_mul_shift8((DCTELEM)(w[2] - w[6]), FIX_1_414213562) -
       tmp13);

    tmp0 = (DCTELEM)(tmp10 + tmp13);
    tmp3 = (DCTELEM)(tmp10 - tmp13);
    tmp1 = (DCTELEM)(tmp11 + tmp12);
    tmp2 = (DCTELEM)(tmp11 - tmp12);

    z13 = (DCTELEM)(w[5] + w[3]);
    z10 = (DCTELEM)(w[5] - w[3]);
    z11 = (DCTELEM)(w[1] + w[7]);
    z12 = (DCTELEM)(w[1] - w[7]);

    tmp7 = (DCTELEM)(z11 + z13);
    tmp11 =
      scalar_mul_shift8((DCTELEM)(z11 - z13), FIX_1_414213562);
    z5 = scalar_mul_shift8((DCTELEM)(z10 + z12), FIX_1_847759065);
    tmp10 = (DCTELEM)(scalar_mul_shift8(z12, FIX_1_082392200) - z5);
    tmp12 = (DCTELEM)(scalar_mul_shift8(z10, -FIX_2_613125930) + z5);

    tmp6 = (DCTELEM)(tmp12 - tmp7);
    tmp5 = (DCTELEM)(tmp11 - tmp6);
    tmp4 = (DCTELEM)(tmp10 + tmp5);

#define STORE_IDCT(POS, EXPR) \
    do { \
      v = idct_descale((DCTELEM)(EXPR), PASS1_BITS + 3); \
      out[(POS)] = idct_clamp(v); \
    } while (0)

    STORE_IDCT(0, tmp0 + tmp7);
    STORE_IDCT(7, tmp0 - tmp7);
    STORE_IDCT(1, tmp1 + tmp6);
    STORE_IDCT(6, tmp1 - tmp6);
    STORE_IDCT(2, tmp2 + tmp5);
    STORE_IDCT(5, tmp2 - tmp5);
    STORE_IDCT(4, tmp3 + tmp4);
    STORE_IDCT(3, tmp3 - tmp4);
#undef STORE_IDCT
  }
}
