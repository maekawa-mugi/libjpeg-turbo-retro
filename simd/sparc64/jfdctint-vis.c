/*
 * Accurate integer forward DCT (SPARC VIS 1)
 *
 * Pass 1 is scalar to avoid an expensive VIS1-only 16-bit transpose.
 * Pass 2 processes four columns at once.  All fixed-point multiplications are
 * exact signed 16x16->32 products and all rounding is applied explicitly
 * before fpackfix(), matching jpeg_fdct_islow().
 */

#include "../jsimdint.h"
#include <stdint.h>
#include <visintrin.h>


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

#define CONST_BITS  13
#define PASS1_BITS  2
#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS)


typedef union {
  __v2hi h;
  __v4qi b;
} vis_halfword_bytes;

typedef union {
  __v4hi v;
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
vis_descale16(__v4hi a, int n)
{
  vis_4w w = vis_mul16_const(a, 1);

  return vis_packw(w, 1 << (n - 1));
}


static inline int32_t
descale32(int32_t x, int n)
{
  return (x + (1 << (n - 1))) >> n;
}


static void
fdct_islow_pass1_scalar(DCTELEM *data)
{
  DCTELEM *dataptr = data;
  int ctr;

  for (ctr = DCTSIZE - 1; ctr >= 0; ctr--) {
    int32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    int32_t tmp10, tmp11, tmp12, tmp13;
    int32_t z1, z2, z3, z4, z5;

    tmp0 = dataptr[0] + dataptr[7];
    tmp7 = dataptr[0] - dataptr[7];
    tmp1 = dataptr[1] + dataptr[6];
    tmp6 = dataptr[1] - dataptr[6];
    tmp2 = dataptr[2] + dataptr[5];
    tmp5 = dataptr[2] - dataptr[5];
    tmp3 = dataptr[3] + dataptr[4];
    tmp4 = dataptr[3] - dataptr[4];

    tmp10 = tmp0 + tmp3;
    tmp13 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp12 = tmp1 - tmp2;

    dataptr[0] = (DCTELEM)((tmp10 + tmp11) << PASS1_BITS);
    dataptr[4] = (DCTELEM)((tmp10 - tmp11) << PASS1_BITS);

    z1 = (tmp12 + tmp13) * F_0_541;
    dataptr[2] =
      (DCTELEM)descale32(z1 + tmp13 * F_0_765, DESCALE_P1);
    dataptr[6] =
      (DCTELEM)descale32(z1 - tmp12 * F_1_847, DESCALE_P1);

    z1 = tmp4 + tmp7;
    z2 = tmp5 + tmp6;
    z3 = tmp4 + tmp6;
    z4 = tmp5 + tmp7;
    z5 = (z3 + z4) * F_1_175;

    tmp4 *= F_0_298;
    tmp5 *= F_2_053;
    tmp6 *= F_3_072;
    tmp7 *= F_1_501;
    z1 *= -F_0_899;
    z2 *= -F_2_562;
    z3 *= -F_1_961;
    z4 *= -F_0_390;

    z3 += z5;
    z4 += z5;

    dataptr[7] = (DCTELEM)descale32(tmp4 + z1 + z3, DESCALE_P1);
    dataptr[5] = (DCTELEM)descale32(tmp5 + z2 + z4, DESCALE_P1);
    dataptr[3] = (DCTELEM)descale32(tmp6 + z2 + z3, DESCALE_P1);
    dataptr[1] = (DCTELEM)descale32(tmp7 + z1 + z4, DESCALE_P1);

    dataptr += DCTSIZE;
  }
}


HIDDEN void
jsimd_fdct_islow_vis(DCTELEM *data)
{
  int col;

  fdct_islow_pass1_scalar(data);

  for (col = 0; col < DCTSIZE; col += 4) {
    __v4hi row0 = *(__v4hi *)(void *)(data + 0 * DCTSIZE + col);
    __v4hi row1 = *(__v4hi *)(void *)(data + 1 * DCTSIZE + col);
    __v4hi row2 = *(__v4hi *)(void *)(data + 2 * DCTSIZE + col);
    __v4hi row3 = *(__v4hi *)(void *)(data + 3 * DCTSIZE + col);
    __v4hi row4 = *(__v4hi *)(void *)(data + 4 * DCTSIZE + col);
    __v4hi row5 = *(__v4hi *)(void *)(data + 5 * DCTSIZE + col);
    __v4hi row6 = *(__v4hi *)(void *)(data + 6 * DCTSIZE + col);
    __v4hi row7 = *(__v4hi *)(void *)(data + 7 * DCTSIZE + col);

    __v4hi tmp0 = __vis_fpadd16(row0, row7);
    __v4hi tmp7 = __vis_fpsub16(row0, row7);
    __v4hi tmp1 = __vis_fpadd16(row1, row6);
    __v4hi tmp6 = __vis_fpsub16(row1, row6);
    __v4hi tmp2 = __vis_fpadd16(row2, row5);
    __v4hi tmp5 = __vis_fpsub16(row2, row5);
    __v4hi tmp3 = __vis_fpadd16(row3, row4);
    __v4hi tmp4 = __vis_fpsub16(row3, row4);

    __v4hi tmp10 = __vis_fpadd16(tmp0, tmp3);
    __v4hi tmp13 = __vis_fpsub16(tmp0, tmp3);
    __v4hi tmp11 = __vis_fpadd16(tmp1, tmp2);
    __v4hi tmp12 = __vis_fpsub16(tmp1, tmp2);

    /* PASS1_BITS descale for outputs 0 and 4. */
    __builtin_vis_write_gsr((16 - PASS1_BITS) << 3);
    row0 = vis_descale16(__vis_fpadd16(tmp10, tmp11), PASS1_BITS);
    row4 = vis_descale16(__vis_fpsub16(tmp10, tmp11), PASS1_BITS);

    /* The remaining outputs are 32-bit fixed-point values descaled by 15. */
    __builtin_vis_write_gsr((16 - DESCALE_P2) << 3);

    vis_4w z1 =
      vis_mul16_const(__vis_fpadd16(tmp12, tmp13), F_0_541);
    vis_4w out2 = vis_addw(z1, vis_mul16_const(tmp13, F_0_765));
    vis_4w out6 = vis_addw(z1, vis_mul16_const(tmp12, -F_1_847));
    row2 = vis_packw(out2, 1 << (DESCALE_P2 - 1));
    row6 = vis_packw(out6, 1 << (DESCALE_P2 - 1));

    __v4hi z1v = __vis_fpadd16(tmp4, tmp7);
    __v4hi z2v = __vis_fpadd16(tmp5, tmp6);
    __v4hi z3v = __vis_fpadd16(tmp4, tmp6);
    __v4hi z4v = __vis_fpadd16(tmp5, tmp7);
    vis_4w z5 =
      vis_mul16_const(__vis_fpadd16(z3v, z4v), F_1_175);

    vis_4w tmp4w = vis_mul16_const(tmp4, F_0_298);
    vis_4w tmp5w = vis_mul16_const(tmp5, F_2_053);
    vis_4w tmp6w = vis_mul16_const(tmp6, F_3_072);
    vis_4w tmp7w = vis_mul16_const(tmp7, F_1_501);
    vis_4w z1w = vis_mul16_const(z1v, -F_0_899);
    vis_4w z2w = vis_mul16_const(z2v, -F_2_562);
    vis_4w z3w = vis_addw(vis_mul16_const(z3v, -F_1_961), z5);
    vis_4w z4w = vis_addw(vis_mul16_const(z4v, -F_0_390), z5);

    row7 = vis_packw(vis_addw(vis_addw(tmp4w, z1w), z3w),
                     1 << (DESCALE_P2 - 1));
    row5 = vis_packw(vis_addw(vis_addw(tmp5w, z2w), z4w),
                     1 << (DESCALE_P2 - 1));
    row3 = vis_packw(vis_addw(vis_addw(tmp6w, z2w), z3w),
                     1 << (DESCALE_P2 - 1));
    row1 = vis_packw(vis_addw(vis_addw(tmp7w, z1w), z4w),
                     1 << (DESCALE_P2 - 1));

    *(__v4hi *)(void *)(data + 0 * DCTSIZE + col) = row0;
    *(__v4hi *)(void *)(data + 1 * DCTSIZE + col) = row1;
    *(__v4hi *)(void *)(data + 2 * DCTSIZE + col) = row2;
    *(__v4hi *)(void *)(data + 3 * DCTSIZE + col) = row3;
    *(__v4hi *)(void *)(data + 4 * DCTSIZE + col) = row4;
    *(__v4hi *)(void *)(data + 5 * DCTSIZE + col) = row5;
    *(__v4hi *)(void *)(data + 6 * DCTSIZE + col) = row6;
    *(__v4hi *)(void *)(data + 7 * DCTSIZE + col) = row7;
  }
}
