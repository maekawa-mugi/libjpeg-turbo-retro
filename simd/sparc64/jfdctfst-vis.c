/*
 * Fast integer forward DCT (SPARC VIS 1)
 *
 * The first pass is scalar because VIS1 lacks a cheap arbitrary 16-bit
 * transpose.  The second pass processes four columns at once.  Multiplication
 * uses the VIS1 full 16x16->32 construction so the explicit descale matches
 * the scalar C implementation bit-for-bit.
 */

#include "../jsimdint.h"
#include <stdint.h>
#include <visintrin.h>


#define IFIX_0_382  98
#define IFIX_0_541  139
#define IFIX_0_707  181
#define IFIX_1_306  334


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


/* Exact two-lane signed 16x16 -> signed 32-bit multiply. */
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


/* GSR.scale must be 8.  fpackfix then performs signed >> 8. */
static inline __v4hi
vis_mul_descale8(__v4hi a, int16_t c)
{
  vis_4h in, out;
  const __v2hi k = { c, c };

  in.v = a;
  out.p.hi = __vis_fpackfix(vis_mul16x16_2(in.p.hi, k));
  out.p.lo = __vis_fpackfix(vis_mul16x16_2(in.p.lo, k));
  return out.v;
}


static inline DCTELEM
scalar_mul_descale8(DCTELEM x, int c)
{
  return (DCTELEM)(((int32_t)x * c) >> 8);
}


static void
fdct_ifast_pass1_scalar(DCTELEM *data)
{
  DCTELEM *dataptr = data;
  int ctr;

  for (ctr = DCTSIZE - 1; ctr >= 0; ctr--) {
    DCTELEM tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    DCTELEM tmp10, tmp11, tmp12, tmp13;
    DCTELEM z1, z2, z3, z4, z5, z11, z13;

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

    dataptr[0] = tmp10 + tmp11;
    dataptr[4] = tmp10 - tmp11;

    z1 = scalar_mul_descale8((DCTELEM)(tmp12 + tmp13), IFIX_0_707);
    dataptr[2] = tmp13 + z1;
    dataptr[6] = tmp13 - z1;

    tmp10 = tmp4 + tmp5;
    tmp11 = tmp5 + tmp6;
    tmp12 = tmp6 + tmp7;

    z5 = scalar_mul_descale8((DCTELEM)(tmp10 - tmp12), IFIX_0_382);
    z2 = scalar_mul_descale8(tmp10, IFIX_0_541) + z5;
    z4 = scalar_mul_descale8(tmp12, IFIX_1_306) + z5;
    z3 = scalar_mul_descale8(tmp11, IFIX_0_707);

    z11 = tmp7 + z3;
    z13 = tmp7 - z3;

    dataptr[5] = z13 + z2;
    dataptr[3] = z13 - z2;
    dataptr[1] = z11 + z4;
    dataptr[7] = z11 - z4;

    dataptr += DCTSIZE;
  }
}


HIDDEN void
jsimd_fdct_ifast_vis(DCTELEM *data)
{
  int col;

  fdct_ifast_pass1_scalar(data);

  /* fpackfix computes (x << scale) >> 16.  scale=8 gives x >> 8. */
  __builtin_vis_write_gsr(8 << 3);

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

    row0 = __vis_fpadd16(tmp10, tmp11);
    row4 = __vis_fpsub16(tmp10, tmp11);

    __v4hi z1 =
      vis_mul_descale8(__vis_fpadd16(tmp12, tmp13), IFIX_0_707);
    row2 = __vis_fpadd16(tmp13, z1);
    row6 = __vis_fpsub16(tmp13, z1);

    tmp10 = __vis_fpadd16(tmp4, tmp5);
    tmp11 = __vis_fpadd16(tmp5, tmp6);
    tmp12 = __vis_fpadd16(tmp6, tmp7);

    __v4hi z5 =
      vis_mul_descale8(__vis_fpsub16(tmp10, tmp12), IFIX_0_382);
    __v4hi z2 = __vis_fpadd16(vis_mul_descale8(tmp10, IFIX_0_541), z5);
    __v4hi z4 = __vis_fpadd16(vis_mul_descale8(tmp12, IFIX_1_306), z5);
    __v4hi z3 = vis_mul_descale8(tmp11, IFIX_0_707);

    __v4hi z11 = __vis_fpadd16(tmp7, z3);
    __v4hi z13 = __vis_fpsub16(tmp7, z3);

    row5 = __vis_fpadd16(z13, z2);
    row3 = __vis_fpsub16(z13, z2);
    row1 = __vis_fpadd16(z11, z4);
    row7 = __vis_fpsub16(z11, z4);

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
