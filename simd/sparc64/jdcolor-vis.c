/*
 * YCbCr -> 4-byte RGB-family conversion (SPARC VIS 1)
 *
 * VIS1 has no bshuffle, but four-byte pixels can be stored efficiently with
 * two levels of fpmerge().  RGB/BGR 24-bit output is intentionally left to
 * the scalar converter for now.
 */

#include "../jsimdint.h"
#include <stdint.h>
#include <visintrin.h>


#define F_0_344  11277
#define F_0_714  23401
#define F_1_402  22971
#define F_1_772  29033


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

typedef union {
  __v8qi v;
  struct {
    __v4qi hi;
    __v4qi lo;
  } p;
} vis_8b;


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
vis_round_shift(vis_4w a, int shift)
{
  vis_4h out;
  const __v2si bias = { 1 << (shift - 1), 1 << (shift - 1) };

  __builtin_vis_write_gsr((16 - shift) << 3);
  out.p.hi = __vis_fpackfix(__vis_fpadd32(a.hi, bias));
  out.p.lo = __vis_fpackfix(__vis_fpadd32(a.lo, bias));
  return out.v;
}


static inline __v4hi
vis_widen_u8(__v4qi x)
{
  const __v2hi exact = { 256, 256 };

  return __vis_fmul8x16au(x, exact);
}


static inline __v4qi
vis_clamp_u8(__v4hi x)
{
  /* fpack16((x * 16), scale=3) == clamp(x, 0, 255). */
  x = __vis_fpadd16(x, x);
  x = __vis_fpadd16(x, x);
  x = __vis_fpadd16(x, x);
  x = __vis_fpadd16(x, x);
  __builtin_vis_write_gsr(3 << 3);
  return __vis_fpack16(x);
}


static inline void
vis_store_4pixel(JSAMPLE *dst, __v4qi c0, __v4qi c1,
                 __v4qi c2, __v4qi c3)
{
  vis_8b a, b;
  __v8qi x02 = __vis_fpmerge(c0, c2);
  __v8qi x13 = __vis_fpmerge(c1, c3);

  a.v = x02;
  b.v = x13;

  *(__v8qi *)(void *)(dst + 0) = __vis_fpmerge(a.p.hi, b.p.hi);
  *(__v8qi *)(void *)(dst + 8) = __vis_fpmerge(a.p.lo, b.p.lo);
}


static inline int
descale(int32_t x, int shift)
{
  return (x + (1 << (shift - 1))) >> shift;
}


static inline JSAMPLE
clamp8(int x)
{
  if (x < 0)
    return 0;
  if (x > 255)
    return 255;
  return (JSAMPLE)x;
}


static inline void
scalar_pixel(JSAMPLE y, JSAMPLE cbv, JSAMPLE crv, JSAMPLE *dst,
             int ro, int go, int bo, int ao)
{
  int cb = GETJSAMPLE(cbv) - CENTERJSAMPLE;
  int cr = GETJSAMPLE(crv) - CENTERJSAMPLE;
  int yy = GETJSAMPLE(y);
  int r = yy + descale(cr * F_1_402, 14);
  int g = yy + descale(cb * -F_0_344 + cr * -F_0_714, 15);
  int b = yy + descale(cb * F_1_772, 14);

  dst[ro] = clamp8(r);
  dst[go] = clamp8(g);
  dst[bo] = clamp8(b);
  dst[ao] = 0xff;
}


static void
ycc_rgbx_vis(JDIMENSION output_width, JSAMPIMAGE input_buf,
             JDIMENSION input_row, JSAMPARRAY output_buf, int num_rows,
             int ro, int go, int bo, int ao)
{
  const __v4hi center = { CENTERJSAMPLE, CENTERJSAMPLE,
                          CENTERJSAMPLE, CENTERJSAMPLE };
  const __v4qi opaque = { -1, -1, -1, -1 };

  while (--num_rows >= 0) {
    JSAMPROW yptr = input_buf[0][input_row];
    JSAMPROW cbptr = input_buf[1][input_row];
    JSAMPROW crptr = input_buf[2][input_row];
    JSAMPROW out = *output_buf++;
    JDIMENSION col = 0;

    input_row++;

    /* One scalar pixel is enough to fix the common 4-byte misalignment. */
    if (((JUINTPTR)out & 7) == 4 && output_width) {
      scalar_pixel(yptr[0], cbptr[0], crptr[0], out, ro, go, bo, ao);
      col = 1;
    }

    if (((JUINTPTR)(out + 4 * col) & 7) == 0) {
      for (; col + 4 <= output_width; col += 4) {
        __v4qi y8 = *(const __v4qi *)(const void *)(yptr + col);
        __v4qi cb8 = *(const __v4qi *)(const void *)(cbptr + col);
        __v4qi cr8 = *(const __v4qi *)(const void *)(crptr + col);
        __v4hi y = vis_widen_u8(y8);
        __v4hi cb = __vis_fpsub16(vis_widen_u8(cb8), center);
        __v4hi cr = __vis_fpsub16(vis_widen_u8(cr8), center);

        __v4hi rd =
          vis_round_shift(vis_mul16_const(cr, F_1_402), 14);
        __v4hi gd =
          vis_round_shift(vis_addw(vis_mul16_const(cb, -F_0_344),
                                   vis_mul16_const(cr, -F_0_714)), 15);
        __v4hi bd =
          vis_round_shift(vis_mul16_const(cb, F_1_772), 14);

        __v4qi r8 = vis_clamp_u8(__vis_fpadd16(y, rd));
        __v4qi g8 = vis_clamp_u8(__vis_fpadd16(y, gd));
        __v4qi b8 = vis_clamp_u8(__vis_fpadd16(y, bd));
        __v4qi c[4];

        c[ro] = r8;
        c[go] = g8;
        c[bo] = b8;
        c[ao] = opaque;
        vis_store_4pixel(out + 4 * col, c[0], c[1], c[2], c[3]);
      }
    }

    for (; col < output_width; col++)
      scalar_pixel(yptr[col], cbptr[col], crptr[col], out + 4 * col,
                   ro, go, bo, ao);
  }
}


HIDDEN void
jsimd_ycc_extrgbx_convert_vis(JDIMENSION output_width, JSAMPIMAGE input_buf,
                              JDIMENSION input_row, JSAMPARRAY output_buf,
                              int num_rows)
{
  ycc_rgbx_vis(output_width, input_buf, input_row, output_buf, num_rows,
               0, 1, 2, 3);
}


HIDDEN void
jsimd_ycc_extbgrx_convert_vis(JDIMENSION output_width, JSAMPIMAGE input_buf,
                              JDIMENSION input_row, JSAMPARRAY output_buf,
                              int num_rows)
{
  ycc_rgbx_vis(output_width, input_buf, input_row, output_buf, num_rows,
               2, 1, 0, 3);
}


HIDDEN void
jsimd_ycc_extxbgr_convert_vis(JDIMENSION output_width, JSAMPIMAGE input_buf,
                              JDIMENSION input_row, JSAMPARRAY output_buf,
                              int num_rows)
{
  ycc_rgbx_vis(output_width, input_buf, input_row, output_buf, num_rows,
               3, 2, 1, 0);
}


HIDDEN void
jsimd_ycc_extxrgb_convert_vis(JDIMENSION output_width, JSAMPIMAGE input_buf,
                              JDIMENSION input_row, JSAMPARRAY output_buf,
                              int num_rows)
{
  ycc_rgbx_vis(output_width, input_buf, input_row, output_buf, num_rows,
               1, 2, 3, 0);
}
