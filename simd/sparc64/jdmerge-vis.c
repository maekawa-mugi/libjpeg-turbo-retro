/*
 * Merged 2:1 upsampling + YCbCr -> 4-byte RGB conversion (SPARC VIS 1)
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

typedef union {
  __v8qi v;
  struct {
    __v4qi hi;
    __v4qi lo;
  } p;
} vis_8b;

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

  dst[ro] = clamp8(yy + descale(cr * F_1_402, 14));
  dst[go] = clamp8(yy + descale(cb * -F_0_344 + cr * -F_0_714, 15));
  dst[bo] = clamp8(yy + descale(cb * F_1_772, 14));
  dst[ao] = 0xff;
}


static inline void
chroma_deltas2(JSAMPLE cb0, JSAMPLE cb1, JSAMPLE cr0, JSAMPLE cr1,
               __v4hi *rd, __v4hi *gd, __v4hi *bd)
{
  __v4qi cb8 = { (int8_t)cb0, (int8_t)cb0, (int8_t)cb1, (int8_t)cb1 };
  __v4qi cr8 = { (int8_t)cr0, (int8_t)cr0, (int8_t)cr1, (int8_t)cr1 };
  const __v4hi center = { CENTERJSAMPLE, CENTERJSAMPLE,
                          CENTERJSAMPLE, CENTERJSAMPLE };
  __v4hi cb = __vis_fpsub16(vis_widen_u8(cb8), center);
  __v4hi cr = __vis_fpsub16(vis_widen_u8(cr8), center);

  *rd = vis_round_shift(vis_mul16_const(cr, F_1_402), 14);
  *gd = vis_round_shift(vis_addw(vis_mul16_const(cb, -F_0_344),
                                 vis_mul16_const(cr, -F_0_714)), 15);
  *bd = vis_round_shift(vis_mul16_const(cb, F_1_772), 14);
}


static inline void
store_rgbx4(JSAMPLE *out, __v4qi y8, __v4hi rd, __v4hi gd, __v4hi bd,
            int ro, int go, int bo, int ao)
{
  const __v4qi opaque = { -1, -1, -1, -1 };
  __v4hi y = vis_widen_u8(y8);
  __v4qi c[4];

  c[ro] = vis_clamp_u8(__vis_fpadd16(y, rd));
  c[go] = vis_clamp_u8(__vis_fpadd16(y, gd));
  c[bo] = vis_clamp_u8(__vis_fpadd16(y, bd));
  c[ao] = opaque;
  vis_store_4pixel(out, c[0], c[1], c[2], c[3]);
}


static void
h2v1_merged_rgbx_vis(JDIMENSION output_width, JSAMPIMAGE input_buf,
                     JDIMENSION group, JSAMPARRAY output_buf,
                     int ro, int go, int bo, int ao)
{
  JSAMPROW y = input_buf[0][group];
  JSAMPROW cb = input_buf[1][group];
  JSAMPROW cr = input_buf[2][group];
  JSAMPROW out = output_buf[0];
  JDIMENSION col = 0;

  if (((JUINTPTR)out & 7) == 0) {
    for (; col + 4 <= output_width; col += 4) {
      JDIMENSION cc = col >> 1;
      __v4hi rd, gd, bd;
      __v4qi y8 = *(const __v4qi *)(const void *)(y + col);

      chroma_deltas2(cb[cc], cb[cc + 1], cr[cc], cr[cc + 1],
                     &rd, &gd, &bd);
      store_rgbx4(out + 4 * col, y8, rd, gd, bd, ro, go, bo, ao);
    }
  }

  for (; col < output_width; col++) {
    JDIMENSION cc = col >> 1;
    scalar_pixel(y[col], cb[cc], cr[cc], out + 4 * col, ro, go, bo, ao);
  }
}


static void
h2v2_merged_rgbx_vis(JDIMENSION output_width, JSAMPIMAGE input_buf,
                     JDIMENSION group, JSAMPARRAY output_buf,
                     int ro, int go, int bo, int ao)
{
  JSAMPROW y0 = input_buf[0][group * 2];
  JSAMPROW y1 = input_buf[0][group * 2 + 1];
  JSAMPROW cb = input_buf[1][group];
  JSAMPROW cr = input_buf[2][group];
  JSAMPROW out0 = output_buf[0];
  JSAMPROW out1 = output_buf[1];
  JDIMENSION col = 0;

  if (((JUINTPTR)out0 & 7) == 0 && ((JUINTPTR)out1 & 7) == 0) {
    for (; col + 4 <= output_width; col += 4) {
      JDIMENSION cc = col >> 1;
      __v4hi rd, gd, bd;
      __v4qi yv0 = *(const __v4qi *)(const void *)(y0 + col);
      __v4qi yv1 = *(const __v4qi *)(const void *)(y1 + col);

      chroma_deltas2(cb[cc], cb[cc + 1], cr[cc], cr[cc + 1],
                     &rd, &gd, &bd);
      store_rgbx4(out0 + 4 * col, yv0, rd, gd, bd, ro, go, bo, ao);
      store_rgbx4(out1 + 4 * col, yv1, rd, gd, bd, ro, go, bo, ao);
    }
  }

  for (; col < output_width; col++) {
    JDIMENSION cc = col >> 1;
    scalar_pixel(y0[col], cb[cc], cr[cc], out0 + 4 * col, ro, go, bo, ao);
    scalar_pixel(y1[col], cb[cc], cr[cc], out1 + 4 * col, ro, go, bo, ao);
  }
}


#define DECL_MERGED_FUNCS(PREFIX, RO, GO, BO, AO) HIDDEN void jsimd_h2v1_##PREFIX##_merged_upsample_vis(   JDIMENSION w, JSAMPIMAGE in, JDIMENSION g, JSAMPARRAY out) {   h2v1_merged_rgbx_vis(w, in, g, out, RO, GO, BO, AO); } HIDDEN void jsimd_h2v2_##PREFIX##_merged_upsample_vis(   JDIMENSION w, JSAMPIMAGE in, JDIMENSION g, JSAMPARRAY out) {   h2v2_merged_rgbx_vis(w, in, g, out, RO, GO, BO, AO); }

DECL_MERGED_FUNCS(extrgbx, 0, 1, 2, 3)
DECL_MERGED_FUNCS(extbgrx, 2, 1, 0, 3)
DECL_MERGED_FUNCS(extxbgr, 3, 2, 1, 0)
DECL_MERGED_FUNCS(extxrgb, 1, 2, 3, 0)

#undef DECL_MERGED_FUNCS


typedef union {
  __v4qi v;
  uint8_t lane[4];
} vis_4b;


static inline void
store_rgb24_from_deltas(JSAMPLE *out, __v4qi y8,
                        __v4hi rd, __v4hi gd, __v4hi bd,
                        int ro, int go, int bo)
{
  vis_4b r, g, b;
  __v4hi y = vis_widen_u8(y8);
  int i;

  r.v = vis_clamp_u8(__vis_fpadd16(y, rd));
  g.v = vis_clamp_u8(__vis_fpadd16(y, gd));
  b.v = vis_clamp_u8(__vis_fpadd16(y, bd));

  for (i = 0; i < 4; i++) {
    out[3 * i + ro] = r.lane[i];
    out[3 * i + go] = g.lane[i];
    out[3 * i + bo] = b.lane[i];
  }
}


static inline void
scalar_pixel24(JSAMPLE y, JSAMPLE cbv, JSAMPLE crv, JSAMPLE *dst,
               int ro, int go, int bo)
{
  int cb = GETJSAMPLE(cbv) - CENTERJSAMPLE;
  int cr = GETJSAMPLE(crv) - CENTERJSAMPLE;
  int yy = GETJSAMPLE(y);

  dst[ro] = clamp8(yy + descale(cr * F_1_402, 14));
  dst[go] = clamp8(yy + descale(cb * -F_0_344 + cr * -F_0_714, 15));
  dst[bo] = clamp8(yy + descale(cb * F_1_772, 14));
}


static void
h2v1_merged_rgb24_vis(JDIMENSION output_width, JSAMPIMAGE input_buf,
                      JDIMENSION group, JSAMPARRAY output_buf,
                      int ro, int go, int bo)
{
  JSAMPROW y = input_buf[0][group];
  JSAMPROW cb = input_buf[1][group];
  JSAMPROW cr = input_buf[2][group];
  JSAMPROW out = output_buf[0];
  JDIMENSION col = 0;

  for (; col + 4 <= output_width; col += 4) {
    JDIMENSION cc = col >> 1;
    __v4hi rd, gd, bd;
    __v4qi y8 = *(const __v4qi *)(const void *)(y + col);

    chroma_deltas2(cb[cc], cb[cc + 1], cr[cc], cr[cc + 1],
                   &rd, &gd, &bd);
    store_rgb24_from_deltas(out + 3 * col, y8, rd, gd, bd, ro, go, bo);
  }

  for (; col < output_width; col++) {
    JDIMENSION cc = col >> 1;

    scalar_pixel24(y[col], cb[cc], cr[cc], out + 3 * col, ro, go, bo);
  }
}


static void
h2v2_merged_rgb24_vis(JDIMENSION output_width, JSAMPIMAGE input_buf,
                      JDIMENSION group, JSAMPARRAY output_buf,
                      int ro, int go, int bo)
{
  JSAMPROW y0 = input_buf[0][group * 2];
  JSAMPROW y1 = input_buf[0][group * 2 + 1];
  JSAMPROW cb = input_buf[1][group];
  JSAMPROW cr = input_buf[2][group];
  JSAMPROW out0 = output_buf[0];
  JSAMPROW out1 = output_buf[1];
  JDIMENSION col = 0;

  for (; col + 4 <= output_width; col += 4) {
    JDIMENSION cc = col >> 1;
    __v4hi rd, gd, bd;
    __v4qi yv0 = *(const __v4qi *)(const void *)(y0 + col);
    __v4qi yv1 = *(const __v4qi *)(const void *)(y1 + col);

    chroma_deltas2(cb[cc], cb[cc + 1], cr[cc], cr[cc + 1],
                   &rd, &gd, &bd);
    store_rgb24_from_deltas(out0 + 3 * col, yv0, rd, gd, bd, ro, go, bo);
    store_rgb24_from_deltas(out1 + 3 * col, yv1, rd, gd, bd, ro, go, bo);
  }

  for (; col < output_width; col++) {
    JDIMENSION cc = col >> 1;

    scalar_pixel24(y0[col], cb[cc], cr[cc], out0 + 3 * col, ro, go, bo);
    scalar_pixel24(y1[col], cb[cc], cr[cc], out1 + 3 * col, ro, go, bo);
  }
}


HIDDEN void
jsimd_h2v1_merged_upsample_vis(JDIMENSION w, JSAMPIMAGE in,
                               JDIMENSION g, JSAMPARRAY out)
{
  h2v1_merged_rgb24_vis(w, in, g, out, 0, 1, 2);
}


HIDDEN void
jsimd_h2v2_merged_upsample_vis(JDIMENSION w, JSAMPIMAGE in,
                               JDIMENSION g, JSAMPARRAY out)
{
  h2v2_merged_rgb24_vis(w, in, g, out, 0, 1, 2);
}


HIDDEN void
jsimd_h2v1_extrgb_merged_upsample_vis(JDIMENSION w, JSAMPIMAGE in,
                                      JDIMENSION g, JSAMPARRAY out)
{
  h2v1_merged_rgb24_vis(w, in, g, out, 0, 1, 2);
}


HIDDEN void
jsimd_h2v2_extrgb_merged_upsample_vis(JDIMENSION w, JSAMPIMAGE in,
                                      JDIMENSION g, JSAMPARRAY out)
{
  h2v2_merged_rgb24_vis(w, in, g, out, 0, 1, 2);
}


HIDDEN void
jsimd_h2v1_extbgr_merged_upsample_vis(JDIMENSION w, JSAMPIMAGE in,
                                      JDIMENSION g, JSAMPARRAY out)
{
  h2v1_merged_rgb24_vis(w, in, g, out, 2, 1, 0);
}


HIDDEN void
jsimd_h2v2_extbgr_merged_upsample_vis(JDIMENSION w, JSAMPIMAGE in,
                                      JDIMENSION g, JSAMPARRAY out)
{
  h2v2_merged_rgb24_vis(w, in, g, out, 2, 1, 0);
}
