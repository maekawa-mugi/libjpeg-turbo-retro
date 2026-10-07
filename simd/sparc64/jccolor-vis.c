/*
 * 4-byte RGB-family -> YCbCr/Gray conversion (SPARC VIS 1)
 *
 * VIS1 has no bshuffle.  Four input pixels are therefore deinterleaved with
 * aligned 32-bit integer loads, after which all color arithmetic is performed
 * four pixels at a time with VIS1 fixed-point operations.
 */

#include "../jsimdint.h"
#include <stdint.h>
#include <visintrin.h>


#define F_0_299  19595
#define F_0_587  38470
#define F_0_114  7471
#define F_0_168  11059
#define F_0_331  21709
#define F_0_418  27439
#define F_0_081  5329


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
  __v4qi v;
  uint8_t lane[4];
} vis_4b;

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
vis_round_shift16(vis_4w a, int32_t bias)
{
  vis_4h out;
  const __v2si b = { bias, bias };

  __builtin_vis_write_gsr(0);
  out.p.hi = __vis_fpackfix(__vis_fpadd32(a.hi, b));
  out.p.lo = __vis_fpackfix(__vis_fpadd32(a.lo, b));
  return out.v;
}


static inline __v4hi
vis_widen_u8(__v4qi x)
{
  const __v2hi exact = { 256, 256 };

  return __vis_fmul8x16au(x, exact);
}


static inline __v4qi
vis_pack_u8(__v4hi x)
{
  x = __vis_fpadd16(x, x);
  x = __vis_fpadd16(x, x);
  x = __vis_fpadd16(x, x);
  x = __vis_fpadd16(x, x);
  __builtin_vis_write_gsr(3 << 3);
  return __vis_fpack16(x);
}


static inline void
load_rgbx4(const JSAMPLE *src, int ro, int go, int bo,
           __v4qi *r, __v4qi *g, __v4qi *b)
{
  vis_4b rv, gv, bv;
  int i;

  for (i = 0; i < 4; i++) {
    uint32_t p = *(const uint32_t *)(const void *)(src + 4 * i);

    rv.lane[i] = (uint8_t)(p >> ((3 - ro) * 8));
    gv.lane[i] = (uint8_t)(p >> ((3 - go) * 8));
    bv.lane[i] = (uint8_t)(p >> ((3 - bo) * 8));
  }

  *r = rv.v;
  *g = gv.v;
  *b = bv.v;
}


static inline int
round_shift16(int32_t x, int32_t bias)
{
  return (x + bias) >> 16;
}


static inline void
scalar_ycc(JSAMPLE r, JSAMPLE g, JSAMPLE b,
           JSAMPLE *y, JSAMPLE *cb, JSAMPLE *cr)
{
  int rr = GETJSAMPLE(r);
  int gg = GETJSAMPLE(g);
  int bb = GETJSAMPLE(b);

  *y = (JSAMPLE)
    (gg + round_shift16((rr - gg) * F_0_299 +
                        (bb - gg) * F_0_114, 32768));
  *cb = (JSAMPLE)
    (CENTERJSAMPLE +
     round_shift16((rr - bb) * -F_0_168 +
                   (gg - bb) * -F_0_331, 32767));
  *cr = (JSAMPLE)
    (CENTERJSAMPLE +
     round_shift16((gg - rr) * -F_0_418 +
                   (bb - rr) * -F_0_081, 32767));
}


static inline JSAMPLE
scalar_gray(JSAMPLE r, JSAMPLE g, JSAMPLE b)
{
  int rr = GETJSAMPLE(r);
  int gg = GETJSAMPLE(g);
  int bb = GETJSAMPLE(b);

  return (JSAMPLE)
    (gg + round_shift16((rr - gg) * F_0_299 +
                        (bb - gg) * F_0_114, 32768));
}


static void
rgbx_ycc_vis(JDIMENSION img_width, JSAMPARRAY input_buf,
             JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows,
             int ro, int go, int bo)
{
  const __v4hi center = { CENTERJSAMPLE, CENTERJSAMPLE,
                          CENTERJSAMPLE, CENTERJSAMPLE };

  while (--num_rows >= 0) {
    JSAMPROW in = *input_buf++;
    JSAMPROW outy = output_buf[0][output_row];
    JSAMPROW outcb = output_buf[1][output_row];
    JSAMPROW outcr = output_buf[2][output_row];
    JDIMENSION col = 0;

    output_row++;

    if (((JUINTPTR)in & 3) == 0) {
      for (; col + 4 <= img_width; col += 4) {
        __v4qi r8, g8, b8;
        __v4hi r, g, b;
        __v4hi y, cb, cr;

        load_rgbx4(in + 4 * col, ro, go, bo, &r8, &g8, &b8);
        r = vis_widen_u8(r8);
        g = vis_widen_u8(g8);
        b = vis_widen_u8(b8);

        y = __vis_fpadd16(
              g,
              vis_round_shift16(
                vis_addw(vis_mul16_const(__vis_fpsub16(r, g), F_0_299),
                         vis_mul16_const(__vis_fpsub16(b, g), F_0_114)),
                32768));

        cb = __vis_fpadd16(
               center,
               vis_round_shift16(
                 vis_addw(vis_mul16_const(__vis_fpsub16(r, b), -F_0_168),
                          vis_mul16_const(__vis_fpsub16(g, b), -F_0_331)),
                 32767));

        cr = __vis_fpadd16(
               center,
               vis_round_shift16(
                 vis_addw(vis_mul16_const(__vis_fpsub16(g, r), -F_0_418),
                          vis_mul16_const(__vis_fpsub16(b, r), -F_0_081)),
                 32767));

        *(__v4qi *)(void *)(outy + col) = vis_pack_u8(y);
        *(__v4qi *)(void *)(outcb + col) = vis_pack_u8(cb);
        *(__v4qi *)(void *)(outcr + col) = vis_pack_u8(cr);
      }
    }

    for (; col < img_width; col++) {
      JSAMPROW p = in + 4 * col;
      scalar_ycc(p[ro], p[go], p[bo],
                 outy + col, outcb + col, outcr + col);
    }
  }
}


static void
rgbx_gray_vis(JDIMENSION img_width, JSAMPARRAY input_buf,
              JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows,
              int ro, int go, int bo)
{
  while (--num_rows >= 0) {
    JSAMPROW in = *input_buf++;
    JSAMPROW out = output_buf[0][output_row++];
    JDIMENSION col = 0;

    if (((JUINTPTR)in & 3) == 0) {
      for (; col + 4 <= img_width; col += 4) {
        __v4qi r8, g8, b8;
        __v4hi r, g, b, y;

        load_rgbx4(in + 4 * col, ro, go, bo, &r8, &g8, &b8);
        r = vis_widen_u8(r8);
        g = vis_widen_u8(g8);
        b = vis_widen_u8(b8);

        y = __vis_fpadd16(
              g,
              vis_round_shift16(
                vis_addw(vis_mul16_const(__vis_fpsub16(r, g), F_0_299),
                         vis_mul16_const(__vis_fpsub16(b, g), F_0_114)),
                32768));
        *(__v4qi *)(void *)(out + col) = vis_pack_u8(y);
      }
    }

    for (; col < img_width; col++) {
      JSAMPROW p = in + 4 * col;
      out[col] = scalar_gray(p[ro], p[go], p[bo]);
    }
  }
}


#define DECL_RGBX_FUNCS(PREFIX, RO, GO, BO) HIDDEN void jsimd_##PREFIX##_ycc_convert_vis(JDIMENSION w, JSAMPARRAY in,                                  JSAMPIMAGE out, JDIMENSION row, int nr) {   rgbx_ycc_vis(w, in, out, row, nr, RO, GO, BO); } HIDDEN void jsimd_##PREFIX##_gray_convert_vis(JDIMENSION w, JSAMPARRAY in,                                   JSAMPIMAGE out, JDIMENSION row, int nr) {   rgbx_gray_vis(w, in, out, row, nr, RO, GO, BO); }

DECL_RGBX_FUNCS(extrgbx, 0, 1, 2)
DECL_RGBX_FUNCS(extbgrx, 2, 1, 0)
DECL_RGBX_FUNCS(extxbgr, 3, 2, 1)
DECL_RGBX_FUNCS(extxrgb, 1, 2, 3)

#undef DECL_RGBX_FUNCS


static inline void
load_rgb24_4(const JSAMPLE *src, int ro, int go, int bo,
             __v4qi *r, __v4qi *g, __v4qi *b)
{
  vis_4b rv, gv, bv;
  int i;

  for (i = 0; i < 4; i++) {
    rv.lane[i] = src[3 * i + ro];
    gv.lane[i] = src[3 * i + go];
    bv.lane[i] = src[3 * i + bo];
  }

  *r = rv.v;
  *g = gv.v;
  *b = bv.v;
}


static inline void
rgb_to_ycc4(__v4qi r8, __v4qi g8, __v4qi b8,
            __v4qi *y8, __v4qi *cb8, __v4qi *cr8)
{
  const __v4hi center = { CENTERJSAMPLE, CENTERJSAMPLE,
                          CENTERJSAMPLE, CENTERJSAMPLE };
  __v4hi r = vis_widen_u8(r8);
  __v4hi g = vis_widen_u8(g8);
  __v4hi b = vis_widen_u8(b8);
  __v4hi y, cb, cr;

  y = __vis_fpadd16(
        g,
        vis_round_shift16(
          vis_addw(vis_mul16_const(__vis_fpsub16(r, g), F_0_299),
                   vis_mul16_const(__vis_fpsub16(b, g), F_0_114)),
          32768));

  cb = __vis_fpadd16(
         center,
         vis_round_shift16(
           vis_addw(vis_mul16_const(__vis_fpsub16(r, b), -F_0_168),
                    vis_mul16_const(__vis_fpsub16(g, b), -F_0_331)),
           32767));

  cr = __vis_fpadd16(
         center,
         vis_round_shift16(
           vis_addw(vis_mul16_const(__vis_fpsub16(g, r), -F_0_418),
                    vis_mul16_const(__vis_fpsub16(b, r), -F_0_081)),
           32767));

  *y8 = vis_pack_u8(y);
  *cb8 = vis_pack_u8(cb);
  *cr8 = vis_pack_u8(cr);
}


static inline __v4qi
rgb_to_gray4(__v4qi r8, __v4qi g8, __v4qi b8)
{
  __v4hi r = vis_widen_u8(r8);
  __v4hi g = vis_widen_u8(g8);
  __v4hi b = vis_widen_u8(b8);
  __v4hi y =
    __vis_fpadd16(
      g,
      vis_round_shift16(
        vis_addw(vis_mul16_const(__vis_fpsub16(r, g), F_0_299),
                 vis_mul16_const(__vis_fpsub16(b, g), F_0_114)),
        32768));

  return vis_pack_u8(y);
}


static void
rgb24_ycc_vis(JDIMENSION img_width, JSAMPARRAY input_buf,
              JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows,
              int ro, int go, int bo)
{
  while (--num_rows >= 0) {
    JSAMPROW in = *input_buf++;
    JSAMPROW outy = output_buf[0][output_row];
    JSAMPROW outcb = output_buf[1][output_row];
    JSAMPROW outcr = output_buf[2][output_row];
    JDIMENSION col = 0;

    output_row++;

    for (; col + 4 <= img_width; col += 4) {
      __v4qi r, g, b, y, cb, cr;

      load_rgb24_4(in + 3 * col, ro, go, bo, &r, &g, &b);
      rgb_to_ycc4(r, g, b, &y, &cb, &cr);
      *(__v4qi *)(void *)(outy + col) = y;
      *(__v4qi *)(void *)(outcb + col) = cb;
      *(__v4qi *)(void *)(outcr + col) = cr;
    }

    for (; col < img_width; col++) {
      JSAMPROW p = in + 3 * col;

      scalar_ycc(p[ro], p[go], p[bo],
                 outy + col, outcb + col, outcr + col);
    }
  }
}


static void
rgb24_gray_vis(JDIMENSION img_width, JSAMPARRAY input_buf,
               JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows,
               int ro, int go, int bo)
{
  while (--num_rows >= 0) {
    JSAMPROW in = *input_buf++;
    JSAMPROW out = output_buf[0][output_row++];
    JDIMENSION col = 0;

    for (; col + 4 <= img_width; col += 4) {
      __v4qi r, g, b;

      load_rgb24_4(in + 3 * col, ro, go, bo, &r, &g, &b);
      *(__v4qi *)(void *)(out + col) = rgb_to_gray4(r, g, b);
    }

    for (; col < img_width; col++) {
      JSAMPROW p = in + 3 * col;

      out[col] = scalar_gray(p[ro], p[go], p[bo]);
    }
  }
}


HIDDEN void
jsimd_rgb_ycc_convert_vis(JDIMENSION w, JSAMPARRAY in, JSAMPIMAGE out,
                          JDIMENSION row, int nr)
{
  rgb24_ycc_vis(w, in, out, row, nr, 0, 1, 2);
}


HIDDEN void
jsimd_extrgb_ycc_convert_vis(JDIMENSION w, JSAMPARRAY in, JSAMPIMAGE out,
                             JDIMENSION row, int nr)
{
  rgb24_ycc_vis(w, in, out, row, nr, 0, 1, 2);
}


HIDDEN void
jsimd_extbgr_ycc_convert_vis(JDIMENSION w, JSAMPARRAY in, JSAMPIMAGE out,
                             JDIMENSION row, int nr)
{
  rgb24_ycc_vis(w, in, out, row, nr, 2, 1, 0);
}


HIDDEN void
jsimd_rgb_gray_convert_vis(JDIMENSION w, JSAMPARRAY in, JSAMPIMAGE out,
                           JDIMENSION row, int nr)
{
  rgb24_gray_vis(w, in, out, row, nr, 0, 1, 2);
}


HIDDEN void
jsimd_extrgb_gray_convert_vis(JDIMENSION w, JSAMPARRAY in, JSAMPIMAGE out,
                              JDIMENSION row, int nr)
{
  rgb24_gray_vis(w, in, out, row, nr, 0, 1, 2);
}


HIDDEN void
jsimd_extbgr_gray_convert_vis(JDIMENSION w, JSAMPARRAY in, JSAMPIMAGE out,
                              JDIMENSION row, int nr)
{
  rgb24_gray_vis(w, in, out, row, nr, 2, 1, 0);
}
