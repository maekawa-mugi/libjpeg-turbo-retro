/*
 * PS2 Emotion Engine (R5900) merged 2:1 upsampling and YCbCr to RGBX.
 *
 * The original merged converter generates chroma offsets once per pair of
 * luma pixels.  Preserve its exact 16.16 fixed-point rounding while packing
 * four pixels into a 128-bit MMI store without creating expanded Cb/Cr rows.
 *
 * The color matrix is currently scalar; the saturating 16-bit clamp and
 * pack are MMI.  Benchmark against jdmerge.c before enabling by default.
 *
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <stdint.h>
#include <string.h>

static const short clamp255[8] __attribute__((aligned(16))) =
  { 255, 255, 255, 255, 255, 255, 255, 255 };

static INLINE void
chroma_offsets(int cb, int cr, int *r, int *g, int *b)
{
  cb -= 128;
  cr -= 128;
  *r = (91881 * cr + 32768) >> 16;
  *g = (-22554 * cb - 46802 * cr + 32768) >> 16;
  *b = (116130 * cb + 32768) >> 16;
}

/*
 * Two adjacent 2:1 chroma samples need eight independent 16x16 products.
 * Decompose large IJG color constants into a 16-bit fractional component:
 *   R = Cr + ((26345 * Cr + 32768) >> 16)
 *   G = -Cr + ((-22554 * Cb + 18734 * Cr + 32768) >> 16)
 *   B = 2*Cb + ((-14942 * Cb + 32768) >> 16)
 * where Cb/Cr are centered around zero.
 *
 * These are algebraically identical to the reference 91881/46802/116130
 * constants, including signed rounding.  This PMULTH path is separately
 * experimental because shuffling the products via HI/LO may cost more than
 * scalar multiplication on the Emotion Engine.
 */
#if defined(PS2_EXPERIMENTAL_MERGED_PMULTH)
static const short merge_coeff[8] __attribute__((aligned(16))) = {
  26345, -14942, -22554, 18734,
  26345, -14942, -22554, 18734
};

static __attribute__((noinline)) void
merge_mul8_mmi(const short lanes[8], JLONG products[8])
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
    "pcpyud $14, $11, $12\n\t"
    "sq $13, 0(%2)\n\t"
    "sq $14, 16(%2)\n\t"
    ".set pop\n\t"
    :
    : "r" (lanes), "r" (merge_coeff), "r" (products)
    : "$8", "$9", "$10", "$11", "$12", "$13", "$14", "memory");
}
#endif

static INLINE void
chroma_offsets2(int cb0, int cr0, int cb1, int cr1,
                int r[2], int g[2], int b[2])
{
#if defined(PS2_EXPERIMENTAL_MERGED_PMULTH)
  short lanes[8] __attribute__((aligned(16)));
  JLONG products[8] __attribute__((aligned(16)));
  int cb[2] = { cb0 - 128, cb1 - 128 };
  int cr[2] = { cr0 - 128, cr1 - 128 };
  int i;

  for (i = 0; i < 2; i++) {
    lanes[4 * i] = (short)cr[i];
    lanes[4 * i + 1] = (short)cb[i];
    lanes[4 * i + 2] = (short)cb[i];
    lanes[4 * i + 3] = (short)cr[i];
  }
  merge_mul8_mmi(lanes, products);
  for (i = 0; i < 2; i++) {
    r[i] = cr[i] + ((products[4 * i] + 32768) >> 16);
    g[i] = -cr[i] +
           ((products[4 * i + 2] + products[4 * i + 3] + 32768) >> 16);
    b[i] = 2 * cb[i] + ((products[4 * i + 1] + 32768) >> 16);
  }
#else
  chroma_offsets(cb0, cr0, &r[0], &g[0], &b[0]);
  chroma_offsets(cb1, cr1, &r[1], &g[1], &b[1]);
#endif
}

static INLINE JSAMPLE
clamp_byte(int x)
{
  return (JSAMPLE)(x < 0 ? 0 : x > 255 ? 255 : x);
}

/* Colors are arranged in final byte order before the MMI packing stage. */
static INLINE void
put_lanes(short lanes[16], unsigned pixel, int y,
          int r, int g, int b, int red, int green, int blue, int alpha)
{
  short *p = lanes + 4 * pixel;
  p[red] = (short)(y + r);
  p[green] = (short)(y + g);
  p[blue] = (short)(y + b);
  p[alpha] = 255;
}

static INLINE void
put_scalar(JSAMPLE *dst, int y, int r, int g, int b,
           int red, int green, int blue, int alpha)
{
  dst[red] = clamp_byte(y + r);
  dst[green] = clamp_byte(y + g);
  dst[blue] = clamp_byte(y + b);
  dst[alpha] = 255;
}

/* Saturate and pack four 4-byte pixels.  All LQ/SQ operands are aligned. */
static INLINE void
pack4_mmi(const short lanes[16], JSAMPLE *dst)
{
  JSAMPLE scratch[16] __attribute__((aligned(16)));
  JSAMPLE *target = (((uintptr_t)dst & 15) == 0) ? dst : scratch;

  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $8, 0(%0)\n\t"
    "lq $9, 16(%0)\n\t"
    "lq $10, 0(%1)\n\t"
    "pmaxh $8, $8, $0\n\t"
    "pmaxh $9, $9, $0\n\t"
    "pminh $8, $8, $10\n\t"
    "pminh $9, $9, $10\n\t"
    "ppacb $8, $9, $8\n\t"
    "sq $8, 0(%2)\n\t"
    ".set pop\n\t"
    :
    : "r" (lanes), "r" (clamp255), "r" (target)
    : "$8", "$9", "$10", "memory");

  if (target == scratch)
    memcpy(dst, scratch, 16);
}

/*
 * 2h1v: one luma row; 2h2v: two luma rows share each chroma sample.
 * The generic merged upsampler only calls these routines for 8-bit YCbCr
 * with 2:1 chroma sampling.  Never read past ceil(width / 2) Cb/Cr bytes.
 */
static void
merged_rows(JDIMENSION width, JSAMPIMAGE input_buf,
            JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf,
            int vertical, int red, int green, int blue, int alpha)
{
  const JSAMPLE *y0 = input_buf[0][in_row_group_ctr * (JDIMENSION)vertical];
  const JSAMPLE *y1 = vertical == 2 ?
    input_buf[0][in_row_group_ctr * 2 + 1] : NULL;
  const JSAMPLE *cb = input_buf[1][in_row_group_ctr];
  const JSAMPLE *cr = input_buf[2][in_row_group_ctr];
  JSAMPLE *dst0 = output_buf[0];
  JSAMPLE *dst1 = vertical == 2 ? output_buf[1] : NULL;
  JDIMENSION col = 0;

  for (; width - col >= 4; col += 4) {
    short lanes0[16] __attribute__((aligned(16)));
    short lanes1[16] __attribute__((aligned(16)));
    int r[2], g[2], b[2];
    unsigned k;

    /* Share each chroma result between adjacent luma pixels and rows. */
    chroma_offsets2(cb[col / 2], cr[col / 2],
                    cb[col / 2 + 1], cr[col / 2 + 1], r, g, b);
    for (k = 0; k < 4; k += 2) {
      unsigned p = k / 2;
      put_lanes(lanes0, k, y0[col + k], r[p], g[p], b[p],
                red, green, blue, alpha);
      put_lanes(lanes0, k + 1, y0[col + k + 1], r[p], g[p], b[p],
                red, green, blue, alpha);
      if (vertical == 2) {
        put_lanes(lanes1, k, y1[col + k], r[p], g[p], b[p],
                  red, green, blue, alpha);
        put_lanes(lanes1, k + 1, y1[col + k + 1], r[p], g[p], b[p],
                  red, green, blue, alpha);
      }
    }
    pack4_mmi(lanes0, dst0 + 4 * col);
    if (vertical == 2)
      pack4_mmi(lanes1, dst1 + 4 * col);
  }

  /* Odd width and any remaining 1..3 pixels use bounded scalar stores. */
  for (; col < width; col++) {
    int r, g, b;
    chroma_offsets(cb[col / 2], cr[col / 2], &r, &g, &b);
    put_scalar(dst0 + 4 * col, y0[col], r, g, b,
               red, green, blue, alpha);
    if (vertical == 2)
      put_scalar(dst1 + 4 * col, y1[col], r, g, b,
                 red, green, blue, alpha);
  }
}

#define PS2_MERGED(name, v, r, g, b, a) \
HIDDEN void \
name(JDIMENSION width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr, \
     JSAMPARRAY output_buf) \
{ \
  merged_rows(width, input_buf, in_row_group_ctr, output_buf, v, r, g, b, a); \
}

#define PS2_MERGED_LAYOUT(layout, r, g, b, a) \
PS2_MERGED(jsimd_h2v1_##layout##_merged_upsample_ps2mmi, 1, r, g, b, a) \
PS2_MERGED(jsimd_h2v2_##layout##_merged_upsample_ps2mmi, 2, r, g, b, a)

PS2_MERGED_LAYOUT(extrgbx, EXT_RGBX_RED, EXT_RGBX_GREEN, EXT_RGBX_BLUE, 3)
PS2_MERGED_LAYOUT(extbgrx, EXT_BGRX_RED, EXT_BGRX_GREEN, EXT_BGRX_BLUE, 3)
PS2_MERGED_LAYOUT(extxbgr, EXT_XBGR_RED, EXT_XBGR_GREEN, EXT_XBGR_BLUE, 0)
PS2_MERGED_LAYOUT(extxrgb, EXT_XRGB_RED, EXT_XRGB_GREEN, EXT_XRGB_BLUE, 0)
