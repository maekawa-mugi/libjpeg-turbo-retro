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

#if defined(PS2_EXPERIMENTAL_MERGED_TABLE)
#include "color-table-mmi.h"
#endif

static const short clamp255[8] __attribute__((aligned(16))) =
  { 255, 255, 255, 255, 255, 255, 255, 255 };

static INLINE void
chroma_offsets(int cb, int cr, int *r, int *g, int *b)
{
#if defined(PS2_EXPERIMENTAL_MERGED_TABLE)
  ps2_table_chroma_offsets((unsigned)cb, (unsigned)cr, r, g, b);
#else
  cb -= 128;
  cr -= 128;
  *r = (91881 * cr + 32768) >> 16;
  *g = (-22554 * cb - 46802 * cr + 32768) >> 16;
  *b = (116130 * cb + 32768) >> 16;
#endif
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


/* Two adjacent four-pixel groups share one coefficient load and inlined
 * PMULTH block.  Keep IJG's original 16.16 signed rounding for G, which
 * must add both products before shifting.
 */
#if defined(PS2_EXPERIMENTAL_MERGED_PMULTH8) && \
    defined(PS2_EXPERIMENTAL_MERGED_PMULTH)
static INLINE void
chroma_offsets4_mmi(const JSAMPLE *cbp, const JSAMPLE *crp,
                    int r[4], int g[4], int b[4])
{
  short inputs[16] __attribute__((aligned(16)));
  int32_t products[16] __attribute__((aligned(16)));
  int i;
  for (i = 0; i < 4; i++) {
    short cb = (short)((int)cbp[i] - 128);
    short cr = (short)((int)crp[i] - 128);
    inputs[4 * i] = cr;
    inputs[4 * i + 1] = cb;
    inputs[4 * i + 2] = cb;
    inputs[4 * i + 3] = cr;
  }
  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $8, 0(%0)\n\t"
    "lq $9, 0(%1)\n\t"
    "pmulth $10, $8, $9\n\t"
    "lq $15, 16(%0)\n\t"
    "pmflo $11\n\t"
    "pmfhi $12\n\t"
    "pcpyld $13, $12, $11\n\t"
    "pcpyud $14, $11, $12\n\t"
    "sq $13, 0(%2)\n\t"
    "sq $14, 16(%2)\n\t"
    "pmulth $10, $15, $9\n\t"
    "pmflo $11\n\t"
    "pmfhi $12\n\t"
    "pcpyld $13, $12, $11\n\t"
    "pcpyud $14, $11, $12\n\t"
    "sq $13, 32(%2)\n\t"
    "sq $14, 48(%2)\n\t"
    ".set pop\n\t"
    :
    : "r" (inputs), "r" (merge_coeff), "r" (products)
    : "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15",
      "memory");
  for (i = 0; i < 4; i++) {
    int cb = (int)cbp[i] - 128;
    int cr = (int)crp[i] - 128;
    r[i] = cr + ((products[4 * i] + 32768) >> 16);
    g[i] = -cr +
           ((products[4 * i + 2] + products[4 * i + 3] + 32768) >> 16);
    b[i] = 2 * cb + ((products[4 * i + 1] + 32768) >> 16);
  }
}
#endif


/*
 * Optional all-MMI chroma offset reconstruction for four chroma samples.
 * PMULTH produces R, G(Cb), B and an unused alpha lane, then PMADDH adds
 * the G(Cr) product in-place in HI/LO.  The 16.16 integer contribution,
 * signed rounding bias and alpha value are pre-built as signed 32-bit
 * words.  PADDW/PSRAW fold all three chroma offsets down to 16 bits and
 * PPACH/PCPYLD/PCPYUD replicate each chroma sample into two output pixels.
 *
 * This avoids spilling the 16 individual 32-bit products and the scalar
 * channel reconstruction, but increases operand preparation.  Keep it
 * disabled until the EE primitive and full merged tests pass and performance
 * on the hardware is known.
 */
#if defined(PS2_EXPERIMENTAL_MERGED_VECTOR_OFFSETS) && \
    defined(PS2_EXPERIMENTAL_MERGED_ADD_PACK)
static INLINE void
chroma_offsets4_vector_mmi(const JSAMPLE *cbp, const JSAMPLE *crp,
                           short offsets[2][16],
                           const short main_coeff[8],
                           const short green_coeff[8],
                           int red, int green, int blue, int alpha)
{
  short main_input[16] __attribute__((aligned(16)));
  short green_input[16] __attribute__((aligned(16)));
  int32_t bias[16] __attribute__((aligned(16)));
  int i;

  for (i = 0; i < 4; i++) {
    int cb = (int)cbp[i] - 128;
    int cr = (int)crp[i] - 128;
    short *in = main_input + 4 * i;
    short *extra = green_input + 4 * i;
    int32_t *base = bias + 4 * i;
    in[red] = (short)cr;
    in[green] = (short)cb;
    in[blue] = (short)cb;
    in[alpha] = 0;
    extra[red] = 0;
    extra[green] = (short)cr;
    extra[blue] = 0;
    extra[alpha] = 0;
    /* Multiplication, not a left shift: shifting a negative signed value
     * would be undefined C.  These intermediates never exceed int32_t.
     */
    base[red] = cr * 65536 + 32768;
    base[green] = -cr * 65536 + 32768;
    base[blue] = 2 * cb * 65536 + 32768;
    base[alpha] = 255 * 65536;
  }

  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $12, 0(%3)\n\t"
    "lq $13, 0(%4)\n\t"
    /* First two chroma samples, four 32-bit channels apiece. */
    "lq $8, 0(%0)\n\t"
    "lq $9, 0(%1)\n\t"
    "pmulth $14, $8, $12\n\t"
    "pmaddh $14, $9, $13\n\t"
    "pmflo $14\n\t"
    "pmfhi $15\n\t"
    "pcpyld $8, $15, $14\n\t"
    "pcpyud $9, $14, $15\n\t"
    "lq $10, 0(%2)\n\t"
    "lq $11, 16(%2)\n\t"
    "paddw $8, $8, $10\n\t"
    "paddw $9, $9, $11\n\t"
    "psraw $8, $8, 16\n\t"
    "psraw $9, $9, 16\n\t"
    "ppach $8, $9, $8\n\t"
    "pcpyld $9, $8, $8\n\t"
    "pcpyud $8, $8, $8\n\t"
    "sq $9, 0(%5)\n\t"
    "sq $8, 16(%5)\n\t"
    /* Remaining two chroma samples reuse both coefficient vectors. */
    "lq $8, 16(%0)\n\t"
    "lq $9, 16(%1)\n\t"
    "pmulth $14, $8, $12\n\t"
    "pmaddh $14, $9, $13\n\t"
    "pmflo $14\n\t"
    "pmfhi $15\n\t"
    "pcpyld $8, $15, $14\n\t"
    "pcpyud $9, $14, $15\n\t"
    "lq $10, 32(%2)\n\t"
    "lq $11, 48(%2)\n\t"
    "paddw $8, $8, $10\n\t"
    "paddw $9, $9, $11\n\t"
    "psraw $8, $8, 16\n\t"
    "psraw $9, $9, 16\n\t"
    "ppach $8, $9, $8\n\t"
    "pcpyld $9, $8, $8\n\t"
    "pcpyud $8, $8, $8\n\t"
    "sq $9, 32(%5)\n\t"
    "sq $8, 48(%5)\n\t"
    ".set pop\n\t"
    :
    : "r" (main_input), "r" (green_input), "r" (bias),
      "r" (main_coeff), "r" (green_coeff), "r" (offsets)
    : "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15", "memory");
}
#endif

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
 * Optional final-stage vectorization: the chroma matrix offsets are already
 * shared between neighboring luma pixels (and both rows of h2v2).  Build
 * 16 signed halfword offsets once for each four-pixel group; form four
 * replicated luma words without reading beyond the row; then apply PADDH,
 * signed clamp, and PPACB entirely in the MMI registers.
 *
 * PEXTLB/PEXTUB with rs=$zero and rt=luma byte vector zero-extend each
 * byte to a halfword.  The alpha lane must receive zero luma so that its
 * pre-filled value of 255 is preserved after PADDH.
 */
#if defined(PS2_EXPERIMENTAL_MERGED_ADD_PACK) && \
    defined(PS2_EXPERIMENTAL_MERGED_PMULTH8)
static INLINE void
pack4_add_offsets_mmi(const short offsets[16], const JSAMPLE *yp,
                      JSAMPLE *dst, int alpha)
{
  uint32_t luma_words[4] __attribute__((aligned(16)));
  JSAMPLE scratch[16] __attribute__((aligned(16)));
  JSAMPLE *target = (((uintptr_t)dst & 15) == 0) ? dst : scratch;
  const uint32_t mask = alpha == 0 ? 0xffffff00u : 0x00ffffffu;
  unsigned k;

  /* Shifts and ORs create four Y bytes without a scalar MULT/MFLO that
   * would contend with the R5900 MMI multiply accumulator.  Clearing alpha
   * allows a single vector add for both RGB and XRGB byte layouts.
   * The EE target is little-endian; output bytes are still in lane order.
   */
  for (k = 0; k < 4; k++) {
    uint32_t v = (uint32_t)yp[k];
    v |= v << 8;
    v |= v << 16;
    luma_words[k] = v & mask;
  }

  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $8, 0(%0)\n\t"
    "lq $9, 16(%0)\n\t"
    "lq $10, 0(%1)\n\t"
    "pextlb $11, $0, $10\n\t"
    "pextub $12, $0, $10\n\t"
    "paddh $8, $8, $11\n\t"
    "paddh $9, $9, $12\n\t"
    "lq $13, 0(%2)\n\t"
    "pmaxh $8, $8, $0\n\t"
    "pmaxh $9, $9, $0\n\t"
    "pminh $8, $8, $13\n\t"
    "pminh $9, $9, $13\n\t"
    "ppacb $8, $9, $8\n\t"
    "sq $8, 0(%3)\n\t"
    ".set pop\n\t"
    :
    : "r" (offsets), "r" (luma_words), "r" (clamp255), "r" (target)
    : "$8", "$9", "$10", "$11", "$12", "$13", "memory");

  if (target == scratch)
    memcpy(dst, scratch, 16);
}
#endif

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
#if defined(PS2_EXPERIMENTAL_MERGED_VECTOR_OFFSETS) && \
    defined(PS2_EXPERIMENTAL_MERGED_ADD_PACK)
  short main_coeff[8] __attribute__((aligned(16)));
  short green_coeff[8] __attribute__((aligned(16)));
  int slot;
  for (slot = 0; slot < 8; slot++) {
    main_coeff[slot] = 0;
    green_coeff[slot] = 0;
  }
  for (slot = 0; slot < 8; slot += 4) {
    main_coeff[slot + red] = 26345;
    main_coeff[slot + green] = -22554;
    main_coeff[slot + blue] = -14942;
    green_coeff[slot + green] = 18734;
  }
#endif

#if defined(PS2_EXPERIMENTAL_MERGED_PMULTH8) && \
    defined(PS2_EXPERIMENTAL_MERGED_PMULTH)
  /* Four chroma samples and eight output pixels per batch.  The second
   * luma row shares the chroma offsets in the h2v2 converter.
   */
  for (; width - col >= 8; col += 8) {
#if defined(PS2_EXPERIMENTAL_MERGED_ADD_PACK)
    short offsets[2][16] __attribute__((aligned(16)));
#else
    short lanes0[2][16] __attribute__((aligned(16)));
    short lanes1[2][16] __attribute__((aligned(16)));
#endif
#if defined(PS2_EXPERIMENTAL_MERGED_VECTOR_OFFSETS) && \
    defined(PS2_EXPERIMENTAL_MERGED_ADD_PACK)
    chroma_offsets4_vector_mmi(cb + col / 2, cr + col / 2, offsets,
                               main_coeff, green_coeff,
                               red, green, blue, alpha);
#else
    int r[4], g[4], b[4];
    unsigned k;
    chroma_offsets4_mmi(cb + col / 2, cr + col / 2, r, g, b);
#endif

#if defined(PS2_EXPERIMENTAL_MERGED_ADD_PACK)
#if !defined(PS2_EXPERIMENTAL_MERGED_VECTOR_OFFSETS)
    /* Build one set of chroma offsets for two output rows. */
    for (k = 0; k < 8; k++) {
      short *out = offsets[k / 4] + 4 * (k % 4);
      unsigned chroma = k / 2;
      out[red] = (short)r[chroma];
      out[green] = (short)g[chroma];
      out[blue] = (short)b[chroma];
      out[alpha] = 255;
    }
#endif
    pack4_add_offsets_mmi(offsets[0], y0 + col, dst0 + 4 * col, alpha);
    pack4_add_offsets_mmi(offsets[1], y0 + col + 4,
                          dst0 + 4 * (col + 4), alpha);
    if (vertical == 2) {
      pack4_add_offsets_mmi(offsets[0], y1 + col, dst1 + 4 * col, alpha);
      pack4_add_offsets_mmi(offsets[1], y1 + col + 4,
                            dst1 + 4 * (col + 4), alpha);
    }
#else
    for (k = 0; k < 8; k++) {
      unsigned group = k / 4;
      unsigned pixel = k % 4;
      unsigned chroma = k / 2;
      put_lanes(lanes0[group], pixel, y0[col + k],
                r[chroma], g[chroma], b[chroma],
                red, green, blue, alpha);
      if (vertical == 2)
        put_lanes(lanes1[group], pixel, y1[col + k],
                  r[chroma], g[chroma], b[chroma],
                  red, green, blue, alpha);
    }
    pack4_mmi(lanes0[0], dst0 + 4 * col);
    pack4_mmi(lanes0[1], dst0 + 4 * (col + 4));
    if (vertical == 2) {
      pack4_mmi(lanes1[0], dst1 + 4 * col);
      pack4_mmi(lanes1[1], dst1 + 4 * (col + 4));
    }
#endif
  }
#endif

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

  /* Consume the remaining luma pair with one chroma calculation,
   * rather than calculating the same offsets twice for adjacent pixels.
   */
  for (; width - col >= 2; col += 2) {
    int r, g, b;
    chroma_offsets(cb[col / 2], cr[col / 2], &r, &g, &b);
    put_scalar(dst0 + 4 * col, y0[col], r, g, b,
               red, green, blue, alpha);
    put_scalar(dst0 + 4 * (col + 1), y0[col + 1], r, g, b,
               red, green, blue, alpha);
    if (vertical == 2) {
      put_scalar(dst1 + 4 * col, y1[col], r, g, b,
                 red, green, blue, alpha);
      put_scalar(dst1 + 4 * (col + 1), y1[col + 1], r, g, b,
                 red, green, blue, alpha);
    }
  }
  if (col < width) {
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
