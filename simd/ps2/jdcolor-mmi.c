/*
 * PlayStation 2 R5900 MMI YCbCr -> 8-bit RGB/RGBA pixel conversion.
 *
 * Use the exact IJG/libjpeg-turbo 16.16 fixed-point YCbCr coefficients
 * and rounding.  RGBX/BGRX/XBGR/XRGB variants use 128-bit MMI signed
 * halfword saturation and byte packing for four pixels per iteration.
 *
 * Scalar color products are the reference path.  The optional four-pixel
 * PMULTH kernel vectorizes the coefficient products while retaining the
 * IJG rounding and scalar channel reconstruction.  Its performance is
 * unmeasured and it must be enabled explicitly.
 *
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <stdint.h>
#include <string.h>

static const short clamp255[8] __attribute__((aligned(16))) =
  { 255, 255, 255, 255, 255, 255, 255, 255 };

/* Identical coefficient rounding to build_ycc_rgb_table() in jdcolor.c.
 * signed right shift is arithmetic on the PS2 R5900 compiler.
 */
static INLINE void
convert_pixel(int y, int cb, int cr, short *r, short *g, short *b)
{
  cb -= 128;
  cr -= 128;
  *r = (short)(y + ((91881 * cr + 32768) >> 16));
  *g = (short)(y + ((-22554 * cb - 46802 * cr + 32768) >> 16));
  *b = (short)(y + ((116130 * cb + 32768) >> 16));
}


/*
 * Optional four-pixel matrix: two PMULTH instructions calculate sixteen
 * signed 16x16 products for four independent Cb/Cr pairs.  Each group of
 * four lanes holds R, B, G(Cb), G(Cr) fractional products for one pixel.
 *
 * 91881 = 65536 + 26345, -46802 = -65536 + 18734,
 * 116130 = 2*65536 - 14942.  Reconstruct *before* the single >> 16
 * rounding shift; separately rounding G(Cb) and G(Cr) changes output.
 *
 * PMULTH writes lanes 0,1,4,5 to LO and 2,3,6,7 to HI.
 * PCPYLD(HI,LO) and PCPYUD(LO,HI) restore the original lane order.
 * These operands were validated in the standalone PS2 primitive test.
 */
#if defined(PS2_EXPERIMENTAL_COLOR_PMULTH)
static const short color_coeff_mmi[8] __attribute__((aligned(16))) = {
  26345, -14942, -22554, 18734,
  26345, -14942, -22554, 18734
};

static INLINE void
convert4_mmi(const JSAMPLE *yp, const JSAMPLE *cbp, const JSAMPLE *crp,
             short lanes[16], int red, int green, int blue, int alpha)
{
  short inputs[16] __attribute__((aligned(16)));
  int32_t products[16] __attribute__((aligned(16)));
  int k;

  for (k = 0; k < 4; ++k) {
    short cb = (short)((int)cbp[k] - 128);
    short cr = (short)((int)crp[k] - 128);
    inputs[4 * k] = cr;
    inputs[4 * k + 1] = cb;
    inputs[4 * k + 2] = cb;
    inputs[4 * k + 3] = cr;
  }

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
    "lq $8, 16(%0)\n\t"
    "pmulth $10, $8, $9\n\t"
    "pmflo $11\n\t"
    "pmfhi $12\n\t"
    "pcpyld $13, $12, $11\n\t"
    "pcpyud $14, $11, $12\n\t"
    "sq $13, 32(%2)\n\t"
    "sq $14, 48(%2)\n\t"
    ".set pop\n\t"
    :
    : "r" (inputs), "r" (color_coeff_mmi), "r" (products)
    : "$8", "$9", "$10", "$11", "$12", "$13", "$14", "memory");

  for (k = 0; k < 4; ++k) {
    int cb = (int)cbp[k] - 128;
    int cr = (int)crp[k] - 128;
    int r = cr + ((products[4 * k] + 32768) >> 16);
    int g = -cr + ((products[4 * k + 2] + products[4 * k + 3] +
                    32768) >> 16);
    int b = 2 * cb + ((products[4 * k + 1] + 32768) >> 16);
    lanes[4 * k + red] = (short)((int)yp[k] + r);
    lanes[4 * k + green] = (short)((int)yp[k] + g);
    lanes[4 * k + blue] = (short)((int)yp[k] + b);
    lanes[4 * k + alpha] = 255;
  }
}
#endif


#if defined(PS2_EXPERIMENTAL_COLOR_PMULTH8) && \
    defined(PS2_EXPERIMENTAL_COLOR_PMULTH)
/* Four PMULTH groups over eight independent samples, but only one
 * coefficient LQ.  Compare on R5900: the larger input/product stack
 * arrays may be slower than two calls to the four-pixel kernel.
 */
static INLINE void
convert8_mmi(const JSAMPLE *yp, const JSAMPLE *cbp, const JSAMPLE *crp,
             short lanes[2][16], int red, int green, int blue, int alpha)
{
  short inputs[32] __attribute__((aligned(16)));
  int32_t products[32] __attribute__((aligned(16)));
  int k;
  for (k = 0; k < 8; k++) {
    short cb = (short)((int)cbp[k] - 128);
    short cr = (short)((int)crp[k] - 128);
    inputs[4 * k] = cr;
    inputs[4 * k + 1] = cb;
    inputs[4 * k + 2] = cb;
    inputs[4 * k + 3] = cr;
  }
  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $9, 0(%1)\n\t"
    "lq $8, 0(%0)\n\t"
    "pmulth $10, $8, $9\n\t"
    "pmflo $11\n\t"
    "pmfhi $12\n\t"
    "pcpyld $13, $12, $11\n\t"
    "pcpyud $14, $11, $12\n\t"
    "sq $13, 0(%2)\n\t"
    "sq $14, 16(%2)\n\t"
    "lq $8, 16(%0)\n\t"
    "pmulth $10, $8, $9\n\t"
    "pmflo $11\n\t"
    "pmfhi $12\n\t"
    "pcpyld $13, $12, $11\n\t"
    "pcpyud $14, $11, $12\n\t"
    "sq $13, 32(%2)\n\t"
    "sq $14, 48(%2)\n\t"
    "lq $8, 32(%0)\n\t"
    "pmulth $10, $8, $9\n\t"
    "pmflo $11\n\t"
    "pmfhi $12\n\t"
    "pcpyld $13, $12, $11\n\t"
    "pcpyud $14, $11, $12\n\t"
    "sq $13, 64(%2)\n\t"
    "sq $14, 80(%2)\n\t"
    "lq $8, 48(%0)\n\t"
    "pmulth $10, $8, $9\n\t"
    "pmflo $11\n\t"
    "pmfhi $12\n\t"
    "pcpyld $13, $12, $11\n\t"
    "pcpyud $14, $11, $12\n\t"
    "sq $13, 96(%2)\n\t"
    "sq $14, 112(%2)\n\t"
    ".set pop\n\t"
    :
    : "r" (inputs), "r" (color_coeff_mmi), "r" (products)
    : "$8", "$9", "$10", "$11", "$12", "$13", "$14", "memory");
  for (k = 0; k < 8; k++) {
    int cb = (int)cbp[k] - 128;
    int cr = (int)crp[k] - 128;
    int r = cr + ((products[4 * k] + 32768) >> 16);
    int g = -cr +
            ((products[4 * k + 2] + products[4 * k + 3] + 32768) >> 16);
    int b = 2 * cb + ((products[4 * k + 1] + 32768) >> 16);
    short *out = lanes[k / 4] + 4 * (k % 4);
    out[red] = (short)((int)yp[k] + r);
    out[green] = (short)((int)yp[k] + g);
    out[blue] = (short)((int)yp[k] + b);
    out[alpha] = 255;
  }
}
#endif

static INLINE JSAMPLE
clip_byte(int value)
{
  if (value < 0)
    return 0;
  if (value > 255)
    return 255;
  return (JSAMPLE)value;
}

/* Input is 16 halfwords: four pixels in their requested byte order.
 * PPACB packs low 8 bytes from RT and high 8 bytes from RS.  P_MAX/MIN
 * are signed-halfword clipping bounds, not modulo-256 truncation.
 * Dest may be unaligned, in which case a temporary aligned store is used.
 */
static INLINE void
pack4_mmi(const short *halfwords, JSAMPLE *dst)
{
  JSAMPLE temporary[16] __attribute__((aligned(16)));
  JSAMPLE *target = (((uintptr_t)dst & 15) == 0) ? dst : temporary;

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
    : "r" (halfwords), "r" (clamp255), "r" (target)
    : "$8", "$9", "$10", "memory");

  if (target == temporary)
    memcpy(dst, temporary, 16);
}

static void
convert_rows(JDIMENSION width, JSAMPIMAGE input_buf, JDIMENSION input_row,
             JSAMPARRAY output_buf, int num_rows,
             int pixel_size, int red, int green, int blue, int alpha)
{
  int row;
  for (row = 0; row < num_rows; row++) {
    const JSAMPLE *yp = input_buf[0][input_row + row];
    const JSAMPLE *cbp = input_buf[1][input_row + row];
    const JSAMPLE *crp = input_buf[2][input_row + row];
    JSAMPLE *dst = output_buf[row];
    JDIMENSION col = 0;

    if (pixel_size == 4) {
#if defined(PS2_EXPERIMENTAL_COLOR_PMULTH8) && \
    defined(PS2_EXPERIMENTAL_COLOR_PMULTH)
      for (; width - col >= 8; col += 8) {
        short lanes[2][16] __attribute__((aligned(16)));
        convert8_mmi(yp + col, cbp + col, crp + col, lanes,
                     red, green, blue, alpha);
        pack4_mmi(lanes[0], dst + col * 4);
        pack4_mmi(lanes[1], dst + (col + 4) * 4);
      }
#endif
      for (; width - col >= 4; col += 4) {
        short lanes[16] __attribute__((aligned(16)));
#if defined(PS2_EXPERIMENTAL_COLOR_PMULTH)
        convert4_mmi(yp + col, cbp + col, crp + col,
                     lanes, red, green, blue, alpha);
#else
        int k;
        for (k = 0; k < 4; k++) {
          short r, g, b;
          convert_pixel((int)yp[col + k], (int)cbp[col + k],
                        (int)crp[col + k], &r, &g, &b);
          lanes[4 * k + red] = r;
          lanes[4 * k + green] = g;
          lanes[4 * k + blue] = b;
          lanes[4 * k + alpha] = 255;
        }
#endif
        pack4_mmi(lanes, dst + col * 4);
      }
    }

    for (; col < width; col++) {
      short r, g, b;
      JSAMPLE *out = dst + col * (JDIMENSION)pixel_size;
      convert_pixel((int)yp[col], (int)cbp[col], (int)crp[col],
                    &r, &g, &b);
      out[red] = clip_byte(r);
      out[green] = clip_byte(g);
      out[blue] = clip_byte(b);
      if (alpha >= 0)
        out[alpha] = 255;
    }
  }
}

#define PS2_YCC_CONVERTER(name, size, red, green, blue, alpha) \
HIDDEN void \
name(JDIMENSION width, JSAMPIMAGE input_buf, JDIMENSION input_row, \
     JSAMPARRAY output_buf, int num_rows) \
{ \
  convert_rows(width, input_buf, input_row, output_buf, num_rows, \
               size, red, green, blue, alpha); \
}

PS2_YCC_CONVERTER(jsimd_ycc_rgb_convert_ps2mmi,
                  RGB_PIXELSIZE, RGB_RED, RGB_GREEN, RGB_BLUE, -1)
PS2_YCC_CONVERTER(jsimd_ycc_extrgb_convert_ps2mmi,
                  EXT_RGB_PIXELSIZE,
                  EXT_RGB_RED, EXT_RGB_GREEN, EXT_RGB_BLUE, -1)
PS2_YCC_CONVERTER(jsimd_ycc_extbgr_convert_ps2mmi,
                  EXT_BGR_PIXELSIZE,
                  EXT_BGR_RED, EXT_BGR_GREEN, EXT_BGR_BLUE, -1)
PS2_YCC_CONVERTER(jsimd_ycc_extrgbx_convert_ps2mmi,
                  EXT_RGBX_PIXELSIZE,
                  EXT_RGBX_RED, EXT_RGBX_GREEN, EXT_RGBX_BLUE, 3)
PS2_YCC_CONVERTER(jsimd_ycc_extbgrx_convert_ps2mmi,
                  EXT_BGRX_PIXELSIZE,
                  EXT_BGRX_RED, EXT_BGRX_GREEN, EXT_BGRX_BLUE, 3)
PS2_YCC_CONVERTER(jsimd_ycc_extxbgr_convert_ps2mmi,
                  EXT_XBGR_PIXELSIZE,
                  EXT_XBGR_RED, EXT_XBGR_GREEN, EXT_XBGR_BLUE, 0)
PS2_YCC_CONVERTER(jsimd_ycc_extxrgb_convert_ps2mmi,
                  EXT_XRGB_PIXELSIZE,
                  EXT_XRGB_RED, EXT_XRGB_GREEN, EXT_XRGB_BLUE, 0)
