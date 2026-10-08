/*
 * Fancy (triangle-filter) chroma upsampling on the PlayStation 2 EE.
 *
 * Eight horizontally adjacent samples are processed in 128-bit MMI registers.
 * R5900 MMI and Loongson MMI are different instruction sets.
 *
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <stdint.h>

/* Exact IJG/libjpeg-turbo rounding biases, one value per halfword. */
static const unsigned short bias1[8] __attribute__((aligned(16))) =
  { 1, 1, 1, 1, 1, 1, 1, 1 };
static const unsigned short bias2[8] __attribute__((aligned(16))) =
  { 2, 2, 2, 2, 2, 2, 2, 2 };

/*
 * Consume eight *interior* samples and emit sixteen interleaved output
 * samples.  All source loads are unaligned-safe (LDL/LDR).  The destination
 * must be 16-byte aligned; the caller handles prefix and tail in scalar C.
 *
 * dst[2*i]   = (3*src[i] + src[i-1] + 1) >> 2
 * dst[2*i+1] = (3*src[i] + src[i+1] + 2) >> 2
 */
static void
fancy8_mmi(const JSAMPLE *center, JSAMPLE *dst)
{
  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "ldl $8, 7(%0)\n\t"
    "ldr $8, 0(%0)\n\t"
    "ldl $9, 7(%1)\n\t"
    "ldr $9, 0(%1)\n\t"
    "ldl $10, 7(%2)\n\t"
    "ldr $10, 0(%2)\n\t"
    "pextlb $8, $0, $8\n\t"
    "pextlb $9, $0, $9\n\t"
    "pextlb $10, $0, $10\n\t"
    "psllh $11, $8, 1\n\t"
    "paddh $11, $11, $8\n\t"
    "paddh $9, $11, $9\n\t"
    "paddh $10, $11, $10\n\t"
    "lq $12, 0(%3)\n\t"
    "lq $13, 0(%4)\n\t"
    "paddh $9, $9, $12\n\t"
    "paddh $10, $10, $13\n\t"
    "psrlh $9, $9, 2\n\t"
    "psrlh $10, $10, 2\n\t"
    "ppacb $9, $0, $9\n\t"
    "ppacb $10, $0, $10\n\t"
    "pextlb $9, $10, $9\n\t"
    "sq $9, 0(%5)\n\t"
    ".set pop\n\t"
    :
    : "r" (center), "r" (center - 1), "r" (center + 1),
      "r" (bias1), "r" (bias2), "r" (dst)
    : "$8", "$9", "$10", "$11", "$12", "$13", "memory");
}

/*
 * Fused 4:2:0 fancy upsampling for eight interior chroma samples.
 * Compute the vertical 3:1 filter for left/center/right neighbors, then
 * horizontally interpolate and pack 16 bytes without a temporary array.
 *
 * Each LDL/LDR pair loads exactly eight bytes from an arbitrary alignment.
 * The caller guarantees center[-1] and center[8] exist and that dst is
 * 16-byte aligned; endpoints are handled in scalar C.
 *
 * left  = 3 * near[-1] + far[-1]
 * mid   = 3 * near[ 0] + far[ 0]
 * right = 3 * near[+1] + far[+1]
 * out[2*i]   = (3*mid + left  + 8) >> 4
 * out[2*i+1] = (3*mid + right + 7) >> 4
 */
static const unsigned short bias8[8] __attribute__((aligned(16))) =
  { 8, 8, 8, 8, 8, 8, 8, 8 };
static const unsigned short bias7[8] __attribute__((aligned(16))) =
  { 7, 7, 7, 7, 7, 7, 7, 7 };

static void
fancy8_h2v2_mmi(const JSAMPLE *near, const JSAMPLE *far, JSAMPLE *dst)
{
  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    /* 8 samples starting one byte before the center. */
    "ldl $8, 6(%0)\n\t"
    "ldr $8, -1(%0)\n\t"
    "ldl $9, 6(%1)\n\t"
    "ldr $9, -1(%1)\n\t"
    "pextlb $8, $0, $8\n\t"
    "pextlb $9, $0, $9\n\t"
    "psllh $10, $8, 1\n\t"
    "paddh $8, $8, $10\n\t"
    "paddh $8, $8, $9\n\t"
    /* Eight vertical sums at the current chroma position. */
    "ldl $9, 7(%0)\n\t"
    "ldr $9, 0(%0)\n\t"
    "ldl $10, 7(%1)\n\t"
    "ldr $10, 0(%1)\n\t"
    "pextlb $9, $0, $9\n\t"
    "pextlb $10, $0, $10\n\t"
    "psllh $11, $9, 1\n\t"
    "paddh $9, $9, $11\n\t"
    "paddh $9, $9, $10\n\t"
    /* Eight vertical sums starting one byte after the center. */
    "ldl $10, 8(%0)\n\t"
    "ldr $10, 1(%0)\n\t"
    "ldl $11, 8(%1)\n\t"
    "ldr $11, 1(%1)\n\t"
    "pextlb $10, $0, $10\n\t"
    "pextlb $11, $0, $11\n\t"
    "psllh $12, $10, 1\n\t"
    "paddh $10, $10, $12\n\t"
    "paddh $10, $10, $11\n\t"
    /* Horizontal 3:1 filter, keeping the IJG rounding asymmetry. */
    "psllh $11, $9, 1\n\t"
    "paddh $11, $11, $9\n\t"
    "paddh $12, $11, $8\n\t"
    "paddh $13, $11, $10\n\t"
    "lq $14, 0(%3)\n\t"
    "lq $15, 0(%4)\n\t"
    "paddh $12, $12, $14\n\t"
    "paddh $13, $13, $15\n\t"
    "psrlh $12, $12, 4\n\t"
    "psrlh $13, $13, 4\n\t"
    "ppacb $12, $0, $12\n\t"
    "ppacb $13, $0, $13\n\t"
    "pextlb $12, $13, $12\n\t"
    "sq $12, 0(%2)\n\t"
    ".set pop\n\t"
    :
    : "r" (near), "r" (far), "r" (dst), "r" (bias8), "r" (bias7)
    : "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15", "memory");
}

HIDDEN void
jsimd_h2v1_fancy_upsample_ps2mmi(int max_v_samp_factor,
                                 JDIMENSION downsampled_width,
                                 JSAMPARRAY input_data,
                                 JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output = *output_data_ptr;
  int row;

  for (row = 0; row < max_v_samp_factor; row++) {
    const JSAMPLE *src = input_data[row];
    JSAMPLE *dst = output[row];
    JDIMENSION i = 0;
    JDIMENSION width = downsampled_width;

    /* Use the fast path only with an aligned store address.  Starting at a
     * multiple of eight also ensures the left neighbor exists. */
    if (width > 8 && (((uintptr_t)dst & 15) == 0)) {
      for (; i < 8; i++) {
        int x = src[i];
        int left = i ? src[i - 1] : x;
        int right = i + 1 < width ? src[i + 1] : x;
        dst[2 * i] = (JSAMPLE)((3 * x + left + 1) >> 2);
        dst[2 * i + 1] = (JSAMPLE)((3 * x + right + 2) >> 2);
      }
      for (; width - i > 8; i += 8)
        fancy8_mmi(src + i, dst + 2 * i);
    }

    for (; i < width; i++) {
      int x = src[i];
      int left = i ? src[i - 1] : x;
      int right = i + 1 < width ? src[i + 1] : x;
      dst[2 * i] = (JSAMPLE)((3 * x + left + 1) >> 2);
      dst[2 * i + 1] = (JSAMPLE)((3 * x + right + 2) >> 2);
    }
  }
}

/* Two-dimensional fancy filter.  Interior pixels use a fused
 * vertical/horizontal 3:1 filter and exact IJG rounding in eight MMI lanes.
 * The first and last samples, plus rows without aligned destinations, use
 * scalar C.  Near/far context rows are provided by the JPEG upsampler.
 */
HIDDEN void
jsimd_h2v2_fancy_upsample_ps2mmi(int max_v_samp_factor,
                                 JDIMENSION downsampled_width,
                                 JSAMPARRAY input_data,
                                 JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output = *output_data_ptr;
  int inrow, outrow, v;

  for (inrow = outrow = 0; outrow + 1 < max_v_samp_factor; inrow++) {
    for (v = 0; v < 2; v++, outrow++) {
      const JSAMPLE *near = input_data[inrow];
      const JSAMPLE *far = input_data[inrow + (v ? 1 : -1)];
      JSAMPLE *dst = output[outrow];
      JDIMENSION width = downsampled_width, i = 0;

      /* Aligned output allows sixteen bytes per MMI store.  Skip the
       * first eight positions so every SIMD lane has a left neighbor;
       * the strict >8 condition guarantees a right neighbor as well. */
      if (width > 16 && (((uintptr_t)dst & 15) == 0)) {
        for (; i < 8; i++) {
          unsigned int current = 3 * near[i] + far[i];
          unsigned int left = i ? 3 * near[i - 1] + far[i - 1] : current;
          unsigned int right = 3 * near[i + 1] + far[i + 1];
          dst[2 * i] = (JSAMPLE)((3 * current + left + 8) >> 4);
          dst[2 * i + 1] = (JSAMPLE)((3 * current + right + 7) >> 4);
        }
        for (; width - i > 8; i += 8)
          fancy8_h2v2_mmi(near + i, far + i, dst + 2 * i);
      }
      /* Scalar tail, including the boundary sample and unaligned rows.
       * Recompute the left neighbor so the MMI/scalar boundary is exact. */
      for (; i < width; i++) {
        unsigned int current = 3 * near[i] + far[i];
        unsigned int left = i ? 3 * near[i - 1] + far[i - 1] : current;
        unsigned int right = i + 1 < width ?
          (unsigned int)(3 * near[i + 1] + far[i + 1]) : current;
        dst[2 * i] = (JSAMPLE)((3 * current + left + 8) >> 4);
        dst[2 * i + 1] = (JSAMPLE)((3 * current + right + 7) >> 4);
      }
    }
  }
}
