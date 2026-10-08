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

/* Vertical 3:1 filter in eight 16-bit lanes.  Result <= 1020. */
static void
vertical8_mmi(const JSAMPLE *near, const JSAMPLE *far,
              unsigned short sums[8])
{
  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "ldl $8, 7(%0)\n\t"
    "ldr $8, 0(%0)\n\t"
    "ldl $9, 7(%1)\n\t"
    "ldr $9, 0(%1)\n\t"
    "pextlb $8, $0, $8\n\t"
    "pextlb $9, $0, $9\n\t"
    "psllh $10, $8, 1\n\t"
    "paddh $10, $10, $8\n\t"
    "paddh $10, $10, $9\n\t"
    "sq $10, 0(%2)\n\t"
    ".set pop\n\t"
    :
    : "r" (near), "r" (far), "r" (sums)
    : "$8", "$9", "$10", "memory");
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
      for (; i + 8 < width; i += 8)
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

/* Two-dimensional fancy filter.  The vertical 3:1 filtering is done by
 * eight-wide MMI, followed by scalar horizontal filtering with the exact
 * IJG biases.  Near/far context rows are guaranteed by the JPEG upsampler
 * when fancy h2v2 mode is selected.
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
      unsigned int previous = 0;

      while (width - i >= 8) {
        unsigned short vals[8] __attribute__((aligned(16)));
        int k;
        vertical8_mmi(near + i, far + i, vals);
        for (k = 0; k < 8; k++) {
          JDIMENSION index = i + (JDIMENSION)k;
          unsigned int current = vals[k];
          unsigned int left = index == 0 ? current : previous;
          unsigned int right = index + 1 == width ? current :
            k < 7 ? vals[k + 1] :
            (unsigned int)(3 * near[index + 1] + far[index + 1]);
          dst[2 * index] = (JSAMPLE)((3 * current + left + 8) >> 4);
          dst[2 * index + 1] = (JSAMPLE)((3 * current + right + 7) >> 4);
          previous = current;
        }
        i += 8;
      }
      for (; i < width; i++) {
        unsigned int current = 3 * near[i] + far[i];
        unsigned int left = i == 0 ? current : previous;
        unsigned int right = i + 1 < width ?
          (unsigned int)(3 * near[i + 1] + far[i + 1]) : current;
        dst[2 * i] = (JSAMPLE)((3 * current + left + 8) >> 4);
        dst[2 * i + 1] = (JSAMPLE)((3 * current + right + 7) >> 4);
        previous = current;
      }
    }
  }
}
