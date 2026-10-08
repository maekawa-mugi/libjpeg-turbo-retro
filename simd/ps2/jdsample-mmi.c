/*
 * Plain chroma upsampling for the PlayStation 2 Emotion Engine (R5900 MMI).
 *
 * SPDX-License-Identifier: Zlib
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  Permission is granted to use, modify, and redistribute it.
 */

#include "../jsimdint.h"
#include <stdint.h>

/*
 * Expand 16 source bytes to 32 bytes by duplicating each source sample.
 *
 * PEXTLB/PEXTUB operate on all 128 bits of the EE general-purpose registers.
 * Passing the same source in both operands interleaves each byte with itself.
 * LQ and SQ require 16-byte alignment.  The caller checks BOTH pointers
 * before entering this fast path, and uses a bounded scalar tail otherwise.
 *
 * This is specifically PS2 R5900 MMI, not Loongson MMI.
 */
static void
expand16_mmi(const JSAMPLE *src, JSAMPLE *dst)
{
  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $8, 0(%0)\n\t"
    "pextlb $9, $8, $8\n\t"
    "pextub $10, $8, $8\n\t"
    "sq $9, 0(%1)\n\t"
    "sq $10, 16(%1)\n\t"
    ".set pop\n\t"
    :
    : "r" (src), "r" (dst)
    : "$8", "$9", "$10", "memory");
}

/*
 * One horizontal expansion feeds two output rows for h2v2 sampling.
 * Reuse the expanded MMI registers instead of reading the first row back
 * via memcpy.  The caller guarantees 16-byte alignment for all pointers.
 */
static void
expand16_pair_mmi(const JSAMPLE *src, JSAMPLE *dst0, JSAMPLE *dst1)
{
  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $8, 0(%0)\n\t"
    "pextlb $9, $8, $8\n\t"
    "pextub $10, $8, $8\n\t"
    "sq $9, 0(%1)\n\t"
    "sq $10, 16(%1)\n\t"
    "sq $9, 0(%2)\n\t"
    "sq $10, 16(%2)\n\t"
    ".set pop\n\t"
    :
    : "r" (src), "r" (dst0), "r" (dst1)
    : "$8", "$9", "$10", "memory");
}

/* Write exactly width bytes, including odd-width images.  No reads beyond
 * ceil(width / 2) input samples and no writes into JPEG's row padding. */
static void
expand_row(JDIMENSION width, const JSAMPLE *src, JSAMPLE *dst)
{
  JDIMENSION col = 0;
  JDIMENSION src_col = 0;

  if ((((uintptr_t)src | (uintptr_t)dst) & 15) == 0) {
    /* width - col >= 32 implies at least 16 source samples exist. */
    while (width - col >= 32) {
      expand16_mmi(src + src_col, dst + col);
      col += 32;
      src_col += 16;
    }
  }

  for (; col < width; src_col++) {
    JSAMPLE value = src[src_col];
    dst[col++] = value;
    if (col < width)
      dst[col++] = value;
  }
}

HIDDEN void
jsimd_h2v1_upsample_ps2mmi(int max_v_samp_factor, JDIMENSION output_width,
                           JSAMPARRAY input_data,
                           JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  int row;

  for (row = 0; row < max_v_samp_factor; row++)
    expand_row(output_width, input_data[row], output_data[row]);
}

HIDDEN void
jsimd_h2v2_upsample_ps2mmi(int max_v_samp_factor, JDIMENSION output_width,
                           JSAMPARRAY input_data,
                           JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  int inrow, outrow;

  for (inrow = outrow = 0; outrow + 1 < max_v_samp_factor;
       inrow++, outrow += 2) {
    const JSAMPLE *src = input_data[inrow];
    JSAMPLE *dst0 = output_data[outrow];
    JSAMPLE *dst1 = output_data[outrow + 1];
    JDIMENSION col = 0, src_col = 0;

    if ((((uintptr_t)src | (uintptr_t)dst0 | (uintptr_t)dst1) & 15) == 0) {
      for (; output_width - col >= 32; col += 32, src_col += 16)
        expand16_pair_mmi(src + src_col, dst0 + col, dst1 + col);
    }

    /* For unaligned rows and the bounded tail, write both rows directly. */
    for (; col < output_width; src_col++) {
      JSAMPLE value = src[src_col];
      dst0[col] = value;
      dst1[col++] = value;
      if (col < output_width) {
        dst0[col] = value;
        dst1[col++] = value;
      }
    }
  }
}
