/*
 * Plain 2:1 downsampling for the PlayStation 2 Emotion Engine (R5900).
 *
 * 128-bit MMI handles 16 output samples per iteration.  IJG alternating
 * rounding biases are preserved.  16-byte aligned loads/stores are used
 * only when both input rows and the destination row are aligned.
 *
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <stdint.h>

static const unsigned short low_byte_mask[8] __attribute__((aligned(16))) =
  { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
static const unsigned short bias_h1[8] __attribute__((aligned(16))) =
  { 0, 1, 0, 1, 0, 1, 0, 1 };
static const unsigned short bias_h2[8] __attribute__((aligned(16))) =
  { 1, 2, 1, 2, 1, 2, 1, 2 };

/* 32 input bytes -> 16 output bytes.  Source and destination aligned. */
static void
downsample16_h1(const JSAMPLE *input, JSAMPLE *output)
{
  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $8, 0(%0)\n\t"
    "lq $9, 16(%0)\n\t"
    "lq $10, 0(%2)\n\t"
    "lq $13, 0(%3)\n\t"
    "pand $11, $8, $10\n\t"
    "pand $12, $9, $10\n\t"
    "psrlh $8, $8, 8\n\t"
    "psrlh $9, $9, 8\n\t"
    "paddh $11, $11, $8\n\t"
    "paddh $12, $12, $9\n\t"
    "paddh $11, $11, $13\n\t"
    "paddh $12, $12, $13\n\t"
    "psrlh $11, $11, 1\n\t"
    "psrlh $12, $12, 1\n\t"
    "ppacb $11, $12, $11\n\t"
    "sq $11, 0(%1)\n\t"
    ".set pop\n\t"
    :
    : "r" (input), "r" (output), "r" (low_byte_mask), "r" (bias_h1)
    : "$8", "$9", "$10", "$11", "$12", "$13", "memory");
}

/* 2 x 32 input bytes -> 16 output bytes (4:2:0). */
static void
downsample16_h2(const JSAMPLE *in0, const JSAMPLE *in1, JSAMPLE *output)
{
  __asm__ volatile(
    ".set push\n\t"
    ".set noreorder\n\t"
    "lq $8, 0(%0)\n\t"
    "lq $9, 16(%0)\n\t"
    "lq $10, 0(%1)\n\t"
    "lq $11, 16(%1)\n\t"
    "paddh $8, $8, $10\n\t"
    "paddh $9, $9, $11\n\t"
    "lq $12, 0(%3)\n\t"
    "lq $13, 0(%4)\n\t"
    "pand $10, $8, $12\n\t"
    "pand $11, $9, $12\n\t"
    "psrlh $8, $8, 8\n\t"
    "psrlh $9, $9, 8\n\t"
    "paddh $10, $10, $8\n\t"
    "paddh $11, $11, $9\n\t"
    "paddh $10, $10, $13\n\t"
    "paddh $11, $11, $13\n\t"
    "psrlh $10, $10, 2\n\t"
    "psrlh $11, $11, 2\n\t"
    "ppacb $10, $11, $10\n\t"
    "sq $10, 0(%2)\n\t"
    ".set pop\n\t"
    :
    : "r" (in0), "r" (in1), "r" (output),
      "r" (low_byte_mask), "r" (bias_h2)
    : "$8", "$9", "$10", "$11", "$12", "$13", "memory");
}

/* The IJG scalar downsampler replicates the last input sample to the
 * padded MCU width *before* averaging.  Preserve this behavior, including
 * padded columns in the output that are subsequently encoded. */
static void
pad_right(JSAMPARRAY input, int rows, JDIMENSION width,
          JDIMENSION padded_width)
{
  int row;
  if (width >= padded_width)
    return;
  for (row = 0; row < rows; row++) {
    JSAMPLE value = input[row][width - 1];
    JDIMENSION col;
    for (col = width; col < padded_width; col++)
      input[row][col] = value;
  }
}

HIDDEN void
jsimd_h2v1_downsample_ps2mmi(JDIMENSION image_width, int max_v_samp_factor,
                             JDIMENSION v_samp_factor,
                             JDIMENSION width_in_blocks, JSAMPARRAY input_data,
                             JSAMPARRAY output_data)
{
  JDIMENSION cols = width_in_blocks * DCTSIZE;
  int row;
  pad_right(input_data, max_v_samp_factor, image_width, cols * 2);

  for (row = 0; row < (int)v_samp_factor; row++) {
    const JSAMPLE *src = input_data[row];
    JSAMPLE *dst = output_data[row];
    JDIMENSION col = 0;

    if ((((uintptr_t)src | (uintptr_t)dst) & 15) == 0) {
      for (; cols - col >= 16; col += 16)
        downsample16_h1(src + 2 * col, dst + col);
    }
    for (; col < cols; col++)
      dst[col] = (JSAMPLE)((src[2 * col] + src[2 * col + 1] +
                             (col & 1)) >> 1);
  }
}

HIDDEN void
jsimd_h2v2_downsample_ps2mmi(JDIMENSION image_width, int max_v_samp_factor,
                             JDIMENSION v_samp_factor,
                             JDIMENSION width_in_blocks, JSAMPARRAY input_data,
                             JSAMPARRAY output_data)
{
  JDIMENSION cols = width_in_blocks * DCTSIZE;
  int row;
  pad_right(input_data, max_v_samp_factor, image_width, cols * 2);

  for (row = 0; row < (int)v_samp_factor; row++) {
    const JSAMPLE *src0 = input_data[2 * row];
    const JSAMPLE *src1 = input_data[2 * row + 1];
    JSAMPLE *dst = output_data[row];
    JDIMENSION col = 0;

    if ((((uintptr_t)src0 | (uintptr_t)src1 |
           (uintptr_t)dst) & 15) == 0) {
      for (; cols - col >= 16; col += 16)
        downsample16_h2(src0 + 2 * col, src1 + 2 * col, dst + col);
    }
    for (; col < cols; col++)
      dst[col] = (JSAMPLE)((src0[2 * col] + src0[2 * col + 1] +
                             src1[2 * col] + src1[2 * col + 1] +
                             (1 + (col & 1))) >> 2);
  }
}
