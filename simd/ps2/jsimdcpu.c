/*
 * PlayStation 2 Emotion Engine SIMD feature detection.
 *
 * The PS2 EE is a fixed hardware target.  Do not confuse its 128-bit
 * R5900 MMI extension with the unrelated 64-bit Loongson MMI.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  Permission is granted to use, modify, and redistribute it.
 */

#include "../jsimdint.h"

HIDDEN unsigned int
jpeg_simd_cpu_support(void)
{
  /* The PS2 EE toolchain and the compile test in CMake have already
   * established the presence of R5900 MMI instructions. */
  return JSIMD_PS2_MMI;
}
