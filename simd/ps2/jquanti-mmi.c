/* Experimental EE R5900 MMI integer JPEG quantizer.
 *
 * Unlike Loongson MMI, R5900 PMULTH produces eight full signed 32-bit
 * products in HI/LO.  The reciprocal table is unsigned 16-bit, so its
 * signed PMULTH result must be corrected when bit 15 is set:
 *    unsigned_product = signed_product + (input << 16).
 *
 * This follows the scalar IJG/libjpeg-turbo reciprocal, correction and
 * shift tables exactly, including divisor=1 (shift=-16) and negative
 * coefficients.  An out-of-range 16-bit PMULTH input falls back to C.
 * Test against src/jcdctmgr.c on the EE before enabling dispatch.
 *
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <stdint.h>

static INLINE JCOEF
quantize_one(DCTELEM value, DCTELEM recip,
             DCTELEM correction, DCTELEM shift)
{
  int magnitude = (int)value;
  uint32_t product;
  int bits = (int)shift + 16;
  if (magnitude < 0)
    magnitude = -magnitude;
  product = ((uint32_t)magnitude + (uint16_t)correction) *
            (uint16_t)recip;
  /* Real tables produce 0 <= bits < 32.  A malformed table is unsupported. */
  if (bits < 0 || bits >= 32)
    return (JCOEF)0;
  magnitude = (int)(product >> bits);
  if (value < 0)
    magnitude = -magnitude;
  return (JCOEF)magnitude;
}

HIDDEN void
jsimd_quantize_ps2mmi(JCOEFPTR coef_block, DCTELEM *divisors,
                      DCTELEM *workspace)
{
  unsigned i, lane;
  if (sizeof(DCTELEM) != 2 || sizeof(JCOEF) != 2)
    return;

  for (i = 0; i < DCTSIZE2; i += 8) {
    short magnitudes[8] __attribute__((aligned(16)));
    short reciprocal_copy[8] __attribute__((aligned(16)));
    const short *reciprocal_src = (const short *)(divisors + i);
    int32_t products[8] __attribute__((aligned(16)));
    int valid = 1;

    for (lane = 0; lane < 8; lane++) {
      unsigned n = i + lane;
      int value = (int)workspace[n];
      int mag = value < 0 ? -value : value;
      unsigned corrected = (unsigned)mag + (uint16_t)divisors[n + 64];
      int shift = (int)divisors[n + 192] + 16;
      if (corrected > 32767u || shift < 0 || shift >= 32) {
        valid = 0;
        break;
      }
      magnitudes[lane] = (short)corrected;
    }

    if (!valid) {
      for (lane = 0; lane < 8; lane++)
        coef_block[i + lane] =
          quantize_one(workspace[i + lane], divisors[i + lane],
                       divisors[i + lane + 64],
                       divisors[i + lane + 192]);
      continue;
    }

    /* IJG's reciprocal table is aligned in the ordinary compressor.
     * Read it directly by LQ instead of copying eight halfwords to
     * the stack on every block.  Preserve an aligned temporary for
     * callers that supply a differently aligned divisor table.
     */
    if (((uintptr_t)reciprocal_src & 15u) != 0) {
      for (lane = 0; lane < 8; lane++)
        reciprocal_copy[lane] = (short)divisors[i + lane];
      reciprocal_src = reciprocal_copy;
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
      ".set pop\n\t"
      :
      : "r" (magnitudes), "r" (reciprocal_src), "r" (products)
      : "$8", "$9", "$10", "$11", "$12", "$13", "$14", "memory");

    for (lane = 0; lane < 8; lane++) {
      unsigned n = i + lane;
      uint32_t product = (uint32_t)products[lane];
      int bits = (int)divisors[n + 192] + 16;
      int value;
      if (((uint16_t)divisors[n] & 0x8000u) != 0)
        product += (uint32_t)(uint16_t)magnitudes[lane] << 16;
      value = (int)(product >> bits);
      if (workspace[n] < 0)
        value = -value;
      coef_block[n] = (JCOEF)value;
    }
  }
}
