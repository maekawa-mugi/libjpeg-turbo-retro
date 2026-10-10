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

/* Aligned eight-lane path: retain magnitudes, unsigned products, shifts
 * and signs in registers.  Reject overflowing corrections and invalid
 * shift tables before touching the output.  In particular -32768 must
 * not pass through saturating PABSH.  XOR/sub gives its exact magnitude,
 * whose sign bit then routes this batch through the C fallback.
 * PSRLVW shifts TWO words (low word of each doubleword), not four. */
static __attribute__((noinline)) int
quantize8_register(JCOEFPTR output, const DCTELEM *table,
                    const DCTELEM *input)
{
  static const short limits[16] __attribute__((aligned(16))) = {
    -16, -16, -16, -16, -16, -16, -16, -16,
    15, 15, 15, 15, 15, 15, 15, 15
  };
  static const int32_t shift_bias[4] __attribute__((aligned(16))) =
    {16, 16, 16, 16};
  int ok;
  __asm__ volatile(
    ".set push\n\t.set noreorder\n\t"
    "lq $8, 0(%3)\n\t"
    "lq $9, 0(%2)\n\t"
    "lq $10, 128(%2)\n\t"
    "lq $11, 384(%2)\n\t"
    "psrah $12, $8, 15\n\t"
    "pxor $13, $8, $12\n\t"
    "psubh $13, $13, $12\n\t"
    "paddh $13, $13, $10\n\t"
    "psrah $14, $13, 15\n\t"
    "psrah $15, $10, 15\n\t"
    "por $14, $14, $15\n\t"
    "lq $16, 0(%4)\n\t"
    "lq $17, 16(%4)\n\t"
    "pcgth $18, $11, $17\n\t"
    "pcgth $19, $16, $11\n\t"
    "por $14, $14, $18\n\t"
    "por $14, $14, $19\n\t"
    "pcpyud $19, $14, $14\n\t"
    "por $14, $14, $19\n\t"
    "bnez $14, 1f\n\t"
    "or %0, $0, $0\n\t"
    "pmulth $14, $13, $9\n\t"
    "pextlh $16, $0, $13\n\t"
    "pextuh $17, $0, $13\n\t"
    "psrah $15, $9, 15\n\t"
    "pextlh $18, $15, $15\n\t"
    "pextuh $19, $15, $15\n\t"
    "psllw $16, $16, 16\n\t"
    "psllw $17, $17, 16\n\t"
    "pand $16, $16, $18\n\t"
    "pand $17, $17, $19\n\t"
    "pmflo $20\n\t"
    "pmfhi $21\n\t"
    "pcpyld $22, $21, $20\n\t"
    "pcpyud $23, $20, $21\n\t"
    "paddw $22, $22, $16\n\t"
    "paddw $23, $23, $17\n\t"
    "psrah $24, $11, 15\n\t"
    "pextlh $16, $24, $11\n\t"
    "pextuh $17, $24, $11\n\t"
    "lq $18, 0(%5)\n\t"
    "paddw $16, $16, $18\n\t"
    "paddw $17, $17, $18\n\t"
    "pextlw $18, $0, $22\n\t"
    "pextlw $19, $0, $16\n\t"
    "psrlvw $20, $18, $19\n\t"
    "pextuw $18, $0, $22\n\t"
    "pextuw $19, $0, $16\n\t"
    "psrlvw $21, $18, $19\n\t"
    "ppacw $22, $21, $20\n\t"
    "pextlw $18, $0, $23\n\t"
    "pextlw $19, $0, $17\n\t"
    "psrlvw $20, $18, $19\n\t"
    "pextuw $18, $0, $23\n\t"
    "pextuw $19, $0, $17\n\t"
    "psrlvw $21, $18, $19\n\t"
    "ppacw $23, $21, $20\n\t"
    "pextlh $16, $12, $12\n\t"
    "pextuh $17, $12, $12\n\t"
    "pxor $22, $22, $16\n\t"
    "pxor $23, $23, $17\n\t"
    "psubw $22, $22, $16\n\t"
    "psubw $23, $23, $17\n\t"
    "ppach $22, $23, $22\n\t"
    "sq $22, 0(%1)\n\t"
    "addiu %0, $0, 1\n\t"
    "1:\n\t.set pop\n\t"
    : "=&r" (ok)
    : "r" (output), "r" (table), "r" (input), "r" (limits),
      "r" (shift_bias)
    : "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15",
      "$16", "$17", "$18", "$19", "$20", "$21", "$22", "$23",
      "$24", "memory");
  return ok;
}

static void
quantize_impl(JCOEFPTR coef_block, DCTELEM *divisors,
                DCTELEM *workspace, int register_path)
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

    if (register_path &&
        (((uintptr_t)coef_block | (uintptr_t)divisors |
          (uintptr_t)workspace) & 15u) == 0) {
      if (!quantize8_register(coef_block + i, divisors + i, workspace + i))
        for (lane = 0; lane < 8; lane++)
          coef_block[i + lane] =
            quantize_one(workspace[i + lane], divisors[i + lane],
                         divisors[i + lane + 64], divisors[i + lane + 192]);
      continue;
    }

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

HIDDEN void
jsimd_quantize_ps2mmi(JCOEFPTR output, DCTELEM *table, DCTELEM *input)
{
  quantize_impl(output, table, input, 1);
}

/* Keep the previous MMI implementation in the same ELF for honest A/B. */
HIDDEN void
jsimd_quantize_legacy_ps2mmi(JCOEFPTR output, DCTELEM *table, DCTELEM *input)
{
  quantize_impl(output, table, input, 0);
}
