/* Portable test of unsigned-reciprocal correction after R5900 PMULTH.
 * This is arithmetic validation, not an R5900 instruction test.
 *
 * cc -O2 -std=c99 -Wall -Wextra -Werror -fsanitize=undefined \
 *   simd/ps2/test-quantize-host.c -o /tmp/ps2-quant-math
 * /tmp/ps2-quant-math
 *
 * SPDX-License-Identifier: Zlib
 */
#include <stdint.h>
#include <stdio.h>

static uint32_t state = 0x712301feu;

static uint32_t
next_rand(void)
{
  state = 1664525u * state + 1013904223u;
  return state;
}

static uint32_t
mmi_unsigned_product(unsigned magnitude, uint16_t reciprocal)
{
  int32_t raw = (int32_t)(int16_t)magnitude *
                (int32_t)(int16_t)reciprocal;
  uint32_t corrected = (uint32_t)raw;
  if (reciprocal & 0x8000u)
    corrected += (uint32_t)magnitude << 16;
  return corrected;
}

static void
get_reciprocal(unsigned divisor, uint16_t *reciprocal,
               uint16_t *correction, int *shift)
{
  unsigned b = 0, r, c;
  uint32_t fq, fr;
  if (divisor <= 1) {
    *reciprocal = 1;
    *correction = 0;
    *shift = -16;
    return;
  }
  while ((1u << (b + 1)) <= divisor)
    b++;
  r = 16 + b;
  fq = ((uint32_t)1 << r) / divisor;
  fr = ((uint32_t)1 << r) % divisor;
  c = divisor / 2;
  if (!fr) {
    fq >>= 1;
    r--;
  } else if (fr <= divisor / 2) {
    c++;
  } else {
    fq++;
  }
  *reciprocal = (uint16_t)fq;
  *correction = (uint16_t)c;
  *shift = (int)r - 16;
}

int
main(void)
{
  static const unsigned magnitudes[] = {
    0, 1, 2, 7, 127, 128, 32766, 32767
  };
  uint32_t cases = 0;
  unsigned recip, i, q, round;
  for (recip = 0; recip <= 65535u; recip++) {
    for (i = 0; i < sizeof(magnitudes) / sizeof(magnitudes[0]); i++) {
      unsigned magnitude = magnitudes[i];
      uint32_t expected = (uint32_t)magnitude * recip;
      if (mmi_unsigned_product(magnitude, (uint16_t)recip) != expected) {
        printf("FAIL PMULTH unsigned correction rec=%u mag=%u\n",
               recip, magnitude);
        return 1;
      }
      cases++;
    }
  }
  for (round = 0; round < 1000000u; round++) {
    unsigned magnitude = next_rand() & 32767u;
    uint16_t reciprocal = (uint16_t)next_rand();
    uint32_t expected = magnitude * (uint32_t)reciprocal;
    if (mmi_unsigned_product(magnitude, reciprocal) != expected) {
      puts("FAIL random PMULTH unsigned correction");
      return 1;
    }
    cases++;
  }

  for (q = 1; q <= 255; q++) {
    uint16_t reciprocal, correction;
    int shift;
    get_reciprocal(q, &reciprocal, &correction, &shift);
    for (round = 0; round < 32768u; round += 117) {
      unsigned magnitude = round;
      uint32_t product = (magnitude + correction) * reciprocal;
      int bits = shift + 16;
      if (bits < 0 || bits >= 32) {
        puts("FAIL shift range");
        return 1;
      }
      if (magnitude + correction <= 32767u) {
        uint32_t emulated = mmi_unsigned_product(
          magnitude + correction, reciprocal
        );
        if ((emulated >> bits) != (product >> bits)) {
          printf("FAIL q=%u mag=%u shift=%d\n", q, magnitude, shift);
          return 1;
        }
      }
      cases++;
    }
  }
  printf("PS2 MMI reciprocal correction host: PASS (%lu comparisons)\n",
         (unsigned long)cases);
  return 0;
}
