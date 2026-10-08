/*
 * Standalone R5900 MMI instruction-semantic regression tests.
 *
 * These tests are intentionally independent of libjpeg's numerical code,
 * so a failing IDCT, chroma, or color test can be narrowed to a specific
 * EE MMI operation.  Run on an actual PlayStation 2 or EE emulator.
 *
 * SPDX-License-Identifier: Zlib
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ALIGN16 __attribute__((aligned(16)))
#define ITERATIONS 1024

static uint32_t rand_state = 0x5a71f00dU;

static uint32_t
next_random(void)
{
  rand_state = rand_state * 1664525U + 1013904223U;
  return rand_state;
}

/* Test PEXTLB/PEXTUB in their actual 128-bit R5900 register layout. */
static void
mmi_duplicate16(const uint8_t *src, uint8_t *dst)
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

/* PMULTH distributes its products across LO and HI; restore lane order. */
static void
mmi_multiply8(const int16_t *a, const int16_t *b, int32_t *out)
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
    "pcpyud $14, $12, $11\n\t"
    "sq $13, 0(%2)\n\t"
    "sq $14, 16(%2)\n\t"
    ".set pop\n\t"
    :
    : "r" (a), "r" (b), "r" (out)
    : "$8", "$9", "$10", "$11", "$12", "$13", "$14", "memory");
}

static const int32_t round11[4] ALIGN16 = { 1024, 1024, 1024, 1024 };
static const int32_t round18[4] ALIGN16 = { 131072, 131072, 131072, 131072 };

/* Final LL&M butterfly and IJG DESCALE with four 32-bit MMI lanes. */
static void
mmi_butterfly4(const int32_t *a, const int32_t *b,
               int32_t *plus, int32_t *minus, int shift)
{
  const int32_t *bias = (shift == 11) ? round11 : round18;

  if (shift == 11) {
    __asm__ volatile(
      ".set push\n\t"
      ".set noreorder\n\t"
      "lq $8, 0(%0)\n\t"
      "lq $9, 0(%1)\n\t"
      "lq $10, 0(%2)\n\t"
      "paddw $11, $8, $9\n\t"
      "psubw $12, $8, $9\n\t"
      "paddw $11, $11, $10\n\t"
      "paddw $12, $12, $10\n\t"
      "psraw $11, $11, 11\n\t"
      "psraw $12, $12, 11\n\t"
      "sq $11, 0(%3)\n\t"
      "sq $12, 0(%4)\n\t"
      ".set pop\n\t"
      :
      : "r" (a), "r" (b), "r" (bias), "r" (plus), "r" (minus)
      : "$8", "$9", "$10", "$11", "$12", "memory");
  } else {
    __asm__ volatile(
      ".set push\n\t"
      ".set noreorder\n\t"
      "lq $8, 0(%0)\n\t"
      "lq $9, 0(%1)\n\t"
      "lq $10, 0(%2)\n\t"
      "paddw $11, $8, $9\n\t"
      "psubw $12, $8, $9\n\t"
      "paddw $11, $11, $10\n\t"
      "paddw $12, $12, $10\n\t"
      "psraw $11, $11, 18\n\t"
      "psraw $12, $12, 18\n\t"
      "sq $11, 0(%3)\n\t"
      "sq $12, 0(%4)\n\t"
      ".set pop\n\t"
      :
      : "r" (a), "r" (b), "r" (bias), "r" (plus), "r" (minus)
      : "$8", "$9", "$10", "$11", "$12", "memory");
  }
}

/* Four RGBX pixels: signed halfword clipping followed by PPACB. */
static const int16_t limit255[8] ALIGN16 =
  { 255, 255, 255, 255, 255, 255, 255, 255 };

static void
mmi_clip_pack16(const int16_t *src, uint8_t *dst)
{
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
    : "r" (src), "r" (limit255), "r" (dst)
    : "$8", "$9", "$10", "memory");
}

int
main(void)
{
  uint8_t pixels[16] ALIGN16;
  uint8_t expanded[48] ALIGN16;
  int16_t lhs[8] ALIGN16, rhs[8] ALIGN16;
  int32_t multiplied[12] ALIGN16;
  int32_t a[4] ALIGN16, b[4] ALIGN16;
  int32_t positive[8] ALIGN16, negative[8] ALIGN16;
  int16_t channels[16] ALIGN16;
  uint8_t packed[32] ALIGN16;
  unsigned n, i;
  int shifts[2] = { 11, 18 };

  for (n = 0; n < ITERATIONS; n++) {
    for (i = 0; i < 16; i++)
      pixels[i] = (uint8_t)(i * 17 + next_random());
    memset(expanded, 0xa5, sizeof(expanded));
    mmi_duplicate16(pixels, expanded);

    for (i = 0; i < 16; i++) {
      if (expanded[2 * i] != pixels[i] ||
          expanded[2 * i + 1] != pixels[i]) {
        printf("FAIL pextlb/pextub iter=%u lane=%u\n", n, i);
        return 1;
      }
    }
    for (i = 32; i < sizeof(expanded); i++)
      if (expanded[i] != 0xa5) {
        puts("FAIL PEXT source guard");
        return 1;
      }

    for (i = 0; i < 8; i++) {
      lhs[i] = (int16_t)next_random();
      rhs[i] = (int16_t)next_random();
    }
    if (n % 8 == 0) {
      lhs[0] = -32768;
      rhs[0] = -32768;
      lhs[7] = 32767;
      rhs[7] = -32768;
    }

    for (i = 0; i < 12; i++)
      multiplied[i] = 0x5a5a5a5a;
    mmi_multiply8(lhs, rhs, multiplied);
    for (i = 0; i < 8; i++) {
      int32_t reference = (int32_t)lhs[i] * (int32_t)rhs[i];
      if (multiplied[i] != reference) {
        printf("FAIL PMULTH iter=%u lane=%u: %ld != %ld\n",
               n, i, (long)multiplied[i], (long)reference);
        return 1;
      }
    }
    for (i = 8; i < 12; i++)
      if (multiplied[i] != 0x5a5a5a5a) {
        puts("FAIL PMULTH guard");
        return 1;
      }

    for (i = 0; i < 4; i++) {
      /* Deliberately avoid signed-overflow undefined behavior in C. */
      a[i] = (int32_t)(next_random() % 100000001U) - 50000000;
      b[i] = (int32_t)(next_random() % 100000001U) - 50000000;
    }
    for (i = 0; i < 2; i++) {
      unsigned lane;
      int shift = shifts[i];
      int32_t bias = (int32_t)1 << (shift - 1);

      for (lane = 0; lane < 8; lane++)
        positive[lane] = negative[lane] = 0x5a5a5a5a;
      mmi_butterfly4(a, b, positive, negative, shift);
      for (lane = 0; lane < 4; lane++) {
        int32_t ref_plus = (a[lane] + b[lane] + bias) >> shift;
        int32_t ref_minus = (a[lane] - b[lane] + bias) >> shift;
        if (positive[lane] != ref_plus || negative[lane] != ref_minus) {
          printf("FAIL PSRAW iter=%u shift=%d lane=%u\n",
                 n, shift, lane);
          return 1;
        }
      }
      for (lane = 4; lane < 8; lane++)
        if (positive[lane] != 0x5a5a5a5a ||
            negative[lane] != 0x5a5a5a5a) {
          puts("FAIL butterfly guard");
          return 1;
        }
    }

    for (i = 0; i < 16; i++)
      channels[i] = (int16_t)((int)(next_random() % 1201U) - 500);
    if (n % 16 == 0) {
      channels[0] = -32768;
      channels[1] = 32767;
      channels[15] = 255;
    }
    memset(packed, 0xa5, sizeof(packed));
    mmi_clip_pack16(channels, packed);
    for (i = 0; i < 16; i++) {
      int16_t value = channels[i];
      uint8_t reference =
        (uint8_t)(value < 0 ? 0 : (value > 255 ? 255 : value));
      if (packed[i] != reference) {
        printf("FAIL PMAXH/PMINH/PPACB iter=%u lane=%u: %u != %u\n",
               n, i, (unsigned)packed[i], (unsigned)reference);
        return 1;
      }
    }
    for (i = 16; i < sizeof(packed); i++)
      if (packed[i] != 0xa5) {
        puts("FAIL PPACB guard");
        return 1;
      }
  }

  printf("PS2 MMI primitives: PASS (%u iterations, five kernel checks each)\n",
         (unsigned)ITERATIONS);
  return 0;
}
