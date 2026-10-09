/*
 * Host-side simulation of the EE PEXTLB/PEXTUB + PADDH + PPACB luma
 * packing data layout.  It does not execute MMI instructions.  Run the
 * on-device primitive and merged JPEG tests before claiming correctness.
 *
 * cc -O2 -std=c99 -Wall -Wextra -Werror -fsanitize=undefined \
 *   simd/ps2/test-add-pack-host.c -o /tmp/test-add-pack-host
 * /tmp/test-add-pack-host
 *
 * SPDX-License-Identifier: Zlib
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t seed = 0x4a67616du;

static uint32_t
next_random(void)
{
  seed = seed * 1664525u + 1013904223u;
  return seed;
}

static uint8_t
clip_byte(int val)
{
  return (uint8_t)(val < 0 ? 0 : val > 255 ? 255 : val);
}

static int
check_four_pixels(int alpha, unsigned iteration)
{
  int16_t offsets[16];
  uint8_t y[4], expected[16], actual[16];
  uint32_t words[4];
  uint32_t mask = alpha == 0 ? 0xffffff00u : 0x00ffffffu;
  unsigned i;

  for (i = 0; i < 4; i++) {
    uint32_t v;
    y[i] = (uint8_t)next_random();
    v = (uint32_t)y[i];
    v |= v << 8;
    v |= v << 16;
    words[i] = v & mask;
  }
  for (i = 0; i < 16; i++) {
    int value = (int)(next_random() % 1601u) - 800;
    int luma = (i % 4 == (unsigned)alpha) ? 0 : y[i / 4];
    offsets[i] = (int16_t)((i % 4 == (unsigned)alpha) ? 255 : value);
    expected[i] = clip_byte((int)offsets[i] + luma);
  }
  /* Model the 16 output halfwords from the two PEXT instructions:
   * the low byte of each Y word enters a halfword's low byte, while
   * its high byte is zero.  PADDH adds offsets, then clamp/pack.
   */
  for (i = 0; i < 16; i++) {
    uint32_t v = words[i / 4];
    int expanded = (int)((v >> (8 * (i % 4))) & 255u);
    actual[i] = clip_byte((int)offsets[i] + expanded);
  }
  if (memcmp(expected, actual, sizeof(actual))) {
    printf("FAIL add/pack alpha=%d iteration=%u\n", alpha, iteration);
    return 1;
  }
  for (i = 0; i < 4; i++) {
    if (((words[i] >> (8 * alpha)) & 255u) != 0) {
      printf("FAIL alpha mask alpha=%d iteration=%u\n", alpha, iteration);
      return 1;
    }
  }
  return 0;
}

int
main(void)
{
  unsigned iteration;
  int alpha;
  for (alpha = 0; alpha <= 3; alpha += 3)
    for (iteration = 0; iteration < 100000; iteration++)
      if (check_four_pixels(alpha, iteration))
        return 1;

  puts("PS2 fused add/pack host model: PASS (200000 vectors)");
  return 0;
}
