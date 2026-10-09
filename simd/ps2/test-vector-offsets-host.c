/*
 * Host mathematical model of the R5900 PMULTH + PMADDH chroma
 * reconstruction used by the experimental eight-pixel merged converter.
 *
 * This tests color coefficients, lane layout, signed rounding, byte clipping
 * and both alpha positions.  The real R5900 instruction sequence still
 * requires the PS2 primitive and full merged test suite.
 *
 * cc -O2 -std=c99 -Wall -Wextra -Werror -fsanitize=undefined \
 *   simd/ps2/test-vector-offsets-host.c -o /tmp/test-vector-offsets-host
 * /tmp/test-vector-offsets-host
 *
 * SPDX-License-Identifier: Zlib
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
  const char *name;
  int r, g, b, a;
} layout_t;

static const layout_t layouts[4] = {
  { "RGBX", 0, 1, 2, 3 },
  { "BGRX", 2, 1, 0, 3 },
  { "XBGR", 3, 2, 1, 0 },
  { "XRGB", 1, 2, 3, 0 }
};

static uint8_t
clip(int x)
{
  return (uint8_t)(x < 0 ? 0 : x > 255 ? 255 : x);
}

/* Emulate the signed 32-bit PMADDH lanes, followed by PADDW, PSRAW(16),
 * PPACH and the two doubleword-copy instructions.  Source bytes are in
 * chroma order, and the 32 signed halfword output lanes are pixel order.
 */
static void
vector_offsets_model(const layout_t *l, const uint8_t cb[4],
                     const uint8_t cr[4], int16_t output[32])
{
  int16_t main_coeff[4] = { 0, 0, 0, 0 };
  int16_t green_coeff[4] = { 0, 0, 0, 0 };
  int i, channel;
  main_coeff[l->r] = 26345;
  main_coeff[l->g] = -22554;
  main_coeff[l->b] = -14942;
  green_coeff[l->g] = 18734;

  for (i = 0; i < 4; i++) {
    int c = (int)cb[i] - 128;
    int d = (int)cr[i] - 128;
    int16_t primary[4] = { 0, 0, 0, 0 };
    int16_t extra[4] = { 0, 0, 0, 0 };
    int32_t bias[4] = { 0, 0, 0, 0 };
    int16_t color[4];

    primary[l->r] = (int16_t)d;
    primary[l->g] = (int16_t)c;
    primary[l->b] = (int16_t)c;
    extra[l->g] = (int16_t)d;
    bias[l->r] = d * 65536 + 32768;
    bias[l->g] = -d * 65536 + 32768;
    bias[l->b] = 2 * c * 65536 + 32768;
    bias[l->a] = 255 * 65536;

    for (channel = 0; channel < 4; channel++) {
      int32_t acc = (int32_t)primary[channel] * main_coeff[channel];
      acc += (int32_t)extra[channel] * green_coeff[channel];
      acc += bias[channel];
      color[channel] = (int16_t)(acc >> 16);
    }
    /* PCPYLD/PCPYUD copy the 64-bit chroma RGBA group twice. */
    memcpy(output + 8 * i, color, 4 * sizeof(int16_t));
    memcpy(output + 8 * i + 4, color, 4 * sizeof(int16_t));
  }
}

static int
verify(const layout_t *l, const uint8_t cb[4], const uint8_t cr[4],
       unsigned iteration)
{
  int16_t offsets[32];
  int pixel, channel;

  vector_offsets_model(l, cb, cr, offsets);

  for (pixel = 0; pixel < 8; pixel++) {
    int ch = pixel / 2;
    int c = (int)cb[ch] - 128;
    int d = (int)cr[ch] - 128;
    int y = (int)((iteration + 29u * (unsigned)pixel +
                   (unsigned)(pixel * 17)) & 255u);
    int ref[4] = { 0, 0, 0, 0 };
    int r = (91881 * d + 32768) >> 16;
    int g = (-22554 * c - 46802 * d + 32768) >> 16;
    int b = (116130 * c + 32768) >> 16;
    ref[l->r] = r;
    ref[l->g] = g;
    ref[l->b] = b;
    ref[l->a] = 255;

    for (channel = 0; channel < 4; channel++) {
      int stored = offsets[4 * pixel + channel];
      int expected = ref[channel];
      int actual_pixel;
      int ref_pixel;

      if (stored != expected) {
        printf("FAIL offset %s iter=%u pixel=%d channel=%d: %d != %d\n",
               l->name, iteration, pixel, channel, stored, expected);
        return 1;
      }
      actual_pixel = clip(stored + (channel == l->a ? 0 : y));
      ref_pixel = clip(expected + (channel == l->a ? 0 : y));
      if (actual_pixel != ref_pixel) {
        printf("FAIL pixel %s iter=%u pixel=%d channel=%d\n",
               l->name, iteration, pixel, channel);
        return 1;
      }
    }
  }
  return 0;
}

int
main(void)
{
  unsigned cb0, cr0, li;
  unsigned tests = 0;
  uint8_t cb[4], cr[4];
  int i;

  for (li = 0; li < 4; li++) {
    for (cb0 = 0; cb0 < 256; cb0++) {
      for (cr0 = 0; cr0 < 256; cr0++) {
        for (i = 0; i < 4; i++) {
          cb[i] = (uint8_t)((cb0 + (unsigned)(i * 37)) & 255u);
          cr[i] = (uint8_t)((cr0 + (unsigned)(i * 53)) & 255u);
        }
        if (verify(&layouts[li], cb, cr, cb0 * 256 + cr0))
          return 1;
        tests++;
      }
    }
  }
  printf("PS2 merged vector offsets (host): PASS (%u batches, %u pixels)\n",
         tests, tests * 8u);
  return 0;
}
