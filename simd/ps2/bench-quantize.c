/* Dedicated on-EE integer quantizer correctness and throughput benchmark.
 * Rebuild the same four reciprocal/correction/scale/shift tables as IJG.
 * Includes q=1, powers of 2, signed extremes, sparse and dense FDCT data.
 * SPDX-License-Identifier: Zlib
 */
#include "bench-harness.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define Q_COUNT 128

/* Default IJG luminance quantization matrix in natural DCT order.
 * At quality 75/95 the standard quality scaling is 50/10 percent.
 * The islow FDCT uses 8 times the actual quantization coefficient.
 */
static const unsigned char jpeg_luma_quant[64] = {
  16, 11, 10, 16, 24, 40, 51, 61,
  12, 12, 14, 19, 26, 58, 60, 55,
  14, 13, 16, 24, 40, 57, 69, 56,
  14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77,
  24, 35, 55, 64, 81, 104, 113, 92,
  49, 64, 78, 87, 103, 121, 120, 101,
  72, 92, 95, 98, 112, 100, 103, 99
};
static DCTELEM divisor_mem[4 * 64 + 16] __attribute__((aligned(16)));
static DCTELEM workspace_mem[64 + 16] __attribute__((aligned(16)));
static JCOEF ref_mem[64 + 16] __attribute__((aligned(16)));
static JCOEF mmi_mem[64 + 16] __attribute__((aligned(16)));

typedef struct {
  DCTELEM *divisors, *workspace;
  JCOEFPTR output;
} q_ctx;

static uint32_t
next_q_random(uint32_t *state)
{
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

static void
set_reciprocal(DCTELEM *dtbl, unsigned i, unsigned d)
{
  unsigned b = 0;
  unsigned r;
  uint32_t fq, fr;
  unsigned c;
  if (d <= 1u) {
    dtbl[i] = 1;
    dtbl[i + 64] = 0;
    dtbl[i + 128] = 1;
    dtbl[i + 192] = -16;
    return;
  }
  while ((1u << (b + 1)) <= d)
    b++;
  r = 16 + b;
  fq = ((uint32_t)1 << r) / d;
  fr = ((uint32_t)1 << r) % d;
  c = d / 2u;
  if (fr == 0) {
    fq >>= 1;
    r--;
  } else if (fr <= d / 2u) {
    c++;
  } else {
    fq++;
  }
  dtbl[i] = (DCTELEM)fq;
  dtbl[i + 64] = (DCTELEM)c;
  dtbl[i + 128] = (DCTELEM)(1u << (32 - r));
  dtbl[i + 192] = (DCTELEM)(r - 16);
}

static void
prepare_quant(unsigned profile, unsigned align, unsigned seed, q_ctx *q)
{
  uint32_t state = 0x91c2fe0bu + seed * 97u;
  unsigned i;
  q->divisors = divisor_mem + (align ? 1 : 0);
  q->workspace = workspace_mem + (align ? 1 : 0);
  q->output = mmi_mem + (align ? 1 : 0);
  for (i = 0; i < 64; i++) {
    unsigned d;
    int value;
    state = next_q_random(&state);
    if (profile >= 4) {
      unsigned scale = profile == 4 ? 50u : 10u;
      unsigned quant = ((unsigned)jpeg_luma_quant[i] * scale + 50u) / 100u;
      if (quant == 0)
        quant = 1;
      /* Mirrors the islow compressor's quantval << 3. */
      d = quant * 8u;
    } else {
      /* Retain artificial divisor=1, power-of-two and broad edge tests. */
      d = i % 11u == 0 ? 1u : i % 13u == 0 ? 16u :
          i % 17u == 0 ? 128u : 2u + (state % 254u);
    }
    set_reciprocal(q->divisors, i, d);
    state = next_q_random(&state);
    value = (int)((state >> 8) % 16385u) - 8192;
    if (profile == 0)
      value = (i == 0) ? ((seed & 1u) ? 100 : -100) : 0;
    else if (profile == 1 && (i & 7u) != 0)
      value = 0;
    else if (profile == 3)
      value = (i & 1u) ? -32768 : 32767;
    else if (profile >= 4) {
      /* JPEG-like blocks: stronger DC and mostly small or zero AC. */
      if (i == 0)
        value = (int)((state >> 8) % 4097u) - 2048;
      else if ((state & 3u) != 0)
        value = 0;
      else
        value = (int)((state >> 8) % 1025u) - 512;
    }
    q->workspace[i] = (DCTELEM)value;
  }
  for (i = 64; i < 80; i++) {
    ref_mem[i] = (JCOEF)0x5a5a;
    mmi_mem[i] = (JCOEF)0x5a5a;
  }
}

static void
quantize_reference(void *arg)
{
  q_ctx *q = (q_ctx *)arg;
  unsigned i;
  for (i = 0; i < 64; i++) {
    int value = q->workspace[i];
    int mag = value < 0 ? -value : value;
    int shift = (int)q->divisors[i + 192] + 16;
    uint32_t product =
      ((uint32_t)mag + (uint16_t)q->divisors[i + 64]) *
      (uint16_t)q->divisors[i];
    int result = (int)(product >> shift);
    q->output[i] = (JCOEF)(value < 0 ? -result : result);
  }
}

static void
quantize_mmi(void *arg)
{
  q_ctx *q = (q_ctx *)arg;
  jsimd_quantize_ps2mmi(q->output, q->divisors, q->workspace);
}


static uint32_t
digest_quant(void *context)
{
  q_ctx *q = (q_ctx *)context;
  return ps2_bench_fnv((const JSAMPLE *)q->output,
                       64u * sizeof(JCOEF), 2166136261u);
}

static void
reset_quant(void *context)
{
  q_ctx *q = (q_ctx *)context;
  unsigned i;
  for (i = 0; i < 64; i++)
    q->output[i] = (JCOEF)0x5a5a;
}

int
ps2_bench_run_quantize(void)
{
  const char *names[6] = { "dc_only", "sparse", "dense", "extreme",
                           "jpeg_q75", "jpeg_q95" };
  unsigned p, align, seed, i;
  int failures = 0;
  for (p = 0; p < 6; p++)
    for (align = 0; align < 2; align++)
      for (seed = 0; seed < 256; seed++) {
        q_ctx q, ref;
        prepare_quant(p, align, seed, &q);
        ref = q;
        ref.output = ref_mem + (align ? 1 : 0);
        for (i = 0; i < 64; i++)
          ref.output[i] = q.output[i] = (JCOEF)0x5a5a;
        quantize_reference(&ref);
        quantize_mmi(&q);
        for (i = 0; i < 80 - (align ? 1u : 0u); i++) {
          if (q.output[i] != ref.output[i]) {
            printf("FAIL,quantize,%s,align=%u,seed=%u,lane=%u,got=%d,expected=%d\n",
                   names[p], align, seed, i,
                   (int)q.output[i], (int)ref.output[i]);
            failures++;
            goto quant_done;
          }
        }
      }
quant_done:
  if (failures) {
    puts("SKIP,quantize,correctness_failed");
    return 1;
  }
  puts("PASS,quantize,correctness,3072_cases");
  for (p = 0; p < 6; p++)
    for (align = 0; align < 2; align++) {
      q_ctx q[2];
      ps2_bench_variant entries[2];
      prepare_quant(p, align, 47, &q[1]);
      q[0] = q[1];
      q[0].output = ref_mem + (align ? 1 : 0);

      entries[0].name = "ijg_c";
      entries[0].run = quantize_reference;
      entries[0].context = &q[0];
      entries[0].digest = digest_quant;
      entries[0].reset = reset_quant;
      entries[1].name = "mmi";
      entries[1].run = quantize_mmi;
      entries[1].context = &q[1];
      entries[1].digest = digest_quant;
      entries[1].reset = reset_quant;
      if (ps2_bench_compare("quantize", names[p], 64,
                            (int)align, entries, 2, Q_COUNT))
        failures++;
    }
  return failures ? 1 : 0;
}
