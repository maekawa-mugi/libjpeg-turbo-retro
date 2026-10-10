/*
 * Same-ELF scalar/A/B timing adapted from the measurement design in
 * maekawa-mugi/openssl-retro, test/ps2/main.c (Apache-2.0).
 *
 * Correctness is checked before timing by the individual test suites.
 * We additionally compare the complete deterministic output hash before
 * timing and after each timed batch.  Repeated runs use rotated orders:
 * 2 variants AB/BA, 3 variants ABC/BCA/CAB, 5 variants all rotations.
 * The median of an even number of timer samples is the result.
 *
 * Never print, hash, or reset buffers inside the timed interval.
 * SPDX-License-Identifier: Zlib
 */
#include "bench-harness.h"
#include <timer.h>
#include <stdio.h>
#include <string.h>

#define BENCH_WARMUP 24
#define BENCH_MAX_VARIANTS 8
#define BENCH_MAX_SAMPLES 16

static volatile uint32_t bench_sink;

uint32_t
ps2_bench_fnv(const JSAMPLE *data, size_t length, uint32_t hash)
{
  size_t i;
  for (i = 0; i < length; i++) {
    hash ^= (uint32_t)data[i];
    hash *= 16777619u;
  }
  return hash;
}

static uint64_t
median_even(uint64_t *samples, unsigned count)
{
  unsigned i, j;
  for (i = 1; i < count; i++) {
    uint64_t x = samples[i];
    for (j = i; j && x < samples[j - 1]; j--)
      samples[j] = samples[j - 1];
    samples[j] = x;
  }
  return (samples[count / 2 - 1] + samples[count / 2]) / 2u;
}

/* The opt-in FPU IDCT is deliberately not byte-exact with IJG.
 * Validate each contender's output stability, but allow only the final
 * contender to have a different digest.  Other contenders MUST match.
 */
static int
bench_compare_internal(const char *category, const char *workload,
                       unsigned width, int alignment,
                       const ps2_bench_variant *variants, unsigned n,
                       unsigned repetitions, int approximate_last)
{
  uint64_t samples[BENCH_MAX_VARIANTS][BENCH_MAX_SAMPLES];
  uint32_t expected[BENCH_MAX_VARIANTS] = { 0 };
  unsigned sample_count, j, i, sample, step, k;
  int failures = 0;

  if (!variants || n < 2 || n > BENCH_MAX_VARIANTS ||
      repetitions == 0)
    return 1;

  /* Two full scheduling rotations; at least six samples each.
   * This balances order for N=2,3,5 in our JPEG kernel groups.
   */
  sample_count = n < 3 ? 6 : 2 * n;
  if (sample_count > BENCH_MAX_SAMPLES)
    return 1;
  memset(samples, 0, sizeof(samples));

  for (i = 0; i < n; i++) {
    uint32_t digest;
    if (variants[i].reset)
      variants[i].reset(variants[i].context);
    variants[i].run(variants[i].context);
    digest = variants[i].digest(variants[i].context);
    expected[i] = digest;
    if (i != 0 && !(approximate_last && i == n - 1) &&
        digest != expected[0]) {
      printf("FAIL,ab_digest,%s,%s,%s,expected=%08lx,actual=%08lx\n",
             category, workload, variants[i].name,
             (unsigned long)expected[0], (unsigned long)digest);
      failures++;
    }
  }
  if (failures)
    return 1;

  /* Warm up EVERY contender before any timed samples. */
  for (i = 0; i < n; i++) {
    for (j = 0; j < BENCH_WARMUP; j++)
      variants[i].run(variants[i].context);
  }

  for (sample = 0; sample < sample_count; sample++) {
    for (step = 0; step < n; step++) {
      uint64_t before, elapsed;
      uint32_t digest;
      i = (sample + step) % n;

      if (variants[i].reset)
        variants[i].reset(variants[i].context);
      __asm__ volatile("" : : : "memory");
      before = GetTimerSystemTime();
      for (k = 0; k < repetitions; k++)
        variants[i].run(variants[i].context);
      elapsed = GetTimerSystemTime() - before;
      __asm__ volatile("" : : : "memory");
      digest = variants[i].digest(variants[i].context);
      if (elapsed == 0 || digest != expected[i]) {
        printf("FAIL,ab_sample,%s,%s,%s,sample=%u,ticks=%llu,digest=%08lx,expected=%08lx\n",
               category, workload, variants[i].name, sample,
               (unsigned long long)elapsed, (unsigned long)digest,
               (unsigned long)expected[i]);
        failures++;
      }
      samples[i][sample] = elapsed;
    }
  }
  if (failures) {
    printf("SKIP,ab_timing,%s,%s,invalid_digest_or_timer\n",
           category, workload);
    return 1;
  }

  for (i = 0; i < n; i++) {
    uint64_t median = median_even(samples[i], sample_count);
    ps2_bench_csv(category, variants[i].name, workload,
                  width, alignment, median, repetitions);
    bench_sink += (uint32_t)median ^ expected[i];
  }
  return 0;
}

int
ps2_bench_compare(const char *category, const char *workload,
                  unsigned width, int alignment,
                  const ps2_bench_variant *variants, unsigned n,
                  unsigned repetitions)
{
  return bench_compare_internal(category, workload, width, alignment,
                                variants, n, repetitions, 0);
}

int
ps2_bench_compare_approx_last(const char *category, const char *workload,
                              unsigned width, int alignment,
                              const ps2_bench_variant *variants, unsigned n,
                              unsigned repetitions)
{
  return bench_compare_internal(category, workload, width, alignment,
                                variants, n, repetitions, 1);
}
