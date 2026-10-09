/* Single-boot correctness + CSV kernel benchmark orchestration.
 * A/B timing uses PS2SDK GetTimerSystemTime and rotated-order medians.
 * The SAME ELF and inputs are used for every contender.
 *
 * CSV is written to stdout (PCSX2 console/serial capture).  The small
 * GS screen only shows pass/fail summary to avoid overflowing the screen.
 * SPDX-License-Identifier: Zlib
 */
#include "bench-harness.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern int ps2_test_printf(const char *, ...);

#define MAX_TIMINGS 512
static struct {
  char category[20], variant[20], workload[32];
  unsigned width;
  int alignment;
  uint64_t ticks;
  unsigned repeats;
} timings[MAX_TIMINGS];
static unsigned timing_count;

static void
screen_best(const char *category, const char *workload,
            unsigned width, int alignment, const char *baseline)
{
  unsigned i;
  uint64_t base = ~(uint64_t)0, best = ~(uint64_t)0;
  const char *winner = "none";
  for (i = 0; i < timing_count; i++) {
    uint64_t cost;
    if (strcmp(timings[i].category, category) ||
        strcmp(timings[i].workload, workload) ||
        timings[i].width != width ||
        timings[i].alignment != alignment)
      continue;
    cost = timings[i].repeats ?
           timings[i].ticks / timings[i].repeats : 0;
    if (!cost)
      continue;
    if (!strcmp(timings[i].variant, baseline))
      base = cost;
    if (cost < best) {
      best = cost;
      winner = timings[i].variant;
    }
  }
  if (base == ~(uint64_t)0 || best == ~(uint64_t)0) {
    ps2_test_printf("BENCH %-11s %-8s n/a\n", category, workload);
  } else {
    unsigned speed100 = (unsigned)(((uint64_t)base * 100u) / best);
    ps2_test_printf("BENCH %-11s %-8s %-8s %u.%02ux\n",
                    category, workload, winner,
                    speed100 / 100u, speed100 % 100u);
  }
}


void
ps2_bench_csv(const char *category, const char *variant,
              const char *workload, unsigned width, int alignment,
              uint64_t ticks, unsigned repeats)
{
  unsigned scaled = repeats ? (unsigned)((ticks + repeats / 2) / repeats) : 0;
  printf("CSV,%s,%s,%s,%u,%d,%llu,%u,%u\n",
         category, variant, workload, width, alignment,
         (unsigned long long)ticks, repeats, scaled);
  if (timing_count < MAX_TIMINGS) {
    unsigned i = timing_count++;
    snprintf(timings[i].category, sizeof(timings[i].category),
             "%s", category);
    snprintf(timings[i].variant, sizeof(timings[i].variant), "%s", variant);
    snprintf(timings[i].workload, sizeof(timings[i].workload),
             "%s", workload);
    timings[i].width = width;
    timings[i].alignment = alignment;
    timings[i].ticks = ticks;
    timings[i].repeats = repeats;
  }
}

void
ps2_bench_screen_summary(void)
{
  ps2_test_printf("A/B same-ELF relative results:\n");
  screen_best("merged", "h2v2_RGBX", 128, 0, "portable_c");
  screen_best("color", "RGBX", 128, 0, "portable_c");
  screen_best("plain_up", "h2v2", 128, 0, "portable_c");
  screen_best("fancy_up", "h2v2", 64, 0, "portable_c");
  screen_best("downsample", "h2v2", 256, 0, "portable_c");
  screen_best("idct", "dense", 8, 0, "ijg_c");
  screen_best("quantize", "dense", 64, 0, "ijg_c");
}

int
ps2_bench_execute(void)
{
  int failures = 0;
  puts("BENCH_START,R5900,PS2SDK_TIMER,warm24,rotating_order,median");
  puts("CSV_HEADER,category,variant,workload,width,alignment,ticks,repeats,ticks_per_call_rounded");
  /* Failure in one category does not skip the remaining categories. */
  failures += ps2_bench_run_merged();
  failures += ps2_bench_run_color();
  failures += ps2_bench_run_sampling();
  failures += ps2_bench_run_idct();
  failures += ps2_bench_run_quantize();
  /* The final summary is redrawn by the outer test runner. */
  printf("BENCH_END,failures=%d\n", failures);
  fflush(stdout);
  return failures ? 1 : 0;
}
