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
extern void ps2_ui_bench_progress(const char *, const char *, unsigned);
extern void ps2_ui_bench_result(unsigned, const char *, const char *, unsigned, int);

#define MAX_TIMINGS 512
static struct {
  char category[20], variant[20], workload[32];
  unsigned width;
  int alignment;
  uint64_t ticks;
  unsigned repeats;
} timings[MAX_TIMINGS];
static unsigned timing_count;
static int total_bench_failures;

static void
screen_best(unsigned row, const char *category, const char *workload,
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
    ps2_ui_bench_result(row,category,"N/A",0,0);
  } else {
    unsigned speed100 = (unsigned)(((uint64_t)base * 100u) / best);
    ps2_test_printf("BENCH %-11s %-8s %-8s %u.%02ux\n",
                    category, workload, winner,
                    speed100 / 100u, speed100 % 100u);
    ps2_ui_bench_result(row,category,winner,speed100,1);
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
  /* Called after sampling, never from inside a timed region. */
  ps2_ui_bench_progress(category,variant,timing_count + 1);
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
  screen_best(0,"merged", "h2v2_RGBX", 128, 0, "portable_c");
  screen_best(1,"color", "RGBX", 128, 0, "portable_c");
  screen_best(2,"plain_up", "h2v2", 128, 0, "portable_c");
  screen_best(3,"fancy_up", "h2v2", 64, 0, "portable_c");
  screen_best(4,"downsample", "h2v2", 256, 0, "portable_c");
  screen_best(5,"idct", "dense", 8, 0, "ijg_c");
  screen_best(6,"quantize", "dense", 64, 0, "ijg_c");
}

/* The five benchmark families run alongside their related regression
 * tests in test-suite.c, not as one opaque final batch. */
void ps2_bench_begin(void)
{
  timing_count=0;
  total_bench_failures=0;
  puts("BENCH_START,R5900,PS2SDK_TIMER,warm24,rotating_order,median");
  puts("CSV_HEADER,category,variant,workload,width,alignment,ticks,repeats,ticks_per_call_rounded");
  fflush(stdout);
}
int ps2_bench_run_group(unsigned group)
{
  int rc;
  const char *const names[]={"quantize","sampling","idct","color","merged"};
  if(group>=5)return 1;
  ps2_ui_bench_progress(names[group],"starting",timing_count);
  switch(group){
  case 0: rc=ps2_bench_run_quantize(); break;
  case 1: rc=ps2_bench_run_sampling(); break;
  case 2: rc=ps2_bench_run_idct(); break;
  case 3: rc=ps2_bench_run_color(); break;
  default: rc=ps2_bench_run_merged(); break;
  }
  total_bench_failures+=rc!=0;
  /* Publish the winners as soon as their family has completed. */
  switch(group){
  case 0: screen_best(6,"quantize","dense",64,0,"ijg_c"); break;
  case 1:
    screen_best(2,"plain_up","h2v2",128,0,"portable_c");
    screen_best(3,"fancy_up","h2v2",64,0,"portable_c");
    screen_best(4,"downsample","h2v2",256,0,"portable_c");
    break;
  case 2: screen_best(5,"idct","dense",8,0,"ijg_c"); break;
  case 3: screen_best(1,"color","RGBX",128,0,"portable_c"); break;
  default: screen_best(0,"merged","h2v2_RGBX",128,0,"portable_c"); break;
  }
  printf("BENCH_GROUP,%s,%s\n",names[group],rc?"FAIL":"PASS");
  fflush(stdout);
  return rc!=0;
}
void ps2_bench_skip_group(unsigned group)
{
  const char *const names[]={"quantize","sampling","idct","color","merged"};
  if(group>=5)return;
  ++total_bench_failures;
  ps2_ui_bench_progress(names[group],"SKIP: failed test",timing_count);
  printf("BENCH_GROUP,%s,SKIP: failed correctness check\n",names[group]);
  fflush(stdout);
}
int ps2_bench_end(void)
{
  printf("BENCH_END,failures=%d\n",total_bench_failures);
  fflush(stdout);
  return total_bench_failures;
}
int
ps2_bench_execute(void)
{
  unsigned i;
  ps2_bench_begin();
  for(i=0;i<5;i++)ps2_bench_run_group(i);
  return ps2_bench_end();
}
