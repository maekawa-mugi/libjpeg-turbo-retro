/* Single-boot correctness + CSV kernel benchmark orchestration.
 * A/B timing uses PS2SDK GetTimerSystemTime and rotated-order medians.
 * The SAME ELF and inputs are used for every contender.
 *
 * CSV is written to stdout (PCSX2 console/serial capture).  The small
 * GS screen only shows pass/fail summary to avoid overflowing the screen.
 * SPDX-License-Identifier: Zlib
 */
#include "bench-harness.h"
#include <debug.h>
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
static unsigned passed_groups;

static void
screen_best(unsigned row, const char *category, const char *workload,
            unsigned width, int alignment, const char *baseline)
{
  unsigned i;
  uint64_t base = ~(uint64_t)0, best = ~(uint64_t)0;
  const char *winner = "none";
  /* A partially failed or skipped family must never publish a winner. */
  static const unsigned family_of_row[9] = {4,3,1,1,1,2,0,5,6};
  if (row >= 9 || !(passed_groups & (1u << family_of_row[row]))) {
    ps2_test_printf("BENCH %-11s %-8s n/a (correctness gate)\n",
                    category,workload);
    if (row < 9) ps2_ui_bench_result(row,category,"N/A",0,0);
    return;
  }
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
    /* Float IDCT is approximate, so NEVER rank it as a bit-exact
     * integer IDCT contender, even if its timing beats every MMI kernel.
     * Its own relative speed is printed in the experimental verdict.
     */
    if (!strcmp(category, "idct") &&
        !strcmp(timings[i].variant, "fpu_approx"))
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


/* Report the requested experimental contender even when its speedup is
 * below 1.00x, rather than displaying the scalar winner at 1.00x.
 */
#if defined(PS2_EXPERIMENTAL_VU0) || defined(PS2_EXPERIMENTAL_VIF0_DMA)
static void
screen_experiment(unsigned row, unsigned group, const char *category,
                  const char *workload, unsigned width,
                  const char *baseline, const char *candidate)
{
  unsigned i;
  uint64_t base = 0, new_cost = 0;
  if (!(passed_groups & (1u << group))) {
    ps2_ui_bench_result(row, category, "N/A", 0, 0);
    return;
  }
  for (i = 0; i < timing_count; i++) {
    uint64_t per_call;
    if (strcmp(timings[i].category, category) ||
        strcmp(timings[i].workload, workload) ||
        timings[i].width != width || timings[i].alignment != 0 ||
        !timings[i].repeats) continue;
    per_call = timings[i].ticks / timings[i].repeats;
    if (!strcmp(timings[i].variant, baseline)) base = per_call;
    if (!strcmp(timings[i].variant, candidate)) new_cost = per_call;
  }
  if (!base || !new_cost) {
    ps2_ui_bench_result(row, category, "N/A", 0, 0);
    return;
  }
  {
    unsigned speed100 = (unsigned)(base * 100u / new_cost);
    ps2_test_printf("BENCH_EXPERIMENT,%s,%s,baseline=%s,candidate=%s,speed=%u.%02ux\n",
                    category, workload, baseline, candidate,
                    speed100 / 100, speed100 % 100);
    ps2_ui_bench_result(row, category, candidate, speed100, 1);
  }
}
#endif

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
#ifdef PS2_EXPERIMENTAL_VU0
  screen_experiment(7, 5, "vu_idct", "dense", 8,
                    "scalar_matrix", "vu0_macro");
#endif
#ifdef PS2_EXPERIMENTAL_VIF0_DMA
  screen_experiment(8, 6, "vif0_dma", "upload256", 256,
                    "cpu_store", "vif0_dma");
#endif
}

/* Results are printed only after the complete correctness and timing
 * matrix. Keep experimental speedups in their own domains: VU0 vs float
 * matrix, DMA vs CPU upload, COP1 float vs IJG integer (approximate).
 */
static uint64_t
find_cost(const char *cat, const char *variant,
          const char *workload, unsigned width)
{
  unsigned i;
  for (i = 0; i < timing_count; i++)
    if (!strcmp(timings[i].category, cat) &&
        !strcmp(timings[i].variant, variant) &&
        !strcmp(timings[i].workload, workload) &&
        timings[i].width == width && timings[i].alignment == 0 &&
        timings[i].repeats)
      return timings[i].ticks / timings[i].repeats;
  return 0;
}

static unsigned
relative100(uint64_t base, uint64_t candidate)
{
  if (!base || !candidate)
    return 0;
  return (unsigned)(base * 100u / candidate);
}

static void
verdict_line(int y, unsigned color, const char *message)
{
  scr_setXY(0, y);
  scr_setfontcolor(color);
  /* Fixed screen-cell limit: never wrap onto the test/benchmark table. */
  scr_printf("%-62.62s", message);
}

void
ps2_bench_one_shot_verdict(int valid)
{
  static const char *const exact_names[] = {
    "ijg_c", "evenoff", "evenon", "batch", "direct"
  };
  uint64_t ijg, best_cost, candidate;
  const char *best_name = "N/A";
  unsigned i, exact_ratio, fpu_ratio = 0, vu_ratio = 0, dma_ratio = 0;
  const char *choice;
  char text[96];

  if (!valid) {
    ps2_test_printf("CONCLUSION,INVALID,fix_correctness_or_benchmark_first\n");
    verdict_line(1, 0x000000ffu, "VERDICT INVALID: check failed test or benchmark");
    verdict_line(2, 0x000000ffu, "Full failure details: console stdout");
    verdict_line(13, 0x000000ffu, "DO NOT USE PERFORMANCE RESULTS UNTIL ALL CHECKS PASS");
    return;
  }

  ijg = find_cost("idct", "ijg_c", "dense", 8);
  best_cost = ijg;
  if (ijg && (passed_groups & (1u << 2))) {
    best_name = "ijg_c";
    for (i = 1; i < sizeof(exact_names) / sizeof(exact_names[0]); i++) {
      candidate = find_cost("idct", exact_names[i], "dense", 8);
      if (candidate && candidate < best_cost) {
        best_cost = candidate;
        best_name = exact_names[i];
      }
    }
    fpu_ratio = relative100(ijg, find_cost("idct", "fpu_approx",
                                          "dense", 8));
  }
  exact_ratio = relative100(ijg, best_cost);
#ifdef PS2_EXPERIMENTAL_VU0
  if (passed_groups & (1u << 5))
    vu_ratio = relative100(find_cost("vu_idct", "scalar_matrix",
                                    "dense", 8),
                           find_cost("vu_idct", "vu0_macro", "dense", 8));
#endif
#ifdef PS2_EXPERIMENTAL_VIF0_DMA
  if (passed_groups & (1u << 6))
    dma_ratio = relative100(find_cost("vif0_dma", "cpu_store",
                                     "upload256", 256),
                            find_cost("vif0_dma", "vif0_dma",
                                      "upload256", 256));
#endif
  if (!exact_ratio) {
    ps2_test_printf("CONCLUSION,INVALID,missing_exact_IDCT_reference\n");
    verdict_line(1, 0x000000ffu, "VERDICT INCOMPLETE: missing exact IDCT reference");
    verdict_line(13, 0x000000ffu, "DO NOT USE PERFORMANCE RESULTS");
    return;
  }

  if (fpu_ratio)
    snprintf(text, sizeof(text),
             "EXACT IDCT: %-7s %u.%02ux | FPU~: %u.%02ux",
             best_name, exact_ratio / 100, exact_ratio % 100,
             fpu_ratio / 100, fpu_ratio % 100);
  else
    snprintf(text, sizeof(text),
             "EXACT IDCT: %-7s %u.%02ux | FPU~: N/A (quality)",
             best_name, exact_ratio / 100, exact_ratio % 100);
  verdict_line(1, 0x0000ff00u, text);
  if (vu_ratio && dma_ratio)
    snprintf(text, sizeof(text),
             "VU0 vs float: %u.%02ux | DMA vs CPU upload: %u.%02ux",
             vu_ratio / 100, vu_ratio % 100,
             dma_ratio / 100, dma_ratio % 100);
  else
    snprintf(text, sizeof(text),
             "VU0 vs float: %s | DMA vs upload: %s",
             vu_ratio ? "measured" : "N/A",
             dma_ratio ? "measured" : "N/A");
  verdict_line(2, 0x00ffffffu, text);

  /* A 5% advantage on this dense sample is only a heuristic.
   * Do NOT enable approximation or DMA in production from this score. */
  choice = exact_ratio >= 105 ? best_name : "ijg_c";
  if (dma_ratio && dma_ratio < 100)
    snprintf(text, sizeof(text),
             "TAKEAWAY: exact %-7s; VU0 experimental; skip DMA",
             choice);
  else
    snprintf(text, sizeof(text),
             "TAKEAWAY: exact %-7s; verify experiments on EE",
             choice);
  verdict_line(13, 0x0000ffffu, text);

  ps2_test_printf(
    "CONCLUSION,exact_idct,%s,baseline_ijg_c,speed=%u.%02ux\n",
    best_name, exact_ratio / 100, exact_ratio % 100);
  if (fpu_ratio)
    ps2_test_printf("CONCLUSION,fpu_approx,vs_ijg_c,speed=%u.%02ux,NOT_EXACT\n",
                    fpu_ratio / 100, fpu_ratio % 100);
  else
    ps2_test_printf("CONCLUSION,fpu_approx,NO_QUALIFIED_RESULT\n");
  if (vu_ratio)
    ps2_test_printf("CONCLUSION,vu0_macro,vs_scalar_float_matrix,speed=%u.%02ux,EXPERIMENT_ONLY\n",
                    vu_ratio / 100, vu_ratio % 100);
  else
    ps2_test_printf("CONCLUSION,vu0_macro,NO_QUALIFIED_RESULT\n");
  if (dma_ratio)
    ps2_test_printf("CONCLUSION,vif0_dma,vs_cpu_upload,speed=%u.%02ux,TRANSFER_ONLY\n",
                    dma_ratio / 100, dma_ratio % 100);
  else
    ps2_test_printf("CONCLUSION,vif0_dma,NO_QUALIFIED_RESULT\n");
  ps2_test_printf("CONCLUSION,recommend_exact,%s,REPRESENTATIVE_DENSE_ONLY\n",
                  choice);
}

/* The five benchmark families run alongside their related regression
 * tests in test-suite.c, not as one opaque final batch. */
void ps2_bench_begin(void)
{
  timing_count=0;
  total_bench_failures=0;
  passed_groups=0;
  puts("BENCH_START,R5900,PS2SDK_TIMER,warm24,rotating_order,median");
  puts("CSV_HEADER,category,variant,workload,width,alignment,ticks,repeats,ticks_per_call_rounded");
  fflush(stdout);
}
int ps2_bench_run_group(unsigned group)
{
  int rc;
  const char *const names[]={"quantize","sampling","idct","color","merged",
                             "vu_idct","vif0_dma"};
  if(group>=7)return 1;
  ps2_ui_bench_progress(names[group],"starting",timing_count);
  switch(group){
  case 0: rc=ps2_bench_run_quantize(); break;
  case 1: rc=ps2_bench_run_sampling(); break;
  case 2: rc=ps2_bench_run_idct(); break;
  case 3: rc=ps2_bench_run_color(); break;
  case 4: rc=ps2_bench_run_merged(); break;
#ifdef PS2_EXPERIMENTAL_VU0
  case 5: rc=ps2_bench_run_vu0(); break;
#endif
#ifdef PS2_EXPERIMENTAL_VIF0_DMA
  case 6: rc=ps2_bench_run_vif0_dma(); break;
#endif
  default: return 1;
  }
  total_bench_failures+=rc!=0;
  if(!rc)passed_groups|=1u<<group;
  /* Publish the winners only when every family check succeeds. */
  switch(group){
  case 0: screen_best(6,"quantize","dense",64,0,"ijg_c"); break;
  case 1:
    screen_best(2,"plain_up","h2v2",128,0,"portable_c");
    screen_best(3,"fancy_up","h2v2",64,0,"portable_c");
    screen_best(4,"downsample","h2v2",256,0,"portable_c");
    break;
  case 2: screen_best(5,"idct","dense",8,0,"ijg_c"); break;
  case 3: screen_best(1,"color","RGBX",128,0,"portable_c"); break;
  case 4: screen_best(0,"merged","h2v2_RGBX",128,0,"portable_c"); break;
#ifdef PS2_EXPERIMENTAL_VU0
  case 5: screen_experiment(7,5,"vu_idct","dense",8,
                            "scalar_matrix","vu0_macro"); break;
#endif
#ifdef PS2_EXPERIMENTAL_VIF0_DMA
  case 6: screen_experiment(8,6,"vif0_dma","upload256",256,
                            "cpu_store","vif0_dma"); break;
#endif
  default: break;
  }
  printf("BENCH_GROUP,%s,%s\n",names[group],rc?"FAIL":"PASS");
  fflush(stdout);
  return rc!=0;
}
void ps2_bench_skip_group(unsigned group)
{
  const char *const names[]={"quantize","sampling","idct","color","merged",
                              "vu_idct","vif0_dma"};
  if(group>=7)return;
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
#ifdef PS2_EXPERIMENTAL_VU0
  ps2_bench_run_group(5);
#endif
#ifdef PS2_EXPERIMENTAL_VIF0_DMA
  ps2_bench_run_group(6);
#endif
  return ps2_bench_end();
}
