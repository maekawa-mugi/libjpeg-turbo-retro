/* Host-only golden-verdict regression: no EE SDK and no PCSX2 required.
 * Check an exact IDCT winner against a faster but approximate float IDCT,
 * including separate VU0-compute and VIF0-transfer comparisons.
 * SPDX-License-Identifier: Zlib
 */
#include "bench-harness.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static int screen_y;
static char screen[25][128];
static char strict_idct[32];

void
scr_setXY(int x, int y)
{
  (void)x;
  screen_y = y;
}

void
scr_setfontcolor(unsigned color)
{
  (void)color;
}

void
scr_printf(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  if (screen_y >= 0 && screen_y < 25)
    vsnprintf(screen[screen_y], sizeof(screen[screen_y]), fmt, args);
  va_end(args);
}

int
ps2_test_printf(const char *fmt, ...)
{
  va_list args;
  int result;
  va_start(args, fmt);
  result = vprintf(fmt, args);
  va_end(args);
  return result;
}

void
ps2_ui_bench_progress(const char *a, const char *b, unsigned n)
{
  (void)a; (void)b; (void)n;
}

void
ps2_ui_bench_result(unsigned row, const char *category,
                    const char *name, unsigned speed100, int valid)
{
  (void)category; (void)speed100;
  if (row == 5 && valid)
    snprintf(strict_idct, sizeof(strict_idct), "%s", name);
}

int ps2_bench_run_quantize(void) { return 0; }
int ps2_bench_run_sampling(void) { return 0; }
int ps2_bench_run_color(void) { return 0; }
int ps2_bench_run_merged(void) { return 0; }

int
ps2_bench_run_idct(void)
{
  ps2_bench_csv("idct", "ijg_c", "dense", 8, 0, 140500, 100);
  ps2_bench_csv("idct", "evenoff", "dense", 8, 0, 280000, 100);
  ps2_bench_csv("idct", "evenon", "dense", 8, 0, 290000, 100);
  ps2_bench_csv("idct", "batch", "dense", 8, 0, 285000, 100);
  ps2_bench_csv("idct", "direct", "dense", 8, 0, 130000, 100);
  ps2_bench_csv("idct", "lut", "dense", 8, 0, 125000, 100);
  ps2_bench_csv("idct", "lut_norow", "dense", 8, 0, 128000, 100);
  /* This is faster, but approximate and must not win 'exact IDCT'. */
  ps2_bench_csv("idct", "fpu_approx", "dense", 8, 0, 110000, 100);
  return 0;
}

int
ps2_bench_run_vu0(void)
{
  ps2_bench_csv("vu_idct", "scalar_matrix", "dense", 8, 0, 176000, 100);
  ps2_bench_csv("vu_idct", "vu0_macro", "dense", 8, 0, 100000, 100);
  return 0;
}

int
ps2_bench_run_vif0_dma(void)
{
  ps2_bench_csv("vif0_dma", "cpu_store", "upload256", 256, 0, 27000, 100);
  ps2_bench_csv("vif0_dma", "vif0_dma", "upload256", 256, 0, 100000, 100);
  return 0;
}

int main(void)
{
  extern void ps2_bench_begin(void);
  extern int ps2_bench_run_group(unsigned group);
  extern void ps2_bench_one_shot_verdict(int valid);
  ps2_bench_begin();
  assert(ps2_bench_run_group(2) == 0);
  assert(!strcmp(strict_idct, "lut"));
  assert(ps2_bench_run_group(5) == 0);
  assert(ps2_bench_run_group(6) == 0);
  ps2_bench_one_shot_verdict(1);
  assert(strstr(screen[1], "EXACT IDCT: lut"));
  assert(strstr(screen[1], "FPU~: 1.27x"));
  assert(strstr(screen[2], "VU0 vs float: 1.76x"));
  assert(strstr(screen[2], "DMA vs CPU upload: 0.27x"));
  assert(strstr(screen[13], "TAKEAWAY: exact lut"));
  assert(strstr(screen[13], "skip DMA"));
  ps2_bench_one_shot_verdict(0);
  assert(strstr(screen[1], "VERDICT INVALID"));
  assert(strstr(screen[13], "DO NOT USE"));
  puts("PASS: one-shot GS verdict separates exact/FPU/VU0/DMA");
  return 0;
}
