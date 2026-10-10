/* Live GS status table for PS2 EE MMI validation and A/B benchmarks.
 * Detailed test diagnostics and machine-readable CSV go to stdout only.
 * SPDX-License-Identifier: Zlib
 */
#include <debug.h>
#include <kernel.h>
#include <stdarg.h>
#include <stdio.h>

#define UI_WHITE  0x00ffffffu
#define UI_GREEN  0x0000ff00u
#define UI_RED    0x000000ffu
#define UI_YELLOW 0x0000ffffu

int ps2_test_primitives(void);
int ps2_test_upsample(void);
int ps2_test_fancy(void);
int ps2_test_downsample(void);
int ps2_test_idct(void);
int ps2_test_color(void);
int ps2_test_merged(void);

#ifdef PS2_MMI_ALL_IN_ONE
int ps2_test_jpeg_stream(void);
void ps2_bench_begin(void);
int ps2_bench_run_group(unsigned);
void ps2_bench_skip_group(unsigned);
int ps2_bench_end(void);
void ps2_bench_screen_summary(void);
#endif

/* Never stream arbitrary test text to the GS. Long diagnostic lines used
 * to wrap past the last screen row and leave cursor/white-block artifacts.
 * Console output is preserved verbatim for capture and automation. */
int ps2_test_printf(const char *format, ...)
{
  char text[512];
  va_list args;
  int result;
  va_start(args, format);
  result = vsnprintf(text, sizeof(text), format, args);
  va_end(args);
  printf("%s", text);
  return result;
}
int ps2_test_puts(const char *text)
{
  return ps2_test_printf("%s\n", text);
}

void ps2_ui_bench_progress(const char *category, const char *variant,
                           unsigned measured)
{
  scr_setXY(0,13);
  scr_setfontcolor(UI_WHITE);
  scr_printf("BENCH %-12.12s %-17.17s %4u samples       ",
             category,variant,measured);
}
void ps2_ui_bench_result(unsigned index, const char *category,
                         const char *winner, unsigned speed100,
                         int valid)
{
  if(index>=9)return;
  scr_setXY(0,15+(int)index);
  scr_setfontcolor(valid?UI_GREEN:UI_YELLOW);
  if(valid)
    scr_printf("%-13.13s %-16.16s %3u.%02ux       ",
               category,winner,speed100/100u,speed100%100u);
  else
    scr_printf("%-13.13s %-16s %s       ",category,"N/A","--");
}

int main(void)
{
  static const struct {
    const char *name;
    int (*run)(void);
  } tests[] = {
    {"MMI primitives",ps2_test_primitives},
    {"plain upsampling",ps2_test_upsample},
    {"fancy upsampling",ps2_test_fancy},
    {"downsampling",ps2_test_downsample},
    {"integer IDCT",ps2_test_idct},
    {"YCbCr to RGB",ps2_test_color},
    {"merged YCbCr/RGBX",ps2_test_merged}
#ifdef PS2_MMI_ALL_IN_ONE
    , {"JPEG stream",ps2_test_jpeg_stream}
#endif
  };
  const unsigned checks=(unsigned)(sizeof(tests)/sizeof(tests[0]));
  unsigned i, failed=0, failed_mask=0;
  int bench_failures=0;
  init_scr();
  scr_setCursor(0);
  scr_setXY(0,0);
  scr_setfontcolor(UI_WHITE);
  scr_printf("LIBJPEG-TURBO RETRO | PS2 EE MMI | VALIDATION + BENCHMARK");
  scr_setXY(0,1);
  scr_printf("Real MMI vs portable C | tests + timings in one ELF");
  scr_setXY(0,3);
  scr_printf("%-3s %-23s %-7s","NO","CORRECTNESS TEST","RESULT");
  for(i=0;i<checks;i++){
    scr_setXY(0,5+(int)i);
    scr_printf("%2u  %-23.23s %-7s",i+1,tests[i].name,"WAIT");
  }
  scr_setXY(0,14);
  scr_printf("%-13s %-16s %s","BENCH FAMILY","PROVISIONAL BEST","SPEED");
  for(i=0;i<7;i++)
    ps2_ui_bench_result(i,"pending","--",0,0);
  ps2_ui_bench_result(7,"vu_idct","N/A",0,0);
  ps2_ui_bench_result(8,"vif0_dma","N/A",0,0);
  ps2_test_puts("LIBJPEG_PS2,START,correctness+benchmark");
#ifdef PS2_EXPERIMENTAL_IDCT_EVEN
  ps2_test_puts("LIBJPEG_PS2,IDCT_EVEN,MMI");
#else
  ps2_test_puts("LIBJPEG_PS2,IDCT_EVEN,scalar");
#endif
#ifdef PS2_MMI_ALL_IN_ONE
  ps2_bench_begin();
#endif
  for(i=0;i<checks;i++){
    int rc;
    scr_setXY(0,5+(int)i);
    scr_setfontcolor(UI_YELLOW);
    scr_printf("%2u  %-23.23s %-7s",i+1,tests[i].name,"RUN");
    printf("TEST_BEGIN,%u,%s\n",i+1,tests[i].name);
    fflush(stdout);
    rc=tests[i].run();
    if(rc){
      ++failed;
      failed_mask|=1u<<i;
    }
    scr_setXY(0,5+(int)i);
    scr_setfontcolor(rc?UI_RED:UI_GREEN);
    scr_printf("%2u  %-23.23s %-7s",i+1,tests[i].name,rc?"FAIL":"PASS");
    printf("TEST_RESULT,%u,%s,%s\n",i+1,tests[i].name,rc?"FAIL":"PASS");
    fflush(stdout);
#ifdef PS2_MMI_ALL_IN_ONE
    /* Run each A/B family immediately after its relevant reference test.
     * Failed correctness gates suppress winner/timing recommendations. */
    {
      int group=-1, allowed=1;
      if(i==0)group=0;             /* quantize */
      else if(i==3){group=1;allowed=(failed_mask & 0x0eu)==0;} /* sampling */
      else if(i==4)group=2;        /* IDCT */
      else if(i==5)group=3;        /* color */
      else if(i==6)group=4;        /* merged */
      if(group>=0){
        if(rc || !allowed)ps2_bench_skip_group((unsigned)group);
        else ps2_bench_run_group((unsigned)group);
      }
    }
#endif
  }
#ifdef PS2_MMI_ALL_IN_ONE
#ifdef PS2_EXPERIMENTAL_VU0
  ps2_bench_run_group(5);
#endif
#ifdef PS2_EXPERIMENTAL_VIF0_DMA
  ps2_bench_run_group(6);
#endif
  bench_failures=ps2_bench_end();
  ps2_bench_screen_summary();
#endif
  scr_setXY(0,24);
  scr_setfontcolor((failed||bench_failures)?UI_RED:UI_GREEN);
  scr_printf("RESULT: %s | tests %u/%u | bench failures %d     ",
             (failed||bench_failures)?"FAIL":"PASS",
             checks-failed,checks,bench_failures);
  /* Preserve all nine score rows and the final PASS/FAIL row on GS.
   * The detailed completion banner is available in console CSV only. */
  printf("LIBJPEG_PS2,DONE,%s,tests=%u,passed=%u,bench_failures=%d\n",
         (failed||bench_failures)?"FAIL":"PASS",checks,checks-failed,
         bench_failures);
  fflush(stdout);
  for(;;)SleepThread();
}
