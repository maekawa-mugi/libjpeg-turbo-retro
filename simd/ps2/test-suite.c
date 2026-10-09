/* On-screen runner for the existing EE MMI reference tests.
 * SPDX-License-Identifier: Zlib
 */
#include <debug.h>
#include <kernel.h>
#include <stdarg.h>
#include <stdio.h>

int ps2_test_primitives(void);
int ps2_test_upsample(void);
int ps2_test_fancy(void);
int ps2_test_downsample(void);
int ps2_test_idct(void);
int ps2_test_color(void);
int ps2_test_merged(void);

#ifdef PS2_MMI_ALL_IN_ONE
int ps2_bench_execute(void);
int ps2_test_jpeg_stream(void);
void ps2_bench_screen_summary(void);
#endif

/* The test objects rename printf/puts to these functions, so detailed
 * failures appear on the GS display as well as the host console. */
int ps2_test_printf(const char *format, ...)
{
  char text[512];
  va_list args;
  int result;
  va_start(args, format);
  result = vsnprintf(text, sizeof(text), format, args);
  va_end(args);
  scr_printf("%s", text);
  printf("%s", text);
  return result;
}

int ps2_test_puts(const char *text)
{
  return ps2_test_printf("%s\n", text);
}

int main(void)
{
  static const struct {
    const char *name;
    int (*run)(void);
  } tests[] = {
    { "MMI primitives", ps2_test_primitives },
    { "plain upsampling", ps2_test_upsample },
    { "fancy upsampling", ps2_test_fancy },
    { "downsampling", ps2_test_downsample },
    { "integer IDCT", ps2_test_idct },
    { "YCbCr to RGB", ps2_test_color },
    { "merged YCbCr/RGBX", ps2_test_merged }
#ifdef PS2_MMI_ALL_IN_ONE
    , { "full JPEG-stream integration", ps2_test_jpeg_stream }
    , { "all variants + A/B medians", ps2_bench_execute }
#endif
  };
  unsigned i, failed = 0;
#ifdef PS2_MMI_ALL_IN_ONE
  unsigned failure_mask = 0;
#endif
  const unsigned total = (unsigned)(sizeof(tests) / sizeof(tests[0]));
  init_scr();
  ps2_test_puts("libjpeg-turbo PS2 EE MMI validation\n");
#ifdef PS2_MMI_ALL_IN_ONE
  ps2_test_puts("All variants and timing in one ELF\n");
  ps2_test_puts("CSV results: stdout / emulator console\n");
#endif
#ifdef PS2_EXPERIMENTAL_IDCT_EVEN
  ps2_test_puts("IDCT even rotation: MMI\n");
#else
  ps2_test_puts("IDCT even rotation: scalar\n");
#endif
  for (i = 0; i < total; i++) {
    ps2_test_printf("[%u/%u] %s\n", i + 1, total, tests[i].name);
    if (tests[i].run()) {
      failed++;
#ifdef PS2_MMI_ALL_IN_ONE
      failure_mask |= 1u << i;
#endif
      ps2_test_printf("TEST FAILED: %s\n", tests[i].name);
    }
  }
#ifdef PS2_MMI_ALL_IN_ONE
  /* Redraw the essential information at the end so a single photograph
   * suffices when serial/stdout capture is unavailable.
   */
  init_scr();
  ps2_test_puts("PS2 EE MMI one-boot summary");
  for (i = 0; i < total; i++)
    ps2_test_printf("%u: %s %s\n", i + 1,
                    (failure_mask & (1u << i)) ? "FAIL" : "OK",
                    tests[i].name);
  ps2_bench_screen_summary();
#endif
  if (failed)
    ps2_test_printf("\nTEST: FAIL! (%u/%u groups failed)\n", failed, total);
  else
    ps2_test_printf("\nTEST: OK! (%u/%u groups passed)\n", total, total);
  ps2_test_puts("Result stays on screen. Reset or stop to exit.");
  fflush(stdout);
  for (;;)
    SleepThread();
}
