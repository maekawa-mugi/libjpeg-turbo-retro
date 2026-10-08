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
    { "YCbCr to RGB", ps2_test_color }
  };
  unsigned i, failed = 0;
  init_scr();
  ps2_test_puts("libjpeg-turbo PS2 EE MMI validation\n");
#ifdef PS2_EXPERIMENTAL_IDCT_EVEN
  ps2_test_puts("IDCT even rotation: MMI\n");
#else
  ps2_test_puts("IDCT even rotation: scalar\n");
#endif
  for (i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
    ps2_test_printf("[%u/6] %s\n", i + 1, tests[i].name);
    if (tests[i].run()) {
      failed++;
      ps2_test_printf("TEST FAILED: %s\n", tests[i].name);
    }
  }
  if (failed)
    ps2_test_printf("\nTEST: FAIL! (%u/6 groups failed)\n", failed);
  else
    ps2_test_puts("\nTEST: OK! (6/6 groups passed)");
  ps2_test_puts("Result stays on screen. Reset or stop to exit.");
  fflush(stdout);
  for (;;)
    SleepThread();
}
