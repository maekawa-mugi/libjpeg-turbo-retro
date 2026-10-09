/* One-ELF PS2 EE SIMD correctness/benchmark declarations.
 * The benchmark objects contain independent copies of the same kernels.
 * SPDX-License-Identifier: Zlib
 */
#ifndef PS2_MMI_BENCH_H
#define PS2_MMI_BENCH_H
#if defined(PS2_BENCH_HOST_CHECK)
#include "bench-host-types.h"
#else
#include "../jsimdint.h"
#endif
#include <stdint.h>

typedef void (*ps2_merged_fn)(JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);
typedef void (*ps2_color_fn)(JDIMENSION, JSAMPIMAGE, JDIMENSION,
                             JSAMPARRAY, int);
typedef void (*ps2_idct_fn)(ISLOW_MULT_TYPE *, JCOEFPTR,
                            JSAMPARRAY, JDIMENSION);
typedef void (*ps2_bench_call)(void *);

#define PS2_DECL_MERGED_ONE(v, layout) \
  extern void ps2_bench_##v##_h2v1_##layout(JDIMENSION, JSAMPIMAGE, \
                                            JDIMENSION, JSAMPARRAY); \
  extern void ps2_bench_##v##_h2v2_##layout(JDIMENSION, JSAMPIMAGE, \
                                            JDIMENSION, JSAMPARRAY);
#define PS2_DECL_MERGED(v) \
  PS2_DECL_MERGED_ONE(v, rgbx) \
  PS2_DECL_MERGED_ONE(v, bgrx) \
  PS2_DECL_MERGED_ONE(v, xbgr) \
  PS2_DECL_MERGED_ONE(v, xrgb)
PS2_DECL_MERGED(scalar)
PS2_DECL_MERGED(pmul4)
PS2_DECL_MERGED(pmul8)
PS2_DECL_MERGED(addpack)
PS2_DECL_MERGED(vector)

#define PS2_DECL_COLOR_ONE(v, name) \
  extern void ps2_bench_##v##_##name(JDIMENSION, JSAMPIMAGE, JDIMENSION, \
                                     JSAMPARRAY, int);
#define PS2_DECL_COLOR(v) \
  PS2_DECL_COLOR_ONE(v, rgb) \
  PS2_DECL_COLOR_ONE(v, extrgb) \
  PS2_DECL_COLOR_ONE(v, extbgr) \
  PS2_DECL_COLOR_ONE(v, rgbx) \
  PS2_DECL_COLOR_ONE(v, bgrx) \
  PS2_DECL_COLOR_ONE(v, xbgr) \
  PS2_DECL_COLOR_ONE(v, xrgb)
PS2_DECL_COLOR(scalar)
PS2_DECL_COLOR(pmul4)
PS2_DECL_COLOR(pmul8)

extern void ps2_bench_idct_evenoff(ISLOW_MULT_TYPE *, JCOEFPTR,
                                    JSAMPARRAY, JDIMENSION);
extern void ps2_bench_idct_evenon(ISLOW_MULT_TYPE *, JCOEFPTR,
                                   JSAMPARRAY, JDIMENSION);

/* Each implementation gets its own output target/context. */
typedef struct {
  const char *name;
  ps2_bench_call run;
  void *context;
  uint32_t (*digest)(void *);
  void (*reset)(void *);
} ps2_bench_variant;

uint32_t ps2_bench_fnv(const JSAMPLE *data, size_t length, uint32_t hash);
int ps2_bench_compare(const char *category, const char *workload,
                      unsigned width, int alignment,
                      const ps2_bench_variant *variants, unsigned n,
                      unsigned repetitions);
void ps2_bench_csv(const char *category, const char *variant,
                   const char *workload, unsigned width, int alignment,
                   uint64_t ticks, unsigned repeats);

int ps2_bench_run_merged(void);
int ps2_bench_run_color(void);
int ps2_bench_run_sampling(void);
int ps2_bench_run_idct(void);
int ps2_bench_run_quantize(void);
int ps2_bench_execute(void);
#endif
