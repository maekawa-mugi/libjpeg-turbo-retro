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
PS2_DECL_MERGED(table)

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
PS2_DECL_COLOR(regpack)
PS2_DECL_COLOR(table)

extern void ps2_bench_idct_evenoff(ISLOW_MULT_TYPE *, JCOEFPTR,
                                    JSAMPARRAY, JDIMENSION);
extern void ps2_bench_idct_evenon(ISLOW_MULT_TYPE *, JCOEFPTR,
                                   JSAMPARRAY, JDIMENSION);

extern void ps2_bench_idct_batch(ISLOW_MULT_TYPE *, JCOEFPTR,
                                  JSAMPARRAY, JDIMENSION);
extern void ps2_bench_idct_direct(ISLOW_MULT_TYPE *, JCOEFPTR,
                                    JSAMPARRAY, JDIMENSION);
extern void ps2_bench_idct_lut(ISLOW_MULT_TYPE *, JCOEFPTR,
                               JSAMPARRAY, JDIMENSION);
extern void ps2_bench_idct_lut_norow(ISLOW_MULT_TYPE *, JCOEFPTR,
                                     JSAMPARRAY, JDIMENSION);
extern void jsimd_quantize_legacy_ps2mmi(JCOEFPTR, DCTELEM *, DCTELEM *);

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
/* Only the final contender may differ from the reference (opt-in FPU). */
int ps2_bench_compare_approx_last(const char *category, const char *workload,
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
#if defined(PS2_EXPERIMENTAL_VU0)
int ps2_bench_run_vu0(void);
#endif
#if defined(PS2_EXPERIMENTAL_VIF0_DMA)
int ps2_bench_run_vif0_dma(void);
#endif
int ps2_bench_execute(void);
#endif
