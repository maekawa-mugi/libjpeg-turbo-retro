/* Standalone integer-IDCT A/B test and benchmark.
 * Scalar IJG, R5900 partial MMI, and experimental even PMULTH are all
 * present in the same ELF.  DC-only, sparse and dense blocks are separate
 * workload classes; not just one misleading average.
 * SPDX-License-Identifier: Zlib
 */
#include "bench-harness.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define DCT_ROW_BYTES 48
#define BENCH_COUNT 96

static struct jpeg_decompress_struct cinfo;
static jpeg_component_info component;
static JSAMPLE sample_range[5 * 256 + 128];
static JCOEF coeff_mem[80] __attribute__((aligned(16)));
static ISLOW_MULT_TYPE quant_mem[80] __attribute__((aligned(16)));
static JSAMPLE refmem[8][DCT_ROW_BYTES] __attribute__((aligned(16)));
static JSAMPLE outmem[6][8][DCT_ROW_BYTES] __attribute__((aligned(16)));
static JSAMPROW refrows[8], outrows[6][8];

typedef struct {
  unsigned int pattern;
  unsigned int iteration;
  int alignment;
  JCOEFPTR coeff;
  ISLOW_MULT_TYPE *quant;
  JDIMENSION outcol;
} idct_case;

typedef struct {
  ps2_idct_fn fn;
  idct_case *sample;
  JSAMPARRAY output;
} idct_call;

#ifdef PS2_APPROX_FPU_IDCT
/* IJG's AA&N floating inverse DCT requires a pre-scaled quantization
 * table.  Prepare it outside timing; the real decoder caches this table
 * in jddctmgr.c as well.  The PS2 FPU can legitimately differ from
 * the bit-exact integer reference, so correctness is reported separately.
 */
static FAST_FLOAT fpu_quant[DCTSIZE2] __attribute__((aligned(16)));
static JSAMPLE fpu_mem[8][DCT_ROW_BYTES] __attribute__((aligned(16)));
static JSAMPROW fpu_rows[8];
static const double fpu_aan[DCTSIZE] = {
  1.0, 1.387039845, 1.306562965, 1.175875602,
  1.0, 0.785694958, 0.541196100, 0.275899379
};

static void
prepare_fpu_quant(idct_case *ctx)
{
  unsigned r, c;
  for (r = 0; r < 8; r++)
    for (c = 0; c < 8; c++) {
      unsigned idx = r * 8 + c;
      fpu_quant[idx] = (FAST_FLOAT)
        ((double)ctx->quant[idx] * fpu_aan[r] * fpu_aan[c]);
    }
}

static void
call_fpu_approx(void *context)
{
  idct_case *ctx = (idct_case *)context;
  component.dct_table = fpu_quant;
  _jpeg_idct_float(&cinfo, &component, ctx->coeff, fpu_rows, ctx->outcol);
  component.dct_table = ctx->quant;
}

static void
reset_fpu_approx(void *context)
{
  unsigned row;
  (void)context;
  for (row = 0; row < 8; row++)
    memset(fpu_rows[row], 0xc9, DCT_ROW_BYTES);
}

static uint32_t
digest_fpu_approx(void *context)
{
  idct_case *ctx = (idct_case *)context;
  uint32_t hash = 2166136261u;
  unsigned row;
  for (row = 0; row < 8; row++)
    hash = ps2_bench_fnv(fpu_rows[row] + ctx->outcol, 8, hash);
  return hash;
}
#endif

static JSAMPLE
range_limit_value(int x)
{
  unsigned int index = (unsigned int)x & 1023u;
  if (index < 128)
    return (JSAMPLE)(index + 128);
  if (index < 512)
    return 255;
  if (index < 896)
    return 0;
  return (JSAMPLE)(index - 896);
}

static void
initialize_idct(void)
{
  unsigned i;
  memset(&cinfo, 0, sizeof(cinfo));
  memset(&component, 0, sizeof(component));
  cinfo.data_precision = 8;
  cinfo.sample_range_limit = sample_range + 256;
  for (i = 0; i < 1024; i++)
    sample_range[256 + 128 + i] = range_limit_value((int)i);
  /* _jpeg_idct_islow uses IDCT_range_limit(cinfo), i.e. the base
   * pointer + CENTERJSAMPLE. _jpeg_idct_float reads the BASE pointer
   * directly. The shared 1024-byte overlapping table must also fill
   * base[0..127] with identity samples, just like jdmaster.c.
   * Omitting these bytes created bogus 255-level FPU mismatches.
   */
  for (i = 0; i < 128; i++)
    sample_range[256 + i] = (JSAMPLE)i;
  for (i = 0; i < 8; i++) {
    refrows[i] = refmem[i];
    outrows[0][i] = outmem[0][i];
    outrows[1][i] = outmem[1][i];
    outrows[2][i] = outmem[2][i];
    outrows[3][i] = outmem[3][i];
    outrows[4][i] = outmem[4][i];
    outrows[5][i] = outmem[5][i];
#ifdef PS2_APPROX_FPU_IDCT
    fpu_rows[i] = fpu_mem[i];
#endif
  }
}

static void
prepare_case(idct_case *ctx, unsigned profile,
             unsigned iteration, int alignment)
{
  unsigned i;
  uint32_t state = 0x91c3e21bu + 3571u * iteration + 53u * profile;
  ctx->pattern = profile;
  ctx->iteration = iteration;
  ctx->alignment = alignment;
  ctx->coeff = coeff_mem + ((alignment & 1) ? 1 : 0);
  ctx->quant = quant_mem + ((alignment & 2) ? 1 : 0);
  ctx->outcol = (alignment & 1) ? 7 : 0;
  component.dct_table = ctx->quant;
  for (i = 0; i < 64; i++) {
    state = state * 1664525u + 1013904223u;
    ctx->quant[i] = (ISLOW_MULT_TYPE)(1u + (state % 31u));
    ctx->coeff[i] = 0;
  }
  ctx->coeff[0] = (JCOEF)((int)(state % 513u) - 256);

  if (profile == 1) {
    /* Two or three AC coefficients, often found in smooth images. */
    ctx->coeff[1 + (iteration % 14u)] =
      (JCOEF)((int)((state >> 5) % 95u) - 47);
    ctx->coeff[45] = (JCOEF)((int)((state >> 12) % 63u) - 31);
  } else if (profile == 2) {
    /* High-entropy DCT block with bounded 32-bit intermediate products. */
    for (i = 1; i < 64; i++) {
      state = state * 1664525u + 1013904223u;
      ctx->coeff[i] = (JCOEF)((int)((state >> 10) % 129u) - 64);
    }
  } else if (profile == 3) {
    /* Even-frequency rotation stress, plus signed range edge cases. */
    for (i = 0; i < 64; i++) {
      ctx->coeff[i] = 0;
      ctx->quant[i] = 1;
    }
    ctx->coeff[0] = (JCOEF)((iteration & 1u) ? -96 : 96);
    ctx->coeff[2] = (JCOEF)((iteration & 2u) ? -120 : 120);
    ctx->coeff[6] = (JCOEF)((iteration & 4u) ? -90 : 90);
    ctx->quant[2] = 8;
    ctx->quant[6] = 10;
  }

  for (i = 0; i < 8; i++) {
    memset(refrows[i], 0xc9, DCT_ROW_BYTES);
    memset(outrows[0][i], 0xc9, DCT_ROW_BYTES);
    memset(outrows[1][i], 0xc9, DCT_ROW_BYTES);
    memset(outrows[2][i], 0xc9, DCT_ROW_BYTES);
    memset(outrows[3][i], 0xc9, DCT_ROW_BYTES);
    memset(outrows[4][i], 0xc9, DCT_ROW_BYTES);
    memset(outrows[5][i], 0xc9, DCT_ROW_BYTES);
  }
}

static void
call_islow_scalar(void *context)
{
  idct_case *c = (idct_case *)context;
  _jpeg_idct_islow(&cinfo, &component, c->coeff, refrows, c->outcol);
}

static void
call_islow_variant(void *context)
{
  idct_call *c = (idct_call *)context;
  c->fn(c->sample->quant, c->sample->coeff,
        c->output, c->sample->outcol);
}


static uint32_t
digest_idct_rows(JSAMPARRAY rows, JDIMENSION column)
{
  uint32_t hash = 2166136261u;
  unsigned row;
  for (row = 0; row < 8; row++)
    hash = ps2_bench_fnv(rows[row] + column, 8, hash);
  return hash;
}

static uint32_t
digest_idct_scalar(void *context)
{
  idct_case *ctx = (idct_case *)context;
  return digest_idct_rows(refrows, ctx->outcol);
}

static uint32_t
digest_idct_variant(void *context)
{
  idct_call *ctx = (idct_call *)context;
  return digest_idct_rows(ctx->output, ctx->sample->outcol);
}

static void
reset_idct_scalar(void *unused)
{
  unsigned i;
  (void)unused;
  for (i = 0; i < 8; i++)
    memset(refrows[i], 0xc9, DCT_ROW_BYTES);
}

static void
reset_idct_variant(void *context)
{
  idct_call *ctx = (idct_call *)context;
  unsigned i;
  for (i = 0; i < 8; i++)
    memset(ctx->output[i], 0xc9, DCT_ROW_BYTES);
}

static int
check_islow_output(JSAMPARRAY actual)
{
  unsigned row, i;
  for (row = 0; row < 8; row++)
    for (i = 0; i < DCT_ROW_BYTES; i++)
      if (actual[row][i] != refrows[row][i])
        return 1;
  return 0;
}

int
ps2_bench_run_idct(void)
{
  const ps2_idct_fn kernels[6] = {
    ps2_bench_idct_evenoff, ps2_bench_idct_evenon,
    ps2_bench_idct_batch, ps2_bench_idct_direct,
    ps2_bench_idct_lut, ps2_bench_idct_lut_norow
  };
  const char *names[6] = { "evenoff", "evenon", "batch", "direct",
                            "lut", "lut_norow" };
  const char *workloads[4] = { "dc_only", "sparse", "dense", "even_stress" };
  int valid[6] = { 1, 1, 1, 1, 1, 1 };
  int failures = 0;
#ifdef PS2_APPROX_FPU_IDCT
  unsigned fpu_max_diff = 0, fpu_differences = 0;
  int fpu_guard_ok = 1;
#endif
  unsigned profile, iteration, alignment, v;

  initialize_idct();

  for (profile = 0; profile < 4; profile++)
    for (iteration = 0; iteration < 128; iteration++)
      for (alignment = 0; alignment < 4; alignment++) {
        idct_case ctx;
        prepare_case(&ctx, profile, iteration, (int)alignment);
        call_islow_scalar(&ctx);
#ifdef PS2_APPROX_FPU_IDCT
        {
          unsigned row, byte;
          prepare_fpu_quant(&ctx);
          reset_fpu_approx(NULL);
          call_fpu_approx(&ctx);
          for (row = 0; row < 8; row++)
            for (byte = 0; byte < DCT_ROW_BYTES; byte++) {
              unsigned delta, x = fpu_rows[row][byte], y = refrows[row][byte];
              if (byte < ctx.outcol || byte >= ctx.outcol + 8) {
                if (x != 0xc9)
                  fpu_guard_ok = 0;
                continue;
              }
              delta = x > y ? x - y : y - x;
              if (delta > fpu_max_diff)
                fpu_max_diff = delta;
              if (delta)
                fpu_differences++;
            }
        }
#endif
        for (v = 0; v < 6; v++) {
          idct_call call;
          if (!valid[v])
            continue;
          call.fn = kernels[v];
          call.sample = &ctx;
          call.output = outrows[v];
          call_islow_variant(&call);
          if (check_islow_output(call.output)) {
            printf("FAIL,idct,%s,%s,iter=%u,alignment=%u\n",
                   names[v], workloads[profile], iteration, alignment);
            failures++;
            valid[v] = 0;
          }
        }
      }

  for (v = 0; v < 6; v++)
    if (valid[v])
      printf("PASS,idct,%s,correctness,2048_cases\n", names[v]);
#ifdef PS2_APPROX_FPU_IDCT
  printf("APPROX,idct,fpu_approx,2048_cases,max_abs_diff=%u,different_pixels=%u,guards=%s\n",
         fpu_max_diff, fpu_differences, fpu_guard_ok ? "PASS" : "FAIL");
  /* Do not label this bit-exact.  More than 3 levels or an out-of-bounds
   * write means that the candidate cannot be included in the benchmark. */
  if (!fpu_guard_ok || fpu_max_diff > 3)
    puts("QUALITY_OMIT,idct,fpu_approx,approx_quality_gate");
#endif

  for (profile = 0; profile < 4; profile++)
    for (alignment = 0; alignment < 4; alignment++) {
      idct_case ctx;
      idct_call contexts[6];
      ps2_bench_variant entries[8];
      unsigned n = 0;
      prepare_case(&ctx, profile, 117, (int)alignment);
#ifdef PS2_APPROX_FPU_IDCT
      prepare_fpu_quant(&ctx);
#endif

      entries[n].name = "ijg_c";
      entries[n].run = call_islow_scalar;
      entries[n].context = &ctx;
      entries[n].digest = digest_idct_scalar;
      entries[n].reset = reset_idct_scalar;
      n++;

      for (v = 0; v < 6; v++) {
        if (!valid[v])
          continue;
        contexts[v].fn = kernels[v];
        contexts[v].sample = &ctx;
        contexts[v].output = outrows[v];
        entries[n].name = names[v];
        entries[n].run = call_islow_variant;
        entries[n].context = &contexts[v];
        entries[n].digest = digest_idct_variant;
        entries[n].reset = reset_idct_variant;
        n++;
      }

#ifdef PS2_APPROX_FPU_IDCT
      if (fpu_guard_ok && fpu_max_diff <= 3) {
        entries[n].name = "fpu_approx";
        entries[n].run = call_fpu_approx;
        entries[n].context = &ctx;
        entries[n].digest = digest_fpu_approx;
        entries[n].reset = reset_fpu_approx;
        n++;
        if (ps2_bench_compare_approx_last("idct", workloads[profile], 8,
                                          (int)alignment, entries, n,
                                          BENCH_COUNT))
          failures++;
      } else
#endif
      if (n < 2 ||
          ps2_bench_compare("idct", workloads[profile], 8,
                            (int)alignment, entries, n, BENCH_COUNT))
        failures++;
    }
  return failures ? 1 : 0;
}
