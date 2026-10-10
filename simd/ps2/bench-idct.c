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
static JSAMPLE outmem[3][8][DCT_ROW_BYTES] __attribute__((aligned(16)));
static JSAMPROW refrows[8], outrows[3][8];

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
  for (i = 0; i < 8; i++) {
    refrows[i] = refmem[i];
    outrows[0][i] = outmem[0][i];
    outrows[1][i] = outmem[1][i];
    outrows[2][i] = outmem[2][i];
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
  const ps2_idct_fn kernels[3] = {
    ps2_bench_idct_evenoff, ps2_bench_idct_evenon, ps2_bench_idct_batch
  };
  const char *names[3] = { "evenoff", "evenon", "batch" };
  const char *workloads[4] = { "dc_only", "sparse", "dense", "even_stress" };
  int valid[3] = { 1, 1, 1 };
  int failures = 0;
  unsigned profile, iteration, alignment, v;

  initialize_idct();

  for (profile = 0; profile < 4; profile++)
    for (iteration = 0; iteration < 128; iteration++)
      for (alignment = 0; alignment < 4; alignment++) {
        idct_case ctx;
        prepare_case(&ctx, profile, iteration, (int)alignment);
        call_islow_scalar(&ctx);
        for (v = 0; v < 3; v++) {
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

  for (v = 0; v < 3; v++)
    if (valid[v])
      printf("PASS,idct,%s,correctness,2048_cases\n", names[v]);

  for (profile = 0; profile < 4; profile++)
    for (alignment = 0; alignment < 4; alignment++) {
      idct_case ctx;
      idct_call contexts[3];
      ps2_bench_variant entries[4];
      unsigned n = 0;
      prepare_case(&ctx, profile, 117, (int)alignment);

      entries[n].name = "ijg_c";
      entries[n].run = call_islow_scalar;
      entries[n].context = &ctx;
      entries[n].digest = digest_idct_scalar;
      entries[n].reset = reset_idct_scalar;
      n++;

      for (v = 0; v < 3; v++) {
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

      if (n < 2 ||
          ps2_bench_compare("idct", workloads[profile], 8,
                            (int)alignment, entries, n, BENCH_COUNT))
        failures++;
    }
  return failures ? 1 : 0;
}
