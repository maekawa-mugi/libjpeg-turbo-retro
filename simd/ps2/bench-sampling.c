/* Benchmark existing MMI plain/fancy upsampling and compressor downsampling
 * against independent portable C formulas.  Correctness is checked before
 * each category is timed.  No kernel options need rebuilding on the console.
 * SPDX-License-Identifier: Zlib
 */
#include "bench-harness.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define NROWS 4
#define IN_CAP 352
#define OUT_CAP 352
#define SAMPLES 192
static JSAMPLE input_storage[NROWS + 2][IN_CAP] __attribute__((aligned(16)));
static JSAMPLE mmi_storage[NROWS][OUT_CAP] __attribute__((aligned(16)));
static JSAMPLE ref_storage[NROWS][OUT_CAP] __attribute__((aligned(16)));
static JSAMPROW inrows[NROWS + 2], mmirows[NROWS], refrows[NROWS];

typedef struct {
  int kind;  /* 0 plain, 1 fancy, 2 compressor downsample */
  int vertical;
  JDIMENSION width;  /* output width for plain, chroma width for fancy */
  JDIMENSION image_width; /* downsample input width */
  JSAMPARRAY dest;
} sampling_ctx;

static JSAMPLE *
sample_offset(JSAMPLE *ptr, unsigned offset)
{
  return (JSAMPLE *)((((uintptr_t)ptr + 15u) & ~(uintptr_t)15u) + offset);
}

static void
setup_sampling(unsigned alignment, unsigned seed)
{
  int row;
  unsigned col;
  for (row = 0; row < NROWS + 2; row++) {
    inrows[row] = sample_offset(input_storage[row], alignment);
    for (col = 0; col < IN_CAP - 32; col++)
      inrows[row][col] = (JSAMPLE)((seed * 31u + col * 37u +
                                      (unsigned)row * 53u) & 255u);
  }
  for (row = 0; row < NROWS; row++) {
    mmirows[row] = sample_offset(mmi_storage[row], alignment);
    refrows[row] = sample_offset(ref_storage[row], alignment);
    memset(mmirows[row], 0xa5, OUT_CAP - 32);
    memset(refrows[row], 0xa5, OUT_CAP - 32);
  }
}

static void
sample_reference(void *opaque)
{
  sampling_ctx *ctx = (sampling_ctx *)opaque;
  JSAMPARRAY input = inrows + (ctx->kind == 2 ? 0 : 1);
  JSAMPARRAY output = ctx->dest;
  unsigned row, col;
  if (ctx->kind == 0) {
    for (row = 0; row < NROWS; row++) {
      unsigned source_row = ctx->vertical == 2 ? row / 2 : row;
      for (col = 0; col < ctx->width; col++)
        output[row][col] = input[source_row][col / 2];
    }
  } else if (ctx->kind == 1) {
    for (row = 0; row < NROWS; row++) {
      int source_row = ctx->vertical == 2 ? (int)(row / 2) : (int)row;
      const JSAMPLE *near = input[source_row];
      const JSAMPLE *far = ctx->vertical == 2 ?
        input[source_row + ((row & 1u) ? 1 : -1)] : near;
      for (col = 0; col < ctx->width; col++) {
        unsigned left = col ? col - 1 : col;
        unsigned right = col + 1 < ctx->width ? col + 1 : col;
        if (ctx->vertical == 1) {
          output[row][2 * col] =
            (JSAMPLE)((3 * near[col] + near[left] + 1) >> 2);
          output[row][2 * col + 1] =
            (JSAMPLE)((3 * near[col] + near[right] + 2) >> 2);
        } else {
          unsigned a = 3 * near[col] + far[col];
          unsigned l = 3 * near[left] + far[left];
          unsigned r = 3 * near[right] + far[right];
          output[row][2 * col] = (JSAMPLE)((3 * a + l + 8) >> 4);
          output[row][2 * col + 1] = (JSAMPLE)((3 * a + r + 7) >> 4);
        }
      }
    }
  } else {
    for (row = 0; row < 2; row++) {
      const JSAMPLE *a = input[ctx->vertical == 2 ? 2 * row : row];
      const JSAMPLE *b = input[2 * row + 1];
      for (col = 0; col < 128; col++) {
        unsigned x = 2 * col, x1 = x + 1;
        unsigned limit = (unsigned)ctx->image_width - 1;
        unsigned v0 = a[x <= limit ? x : limit];
        unsigned v1 = a[x1 <= limit ? x1 : limit];
        unsigned total = v0 + v1;
        if (ctx->vertical == 2)
          total += b[x <= limit ? x : limit] +
                   b[x1 <= limit ? x1 : limit] +
                   1u + (col & 1u);
        else
          total += col & 1u;
        output[row][col] =
          (JSAMPLE)(total >> (ctx->vertical == 2 ? 2 : 1));
      }
    }
  }
}

static void
sample_mmi(void *opaque)
{
  sampling_ctx *ctx = (sampling_ctx *)opaque;
  JSAMPARRAY output = ctx->dest;
  if (ctx->kind == 0) {
    JSAMPARRAY input = inrows + 1;
    if (ctx->vertical == 2)
      jsimd_h2v2_upsample_ps2mmi(NROWS, ctx->width, input, &output);
    else
      jsimd_h2v1_upsample_ps2mmi(NROWS, ctx->width, input, &output);
  } else if (ctx->kind == 1) {
    JSAMPARRAY input = inrows + 1;
    if (ctx->vertical == 2)
      jsimd_h2v2_fancy_upsample_ps2mmi(NROWS, ctx->width, input, &output);
    else
      jsimd_h2v1_fancy_upsample_ps2mmi(NROWS, ctx->width, input, &output);
  } else {
    JSAMPARRAY input = inrows;
    if (ctx->vertical == 2)
      jsimd_h2v2_downsample_ps2mmi(ctx->image_width, NROWS, 2, 16,
                                   input, output);
    else
      jsimd_h2v1_downsample_ps2mmi(ctx->image_width, NROWS, 2, 16,
                                   input, output);
  }
}


static uint32_t
digest_sampling(void *opaque)
{
  const sampling_ctx *ctx = (const sampling_ctx *)opaque;
  unsigned row, rows = ctx->kind == 2 ? 2u : NROWS;
  unsigned count = ctx->kind == 2 ? 128u :
                   ctx->kind == 1 ? 2u * (unsigned)ctx->width :
                   (unsigned)ctx->width;
  uint32_t hash = 2166136261u;
  for (row = 0; row < rows; row++)
    hash = ps2_bench_fnv(ctx->dest[row], count, hash);
  return hash;
}

static void
reset_sampling(void *opaque)
{
  sampling_ctx *ctx = (sampling_ctx *)opaque;
  unsigned row, rows = ctx->kind == 2 ? 2u : NROWS;
  unsigned bytes = ctx->kind == 2 ? 128u :
                   ctx->kind == 1 ? 2u * (unsigned)ctx->width :
                   (unsigned)ctx->width;
  for (row = 0; row < rows; row++)
    memset(ctx->dest[row], 0xa5, bytes + 16);
}

static int
compare_sampling(const sampling_ctx *ctx)
{
  unsigned rows = ctx->kind == 2 ? 2u : NROWS;
  unsigned bytes = ctx->kind == 2 ? 128u :
                   ctx->kind == 1 ? 2u * (unsigned)ctx->width :
                   (unsigned)ctx->width;
  unsigned row, i;
  for (row = 0; row < NROWS; row++) {
    for (i = 0; i < (row < rows ? bytes + 16 : 16); i++) {
      if (mmirows[row][i] != refrows[row][i])
        return 1;
    }
  }
  return 0;
}

int
ps2_bench_run_sampling(void)
{
  static const unsigned sizes[] = { 64, 65 };
  int kind, vertical, off, index, seed;
  int failures = 0;
  for (kind = 0; kind < 3; kind++)
    for (vertical = 1; vertical <= 2; vertical++)
      for (off = 0; off < 2; off++)
        for (index = 0; index < 2; index++)
          for (seed = 0; seed < 2; seed++) {
            sampling_ctx ref, mmi;
            unsigned width = kind == 0 ? (sizes[index] * 2u) :
                             sizes[index];
            setup_sampling((unsigned)off, (unsigned)seed);
            memset(&ref, 0, sizeof(ref));
            ref.kind = kind;
            ref.vertical = vertical;
            ref.width = (JDIMENSION)width;
            ref.image_width = (JDIMENSION)(255 + index);
            ref.dest = refrows;
            mmi = ref;
            mmi.dest = mmirows;
            sample_reference(&ref);
            sample_mmi(&mmi);
            if (compare_sampling(&mmi)) {
              printf("FAIL,sampling,kind=%d,h2v%d,off=%d,width=%u,seed=%d\n",
                     kind, vertical, off, width, seed);
              failures++;
            }
          }

  if (failures) {
    printf("SKIP,sampling,correctness_failed,%d\n", failures);
    return 1;
  }
  puts("PASS,sampling,plain_fancy_downsample,48_cases");

  for (kind = 0; kind < 3; kind++)
    for (vertical = 1; vertical <= 2; vertical++)
      for (off = 0; off < 2; off++)
        for (index = 0; index < 2; index++) {
          sampling_ctx ctx[2];
          ps2_bench_variant variants[2];
          unsigned width = kind == 0 ? sizes[index] * 2u : sizes[index];
          const char *names[] = { "plain_up", "fancy_up", "downsample" };
          const char *orientation = vertical == 1 ? "h2v1" : "h2v2";
          unsigned output_width = kind == 2 ? 255u + (unsigned)index : width;

          setup_sampling((unsigned)off, 1);
          memset(ctx, 0, sizeof(ctx));
          ctx[0].kind = kind;
          ctx[0].vertical = vertical;
          ctx[0].width = (JDIMENSION)width;
          ctx[0].image_width = (JDIMENSION)(255 + index);
          ctx[0].dest = refrows;
          ctx[1] = ctx[0];
          ctx[1].dest = mmirows;

          variants[0].name = "portable_c";
          variants[0].run = sample_reference;
          variants[0].context = &ctx[0];
          variants[0].digest = digest_sampling;
          variants[0].reset = reset_sampling;
          variants[1].name = "mmi";
          variants[1].run = sample_mmi;
          variants[1].context = &ctx[1];
          variants[1].digest = digest_sampling;
          variants[1].reset = reset_sampling;

          if (ps2_bench_compare(names[kind], orientation, output_width, off,
                                variants, 2, SAMPLES))
            failures++;
        }
  return failures ? 1 : 0;
}
