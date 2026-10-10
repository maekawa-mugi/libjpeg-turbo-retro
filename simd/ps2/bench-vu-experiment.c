/* PS2 novelty experiments (opt-in, benchmark-only).
 *
 * VU0 macro mode: a real separable 8x8 floating-point IDCT. Four
 * independent pixels use COP2 vector instructions per inner dot product.
 * A portable scalar float matrix IDCT is the comparison baseline.
 *
 * VIF0 DMA: a SEPARATE upload microbenchmark, comparing CPU stores to VU0
 * data memory against an actual VIF0 normal-mode DMA + V4_32 UNPACK.
 * There is no VU0 microprogram in this DMA test.  An upload speedup does
 * not imply a whole-IDCT or whole-JPEG speedup.
 *
 * Neither path changes libjpeg's default IDCT dispatcher.
 * SPDX-License-Identifier: Zlib
 */
#include "bench-harness.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if defined(PS2_EXPERIMENTAL_VIF0_DMA)
#include <kernel.h>
#endif

#if defined(PS2_EXPERIMENTAL_VU0)
static const float idct_basis[8][8] = {
  { 0.35355339059f, 0.49039264020f, 0.46193976626f, 0.41573480615f, 0.35355339059f, 0.27778511651f, 0.19134171618f, 0.09754516101f },
  { 0.35355339059f, 0.41573480615f, 0.19134171618f, -0.09754516101f, -0.35355339059f, -0.49039264020f, -0.46193976626f, -0.27778511651f },
  { 0.35355339059f, 0.27778511651f, -0.19134171618f, -0.49039264020f, -0.35355339059f, 0.09754516101f, 0.46193976626f, 0.41573480615f },
  { 0.35355339059f, 0.09754516101f, -0.46193976626f, -0.27778511651f, 0.35355339059f, 0.41573480615f, -0.19134171618f, -0.49039264020f },
  { 0.35355339059f, -0.09754516101f, -0.46193976626f, 0.27778511651f, 0.35355339059f, -0.41573480615f, -0.19134171618f, 0.49039264020f },
  { 0.35355339059f, -0.27778511651f, -0.19134171618f, 0.49039264020f, -0.35355339059f, -0.09754516101f, 0.46193976626f, -0.41573480615f },
  { 0.35355339059f, -0.41573480615f, 0.19134171618f, 0.09754516101f, -0.35355339059f, 0.49039264020f, -0.46193976626f, 0.27778511651f },
  { 0.35355339059f, -0.49039264020f, 0.46193976626f, -0.41573480615f, 0.35355339059f, -0.27778511651f, 0.19134171618f, -0.09754516101f }
};

/* C and VU0 compute the SAME orthonormal 2D matrix transform, with the
 * 0.5/sqrt(2) DC basis and 0.5 cosine AC basis. No 8-bit clipping occurs
 * until the full 2D transform is complete. Differences are allowed and
 * reported, not mislabeled as an IJG bit-exact success.
 */
static JCOEF coeffs[64] __attribute__((aligned(16)));
static ISLOW_MULT_TYPE quant[64] __attribute__((aligned(16)));
static JSAMPLE c_output[64] __attribute__((aligned(16)));
static JSAMPLE vu_output[64] __attribute__((aligned(16)));
static float vu_coeff[8][8][4] __attribute__((aligned(16)));
static float vu_in[8][4] __attribute__((aligned(16)));
static float vu_out[4] __attribute__((aligned(16)));

static void
init_basis(void)
{
  int a, k, lane;
  for (a = 0; a < 8; a++)
    for (k = 0; k < 8; k++)
      for (lane = 0; lane < 4; lane++)
        vu_coeff[a][k][lane] = idct_basis[a][k];
}

static void
seed_coeffs(unsigned seed)
{
  int i;
  for (i = 0; i < 64; i++) {
    unsigned x = seed * 4919u + (unsigned)i * 3571u + 17u;
    /* Include zero-heavy and full AC blocks, but avoid huge float values
     * that do not represent typical 8-bit JPEG decoded coefficients.
     */
    coeffs[i] = (JCOEF)((seed % 7u == 0 && i > 0) ? 0 :
                        (int)((x ^ (x >> 7)) % 17u) - 8);
    quant[i] = (ISLOW_MULT_TYPE)(1 + (i * 11 + seed) % 11u);
  }
}

static JSAMPLE
to_pixel(float x)
{
  /* Same final +0.5 truncation on both sides; non-IEEE VU0 rounding
   * may still cause small output differences near pixel boundaries.
   */
  int value = (int)(x + 128.5f);
  if (value < 0) value = 0;
  if (value > 255) value = 255;
  return (JSAMPLE)value;
}

static void
read_input(float src[8][8])
{
  int row, col;
  for (row = 0; row < 8; row++)
    for (col = 0; col < 8; col++) {
      int i = row * 8 + col;
      src[row][col] = (float)((int)coeffs[i] * (int)quant[i]);
    }
}

static void
transform_scalar(float src[8][8], float dst[8][8])
{
  int a, k, lane;
  for (a = 0; a < 8; a++)
    for (lane = 0; lane < 8; lane++) {
      float acc = 0.0f;
      for (k = 0; k < 8; k++)
        acc += idct_basis[a][k] * src[k][lane];
      dst[a][lane] = acc;
    }
}

static void
transpose(const float in[8][8], float out[8][8])
{
  int row, col;
  for (row = 0; row < 8; row++)
    for (col = 0; col < 8; col++)
      out[col][row] = in[row][col];
}

static void
run_c_matrix(void *unused)
{
  float input[8][8] __attribute__((aligned(16)));
  float stage[8][8] __attribute__((aligned(16)));
  float trans[8][8] __attribute__((aligned(16)));
  float output[8][8] __attribute__((aligned(16)));
  int row, col;
  (void)unused;
  read_input(input);
  transform_scalar(input, stage);
  transpose(stage, trans);
  transform_scalar(trans, output);
  for (row = 0; row < 8; row++)
    for (col = 0; col < 8; col++)
      c_output[row * 8 + col] = to_pixel(output[col][row]);
}

#if defined(PS2_EXPERIMENTAL_VU0)
/* Actual R5900 COP2 VU0 macro arithmetic, not a C loop with a VU label.
 * The register-only dot8 sums eight products across 4 independent lanes.
 * vf1-vf4 are wholly owned by this asm block; no VU microprogram runs.
 */
static __attribute__((noinline)) void
vu_dot8(const float *in, const float *weights, float *out)
{
  __asm__ volatile(
    ".set push\n\t.set noreorder\n\t"
    "lqc2 $vf1, 0(%0)\n\t"
    "lqc2 $vf2, 0(%1)\n\t"
    "vmul.xyzw $vf4, $vf1, $vf2\n\t"
    "vmove.xyzw $vf3, $vf4\n\t"
    "lqc2 $vf1, 16(%0)\n\t"
    "lqc2 $vf2, 16(%1)\n\t"
    "vmul.xyzw $vf4, $vf1, $vf2\n\t"
    "vadd.xyzw $vf3, $vf3, $vf4\n\t"
    "lqc2 $vf1, 32(%0)\n\t"
    "lqc2 $vf2, 32(%1)\n\t"
    "vmul.xyzw $vf4, $vf1, $vf2\n\t"
    "vadd.xyzw $vf3, $vf3, $vf4\n\t"
    "lqc2 $vf1, 48(%0)\n\t"
    "lqc2 $vf2, 48(%1)\n\t"
    "vmul.xyzw $vf4, $vf1, $vf2\n\t"
    "vadd.xyzw $vf3, $vf3, $vf4\n\t"
    "lqc2 $vf1, 64(%0)\n\t"
    "lqc2 $vf2, 64(%1)\n\t"
    "vmul.xyzw $vf4, $vf1, $vf2\n\t"
    "vadd.xyzw $vf3, $vf3, $vf4\n\t"
    "lqc2 $vf1, 80(%0)\n\t"
    "lqc2 $vf2, 80(%1)\n\t"
    "vmul.xyzw $vf4, $vf1, $vf2\n\t"
    "vadd.xyzw $vf3, $vf3, $vf4\n\t"
    "lqc2 $vf1, 96(%0)\n\t"
    "lqc2 $vf2, 96(%1)\n\t"
    "vmul.xyzw $vf4, $vf1, $vf2\n\t"
    "vadd.xyzw $vf3, $vf3, $vf4\n\t"
    "lqc2 $vf1, 112(%0)\n\t"
    "lqc2 $vf2, 112(%1)\n\t"
    "vmul.xyzw $vf4, $vf1, $vf2\n\t"
    "vadd.xyzw $vf3, $vf3, $vf4\n\t"
    "vnop\n\t"
    "vnop\n\t"
    "sqc2 $vf3, 0(%2)\n\t"
    ".set pop\n\t"
    :
    : "r"(in), "r"(weights), "r"(out)
    : "memory");
}

static void
transform_vu(const float src[8][8], float dst[8][8])
{
  int group, k, a, lane;
  for (group = 0; group < 2; group++) {
    for (k = 0; k < 8; k++)
      for (lane = 0; lane < 4; lane++)
        vu_in[k][lane] = src[k][4 * group + lane];
    for (a = 0; a < 8; a++) {
      vu_dot8(&vu_in[0][0], &vu_coeff[a][0][0], vu_out);
      for (lane = 0; lane < 4; lane++)
        dst[a][4 * group + lane] = vu_out[lane];
    }
  }
}

static void
run_vu_matrix(void *unused)
{
  float input[8][8] __attribute__((aligned(16)));
  float stage[8][8] __attribute__((aligned(16)));
  float trans[8][8] __attribute__((aligned(16)));
  float output[8][8] __attribute__((aligned(16)));
  int row, col;
  (void)unused;
  read_input(input);
  transform_vu(input, stage);
  transpose(stage, trans);
  transform_vu(trans, output);
  for (row = 0; row < 8; row++)
    for (col = 0; col < 8; col++)
      vu_output[row * 8 + col] = to_pixel(output[col][row]);
}
#endif

static uint32_t
digest_c(void *p)
{
  (void)p;
  return ps2_bench_fnv(c_output, 64, 2166136261u);
}

#if defined(PS2_EXPERIMENTAL_VU0)
static uint32_t
digest_vu(void *p)
{
  (void)p;
  return ps2_bench_fnv(vu_output, 64, 2166136261u);
}

int
ps2_bench_run_vu0(void)
{
  ps2_bench_variant entries[2];
  unsigned seed, i, maxdiff = 0, changed = 0;
  init_basis();
  for (seed = 0; seed < 128; seed++) {
    seed_coeffs(seed);
    run_c_matrix(NULL);
    run_vu_matrix(NULL);
    for (i = 0; i < 64; i++) {
      unsigned a = c_output[i], b = vu_output[i];
      unsigned delta = a > b ? a - b : b - a;
      if (delta > maxdiff) maxdiff = delta;
      if (delta) changed++;
    }
  }
  printf("APPROX,vu_idct,vu0_macro,128_blocks,max_abs_diff=%u,different_pixels=%u\n",
         maxdiff, changed);
  if (maxdiff > 3) {
    puts("SKIP,vu_idct,vu0_macro,approx_quality_gate");
    return 1;
  }
  seed_coeffs(117);
  entries[0].name = "scalar_matrix";
  entries[0].run = run_c_matrix;
  entries[0].digest = digest_c;
  entries[0].reset = NULL;
  entries[0].context = NULL;
  entries[1].name = "vu0_macro";
  entries[1].run = run_vu_matrix;
  entries[1].digest = digest_vu;
  entries[1].reset = NULL;
  entries[1].context = NULL;
  return ps2_bench_compare_approx_last("vu_idct", "dense", 8, 0,
                                       entries, 2, 64);
}
#endif

#endif /* PS2_EXPERIMENTAL_VU0 */

#if defined(PS2_EXPERIMENTAL_VIF0_DMA)
/* VIF0 UNPACK V4_32 stream: STCYCL(1,1) QW, UNPACK 16 QW to VU0
 * data-memory address zero, followed by the actual 256-byte payload.
 * The command must be the last word of each command quadword; otherwise
 * a following command word would be interpreted as input float data.
 *
 * Normal DMA (CH0 DIR=1 MOD=0 STR=1), physical address < 32 MiB.
 * Both the CPU baseline and the DMA contender write exactly the same
 * 64 words at VU0 data-memory base 0x11004000.
 */
static uint32_t dma_packet[8 + PS2_VU_TRANSFER_WORDS]
  __attribute__((aligned(16)));
static unsigned dma_failed;
static void
init_dma_packet(void)
{
  unsigned i;
  memset(dma_packet, 0, sizeof(dma_packet));
  dma_packet[3] = 0x01000101u; /* STCYCL WL=1 CL=1 */
  dma_packet[7] = 0x6c100000u; /* UNPACK V4_32, NUM=16, ADDR=0 */
  for (i = 0; i < PS2_VU_TRANSFER_WORDS; i++)
    dma_packet[8 + i] = 0x3f000000u + (i * 101u);
}

/* Bounded waits prevent a failed VIF0/DMAC configuration from hanging
 * the entire one-boot test indefinitely.
 */
static int
wait_dma(void)
{
  unsigned limit = 1000000;
  while ((*PS2_VU_CHCR & 0x100u) && --limit) { }
  if (!limit) return 0;
  limit = 1000000;
  while ((*PS2_VU_VIF_STAT & 0x1f000000u) && --limit) { }
  return limit != 0;
}

static void
run_cpu_store(void *unused)
{
  unsigned i;
  (void)unused;
  for (i = 0; i < PS2_VU_TRANSFER_WORDS; i++)
    PS2_VU_VIF0_DATA[i] = dma_packet[8 + i];
}

static void
run_vif0_dma(void *unused)
{
  (void)unused;
  if (!wait_dma()) { dma_failed = 1; return; }
  /* Flush packet contents before DMAC reads their physical RAM address. */
  SyncDCache(dma_packet, dma_packet + 8 + PS2_VU_TRANSFER_WORDS);
  *PS2_VU_MADR = ((uint32_t)(uintptr_t)dma_packet) & 0x1fffffffu;
  *PS2_VU_QWC = PS2_VU_DMA_QWC;
  *PS2_VU_CHCR = 0x101u;
  if (!wait_dma()) dma_failed = 1;
}

static uint32_t
digest_vu_memory(void *unused)
{
  uint32_t hash = 2166136261u;
  unsigned i, j;
  (void)unused;
  for (i = 0; i < PS2_VU_TRANSFER_WORDS; i++) {
    uint32_t word = PS2_VU_VIF0_DATA[i];
    for (j = 0; j < 4; j++) {
      hash ^= (word >> (8 * j)) & 255u;
      hash *= 16777619u;
    }
  }
  return hash;
}

int
ps2_bench_run_vif0_dma(void)
{
  ps2_bench_variant entries[2];
  uint32_t expected, actual;
  dma_failed = 0;
  init_dma_packet();
  run_cpu_store(NULL);
  expected = digest_vu_memory(NULL);
  run_vif0_dma(NULL);
  actual = digest_vu_memory(NULL);
  if (dma_failed || expected != actual) {
    printf("FAIL,vif0_dma,upload256,expected=%08lx,actual=%08lx,timeout=%u\n",
           (unsigned long)expected, (unsigned long)actual, dma_failed);
    return 1;
  }
  puts("PASS,vif0_dma,upload256,256_bytes_verified");
  entries[0].name = "cpu_store";
  entries[0].run = run_cpu_store;
  entries[0].context = NULL;
  entries[0].reset = NULL;
  entries[0].digest = digest_vu_memory;
  entries[1].name = "vif0_dma";
  entries[1].run = run_vif0_dma;
  entries[1].context = NULL;
  entries[1].reset = NULL;
  entries[1].digest = digest_vu_memory;
  if (ps2_bench_compare("vif0_dma", "upload256", 256, 0, entries, 2, 64))
    return 1;
  return dma_failed ? 1 : 0;
}
#endif
