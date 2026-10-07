/*
 * Downsampling (SPARC VIS 1)
 *
 * VIS1 has no arbitrary byte shuffle.  The arithmetic is fully vectorized,
 * and a small integer-register compaction step collects the useful even lanes.
 * This keeps the implementation VIS1-only while avoiding scalar arithmetic in
 * the hot averaging path.
 */

#include "../jsimdint.h"
#include <stdint.h>
#include <visintrin.h>


typedef union {
  __v4qi v;
  uint32_t u;
} vis_u32;


/* Pack byte lanes 0 and 2 from a four-byte VIS result into one uint16_t.
 * SPARC is big-endian, so the shifts below preserve memory order.
 */
static inline uint16_t
vis_pack_even_bytes(__v4qi v)
{
  vis_u32 x;
  uint32_t u;

  x.v = v;
  u = x.u;
  return (uint16_t)(((u >> 16) & 0xff00U) | ((u >> 8) & 0x00ffU));
}


static inline void
vis_h2v1_block(const JSAMPLE *src, JSAMPLE *dst)
{
  const __v4hi bias = { 0, 0, 16, 0 };
  __v4hi x0, x1, x2, x3;
  __v4hi n0, n1, n2, n3;
  __v4qi p0, p1, p2, p3;
  uint16_t q0, q1, q2, q3;

  x0 = __vis_fexpand(*(const __v4qi *)(const void *)(src + 0));
  x1 = __vis_fexpand(*(const __v4qi *)(const void *)(src + 4));
  x2 = __vis_fexpand(*(const __v4qi *)(const void *)(src + 8));
  x3 = __vis_fexpand(*(const __v4qi *)(const void *)(src + 12));

  n0 = __vis_faligndatav4hi(x0, x0);
  n1 = __vis_faligndatav4hi(x1, x1);
  n2 = __vis_faligndatav4hi(x2, x2);
  n3 = __vis_faligndatav4hi(x3, x3);

  p0 = __vis_fpack16(__vis_fpadd16(__vis_fpadd16(x0, n0), bias));
  p1 = __vis_fpack16(__vis_fpadd16(__vis_fpadd16(x1, n1), bias));
  p2 = __vis_fpack16(__vis_fpadd16(__vis_fpadd16(x2, n2), bias));
  p3 = __vis_fpack16(__vis_fpadd16(__vis_fpadd16(x3, n3), bias));

  q0 = vis_pack_even_bytes(p0);
  q1 = vis_pack_even_bytes(p1);
  q2 = vis_pack_even_bytes(p2);
  q3 = vis_pack_even_bytes(p3);

  *(uint32_t *)(void *)(dst + 0) = ((uint32_t)q0 << 16) | q1;
  *(uint32_t *)(void *)(dst + 4) = ((uint32_t)q2 << 16) | q3;
}


static inline void
vis_h2v2_block(const JSAMPLE *src0, const JSAMPLE *src1, JSAMPLE *dst)
{
  const __v4hi bias = { 16, 0, 32, 0 };
  __v4hi a0, a1, a2, a3, b0, b1, b2, b3;
  __v4hi an0, an1, an2, an3, bn0, bn1, bn2, bn3;
  __v4qi p0, p1, p2, p3;
  uint16_t q0, q1, q2, q3;

  a0 = __vis_fexpand(*(const __v4qi *)(const void *)(src0 + 0));
  a1 = __vis_fexpand(*(const __v4qi *)(const void *)(src0 + 4));
  a2 = __vis_fexpand(*(const __v4qi *)(const void *)(src0 + 8));
  a3 = __vis_fexpand(*(const __v4qi *)(const void *)(src0 + 12));
  b0 = __vis_fexpand(*(const __v4qi *)(const void *)(src1 + 0));
  b1 = __vis_fexpand(*(const __v4qi *)(const void *)(src1 + 4));
  b2 = __vis_fexpand(*(const __v4qi *)(const void *)(src1 + 8));
  b3 = __vis_fexpand(*(const __v4qi *)(const void *)(src1 + 12));

  an0 = __vis_faligndatav4hi(a0, a0);
  an1 = __vis_faligndatav4hi(a1, a1);
  an2 = __vis_faligndatav4hi(a2, a2);
  an3 = __vis_faligndatav4hi(a3, a3);
  bn0 = __vis_faligndatav4hi(b0, b0);
  bn1 = __vis_faligndatav4hi(b1, b1);
  bn2 = __vis_faligndatav4hi(b2, b2);
  bn3 = __vis_faligndatav4hi(b3, b3);

#define H2V2_PACK(A, AN, B, BN) \
  __vis_fpack16(__vis_fpadd16( \
    __vis_fpadd16(__vis_fpadd16((A), (AN)), \
                  __vis_fpadd16((B), (BN))), bias))

  p0 = H2V2_PACK(a0, an0, b0, bn0);
  p1 = H2V2_PACK(a1, an1, b1, bn1);
  p2 = H2V2_PACK(a2, an2, b2, bn2);
  p3 = H2V2_PACK(a3, an3, b3, bn3);
#undef H2V2_PACK

  q0 = vis_pack_even_bytes(p0);
  q1 = vis_pack_even_bytes(p1);
  q2 = vis_pack_even_bytes(p2);
  q3 = vis_pack_even_bytes(p3);

  *(uint32_t *)(void *)(dst + 0) = ((uint32_t)q0 << 16) | q1;
  *(uint32_t *)(void *)(dst + 4) = ((uint32_t)q2 << 16) | q3;
}


HIDDEN void
jsimd_h2v1_downsample_vis(JDIMENSION image_width, int max_v_samp_factor,
                          JDIMENSION v_samp_factor,
                          JDIMENSION width_in_blocks, JSAMPARRAY input_data,
                          JSAMPARRAY output_data)
{
  JDIMENSION output_cols = width_in_blocks * DCTSIZE;
  JDIMENSION row;

  /* fpack16: (fixed16 << scale) >> 7.  Inputs are sample*16, so scale=2
   * implements the scalar divide-by-two.  align=2 selects the next 16-bit lane.
   */
  __builtin_vis_write_gsr((2 << 3) | 2);

  for (row = 0; row < v_samp_factor; row++) {
    JSAMPROW src = input_data[row];
    JSAMPROW dst = output_data[row];
    JDIMENSION outcol = 0;
    JDIMENSION incol = 0;

    if (((JUINTPTR)src & 3) == 0 && ((JUINTPTR)dst & 3) == 0) {
      for (; outcol + 8 <= output_cols && incol + 16 <= image_width;
           outcol += 8, incol += 16)
        vis_h2v1_block(src + incol, dst + outcol);
    }

    for (; outcol < output_cols; outcol++, incol += 2) {
      JDIMENSION i0 = incol < image_width ? incol : image_width - 1;
      JDIMENSION i1 = incol + 1 < image_width ? incol + 1 : image_width - 1;
      int bias = outcol & 1;

      dst[outcol] = (JSAMPLE)((GETJSAMPLE(src[i0]) +
                               GETJSAMPLE(src[i1]) + bias) >> 1);
    }
  }

  (void)max_v_samp_factor;
}


HIDDEN void
jsimd_h2v2_downsample_vis(JDIMENSION image_width, int max_v_samp_factor,
                          JDIMENSION v_samp_factor,
                          JDIMENSION width_in_blocks, JSAMPARRAY input_data,
                          JSAMPARRAY output_data)
{
  JDIMENSION output_cols = width_in_blocks * DCTSIZE;
  JDIMENSION outrow;
  JDIMENSION inrow = 0;

  /* scale=1 implements divide-by-four for values represented as sample*16. */
  __builtin_vis_write_gsr((1 << 3) | 2);

  for (outrow = 0; outrow < v_samp_factor; outrow++, inrow += 2) {
    JSAMPROW src0 = input_data[inrow];
    JSAMPROW src1 = input_data[inrow + 1];
    JSAMPROW dst = output_data[outrow];
    JDIMENSION outcol = 0;
    JDIMENSION incol = 0;

    if (((JUINTPTR)src0 & 3) == 0 && ((JUINTPTR)src1 & 3) == 0 &&
        ((JUINTPTR)dst & 3) == 0) {
      for (; outcol + 8 <= output_cols && incol + 16 <= image_width;
           outcol += 8, incol += 16)
        vis_h2v2_block(src0 + incol, src1 + incol, dst + outcol);
    }

    for (; outcol < output_cols; outcol++, incol += 2) {
      JDIMENSION i0 = incol < image_width ? incol : image_width - 1;
      JDIMENSION i1 = incol + 1 < image_width ? incol + 1 : image_width - 1;
      int bias = (outcol & 1) ? 2 : 1;

      dst[outcol] =
        (JSAMPLE)((GETJSAMPLE(src0[i0]) + GETJSAMPLE(src0[i1]) +
                   GETJSAMPLE(src1[i0]) + GETJSAMPLE(src1[i1]) + bias) >> 2);
    }
  }

  (void)max_v_samp_factor;
}
