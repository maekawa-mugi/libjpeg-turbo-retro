/*
 * Plain upsampling (SPARC VIS 1)
 *
 * The implementation deliberately uses only VIS1 operations.  In particular,
 * it does not use VIS2 bshuffle, so the same object code can run on first
 * generation UltraSPARC-class VIS implementations.
 */

#include "../jsimdint.h"
#include <visintrin.h>


LOCAL(void)
h2v1_scalar(int max_v_samp_factor, JDIMENSION output_width,
            JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  int inrow;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    JSAMPROW inptr = input_data[inrow];
    JSAMPROW outptr = output_data[inrow];
    JDIMENSION colctr;

    for (colctr = 0; 2 * colctr < output_width; colctr++) {
      JSAMPLE sample = inptr[colctr];

      outptr[2 * colctr] = sample;
      outptr[2 * colctr + 1] = sample;
    }
  }
}


HIDDEN void
jsimd_h2v1_upsample_vis(int max_v_samp_factor, JDIMENSION output_width,
                        JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  int inrow;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    JSAMPROW inptr = input_data[inrow];
    JSAMPROW outptr = output_data[inrow];
    JDIMENSION colctr;

    /*
     * libjpeg-turbo's sample allocator aligns rows and pads them for 2x
     * upsampling.  Keep a scalar fallback for unusual external buffers, while
     * making the normal path completely branch-free inside the hot loop.
     */
    if (((JUINTPTR)inptr & 3) != 0 || ((JUINTPTR)outptr & 7) != 0) {
      JSAMPARRAY in = &input_data[inrow];
      JSAMPARRAY out = &output_data[inrow];

      h2v1_scalar(1, output_width, in, out);
      continue;
    }

    for (colctr = 0; 2 * colctr < output_width; colctr += 16) {
      const __v4qi *src =
        (const __v4qi *)(const void *)(inptr + colctr);
      __v8qi *dst = (__v8qi *)(void *)(outptr + 2 * colctr);
      __v4qi s0 = src[0];
      __v4qi s1 = src[1];
      __v4qi s2 = src[2];
      __v4qi s3 = src[3];
      __v8qi d0 = __vis_fpmerge(s0, s0);
      __v8qi d1 = __vis_fpmerge(s1, s1);
      __v8qi d2 = __vis_fpmerge(s2, s2);
      __v8qi d3 = __vis_fpmerge(s3, s3);

      dst[0] = d0;
      dst[1] = d1;
      dst[2] = d2;
      dst[3] = d3;
    }
  }
}


HIDDEN void
jsimd_h2v2_upsample_vis(int max_v_samp_factor, JDIMENSION output_width,
                        JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  int inrow, outrow;

  for (inrow = 0, outrow = 0; outrow < max_v_samp_factor; inrow++) {
    JSAMPROW inptr = input_data[inrow];
    JSAMPROW outptr0 = output_data[outrow++];
    JSAMPROW outptr1 = output_data[outrow++];
    JDIMENSION colctr;

    if (((JUINTPTR)inptr & 3) != 0 ||
        ((JUINTPTR)outptr0 & 7) != 0 ||
        ((JUINTPTR)outptr1 & 7) != 0) {
      for (colctr = 0; 2 * colctr < output_width; colctr++) {
        JSAMPLE sample = inptr[colctr];

        outptr0[2 * colctr] = sample;
        outptr0[2 * colctr + 1] = sample;
        outptr1[2 * colctr] = sample;
        outptr1[2 * colctr + 1] = sample;
      }
      continue;
    }

    for (colctr = 0; 2 * colctr < output_width; colctr += 16) {
      const __v4qi *src =
        (const __v4qi *)(const void *)(inptr + colctr);
      __v8qi *dst0 = (__v8qi *)(void *)(outptr0 + 2 * colctr);
      __v8qi *dst1 = (__v8qi *)(void *)(outptr1 + 2 * colctr);
      __v4qi s0 = src[0];
      __v4qi s1 = src[1];
      __v4qi s2 = src[2];
      __v4qi s3 = src[3];
      __v8qi d0 = __vis_fpmerge(s0, s0);
      __v8qi d1 = __vis_fpmerge(s1, s1);
      __v8qi d2 = __vis_fpmerge(s2, s2);
      __v8qi d3 = __vis_fpmerge(s3, s3);

      dst0[0] = d0;
      dst0[1] = d1;
      dst0[2] = d2;
      dst0[3] = d3;

      dst1[0] = d0;
      dst1[1] = d1;
      dst1[2] = d2;
      dst1[3] = d3;
    }
  }
}


/*
 * Fancy horizontal upsampling.  VIS fexpand() represents each byte as a
 * 16-bit value shifted left by four.  With GSR.scale = 1, fpack16() then
 * performs exactly the final division by four.  The +1/+2 ordered-dither
 * biases are represented as +16/+32 in that fixed-point domain.
 */

LOCAL(void)
h2v1_fancy_scalar_row(JDIMENSION width, JSAMPROW inptr, JSAMPROW outptr)
{
  JDIMENSION colctr;

  outptr[0] = inptr[0];
  outptr[1] = (JSAMPLE)((3 * GETJSAMPLE(inptr[0]) +
                         GETJSAMPLE(inptr[1]) + 2) >> 2);

  for (colctr = 1; colctr < width - 1; colctr++) {
    int center = 3 * GETJSAMPLE(inptr[colctr]);

    outptr[2 * colctr] =
      (JSAMPLE)((center + GETJSAMPLE(inptr[colctr - 1]) + 1) >> 2);
    outptr[2 * colctr + 1] =
      (JSAMPLE)((center + GETJSAMPLE(inptr[colctr + 1]) + 2) >> 2);
  }

  outptr[2 * width - 2] =
    (JSAMPLE)((3 * GETJSAMPLE(inptr[width - 1]) +
               GETJSAMPLE(inptr[width - 2]) + 1) >> 2);
  outptr[2 * width - 1] = inptr[width - 1];
}


HIDDEN void
jsimd_h2v1_fancy_upsample_vis(int max_v_samp_factor,
                              JDIMENSION downsampled_width,
                              JSAMPARRAY input_data,
                              JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  const __v4hi bias_left = { 16, 16, 16, 16 };
  const __v4hi bias_right = { 32, 32, 32, 32 };
  int inrow;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    JSAMPROW inptr = input_data[inrow];
    JSAMPROW outptr = output_data[inrow];
    JDIMENSION colctr;

    if (downsampled_width < 10 ||
        ((JUINTPTR)inptr & 3) != 0 || ((JUINTPTR)outptr & 7) != 0) {
      h2v1_fancy_scalar_row(downsampled_width, inptr, outptr);
      continue;
    }

    /* Handle the unaligned left edge and make the VIS stores 8-byte aligned. */
    outptr[0] = inptr[0];
    outptr[1] = (JSAMPLE)((3 * GETJSAMPLE(inptr[0]) +
                           GETJSAMPLE(inptr[1]) + 2) >> 2);
    for (colctr = 1; colctr < 4; colctr++) {
      int center = 3 * GETJSAMPLE(inptr[colctr]);

      outptr[2 * colctr] =
        (JSAMPLE)((center + GETJSAMPLE(inptr[colctr - 1]) + 1) >> 2);
      outptr[2 * colctr + 1] =
        (JSAMPLE)((center + GETJSAMPLE(inptr[colctr + 1]) + 2) >> 2);
    }

    /* GSR.scale = 1.  alignaddr() below changes only GSR.align. */
    __builtin_vis_write_gsr(1 << 3);

    for (colctr = 4; colctr + 3 <= downsampled_width - 2; colctr += 4) {
      const __v4qi *src =
        (const __v4qi *)(const void *)(inptr + colctr);
      __v4hi prev = __vis_fexpand(src[-1]);
      __v4hi center = __vis_fexpand(src[0]);
      __v4hi next = __vis_fexpand(src[1]);
      __v4hi three_center;
      __v4hi left_neighbor, right_neighbor;
      __v4hi left, right;
      __v4qi packed_left, packed_right;
      __v8qi packed;
      __v8qi *dst = (__v8qi *)(void *)(outptr + 2 * colctr);

      (void)__vis_alignaddr((void *)(outptr + 2 * colctr), 6);
      left_neighbor = __vis_faligndatav4hi(prev, center);
      (void)__vis_alignaddr((void *)(outptr + 2 * colctr), 2);
      right_neighbor = __vis_faligndatav4hi(center, next);

      three_center = __vis_fpadd16(center, center);
      three_center = __vis_fpadd16(three_center, center);

      left = __vis_fpadd16(three_center, left_neighbor);
      left = __vis_fpadd16(left, bias_left);
      right = __vis_fpadd16(three_center, right_neighbor);
      right = __vis_fpadd16(right, bias_right);

      packed_left = __vis_fpack16(left);
      packed_right = __vis_fpack16(right);
      packed = __vis_fpmerge(packed_left, packed_right);
      *dst = packed;
    }

    for (; colctr < downsampled_width - 1; colctr++) {
      int center = 3 * GETJSAMPLE(inptr[colctr]);

      outptr[2 * colctr] =
        (JSAMPLE)((center + GETJSAMPLE(inptr[colctr - 1]) + 1) >> 2);
      outptr[2 * colctr + 1] =
        (JSAMPLE)((center + GETJSAMPLE(inptr[colctr + 1]) + 2) >> 2);
    }

    outptr[2 * downsampled_width - 2] =
      (JSAMPLE)((3 * GETJSAMPLE(inptr[downsampled_width - 1]) +
                 GETJSAMPLE(inptr[downsampled_width - 2]) + 1) >> 2);
    outptr[2 * downsampled_width - 1] = inptr[downsampled_width - 1];
  }
}


/*
 * Return (3 * near + far) * 8 in four signed 16-bit lanes.  Keeping three
 * fractional bits instead of the four produced by fexpand() is what lets the
 * complete 2-D 9:3:3:1 filter stay below INT16_MAX.  fpack16() with
 * GSR.scale = 0 then performs the exact final division by 16.
 */
static inline __v4hi
vis_vertical_3_1_x8(__v4qi near_samples, __v4qi far_samples,
                    __v2hi times8)
{
  __v4hi near8 = __vis_fmul8x16au(near_samples, times8);
  __v4hi far8 = __vis_fmul8x16au(far_samples, times8);
  __v4hi sum = __vis_fpadd16(near8, near8);

  sum = __vis_fpadd16(sum, near8);
  return __vis_fpadd16(sum, far8);
}


LOCAL(void)
h2v2_fancy_scalar_row(JDIMENSION width, JSAMPROW nearptr,
                      JSAMPROW farptr, JSAMPROW outptr)
{
  JDIMENSION colctr;
  int lastcolsum, thiscolsum, nextcolsum;

  thiscolsum = 3 * GETJSAMPLE(nearptr[0]) + GETJSAMPLE(farptr[0]);
  nextcolsum = 3 * GETJSAMPLE(nearptr[1]) + GETJSAMPLE(farptr[1]);

  outptr[0] = (JSAMPLE)((thiscolsum * 4 + 8) >> 4);
  outptr[1] = (JSAMPLE)((thiscolsum * 3 + nextcolsum + 7) >> 4);
  lastcolsum = thiscolsum;
  thiscolsum = nextcolsum;

  for (colctr = 1; colctr < width - 1; colctr++) {
    nextcolsum =
      3 * GETJSAMPLE(nearptr[colctr + 1]) + GETJSAMPLE(farptr[colctr + 1]);
    outptr[2 * colctr] =
      (JSAMPLE)((thiscolsum * 3 + lastcolsum + 8) >> 4);
    outptr[2 * colctr + 1] =
      (JSAMPLE)((thiscolsum * 3 + nextcolsum + 7) >> 4);
    lastcolsum = thiscolsum;
    thiscolsum = nextcolsum;
  }

  outptr[2 * width - 2] =
    (JSAMPLE)((thiscolsum * 3 + lastcolsum + 8) >> 4);
  outptr[2 * width - 1] = (JSAMPLE)((thiscolsum * 4 + 7) >> 4);
}


HIDDEN void
jsimd_h2v2_fancy_upsample_vis(int max_v_samp_factor,
                              JDIMENSION downsampled_width,
                              JSAMPARRAY input_data,
                              JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  const __v2hi times8 = { 2048, 2048 };
  const __v4hi bias_left = { 64, 64, 64, 64 };
  const __v4hi bias_right = { 56, 56, 56, 56 };
  int inrow, outrow, v;

  inrow = outrow = 0;
  while (outrow < max_v_samp_factor) {
    for (v = 0; v < 2; v++) {
      JSAMPROW nearptr = input_data[inrow];
      JSAMPROW farptr = input_data[inrow + (v ? 1 : -1)];
      JSAMPROW outptr = output_data[outrow++];
      JDIMENSION colctr;

      if (downsampled_width < 10 ||
          ((JUINTPTR)nearptr & 3) != 0 ||
          ((JUINTPTR)farptr & 3) != 0 ||
          ((JUINTPTR)outptr & 7) != 0) {
        h2v2_fancy_scalar_row(downsampled_width, nearptr, farptr, outptr);
        continue;
      }

      /* Exact scalar edge/prelude for columns 0..3. */
      {
        int lastcolsum, thiscolsum, nextcolsum;

        thiscolsum =
          3 * GETJSAMPLE(nearptr[0]) + GETJSAMPLE(farptr[0]);
        nextcolsum =
          3 * GETJSAMPLE(nearptr[1]) + GETJSAMPLE(farptr[1]);
        outptr[0] = (JSAMPLE)((thiscolsum * 4 + 8) >> 4);
        outptr[1] =
          (JSAMPLE)((thiscolsum * 3 + nextcolsum + 7) >> 4);
        lastcolsum = thiscolsum;
        thiscolsum = nextcolsum;

        for (colctr = 1; colctr < 4; colctr++) {
          nextcolsum =
            3 * GETJSAMPLE(nearptr[colctr + 1]) +
            GETJSAMPLE(farptr[colctr + 1]);
          outptr[2 * colctr] =
            (JSAMPLE)((thiscolsum * 3 + lastcolsum + 8) >> 4);
          outptr[2 * colctr + 1] =
            (JSAMPLE)((thiscolsum * 3 + nextcolsum + 7) >> 4);
          lastcolsum = thiscolsum;
          thiscolsum = nextcolsum;
        }
      }

      /* GSR.scale = 0.  alignaddr() below changes only GSR.align. */
      __builtin_vis_write_gsr(0);

      for (colctr = 4; colctr + 3 <= downsampled_width - 2; colctr += 4) {
        const __v4qi *near_src =
          (const __v4qi *)(const void *)(nearptr + colctr);
        const __v4qi *far_src =
          (const __v4qi *)(const void *)(farptr + colctr);
        __v4hi prev =
          vis_vertical_3_1_x8(near_src[-1], far_src[-1], times8);
        __v4hi center =
          vis_vertical_3_1_x8(near_src[0], far_src[0], times8);
        __v4hi next =
          vis_vertical_3_1_x8(near_src[1], far_src[1], times8);
        __v4hi three_center;
        __v4hi left_neighbor, right_neighbor;
        __v4hi left, right;
        __v4qi packed_left, packed_right;
        __v8qi *dst = (__v8qi *)(void *)(outptr + 2 * colctr);

        (void)__vis_alignaddr((void *)(outptr + 2 * colctr), 6);
        left_neighbor = __vis_faligndatav4hi(prev, center);
        (void)__vis_alignaddr((void *)(outptr + 2 * colctr), 2);
        right_neighbor = __vis_faligndatav4hi(center, next);

        three_center = __vis_fpadd16(center, center);
        three_center = __vis_fpadd16(three_center, center);

        left = __vis_fpadd16(three_center, left_neighbor);
        left = __vis_fpadd16(left, bias_left);
        right = __vis_fpadd16(three_center, right_neighbor);
        right = __vis_fpadd16(right, bias_right);

        *dst = __vis_fpmerge(__vis_fpack16(left), __vis_fpack16(right));
      }

      /* Exact scalar tail and right edge. */
      if (colctr < downsampled_width - 1) {
        int lastcolsum =
          3 * GETJSAMPLE(nearptr[colctr - 1]) +
          GETJSAMPLE(farptr[colctr - 1]);
        int thiscolsum =
          3 * GETJSAMPLE(nearptr[colctr]) + GETJSAMPLE(farptr[colctr]);

        for (; colctr < downsampled_width - 1; colctr++) {
          int nextcolsum =
            3 * GETJSAMPLE(nearptr[colctr + 1]) +
            GETJSAMPLE(farptr[colctr + 1]);

          outptr[2 * colctr] =
            (JSAMPLE)((thiscolsum * 3 + lastcolsum + 8) >> 4);
          outptr[2 * colctr + 1] =
            (JSAMPLE)((thiscolsum * 3 + nextcolsum + 7) >> 4);
          lastcolsum = thiscolsum;
          thiscolsum = nextcolsum;
        }

        outptr[2 * downsampled_width - 2] =
          (JSAMPLE)((thiscolsum * 3 + lastcolsum + 8) >> 4);
        outptr[2 * downsampled_width - 1] =
          (JSAMPLE)((thiscolsum * 4 + 7) >> 4);
      } else {
        /*
         * The vector loop can only stop at width - 1 when the last four
         * interior samples exactly fill a vector.  Reconstruct the final
         * column directly in that case.
         */
        int last = (int)downsampled_width - 1;
        int lastsum =
          3 * GETJSAMPLE(nearptr[last - 1]) + GETJSAMPLE(farptr[last - 1]);
        int thissum =
          3 * GETJSAMPLE(nearptr[last]) + GETJSAMPLE(farptr[last]);

        outptr[2 * last] =
          (JSAMPLE)((thissum * 3 + lastsum + 8) >> 4);
        outptr[2 * last + 1] = (JSAMPLE)((thissum * 4 + 7) >> 4);
      }
    }
    inrow++;
  }
}
