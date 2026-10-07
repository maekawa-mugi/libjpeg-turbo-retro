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
