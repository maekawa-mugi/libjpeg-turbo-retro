/* End-to-end JPEG-stream smoke and differential output-layout test.
 * JPEG encode covers RGB->YCbCr and 4:4:4/4:2:2/4:2:0 downsampling.
 * Decode covers Huffman, IDCT, fancy/plain upsampling, and seven-color
 * pipeline layout consistency across odd dimensions.
 *
 * The RGB output from the SAME compressed stream, DCT method and fancy
 * setting is the reference for all four 4-byte formats.  It is not a
 * performance benchmark, nor a SIMD-off reference library.
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  struct jpeg_error_mgr pub;
  jmp_buf jump;
  char message[JMSG_LENGTH_MAX];
} test_error_mgr;

typedef struct {
  JSAMPLE *data;
  size_t stride;
  unsigned width, height, components;
} decoded_image;

static void
test_error_exit(j_common_ptr info)
{
  test_error_mgr *err = (test_error_mgr *)info->err;
  (*info->err->format_message)(info, err->message);
  longjmp(err->jump, 1);
}

static JSAMPLE *
pattern_rgb(unsigned width, unsigned height, unsigned seed)
{
  unsigned x, y;
  JSAMPLE *p = (JSAMPLE *)malloc((size_t)width * height * RGB_PIXELSIZE);
  if (!p)
    return NULL;
  for (y = 0; y < height; y++)
    for (x = 0; x < width; x++) {
      JSAMPLE *dst = p + ((size_t)y * width + x) * RGB_PIXELSIZE;
      dst[RGB_RED] = (JSAMPLE)((29u * x + 41u * y + seed) & 255u);
      dst[RGB_GREEN] = (JSAMPLE)((7u * x + 17u * y + seed * 3u) & 255u);
      dst[RGB_BLUE] = (JSAMPLE)((59u * x + 3u * y + seed * 11u) & 255u);
    }
  return p;
}

static int
encode_jpeg(const JSAMPLE *rgb, unsigned width, unsigned height,
            int subsample, unsigned char **data, unsigned long *bytes)
{
  struct jpeg_compress_struct c;
  test_error_mgr err;
  JSAMPROW row;
  memset(&c, 0, sizeof(c));
  memset(&err, 0, sizeof(err));
  c.err = jpeg_std_error(&err.pub);
  err.pub.error_exit = test_error_exit;
  *data = NULL;
  *bytes = 0;
  if (setjmp(err.jump)) {
    printf("FAIL,jpeg_stream,encode,%s\n", err.message);
    jpeg_destroy_compress(&c);
    free(*data);
    *data = NULL;
    return 1;
  }
  jpeg_create_compress(&c);
  jpeg_mem_dest(&c, data, bytes);
  c.image_width = width;
  c.image_height = height;
  c.input_components = 3;
  c.in_color_space = JCS_RGB;
  jpeg_set_defaults(&c);
  jpeg_set_quality(&c, 82, TRUE);
  c.comp_info[0].h_samp_factor = subsample == 0 ? 1 : 2;
  c.comp_info[0].v_samp_factor = subsample == 2 ? 2 : 1;
  jpeg_start_compress(&c, TRUE);
  while (c.next_scanline < c.image_height) {
    row = (JSAMPROW)(rgb + (size_t)c.next_scanline *
                       width * RGB_PIXELSIZE);
    jpeg_write_scanlines(&c, &row, 1);
  }
  jpeg_finish_compress(&c);
  jpeg_destroy_compress(&c);
  return 0;
}

static void
free_decoded(decoded_image *d)
{
  free(d->data);
  memset(d, 0, sizeof(*d));
}

static int
decode_jpeg(const unsigned char *data, unsigned long bytes,
            J_COLOR_SPACE color, int fancy, J_DCT_METHOD dct,
            decoded_image *out)
{
  struct jpeg_decompress_struct c;
  test_error_mgr err;
  JSAMPROW row;
  unsigned int comp;
  memset(&c, 0, sizeof(c));
  memset(&err, 0, sizeof(err));
  memset(out, 0, sizeof(*out));
  c.err = jpeg_std_error(&err.pub);
  err.pub.error_exit = test_error_exit;
  if (setjmp(err.jump)) {
    printf("FAIL,jpeg_stream,decode,%s\n", err.message);
    jpeg_destroy_decompress(&c);
    free_decoded(out);
    return 1;
  }
  jpeg_create_decompress(&c);
  jpeg_mem_src(&c, data, bytes);
  if (jpeg_read_header(&c, TRUE) != JPEG_HEADER_OK) {
    jpeg_destroy_decompress(&c);
    return 1;
  }
#ifdef PS2_APPROX_FPU_DECODER_DEFAULT
  if (c.dct_method != JDCT_FLOAT) {
    puts("FAIL,jpeg_stream,fpu_default_not_selected");
    jpeg_destroy_decompress(&c);
    return 1;
  }
#endif
  c.out_color_space = color;
  c.dct_method = dct;
  c.do_fancy_upsampling = fancy ? TRUE : FALSE;
  jpeg_start_decompress(&c);
  comp = c.output_components;
  out->width = (unsigned)c.output_width;
  out->height = (unsigned)c.output_height;
  out->components = comp;
  out->stride = (size_t)c.output_width * comp + 16;
  if (comp > 4 || out->stride > 4096 || out->height > 512) {
    jpeg_destroy_decompress(&c);
    return 1;
  }
  out->data = (JSAMPLE *)malloc(out->stride * out->height);
  if (!out->data) {
    jpeg_destroy_decompress(&c);
    return 1;
  }
  memset(out->data, 0xa5, out->stride * out->height);
  while (c.output_scanline < c.output_height) {
    row = out->data + (size_t)c.output_scanline * out->stride;
    jpeg_read_scanlines(&c, &row, 1);
  }
  jpeg_finish_decompress(&c);
  jpeg_destroy_decompress(&c);
  return 0;
}

static int
verify_layout(const decoded_image *ref, const decoded_image *test,
              const char *name, const int pos[4])
{
  unsigned y, x, i;
  if (ref->width != test->width || ref->height != test->height ||
      ref->components != RGB_PIXELSIZE || test->components != 4)
    return 1;
  for (y = 0; y < ref->height; y++) {
    const JSAMPLE *base = ref->data + (size_t)y * ref->stride;
    const JSAMPLE *out = test->data + (size_t)y * test->stride;
    for (x = 0; x < ref->width; x++) {
      JSAMPLE desired[4] = { 0, 0, 0, 0 };
      desired[pos[0]] = base[x * RGB_PIXELSIZE + RGB_RED];
      desired[pos[1]] = base[x * RGB_PIXELSIZE + RGB_GREEN];
      desired[pos[2]] = base[x * RGB_PIXELSIZE + RGB_BLUE];
      desired[pos[3]] = 255;
      for (i = 0; i < 4; i++)
        if (out[4 * x + i] != desired[i]) {
          printf("FAIL,jpeg_stream,layout=%s,row=%u,x=%u,byte=%u,got=%u,expected=%u\n",
                 name, y, x, i, (unsigned)out[4 * x + i],
                 (unsigned)desired[i]);
          return 1;
        }
    }
    for (i = 4 * ref->width; i < 4 * ref->width + 16; i++)
      if (out[i] != 0xa5)
        return 1;
    for (i = RGB_PIXELSIZE * ref->width;
         i < RGB_PIXELSIZE * ref->width + 16; i++)
      if (base[i] != 0xa5)
        return 1;
  }
  return 0;
}

int
ps2_test_jpeg_stream(void)
{
  static const unsigned widths[] = { 1, 7, 16, 31, 32, 33, 65, 127 };
  static const unsigned heights[] = { 1, 9, 17, 63 };
  static const struct {
    const char *name;
    J_COLOR_SPACE color;
    int pos[4]; /* R, G, B, X byte offsets */
  } formats[] = {
    { "RGBX", JCS_EXT_RGBX, { 0, 1, 2, 3 } },
    { "BGRX", JCS_EXT_BGRX, { 2, 1, 0, 3 } },
    { "XBGR", JCS_EXT_XBGR, { 3, 2, 1, 0 } },
    { "XRGB", JCS_EXT_XRGB, { 1, 2, 3, 0 } }
  };
  unsigned wi, sub, fancy, dct, format, comparisons = 0;
#ifdef PS2_APPROX_FPU_IDCT
  const unsigned idct_variants = 3;
#else
  const unsigned idct_variants = 2;
#endif
  int failures = 0;
  for (wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
    unsigned width = widths[wi];
    unsigned height = heights[wi % 4];
    JSAMPLE *rgb = pattern_rgb(width, height, wi);
    if (!rgb) {
      puts("FAIL,jpeg_stream,input_allocation");
      return 1;
    }
    for (sub = 0; sub < 3; sub++) {
      unsigned char *jpeg_bytes = NULL;
      unsigned long jpeg_size = 0;
      if (encode_jpeg(rgb, width, height, (int)sub,
                      &jpeg_bytes, &jpeg_size)) {
        failures++;
        continue;
      }
      for (fancy = 0; fancy < 2; fancy++)
        for (dct = 0; dct < idct_variants; dct++) {
          decoded_image reference;
          J_DCT_METHOD method = dct == 2 ? JDCT_FLOAT :
                                dct ? JDCT_IFAST : JDCT_ISLOW;
          if (decode_jpeg(jpeg_bytes, jpeg_size, JCS_RGB,
                          (int)fancy, method, &reference)) {
            failures++;
            continue;
          }
          for (format = 0; format < 4; format++) {
            decoded_image actual;
            if (decode_jpeg(jpeg_bytes, jpeg_size,
                            formats[format].color,
                            (int)fancy, method, &actual)) {
              failures++;
              continue;
            }
            if (verify_layout(&reference, &actual,
                              formats[format].name,
                              formats[format].pos)) {
              printf("FAIL,jpeg_stream,width=%u,height=%u,sub=%u,fancy=%u,idct=%u,format=%s\n",
                     width, height, sub, fancy, dct,
                     formats[format].name);
              failures++;
            }
            comparisons++;
            free_decoded(&actual);
          }
          free_decoded(&reference);
        }
      free(jpeg_bytes);
    }
    free(rgb);
  }
  if (failures)
    printf("FAIL,jpeg_stream,errors=%d,compared=%u\n", failures, comparisons);
  else
    printf("PASS,jpeg_stream,layouts=%u,8_sizes_3_subsampling_2_fancy_%u_idct\n",
           comparisons, idct_variants);
  return failures ? 1 : 0;
}
