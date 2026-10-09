/* Tiny syntax-check-only model of the libjpeg types used by bench-*.c.
 * NEVER compile this into a PS2 ELF.  The actual bench-harness uses
 * ../jsimdint.h when PS2_BENCH_HOST_CHECK is not defined.
 * SPDX-License-Identifier: Zlib
 */
#ifndef PS2_BENCH_HOST_TYPES_H
#define PS2_BENCH_HOST_TYPES_H
#include <stddef.h>
#include <stdint.h>

typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;
typedef unsigned int JDIMENSION;
typedef short JCOEF;
typedef JCOEF *JCOEFPTR;
typedef short ISLOW_MULT_TYPE;
typedef short DCTELEM;

#define RGB_PIXELSIZE 3
#define RGB_RED 0
#define RGB_GREEN 1
#define RGB_BLUE 2
#define EXT_RGB_PIXELSIZE 3
#define EXT_RGB_RED 0
#define EXT_RGB_GREEN 1
#define EXT_RGB_BLUE 2
#define EXT_BGR_PIXELSIZE 3
#define EXT_BGR_RED 2
#define EXT_BGR_GREEN 1
#define EXT_BGR_BLUE 0
#define EXT_RGBX_PIXELSIZE 4
#define EXT_RGBX_RED 0
#define EXT_RGBX_GREEN 1
#define EXT_RGBX_BLUE 2
#define EXT_BGRX_PIXELSIZE 4
#define EXT_BGRX_RED 2
#define EXT_BGRX_GREEN 1
#define EXT_BGRX_BLUE 0
#define EXT_XBGR_PIXELSIZE 4
#define EXT_XBGR_RED 3
#define EXT_XBGR_GREEN 2
#define EXT_XBGR_BLUE 1
#define EXT_XRGB_PIXELSIZE 4
#define EXT_XRGB_RED 1
#define EXT_XRGB_GREEN 2
#define EXT_XRGB_BLUE 3

struct jpeg_decompress_struct {
  int data_precision;
  JSAMPLE *sample_range_limit;
};
typedef struct jpeg_component_info {
  ISLOW_MULT_TYPE *dct_table;
} jpeg_component_info;

extern void _jpeg_idct_islow(struct jpeg_decompress_struct *,
                              jpeg_component_info *, JCOEFPTR,
                              JSAMPARRAY, JDIMENSION);

extern void jsimd_h2v1_upsample_ps2mmi(JDIMENSION, JDIMENSION,
                                        JSAMPARRAY, JSAMPARRAY *);
extern void jsimd_h2v2_upsample_ps2mmi(JDIMENSION, JDIMENSION,
                                        JSAMPARRAY, JSAMPARRAY *);
extern void jsimd_h2v1_fancy_upsample_ps2mmi(JDIMENSION, JDIMENSION,
                                              JSAMPARRAY, JSAMPARRAY *);
extern void jsimd_h2v2_fancy_upsample_ps2mmi(JDIMENSION, JDIMENSION,
                                              JSAMPARRAY, JSAMPARRAY *);
extern void jsimd_h2v1_downsample_ps2mmi(JDIMENSION, JDIMENSION,
                                          JDIMENSION, JDIMENSION,
                                          JSAMPARRAY, JSAMPARRAY);
extern void jsimd_h2v2_downsample_ps2mmi(JDIMENSION, JDIMENSION,
                                          JDIMENSION, JDIMENSION,
                                          JSAMPARRAY, JSAMPARRAY);
extern void jsimd_quantize_ps2mmi(JCOEFPTR, DCTELEM *, DCTELEM *);
#endif
