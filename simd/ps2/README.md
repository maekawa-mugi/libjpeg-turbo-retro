# PlayStation 2 Emotion Engine SIMD backend

This directory contains the PS2 EE (R5900) **128-bit MMI** backend.
It is independent of `simd/mips64/`, which uses **Loongson's distinct
64-bit MMI instruction set**.  VU0/VU1 are not enabled by this backend.

## Current coverage

- `h2v1` plain upsampling: 16-byte MMI loads, byte interleave, 32-byte
  stores, with a bounded scalar tail.
- `h2v2` plain upsampling: the same horizontal expansion, followed by
  duplication of the expanded output row.
- All other JPEG SIMD hooks return no implementation and use the
  library's generic C routines.  In particular, this is **not yet an
  MMI IDCT, FDCT, fancy upsampler, or color converter**.
- The backend is selected only with `WITH_SIMD=ON` for the PS2 EE
  toolchain.  Building with `WITH_SIMD=OFF` still uses generic C.

## Build with PS2SDK

```sh
cmake -S . -B build-ps2 \
  -DCMAKE_TOOLCHAIN_FILE="$PS2DEV/share/ps2dev.cmake" \
  -DENABLE_SHARED=OFF -DENABLE_STATIC=ON \
  -DWITH_SIMD=ON -DWITH_TOOLS=OFF
cmake --build build-ps2 -j
```

For exercising these two kernels, use 8-bit JPEG images with 4:2:2
or 4:2:0 subsampling and set `cinfo.do_fancy_upsampling = FALSE`
**before** `jpeg_start_decompress()`.  libjpeg normally selects
*fancy* upsampling, which deliberately falls back to generic C here.

## Suggested on-device tests

1. Compare SIMD on/off for 4:2:0 and 4:2:2 images, including odd widths,
   widths below 32, exactly 32, and non-multiples of 32.
2. Verify aligned and deliberately unaligned input/output rows.
3. Compare pixels against the generic C plain upsampler.
4. Measure full-frame time and, separately, the upsampling stage.

R5900 compilation, pixel equivalence, and PS2 execution have **not yet
been tested**.  The toolchain check and the new assembly should be
validated before treating this as a production optimization.
