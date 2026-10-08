# PlayStation 2 Emotion Engine SIMD backend

This directory contains the PS2 EE (R5900) **128-bit MMI** backend.
It is independent of `simd/mips64/`, which uses **Loongson's distinct
64-bit MMI instruction set**.  VU0/VU1 are not enabled by this backend.

## Current coverage

- `h2v1` plain upsampling: 16-byte MMI loads, byte interleave, 32-byte
  stores, with a bounded scalar tail.
- `h2v2` plain upsampling: the same horizontal expansion, followed by
  duplication of the expanded output row.
- `h2v1` fancy upsampling: exact 3:1 triangle interpolation with eight
  horizontal pixels at a time processed in R5900 MMI.
- `h2v2` fancy upsampling: eight-wide MMI vertical interpolation followed
  by bit-exact horizontal interpolation in C.
- All remaining JPEG SIMD hooks use generic C.  In particular, this is
  **not yet an MMI IDCT, FDCT, merged upsampler, or color converter**.
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

For an optional PS2 ELF smoke test, also pass `-DWITH_PS2_MMI_TESTS=ON`
to the configuration command.  Build targets `ps2_mmi_upsample_test` and `ps2_mmi_fancy_test`,
then run the resulting ELFs on PS2 hardware or an emulator.  The
plain test covers 48 combinations of width, sampling ratio, and pointer
alignment; the fancy test covers 128 (including source context rows,
different data patterns, boundaries, and deliberately unaligned rows).
Both tests check that output padding is untouched.  The tests are **not**
executed during cross-compilation.

For exercising the plain kernels, use 8-bit JPEG images with 4:2:2 or
4:2:0 subsampling and set `cinfo.do_fancy_upsampling = FALSE` before
`jpeg_start_decompress()`.  The default fancy upsampling now also has
a PS2 MMI path, including 4:2:0.  Compare both settings with a
`WITH_SIMD=OFF` reference build.

## Suggested on-device tests

1. Compare SIMD on/off for 4:2:0 and 4:2:2 images in both fancy/plain modes, including odd widths,
   widths below 32, exactly 32, and non-multiples of 32.
2. Verify aligned and deliberately unaligned input/output rows.
3. Compare pixels against the generic C plain upsampler.
4. Measure full-frame time and, separately, the upsampling stage.

R5900 compilation, pixel equivalence, and PS2 execution have **not yet
been tested**.  The toolchain check and the new assembly should be
validated before treating this as a production optimization.
