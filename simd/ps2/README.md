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
- `h2v1` and `h2v2` compressor downsampling: process 16 output samples
  with 128-bit MMI, preserving IJG's alternating rounding biases and
  right-edge expansion before downsampling.
- Optional `JDCT_ISLOW` (accurate integer 8x8 IDCT): MMI-accelerated AC-zero
  detection and DC-only shortcut.  Non-DC-only blocks execute the matching
  IJG 8x8 integer algorithm, with exact output range-table wrapping.
  This is a *partial* IDCT optimization, not a complete vector IDCT.
- All remaining JPEG SIMD hooks use generic C.  In particular, this is
  **not yet an MMI FDCT, ifast IDCT, merged upsampler, or color converter**.
- The backend is selected only with `WITH_SIMD=ON` for the PS2 EE
  toolchain.  Building with `WITH_SIMD=OFF` still uses generic C.
- The IDCT dispatcher is **disabled by default** until performance and
  bit-exactness are verified on actual PS2 hardware.  Set
  `-DWITH_PS2_EXPERIMENTAL_IDCT=ON` to exercise it in normal JPEG decoding.
  The IDCT standalone test builds independently of this setting.

## Build with PS2SDK

```sh
cmake -S . -B build-ps2 \
  -DCMAKE_TOOLCHAIN_FILE="$PS2DEV/share/ps2dev.cmake" \
  -DENABLE_SHARED=OFF -DENABLE_STATIC=ON \
  -DWITH_SIMD=ON -DWITH_TOOLS=OFF
cmake --build build-ps2 -j
```

To build the standalone validation ELFs, add `-DWITH_PS2_MMI_TESTS=ON`
to the configure command and run:

```sh
cmake --build build-ps2 --target \
  ps2_mmi_upsample_test ps2_mmi_fancy_test \
  ps2_mmi_downsample_test ps2_mmi_idct_test
```

Run these ELFs on PS2 hardware or an emulator.  The
plain test covers 48 combinations of width, sampling ratio, and pointer
alignment; the fancy test covers 128 (including source context rows,
different data patterns, boundaries, and deliberately unaligned rows).
All tests check that output padding is untouched.  The downsampling
test covers 136 cases (including right-edge padding and alternating
rounding), and the IDCT test compares 512 blocks against the library's
reference integer IDCT.  The tests are **not** executed during
cross-compilation.

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
been tested**.  In addition to byte comparison, benchmark separately
for low-entropy and high-entropy coefficient blocks: the IDCT MMI
shortcut speeds DC-only blocks but requires a complete AC scan for
other blocks.  Keep `WITH_PS2_EXPERIMENTAL_IDCT=OFF` unless testing or
benchmarking the IDCT path.  The toolchain check and the new assembly should be
validated before treating this as a production optimization.
