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
- Optional `JDCT_ISLOW` (accurate integer 8x8 IDCT): 128-bit MMI
  zero-AC detection for the DC-only shortcut; for non-DC-only blocks,
  `PMULTH` dequantizes eight 16-bit coefficient/quant pairs into 32-bit
  products per iteration and `PCPYLD`/`PCPYUD` reorder the R5900
  accumulators into the original block order.  Each IDCT pass now uses
  four 32-bit MMI lanes for the final Loeffler butterflies and IJG
  rounding (`PADDW`, `PSUBW`, `PSRAW`).  The odd-part rotation stage
  in each pass now also batches eight signed 16-bit multiplies with
  `PMULTH` when all intermediate operands fit in a signed halfword;
  values outside that range use the reference `MULTIPLY16C16` path.
  The remaining middle stages retain the IJG fixed-point operations
  and 10-bit range wrapping.
  An unaligned coefficient/quant table or non-16-bit quant type uses
  the scalar dequantizer.  This is a *partial-vector* transform, not
  a fully vectorized 8x8 IDCT.
- Optional YCbCr to 4-byte RGBX/RGBA/BGRX/BGRA/XBGR/ABGR/XRGB/ARGB:
  16.16 fixed-point color calculation, followed by 128-bit MMI
  halfword clipping and packing of four pixels.  This is an experimental
  *partial* SIMD converter, not yet a vectorized color matrix.
  Three-byte RGB and BGR continue to use the existing C converter.
- Remaining JPEG SIMD hooks use generic C, including
  **MMI FDCT, ifast IDCT, and merged upsampling**.
- The backend is selected only with `WITH_SIMD=ON` for the PS2 EE
  toolchain.  Building with `WITH_SIMD=OFF` still uses generic C.
- The IDCT dispatcher is **disabled by default** until performance and
  bit-exactness are verified on actual PS2 hardware.  Set
  `-DWITH_PS2_EXPERIMENTAL_IDCT=ON` to exercise it in normal JPEG decoding.
  The IDCT standalone test builds independently of this setting.

- The color converter is **disabled by default** until compiled and
  benchmarked on PS2.  Pass `-DWITH_PS2_EXPERIMENTAL_COLOR=ON` to enable
  the four-byte YCbCr-to-RGB MMI output path.  It is a separate switch
  from `WITH_PS2_EXPERIMENTAL_IDCT`.

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
  ps2_mmi_downsample_test ps2_mmi_idct_test \
  ps2_mmi_color_test
```

Run these ELFs on PS2 hardware or an emulator.  The
plain test covers 48 combinations of width, sampling ratio, and pointer
alignment; the fancy test covers 128 (including source context rows,
different data patterns, boundaries, and deliberately unaligned rows).
All tests check that output padding is untouched.  The downsampling
test covers 136 cases (including right-edge padding and alternating
rounding), and the IDCT test compares 2048 blocks (DC-only, sparse, dense,
single-AC, signed odd-frequency stress, and aligned/unaligned
quant/coefficient tables) against the library's
reference integer IDCT.  The color test covers 420 image-row/layout
cases, including all four-byte output layouts, three-byte scalar
reference layouts, unaligned destinations, chroma extremes, and buffer
guard bytes.  The tests are **not** executed during
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
shortcut can speed DC-only blocks but requires a complete AC scan for
other blocks.  The PMULTH odd-part rotations and butterfly paths may have different
speed tradeoffs due to stack-buffer traffic and reordering.  The assembler
probe also checks `PMULTH`, `PMFLO`/`PMFHI`, `PCPYLD`/`PCPYUD`,
`PADDW`, `PSRAW`, and `PMAXH`.  Passing that probe verifies syntax
support only; it does not verify runtime semantics or timing.  Keep `WITH_PS2_EXPERIMENTAL_IDCT=OFF` unless testing or
benchmarking the IDCT path.  The toolchain check and the new assembly should be
validated before treating this as a production optimization.
