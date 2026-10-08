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
  The even-part rotation can also use `PMULTH` for its three fixed-point
  products, but that consumes only three of eight lanes and may be
  slower than the scalar path.  It is therefore gated separately by
  `WITH_PS2_EXPERIMENTAL_IDCT_EVEN=ON` (default OFF) for benchmarking.
  Both vector paths preserve the scalar fallback for operands outside
  the signed 16-bit range.  Remaining middle stages use IJG fixed-point
  arithmetic and 10-bit range wrapping.
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
  `WITH_PS2_EXPERIMENTAL_IDCT_EVEN` is a separate option that controls
  only the even-rotation inner kernel.  It also affects the standalone
  IDCT test when built with `WITH_PS2_MMI_TESTS=ON`.

- The color converter is **disabled by default** until compiled and
  benchmarked on PS2.  Pass `-DWITH_PS2_EXPERIMENTAL_COLOR=ON` to enable
  the four-byte YCbCr-to-RGB MMI output path.  It is a separate switch
  from `WITH_PS2_EXPERIMENTAL_IDCT`.

## Build with PS2SDK

### Single ELF with an on-screen result

With the GCC/newlib EE toolchain and PS2SDK available in WSL, run from
the repository root:

```sh
bash simd/ps2/build-test-elf.sh
```

Set `PS2DEV` and `PS2SDK` if they are installed outside `/usr/local/ps2dev`.
If the SDK is not installed, the script can build the required EE libraries
from an existing checkout specified by `PS2SDK_SOURCE` (default:
`$PS2DEV/ps2dev/build/ps2sdk`). This requires no dependency download.
The copied SDK source, libraries, and startup object stay in
`build-ps2-sdk/`; the compiler installation is not modified.

Open `build-ps2/simd/ps2_mmi_test_suite.elf` in PCSX2 to run all six
existing tests: MMI primitives, plain/fancy upsampling, downsampling,
integer IDCT, and YCbCr color conversion. The runner uses PS2SDK's debug
screen to display detailed failures and the final result:

```text
TEST: OK! (6/6 groups passed)
```

A failed comparison instead produces `TEST: FAIL!`, with the failing
group and comparison details above it. The ELF sleeps after testing so
the result remains visible until the emulator is reset or stopped.
It uses deterministic generated test data and needs no external images.
These are instruction and kernel tests, not a full JPEG-stream test or
a performance benchmark.

The default suite uses the scalar even-rotation stage. To test the
experimental even-rotation variant, rebuild with:

```sh
PS2_IDCT_EVEN=ON bash simd/ps2/build-test-elf.sh
```

On 2026-10-09, the default suite and the static JPEG library were
successfully compiled and linked with EE GCC 15.1.0 and the existing
PS2SDK source checkout. The ELF header has the R5900 architecture flag,
entry point `0x100e48`, and a load segment starting at `0x00100000`.
The user then ran this ELF in PCSX2 and reported `TEST: OK!` for all six
groups: 1024 primitive iterations, 48 plain upsampling cases, 128 fancy
upsampling cases, 136 downsampling cases, 2048 IDCT reference comparisons,
and 420 color conversion cases. This run used scalar even rotation;
the experimental MMI even-rotation variant has not been verified.

### Individual validation ELFs

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
  ps2_mmi_color_test ps2_mmi_primitives_test
```

Run these ELFs on PS2 hardware or an emulator.  The
plain test covers 48 combinations of width, sampling ratio, and pointer
alignment; the fancy test covers 128 (including source context rows,
different data patterns, boundaries, and deliberately unaligned rows).
All tests check that output padding is untouched.  The downsampling
test covers 136 cases (including right-edge padding and alternating
rounding), and the IDCT test compares 2048 blocks (DC-only, sparse, dense,
single-AC, signed odd/even-frequency stress, and aligned/unaligned
quant/coefficient tables) against the library's
reference integer IDCT.  The color test covers 420 image-row/layout
cases, including all four-byte output layouts, three-byte scalar
reference layouts, unaligned destinations, chroma extremes, and buffer
guard bytes.  The independent `ps2_mmi_primitives_test` checks 1024
deterministic-random iterations of PEXTLB/PEXTUB byte expansion, all
eight PMULTH 16x16 products and their original lane order, both
11-/18-bit LL&M butterfly rounding shifts, and PMAXH/PMINH/PPACB
saturation/packing.  Guard bytes/words are checked separately.
The tests are **not** executed during cross-compilation.

For exercising the plain kernels, use 8-bit JPEG images with 4:2:2 or
4:2:0 subsampling and set `cinfo.do_fancy_upsampling = FALSE` before
`jpeg_start_decompress()`.  The default fancy upsampling now also has
a PS2 MMI path, including 4:2:0.  Compare both settings with a
`WITH_SIMD=OFF` reference build.

## R5900 PMULTH/PCPYUD lane-order regression

The R5900 `PMULTH` instruction deposits halfword multiplications into
two accumulators: LO contains lanes 0, 1, 4, 5 and HI contains lanes
2, 3, 6, 7.  `PCPYLD rd, HI, LO` restores lanes 0..3; **PCPYUD uses
the upper half of its first source as the LOWER output half**.
Therefore `PCPYUD rd, LO, HI` (not `HI, LO`) restores lanes 4..7.
The earlier code used the reversed operand order in this instruction;
the fix and instruction-level regression test are on this branch.

```sh
cmake --build build-ps2 --target ps2_mmi_primitives_test
# Run ps2_mmi_primitives_test.elf on your PS2 or EE emulator.
# Expected: PS2 MMI primitives: PASS (1024 iterations, five kernel checks each)
```

Run this ELF before the full 2048-case IDCT comparison, so an ISA-lane
problem can be separated from an IDCT math or rounding problem.

## Compare even-part IDCT rotation variants

The even-rotation MMI experiment uses only three multiplication lanes and
has **no measured speedup**.  Build and run the IDCT test twice, keeping
the same JPEG test images, compiler flags, and hardware:

```sh
# Default scalar even-rotation path
cmake -S . -B build-ps2-even-off \
  -DCMAKE_TOOLCHAIN_FILE="$PS2DEV/share/ps2dev.cmake" \
  -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_SIMD=ON \
  -DWITH_TOOLS=OFF -DWITH_PS2_MMI_TESTS=ON \
  -DWITH_PS2_EXPERIMENTAL_IDCT=ON \
  -DWITH_PS2_EXPERIMENTAL_IDCT_EVEN=OFF
cmake --build build-ps2-even-off --target ps2_mmi_idct_test

# Experimental three-lane PMULTH even-rotation path
cmake -S . -B build-ps2-even-on \
  -DCMAKE_TOOLCHAIN_FILE="$PS2DEV/share/ps2dev.cmake" \
  -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_SIMD=ON \
  -DWITH_TOOLS=OFF -DWITH_PS2_MMI_TESTS=ON \
  -DWITH_PS2_EXPERIMENTAL_IDCT=ON \
  -DWITH_PS2_EXPERIMENTAL_IDCT_EVEN=ON
cmake --build build-ps2-even-on --target ps2_mmi_idct_test
```

Both test ELFs should report 2048 bit-exact reference comparisons.
Passing does not imply that the experimental kernel is faster.
Benchmark full decode separately for low- and high-entropy JPEG blocks.

## Suggested on-device tests

1. Compare SIMD on/off for 4:2:0 and 4:2:2 images in both fancy/plain modes, including odd widths,
   widths below 32, exactly 32, and non-multiples of 32.
2. Verify aligned and deliberately unaligned input/output rows.
3. Compare pixels against the generic C plain upsampler.
4. Measure full-frame time and, separately, the upsampling stage.

R5900 compilation, linking, and kernel reference comparisons in PCSX2 have
been verified for the default combined suite as described above. Actual
PS2 hardware, full JPEG-stream processing, and performance have **not yet
been tested**. In addition to byte comparison, benchmark separately
for low-entropy and high-entropy coefficient blocks: the IDCT MMI
shortcut can speed DC-only blocks but requires a complete AC scan for
other blocks.  The PMULTH odd-part rotations and butterfly paths may have different
speed tradeoffs due to stack-buffer traffic and reordering.  The assembler
probe also checks `PMULTH`, `PMFLO`/`PMFHI`, `PCPYLD`/`PCPYUD`,
`PADDW`, `PSRAW`, and `PMAXH`.  Passing that probe verifies syntax
support only; it does not verify runtime semantics or timing.  Keep `WITH_PS2_EXPERIMENTAL_IDCT=OFF` unless testing or
benchmarking the IDCT path.  The toolchain check and the new assembly should be
validated before treating this as a production optimization.
