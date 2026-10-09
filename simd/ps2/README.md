# PlayStation 2 Emotion Engine SIMD backend

This directory contains the PS2 EE (R5900) **128-bit MMI** backend.
It is independent of `simd/mips64/`, which uses **Loongson's distinct
64-bit MMI instruction set**.  VU0/VU1 are not enabled by this backend.

## OpenSSL-retro-style scalar/A/B conclusions

The single-boot ELF now compares a **true portable-C baseline** against
five merged MMI variants and three ordinary YCbCr MMI variants. The old
"scalar" variant only uses scalar chroma calculations while retaining
MMI packing, so it is no longer incorrectly treated as a pure-C baseline.

Timing follows the design of openssl-retro's `test/ps2/main.c` from its
`codex/ps2-ee-mmi-test` branch: 24 warm-ups, rotating AB/BA or A/B/C
orders, six to twelve 64-bit `GetTimerSystemTime()` samples and an even
median. Output hashes are compared outside the timed interval both
before and after each batch. Any failure disables that workload's score.

All 368 comparison rows must be present, with a final 9/9 correctness
PASS. Run `python3 simd/ps2/analyze-bench.py ps2-console.log` to obtain
geometric-mean speedups, worst-case regressions and a gated candidate
(>=1.05x geometric mean, >=0.95x worst by default).

Run `bash simd/ps2/verify-bench-host.sh` before building the EE ELF,
or `CC=clang bash simd/ps2/verify-bench-host.sh` to catch accidental
host-side syntax issues in seven benchmark source files.
## One-boot PS2 validation (recommended)

For the integrated all-kernel test and side-by-side benchmark, read
[ONE_BOOT.md](ONE_BOOT.md) first.  The build script defaults to
`PS2_ALL_IN_ONE=ON` and produces one ELF containing:

- All five merged RGBX variants and all three regular YCbCr converters
  (including the new eight-pixel color PMULTH batch).
- Both integer IDCT even-rotation choices, existing plain/fancy
  upsampling and downsampling, and the new experimental eight-lane
  integer quantization kernel.
- 384 complete JPEG-stream output-layout comparisons, extended
  correctness tests and rotated-order 64-bit median A/B timing.
- CSV output for offline ranking with `analyze-bench.py` and seven
  representative speed ratios on screen.

Use `bash simd/ps2/build-test-elf.sh`, run the generated
`ps2_mmi_test_suite.elf` **once**, and check for `TEST: OK! (9/9
groups passed)`.  The earlier 7/7 instructions below refer to the
legacy non-integrated build.  All new optimizations stay disabled in
normal JPEG dispatch until hardware validation and benchmarking.

## Current coverage

- `h2v1` plain upsampling: 16-byte MMI loads, byte interleave, 32-byte
  stores, with a bounded scalar tail.
- `h2v2` plain upsampling: duplicate the expanded MMI registers directly into
  both output rows, avoiding a subsequent row-sized `memcpy()` for aligned
  buffers.  Unaligned input/output rows and odd-width tails remain scalar.
- `h2v1` fancy upsampling: exact 3:1 triangle interpolation with eight
  horizontal pixels at a time processed in R5900 MMI.
- `h2v2` fancy upsampling: fused vertical 3:1 and horizontal 3:1 filtering
  in eight 16-bit MMI lanes for interior samples, followed by interleaved
  16-byte stores.  Edge pixels and unaligned destinations use bit-exact C.
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
  *partial* SIMD converter by default.  The optional
  `WITH_PS2_EXPERIMENTAL_COLOR_PMULTH=ON` kernel replaces four pixels'
  12 scalar coefficient multiplications with two eight-lane `PMULTH`
  instructions, using the same IJG 16.16 rounding and output clipping.
  Reconstruction, pack, and misalignment handling remain unchanged.
  This is still partial SIMD, not a completely vectorized matrix.
  The option defaults to OFF and must be benchmarked against the scalar
  multiplication variant before deployment.  Three-byte RGB and BGR
  continue to use the existing C converter.
- Optional merged 4:2:2/4:2:0 upsampling plus YCbCr conversion for the
  4-byte RGBX/BGRX/XBGR/XRGB families.  Reuse each chroma sample for two
  horizontal (and optionally two vertical) luma pixels, avoiding separate
  expanded chroma rows.  YCbCr matrix multiplication is still scalar;
  four-pixel saturation and interleaved output packing use R5900 MMI.
  Enable with `WITH_PS2_EXPERIMENTAL_MERGED=ON` (default OFF).
  `WITH_PS2_EXPERIMENTAL_MERGED_PMULTH=ON` (also default OFF) replaces
  the two scalar chroma matrix evaluations per four pixels with one eight-lane
  `PMULTH`.  The fixed-point constants are decomposed into signed 16-bit
  multipliers, preserving IJG rounding.  Benchmark both variants because
  PMULTH HI/LO extraction and temporary arrays may erase the arithmetic gain.
  Optional `WITH_PS2_EXPERIMENTAL_MERGED_PMULTH8=ON` (default OFF)
  processes four chroma samples and eight luma pixels per iteration,
  sharing the coefficient load between two PMULTH instructions and
  reducing loop/control overhead.  It requires
  `WITH_PS2_EXPERIMENTAL_MERGED_PMULTH=ON` and retains the four-pixel
  PMULTH kernel for the final block.  The remaining one or two luma
  pixels now reuse offsets within a chroma pair.
  An additional default-OFF
  `WITH_PS2_EXPERIMENTAL_MERGED_ADD_PACK=ON` option makes the eight-pixel
  path add luma values to these chroma offsets with `PEXT[LU]B + PADDH`
  before `PMAXH/PMINH/PPACB`.  The four-byte Y values are expanded to
  signed halfwords inside MMI rather than performing a separate scalar
  addition for every color channel and output row.  The chroma offsets
  are shared between both output rows for 4:2:0.  This is not a fully
  SIMD matrix, since the PMULTH results are still reconstructed in C.
  The additional luma word/offset preparation can outweigh reduced
  scalar arithmetic; benchmark before enabling.
  The further `WITH_PS2_EXPERIMENTAL_MERGED_VECTOR_OFFSETS=ON` option
  (default OFF; requires ADD_PACK) retains the chroma products inside
  MMI, uses `PMULTH + PMADDH` to accumulate green's two fractional
  terms, then `PADDW/PSRAW/PPACH` for exact signed 16.16 rounding and
  `PCPYLD/PCPYUD` for duplicating each chroma sample across two pixels.
  This eliminates the 16-word intermediate product array and scalar
  chroma offset reconstruction, but requires preparing operand and
  rounding-bias vectors and is not yet measured on hardware.
  The 3-byte formats and RGB565 remain on the portable merged converter.
- Experimental eight-lane reciprocal JPEG quantization is implemented
  in `jquanti-mmi.c`; choose `WITH_PS2_EXPERIMENTAL_QUANTIZE=ON`
  (default OFF) for normal compression dispatch.  It handles unusual
  operand magnitudes with an IJG-equivalent C fallback and has a separate
  2048-case on-EE regression and comparison against the scalar formula.
- Regular four-byte YCbCr conversion has an independent eight-pixel
  `PMULTH` option: `WITH_PS2_EXPERIMENTAL_COLOR_PMULTH8=ON`
  requires `WITH_PS2_EXPERIMENTAL_COLOR_PMULTH=ON`; both default OFF.
- Remaining JPEG SIMD hooks use generic C, including
  **MMI FDCT and ifast IDCT**, RGB-to-YCbCr encode, and RGB565.
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

Open `build-ps2/simd/ps2_mmi_test_suite.elf` in PCSX2 to run all seven
test groups: MMI primitives, plain/fancy upsampling, downsampling,
integer IDCT, YCbCr color conversion, and merged upsampling/color conversion.
The runner uses PS2SDK's debug screen to display failures and the new
expected result:

```text
TEST: OK! (7/7 groups passed)
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

To enable the new merged 4-byte output path in the static library as well,
rebuild with:

```sh
PS2_MERGED=ON bash simd/ps2/build-test-elf.sh
```

To test the optional PMULTH color matrix in the same merged kernel:

```sh
PS2_MERGED=ON PS2_MERGED_PMULTH=ON bash simd/ps2/build-test-elf.sh
```

For the new eight-pixel merged batch (separate A/B benchmark):

```sh
PS2_MERGED=ON PS2_MERGED_PMULTH=ON PS2_MERGED_PMULTH8=ON \
  bash simd/ps2/build-test-elf.sh
```

To test the new fused MMI luma addition and pack, while keeping the
baseline eight-pixel variant available:

```sh
PS2_MERGED=ON PS2_MERGED_PMULTH=ON PS2_MERGED_PMULTH8=ON \
PS2_MERGED_ADD_PACK=ON bash simd/ps2/build-test-elf.sh
```

This last option requires `PS2_MERGED_PMULTH8=ON`.
It is experimental and disabled by default.

To try the additional fused `PMADDH` chroma-offset reconstruction:

```sh
PS2_MERGED=ON PS2_MERGED_PMULTH=ON PS2_MERGED_PMULTH8=ON \
PS2_MERGED_ADD_PACK=ON PS2_MERGED_VECTOR_OFFSETS=ON \
bash simd/ps2/build-test-elf.sh
```

The new option requires `PS2_MERGED_ADD_PACK=ON`.  Compare the
scalar baseline, four-pixel PMULTH, eight-pixel PMULTH, luma-add/pack
and vector-offsets variants individually; do not enable by default.

The 8-pixel path requires the 4-pixel PMULTH path.  Compare the
scalar, 4-pixel PMULTH, and 8-pixel PMULTH variants on real hardware;
the 8-pixel variant has not been cross-compiled or run on PS2 yet.

To test the four-pixel PMULTH YCbCr conversion variant:
  
```sh
PS2_COLOR_PMULTH=ON bash simd/ps2/build-test-elf.sh
```

The color standalone test exercises both variants independently of normal
JPEG SIMD dispatch.  This option only changes the internal four-byte
converter; enable `WITH_PS2_EXPERIMENTAL_COLOR=ON` to dispatch to it
during JPEG decoding.  Compare ON and OFF using the same images and
hardware, including small image widths and unaligned output rows.

The merged standalone test runs even when `PS2_MERGED` is OFF.
All five MMI merged variants plus the pure-C control must pass the 14336 reference cases and be
benchmarked separately; no speedup has yet been measured.

On 2026-10-09, the default suite and the static JPEG library were
successfully compiled and linked with EE GCC 15.1.0 and the existing
PS2SDK source checkout. The ELF header has the R5900 architecture flag,
entry point `0x100e48`, and a load segment starting at `0x00100000`.
The user then ran this ELF in PCSX2 and reported `TEST: OK!` for all six
groups: 1024 primitive iterations, 48 plain upsampling cases, 128 fancy
upsampling cases, 136 downsampling cases, 2048 IDCT reference comparisons,
and 420 color conversion cases. This run used scalar even rotation;
the experimental MMI even-rotation variant has not been verified.
Subsequent h2v2 plain/fancy and merged converter changes have **not**
been rerun in PCSX2.  The updated suite now exercises 272 plain, 320 fancy, 6720 color,
and 14336 merged reference cases; rebuild and rerun the ELF
before claiming a new PASS.  The historical 6/6 result does not validate
the new seventh group.

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
  ps2_mmi_color_test ps2_mmi_primitives_test \
  ps2_mmi_merged_test
```

Run these ELFs on PS2 hardware or an emulator.  The
plain test now covers 272 combinations of width, sampling ratio, and
independent input/output-row alignment (including mixed alignment within a
h2v2 row pair); the fancy test now covers 320 cases, including independently
aligned and unaligned source/destination rows, source context rows, and SIMD
boundary widths.
All tests check that output padding is untouched.  The downsampling
test covers 136 cases (including right-edge padding and alternating
rounding), and the IDCT test compares 2048 blocks (DC-only, sparse, dense,
single-AC, signed odd/even-frequency stress, and aligned/unaligned
quant/coefficient tables) against the library's
reference integer IDCT.  The color test now covers 6720 image-row/layout
cases, including all four-byte output layouts, three-byte scalar
reference layouts, independently unaligned Y/Cb/Cr planes and destinations,
chroma extremes, 16.16 IJG rounding, and buffer guard bytes.
The previous 420-case run does not validate these expanded cases or
the new PMULTH path.  The merged test covers 14336 cases across the four
4-byte layouts, h2v1/h2v2 subsampling, odd widths, independent row
alignments, and guard bytes.  It compares against IJG's 16.16
fixed-point rounding and does not decode full JPEG files.
### Host-side color arithmetic verification

The split 16-bit coefficients used by both PMULTH color converters
can be checked exhaustively on a normal Linux workstation without an EE
cross-compiler.  This is arithmetic-only verification, **not** validation
of the MMI register lane order, assembler scheduling, or memory safety.

```sh
cc -std=c99 -O2 -Wall -Wextra -Werror -fsanitize=undefined \
  simd/ps2/test-color-math-host.c -o /tmp/test-color-math
/tmp/test-color-math
# Expected: PASS (65536 Cb/Cr pairs, 256 Y each)
```

The fused PEXT/PADDH + clamp/pack byte layout also has a portable
simulation that exercises 200000 four-pixel vectors, including signed
over/underflow and alpha bytes at offset 0 or 3.  This verifies only
the arithmetic and byte layout, not actual R5900 MMI semantics:

```sh
cc -std=c99 -O2 -Wall -Wextra -Werror -fsanitize=undefined \
  simd/ps2/test-add-pack-host.c -o /tmp/test-add-pack-host
/tmp/test-add-pack-host
# Expected: PS2 fused add/pack host model: PASS (200000 vectors)
```

The dedicated PS2 `ps2_mmi_primitives_test` now checks the same
PEXT/PADDH/PPACB instruction sequence on the EE, with guard bytes.

The experimental vector-offset path has an additional exhaustive
four-layout mathematical test of 262144 eight-pixel batches.  It
checks R/G/B/A placement, chroma replication, negative rounding,
and the luma/clamp output, but cannot validate R5900 execution:

```sh
cc -std=c99 -O2 -Wall -Wextra -Werror -fsanitize=undefined \
  simd/ps2/test-vector-offsets-host.c -o /tmp/test-vector-offsets-host
/tmp/test-vector-offsets-host
# Expected: PASS (262144 batches, 2097152 pixels)
```


The independent `ps2_mmi_primitives_test` checks 1024
deterministic-random iterations of PEXTLB/PEXTUB byte expansion, all
eight PMULTH 16x16 products and their original lane order, two
consecutive PMULTH instructions for sixteen products with guard checks,
fused zero-extension/PADDH/clamp/PPACB with both alpha byte layouts,
PMADDH/PADDW/PSRAW/PPACH chroma reconstruction and lane replication, both
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
# Expected: PS2 MMI primitives: PASS (1024 iterations, eight kernel checks each)
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

The original main-branch MMI kernels passed the default combined suite in
PCSX2 as described above. The new `mmi` branch kernels and expanded tests
require a fresh cross-build and PCSX2 run. Actual
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
