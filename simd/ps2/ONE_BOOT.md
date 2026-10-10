# PS2 EE MMI: single-boot correctness and performance

This optional test executable links all experimental variants as **separate
symbols**, so a PS2 owner can verify and compare them in **one boot**.  The
normal JPEG library keeps experimental dispatch disabled by default.

## Build on your computer, then transfer just one ELF

```sh
PS2_ALL_IN_ONE=ON bash simd/ps2/build-test-elf.sh
# ELF: build-ps2/simd/libjpeg_turbo_mmi.elf
```

From a clean WSL session with an installed EE compiler and PS2SDK:

~~~sh
sudo apt-get update
sudo apt-get install -y git cmake make gcc g++ python3
export PS2DEV="${PS2DEV:-/usr/local/ps2dev}"
export PS2SDK="${PS2SDK:-$PS2DEV/ps2sdk}"
export PATH="$PS2DEV/ee/bin:$PS2DEV/iop/bin:$PATH"
test -x "$PS2DEV/ee/bin/mips64r5900el-ps2-elf-gcc"
test -f "$PS2SDK/ee/startup/linkfile"
git clone --single-branch --branch mmi \
  https://github.com/maekawa-mugi/libjpeg-turbo-retro.git
cd libjpeg-turbo-retro
PS2_ALL_IN_ONE=ON bash simd/ps2/build-test-elf.sh
ls -lh build-ps2/simd/libjpeg_turbo_mmi.elf
~~~

If PS2SDK is installed elsewhere, set PS2SDK to the actual directory
before building, e.g. a PS2-OtherOS modern-deps SDK in /mnt/c/...
The build script enables this mode by default.  In CMake it is controlled by
`WITH_PS2_MMI_TESTS=ON` and `WITH_PS2_MMI_ALL_IN_ONE=ON`.
The build generates separate objects for pure C, scalar matrix + MMI pack,
PMULTH4, PMULTH8, add/pack and vector merged conversion; pure C,
scalar matrix + MMI pack, PMULTH4 and PMULTH8 YCbCr conversion;
and both even-IDCT variants.  It also links the new quantizer and tests.

After linking, the build script runs `preflight-elf.sh` to verify the MIPS
little-endian ELF header and the expected variant/test symbols before
transferring anything to the console.

The assembler probe for `PMADDH` and `PPACH` runs **at build
time**.  If it fails, fix the toolchain before transferring the ELF.

This ELF targets **PS2SDK / ps2-elf**, not native PS2 Linux userspace.
The upper-register ABI, exception and timing behavior must still be checked
separately for PS2 Linux.

## One launch on PCSX2 or PS2 hardware

Launch `ps2_mmi_test_suite.elf`.  The first seven groups are the
existing kernel correctness checks.  Group 8 encodes actual JPEG streams
and compares RGB with all four 4-byte output orders.  Group 9 checks and
times the experimental variants, existing sampling kernels and quantizer.

Expected final summary:

```text
TEST: OK! (9/9 groups passed)
```

Even if one group fails, later groups still run.  Failing variants are not
timed.  The program stays on screen, including **seven representative
best-variant/baseline ratios**.  Photograph the final screen.

Detailed values are written as `CSV,...` lines to **stdout** for
PCSX2 console or serial/host capture, if the setup has stdout forwarding.
No automatic USB or memory-card output is promised: without storage
drivers it would be unreliable.  Test stdout capture before a real PS2
session if the full table matters.

A typical line is:

```text
CSV,merged,pmul8,h2v2_RGBX,128,0,38000,192,198
```

The columns are category, variant, workload, width, alignment, raw PS2SDK
GetTimerSystemTime() ticks, repetitions and rounded ticks/call.  Copy the *entire* console
transcript, including `BENCH_START` and `BENCH_END`, to a
text file.  Back on the PC:

```sh
python3 simd/ps2/analyze-bench.py pcsx2-console.txt
```

The script requires **all 376 timing rows**, a complete 9/9 PASS, matching
workloads and matching post-timer output digests before recommending a
candidate. It ranks by geometric-mean and worst-case speedup against pure C
(or IJG C). By default a candidate must deliver >= 1.05x geometric mean
without any measured case below 0.95x. Otherwise retain the baseline.

## Complete coverage in this ELF

| Group | Variants and independent correctness checks |
|---|---|
| Actual MMI primitives | 8 regressions including PMULTH, PMADDH, PEXT/PADDH, signed rounding and register order |
| Plain upsampling | h2v1/h2v2 MMI and portable C |
| Fancy upsampling | h2v1/h2v2 MMI and portable C, edges and mixed alignments |
| Compressor downsampling | h2v1/h2v2 MMI and portable C, right-edge extension |
| IDCT | IJG scalar, evenoff and evenon, 2048 cases per MMI variant, DC/sparse/dense distributions |
| YCbCr to RGB | pure C plus three MMI paths, 7 layouts, 560 cases per variant |
| Merged RGBX | pure C plus five MMI paths, h2v1/h2v2 and all 4 layouts, 640 cases per variant |
| End-to-end JPEG streams | 384 RGB vs RGBX/BGRX/XBGR/XRGB comparisons over 8 widths, 3 subsampling modes, fancy on/off, islow/ifast |
| Integer quantization | IJG reciprocal scalar and new 8-lane MMI quantizer, 3072 cases including signed/divisor edges and quality-75/95 JPEG matrices |

The preexisting tests continue to run unchanged.  The full JPEG-stream
test checks output-layout consistency and real compression/decompression,
not against an entirely SIMD-disabled second library.  New compiled variants
are compared by direct calls; enabling them inside ordinary JPEG dispatch
still requires selecting an experimental compile flag.

## Measurements and interpretation

Each contender warms up for 24 calls. The benchmark rotates the
execution order (AB/BA, ABC/BCA/CAB or the equivalent six-way rotation),
then takes an even **median** from 6 to 12 samples. It uses PS2SDK
GetTimerSystemTime() with a 64-bit counter, based on the OpenSSL-retro
A/B/F harness methodology. Buffers are prepared/reset outside timing;
output hashes are compared after each timed batch. Any mismatch makes
the workload invalid. Timer ticks do not automatically equal CPU cycles;
PCSX2 results cannot replace actual EE hardware timing.

Small/odd widths, aligned/unaligned rows, DC-only and dense IDCT are
separate workloads.  Beware improvements on one workload that regress others.

## Preflight to perform without the console

```sh
for stem in color-math add-pack vector-offsets quantize; do
  cc -std=c99 -O1 -g -Wall -Wextra -Werror \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "simd/ps2/test-${stem}-host.c" -o "/tmp/test-${stem}"
  "/tmp/test-${stem}"
done
bash -n simd/ps2/build-test-elf.sh
python3 -m py_compile simd/ps2/analyze-bench.py
```

The branch has a GitHub Actions host preflight workflow for the same tests.
These are **arithmetic-only**, and do not establish R5900 assembly semantics.

## Explicit limits

The all-in-one runner adds a PS2 quantizer, a regular 8-pixel color
conversion path, and extensive sampling/IDCT coverage.  **EE FDCT,
compressor RGB->YCbCr SIMD, RGB565 and Huffman-specific acceleration are
not yet implemented**.  Normal PS2 Linux ABI validation and whole-image
performance comparison to a SIMD-disabled library are also outstanding.

The PR should stay Draft until the complete ELF cross-compiles, shows
**9/9** on PS2/PCSX2, produces no `FAIL` or `SKIP` records,
and real hardware benchmarks justify enabling any experiment.

## Register pipeline candidates (2026-10-10)

The default all-in-one ELF now compares these additional candidates with
both C and the previous MMI implementations:

- `idct/batch`: one assembly loop for the 64 aligned dequantization
  products instead of eight calls.  Unaligned inputs retain the fallback.
- `color/regpack`: PMULTH/PMADDH, channel reconstruction, one rounding
  step, clipping and packing without intermediate product/channel stores.
  Four RGBX layouts and scalar tails are included.
- `quantize/regpipe`: register-based magnitudes, correction, unsigned
  reciprocal adjustment, per-coefficient shifts and sign restoration.
  Unsafe ranges and unaligned buffers retain the reference/fallback paths.
  The old implementation remains the `mmi` contender.

No SPR is used.  The complete CSV matrix is now 420 rows; use the matching
`analyze-bench.py`.  All three candidates are automatically linked into
`PS2_ALL_IN_ONE=ON` builds without additional flags.  The new color and
IDCT candidates are benchmark-only until their measured performance is
established; quantization remains behind the existing experimental
quantization dispatch option.  Faster performance has not yet been
established on either hardware or PCSX2.  Check the same-ELF correctness
results and CSV before selecting a production implementation.
