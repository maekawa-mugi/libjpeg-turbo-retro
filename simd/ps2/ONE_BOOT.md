# PS2 EE MMI: single-boot correctness and performance

This optional test executable links all experimental variants as **separate
symbols**, so a PS2 owner can verify and compare them in **one boot**.  The
normal JPEG library keeps experimental dispatch disabled by default.

## One build, one ELF, one final verdict

```sh
bash simd/ps2/build-test-elf.sh
# Result: ./libjpeg_turbo_mmi.elf
```

This is the **only** PCSX2 executable published by the default build,
at the repository root. Object files and CMake cache stay under
`build-ps2/one-elf/`. The script replaces the old 16-ELF matrix,
validates the binary with EE `readelf`/`nm`, and removes previously
generated `libjpeg_turbo_mmi_even*.elf` files and the obsolete profile
manifest **only after** a successful build. It never deletes unrelated
ELF files. Keep the same binary for all tests.

Every benchmark candidate is compiled into that **same** binary:
the existing scalar/MMI/table color and merged variants, plain/fancy
upsampling and downsampling, integer IDCT `ijg_c/evenoff/evenon/batch/direct`,
quantizer variants, the opt-in accuracy-checked `fpu_approx`, a real
VU0 macro-mode float-matrix IDCT, and the separate VIF0 DMA upload test.

Important: the approximate `fpu_approx` contender is compiled **only
for benchmarking** via `WITH_PS2_BENCH_APPROX_FPU_IDCT=ON`.
The library's default decoder remains IJG integer IDCT
(`WITH_PS2_APPROX_FPU_IDCT=OFF`). Evenoff and evenon are both present
as independent exact contenders, so 16 build-time combinations are
unnecessary. Runtime JPEG stream tests exercise ISLOW, IFAST, and FLOAT
by setting the desired method explicitly.

### Final screen interpretation

After 8/8 correctness checks and all benchmark groups, the GS screen
automatically displays:

```text
EXACT IDCT: ijg_c   1.00x | FPU~: ...x / N/A
VU0 vs float: ...x | DMA vs CPU upload: ...x
...
TAKEAWAY: exact ijg_c ; VU0 experimental; skip DMA
...
RESULT: PASS | tests 8/8 | bench failures 0
```

The precise text and winner depend on the measured PS2/PCSX2 results.
The exact IDCT winner is selected **only** among bit-exact integer
candidates; float FPU is never promoted to an exact integer winner.
The FPU ratio compares the approximate IDCT against `ijg_c` on
the same dense 8x8 block; FPU results are hidden if the maximum
absolute pixel error exceeds 3. The VU0 ratio is vs an **equivalent
scalar float matrix**, *not* vs IJG integer. The DMA ratio measures
only **256-byte memory transfer**, *not* the decode pipeline.

The takeaway uses a representative dense IDCT sample and a provisional
5% benefit threshold. It does not change libjpeg dispatch or claim
whole-image speedups. Full comparable workload results, pixel-error
reports, and `CONCLUSION,...` machine-readable verdict lines remain
available on stdout. A failed test displays "VERDICT INVALID", never
an endorsement of unvalidated timing.

The root ELF can still be launched with existing PCSX2 scripts.
For reproducible host-only checks:
`bash simd/ps2/verify-bench-host.sh`.
The PS2 EE toolchain is required to produce the real ELF.

## Novelty experiment: VU0 arithmetic and VIF0 DMA upload

The one-ELF test suite always includes two additional benchmark experiments.
They are deliberately separate from the exact integer IDCT contenders.

```sh
# Both experiments in the same ELF (no PR or runtime decoder changes):
bash simd/ps2/build-test-elf.sh

# VU0 arithmetic and VIF0 DMA transfer are both always included.
```

**VU0 IDCT (`vu_idct`, experimental on-screen row 8):** Computes a real
8x8 separable inverse DCT. The basis matrix matches the mathematical
orthonormal DCT; each COP2 `LQC2/VMUL/VADD/SQC2` dot product performs four
floating lanes at a time. The baseline `scalar_matrix` executes the
**same float matrix**, including dequantization, two passes, transposition
and output clamping. `vu0_macro` executes the same conversion but
uses VU0 macro instructions for each 4-lane dot. The entire call,
including staging, is timed. A separate 128-block differential test
reports `max_abs_diff` and `different_pixels`; >3 levels disables
timing. This is not IJG integer bit-exact and is not wired to normal
JPEG decode. Comparing its timing against `idct/ijg_c` is informative,
but those implementations do not perform identical arithmetic.

**VIF0 DMA (`vif0_dma`, experimental on-screen row 9):** A real
DMA channel-0 transfer with a two-quadword VIF0 command header
(`STCYCL(1,1)`, `UNPACK V4_32`) and 256 bytes of aligned payload
into VU0's memory-mapped data RAM. `cpu_store` writes the same words
to the same VU0 address; `vif0_dma` includes cache flush, DMA start
and bounded DMA/VIF completion waits in its measured interval.
Both verify the same 256-byte data. The test temporarily masks the
documented VIF0 DMAtag mismatch-detection erratum (`VIF0_ERR.ME0=1`),
requires VIF0's FIFO to be empty and VPS to be idle before CPU reads
VU0 data RAM, and restores the original VIF error mask on exit.
This is a **transfer-only** benchmark, not a DMA-fed IDCT, and must
not be reported as an IDCT speedup.
The VIF0 DMA channel must be idle/owned exclusively during this
opt-in experiment, which should be run in isolation from other VIF0 users.

The GS display shows `vu_idct / vu0_macro / <ratio>x` and
`vif0_dma / vif0_dma / <ratio>x` even when the ratio is less than 1.00.
Detailed measurements are recorded as:

```text
APPROX,vu_idct,vu0_macro,128_blocks,max_abs_diff=...,different_pixels=...
CSV,vu_idct,scalar_matrix,dense,8,0,...
CSV,vu_idct,vu0_macro,dense,8,0,...
PASS,vif0_dma,upload256,256_bytes_verified
CSV,vif0_dma,cpu_store,upload256,256,0,...
CSV,vif0_dma,vif0_dma,upload256,256,0,...
```

The base CSV matrix is **484** rows, with 16 more if the
earlier FPU experiment qualifies, plus 2 each for VU0 and DMA
(488 total with VU0 and VIF0 DMA, 504 including qualifying FPU).
The report parser prints both experimental ratios separately from
the production JPEG candidate verdicts. Neither experiment is
known to be faster on PCSX2 or actual EE until run. The VU0 instruction
set and VIF0 DMA completion behavior must be verified on hardware.

## Exact IDCT direct contender and PS2 COP1 floating opt-in

The one-boot IDCT matrix now includes `direct`, a separate bit-exact
implementation of the existing Loeffler integer IDCT.  The `direct`
contender keeps the MMI global DC-only check but avoids the 64-element
pre-dequantization buffer and the repeated PMULTH/HI/LO/temporary-array
round trips in the non-DC path.  It instead dequantizes only the coefficients
actually needed per active column and computes the rotators and final
butterflies directly with IJG fixed-point arithmetic.  Like the other
integer contenders, it is compared **byte for byte** against `ijg_c`
in all 2048 block cases before its timing can be accepted.  It remains
benchmark-only; do not assume it beats `ijg_c` without the console CSV.

The library-only option `WITH_PS2_APPROX_FPU_IDCT=ON` changes the
**default decoder** IDCT to the existing IJG AA&N floating-point inverse
transform (`JDCT_FLOAT`) on the EE's COP1 hardware.  It is **not** a
bit-exact PS2 MMI kernel and is not a change to JPEG compression.
The R5900 FPU supports only single precision, truncation-style rounding,
and non-IEEE exceptional/denormal behavior.  Applications selecting their
own `cinfo.dct_method` continue to override the new default.  The
accuracy/quality impact must be judged on actual output images.

```sh
# Exact integer IDCT contenders (default) including direct:
bash simd/ps2/build-test-elf.sh

# Optional native float decoder + fpu_approx timing candidate:
bash simd/ps2/build-test-elf.sh
```

In float mode the log prints
`APPROX,idct,fpu_approx,2048_cases,max_abs_diff=...,different_pixels=...,guards=PASS`.
This is not an integer correctness PASS.  The `fpu_approx` candidate
joins the timing matrix only if all 2048 cases have intact sentinels
and a maximum absolute pixel difference of 3 or less.  Its per-variant
output digest must remain stable in the timed repeats, but it is
allowed to differ from IJG's strict integer digest; all other
contenders still require byte-exact agreement.  JPEG-stream layout
testing adds `JDCT_FLOAT` when the option is enabled.

The exact benchmark matrix contains 488 rows with VU0 + DMA, or 504 when
`fpu_approx` passes its quality gate.  The verdict parser
(`simd/ps2/analyze-bench.py`) accepts either complete matrix and
supports the new 8/8 `LIBJPEG_PS2,DONE,PASS` transcript.

## Build on your computer, then choose a root-level ELF

```sh
bash simd/ps2/build-test-elf.sh
# ELFs: ./libjpeg_turbo_mmi_evenoff_exact.elf, ..., ./libjpeg_turbo_mmi.elf
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
bash simd/ps2/build-test-elf.sh
ls -lh libjpeg_turbo_mmi.elf
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

Launch one of the generated `libjpeg_turbo_mmi_*.elf` files.  The first seven groups are the
existing kernel correctness checks.  Group 8 encodes actual JPEG streams
and compares RGB with all four 4-byte output orders.  Group 9 checks and
times the experimental variants, existing sampling kernels and quantizer.

Expected final GS summary: `RESULT: PASS | tests 8/8 | bench failures 0`.

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

The script requires **all applicable timing rows** and a complete 8/8 PASS, matching
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

No SPR is used by this quantizer. The one-ELF CSV matrix contains 488 rows
(504 with qualifying FPU); use the matching
`analyze-bench.py`.  All three candidates are automatically linked into
`PS2_ALL_IN_ONE=ON` builds without additional flags.  The new color and
IDCT candidates are benchmark-only until their measured performance is
established; quantization remains behind the existing experimental
quantization dispatch option.  Faster performance has not yet been
established on either hardware or PCSX2.  Check the same-ELF correctness
results and CSV before selecting a production implementation.
