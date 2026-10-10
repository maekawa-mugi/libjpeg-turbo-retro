#!/usr/bin/env bash
# Native syntax-check of PS2 benchmark runner as in openssl-retro's
# test/ps2/verify-bench-host.sh. No R5900 instructions execute here.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
cc=${CC:-cc}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cat >"$tmp/timer.h" <<'EOF'
#ifndef PS2_BENCH_FAKE_TIMER_H
#define PS2_BENCH_FAKE_TIMER_H
#include <stdint.h>
extern uint64_t GetTimerSystemTime(void);
#endif
EOF
cat >"$tmp/debug.h" <<'EOF'
#ifndef PS2_BENCH_FAKE_DEBUG_H
#define PS2_BENCH_FAKE_DEBUG_H
void init_scr(void);
void scr_printf(const char *, ...);
void scr_setXY(int, int);
void scr_setfontcolor(unsigned int);
void scr_setCursor(int);
#endif
EOF
cat >"$tmp/kernel.h" <<'EOF'
#ifndef PS2_BENCH_FAKE_KERNEL_H
#define PS2_BENCH_FAKE_KERNEL_H
void SleepThread(void);
void SyncDCache(void *, void *);
#endif
EOF
files=(
  simd/ps2/bench-ab.c
  simd/ps2/bench-suite.c
  simd/ps2/bench-merged.c
  simd/ps2/bench-color.c
  simd/ps2/bench-sampling.c
  simd/ps2/bench-idct.c
  simd/ps2/bench-quantize.c
)
for file in "${files[@]}"; do
  echo "SYNTAX: $file ($cc)"
  "$cc" -std=c99 -Wall -Wextra -Werror -fsyntax-only \
    -DPS2_BENCH_HOST_CHECK -I"$tmp" "$file"
done
# Verify the non-bit-exact FPU contender's optional code path as well.
"$cc" -std=c99 -Wall -Wextra -Werror -fsyntax-only \
  -DPS2_BENCH_HOST_CHECK -DPS2_APPROX_FPU_IDCT \
  -I"$tmp" simd/ps2/bench-idct.c
"$cc" -std=c99 -Wall -Wextra -Werror -fsyntax-only \
  -DPS2_BENCH_HOST_CHECK -DPS2_EXPERIMENTAL_VU0=1 \
  -DPS2_EXPERIMENTAL_VIF0_DMA=1 -I"$tmp" \
  simd/ps2/bench-vu-experiment.c
"$cc" -std=c99 -Wall -Wextra -Werror -fsyntax-only \
  -DPS2_MMI_ALL_IN_ONE -I"$tmp" simd/ps2/test-suite.c
bash -n simd/ps2/build-test-elf.sh
bash -n simd/ps2/preflight-elf.sh
bash simd/ps2/test-build-matrix-host.sh
# Execute the one-shot verdict using synthetic but internally consistent
# timings, verifying strict IDCT excludes faster approximate FPU entries.
"$cc" -std=c99 -O1 -Wall -Wextra -Werror \
  -DPS2_BENCH_HOST_CHECK \
  -DPS2_EXPERIMENTAL_VU0=1 -DPS2_EXPERIMENTAL_VIF0_DMA=1 \
  -I"$tmp" simd/ps2/bench-suite.c \
  simd/ps2/test-one-shot-verdict-host.c -o "$tmp/one-shot-verdict"
"$tmp/one-shot-verdict"
python3 -m py_compile simd/ps2/analyze-bench.py
python3 simd/ps2/test_analyze_bench.py
echo "PASS: native benchmark syntax, shell and verdict regression"
