#!/usr/bin/env bash
# Validate one-boot PS2 EE test ELF before transferring it.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
elf=${1:-"$root/libjpeg_turbo_mmi.elf"}
: "${PS2DEV:=/usr/local/ps2dev}"
[[ -s "$elf" ]] || { echo "Missing or empty test ELF: $elf" >&2; exit 1; }

ee_nm="$PS2DEV/ee/bin/mips64r5900el-ps2-elf-nm"
ee_readelf="$PS2DEV/ee/bin/mips64r5900el-ps2-elf-readelf"
[[ -x "$ee_nm" ]] || { echo "EE nm missing: $ee_nm" >&2; exit 1; }
[[ -x "$ee_readelf" ]] || { echo "EE readelf missing: $ee_readelf" >&2; exit 1; }

header=$("$ee_readelf" -h "$elf")
grep -qi "MIPS" <<<"$header" || { echo "Not a MIPS ELF" >&2; exit 1; }
grep -qi "little endian" <<<"$header" || {
  echo "Unexpected EE endianness" >&2
  exit 1
}

symbol_table=$("$ee_nm" --defined-only "$elf" | awk 'NF >= 3 { print $NF }')
required=(
  main
  ps2_test_primitives
  ps2_test_merged
  ps2_test_jpeg_stream
  ps2_bench_execute
  ps2_bench_compare
  ps2_bench_run_quantize
  ps2_bench_scalar_h2v1_rgbx
  ps2_bench_pmul4_h2v1_rgbx
  ps2_bench_pmul8_h2v2_bgrx
  ps2_bench_addpack_h2v2_xbgr
  ps2_bench_vector_h2v2_xrgb
  ps2_bench_scalar_rgbx
  ps2_bench_pmul4_rgbx
  ps2_bench_pmul8_rgbx
  ps2_bench_regpack_rgbx
  ps2_bench_idct_batch
  ps2_bench_idct_direct
  ps2_bench_idct_lut
  ps2_bench_idct_lut_norow
  ps2_bench_idct_evenoff
  ps2_bench_idct_evenon
  jsimd_quantize_ps2mmi
  jsimd_quantize_legacy_ps2mmi
)
case "${PS2_VU0_IDCT:-OFF}" in
  ON|on|TRUE|true|1) required+=(ps2_bench_run_vu0) ;;
esac
case "${PS2_VIF0_DMA:-OFF}" in
  ON|on|TRUE|true|1) required+=(ps2_bench_run_vif0_dma) ;;
esac
for symbol in "${required[@]}"; do
  if ! grep -Fxq "$symbol" <<<"$symbol_table"; then
    echo "ELF preflight: missing symbol $symbol" >&2
    exit 1
  fi
done
echo "ELF preflight: PASS (${#required[@]} required symbols present)"
echo "ELF path: $elf"
echo "This single ELF is ready for transfer."
