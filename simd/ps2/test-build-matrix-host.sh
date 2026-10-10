#!/usr/bin/env bash
# Host-only one-ELF integration test. No cross-toolchain is required.
# The mock CMake records all requested variant flags in the produced ELF.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
temp=$(mktemp -d)
trap 'rm -rf "$temp"' EXIT
mkdir -p "$temp/project/simd/ps2" "$temp/ps2dev/ee/bin" "$temp/ps2sdk/ee/lib"
for lib in libdebug.a libcdvd.a libpthreadglue.a; do
  : > "$temp/ps2sdk/ee/lib/$lib"
done
cp "$root/simd/ps2/build-test-elf.sh" "$temp/project/simd/ps2/build-test-elf.sh"
cat > "$temp/project/simd/ps2/preflight-elf.sh" <<'MOCK_PREFLIGHT'
#!/usr/bin/env bash
set -euo pipefail
[[ -s "$1" ]]
[[ "${PS2_VU0_IDCT:-}" == ON ]]
[[ "${PS2_VIF0_DMA:-}" == ON ]]
[[ "${MOCK_FAIL_PREFLIGHT:-OFF}" != ON ]]
MOCK_PREFLIGHT
cat > "$temp/ps2dev/ee/bin/cmake" <<'MOCK_CMAKE'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == -S ]]; then
  while (($#)); do
    case "$1" in
      -B) directory=$2; shift 2;;
      -DWITH_PS2_*) flags="${flags:-}$1,"; shift;;
      *) shift;;
    esac
  done
  mkdir -p "$directory"
  printf '%s\n' "$flags" > "$directory/mock-config.txt"
else
  [[ "$1" == --build ]]
  [[ "${MOCK_FAIL_BUILD:-OFF}" != ON ]]
  mkdir -p "$2/simd"
  cp "$2/mock-config.txt" "$2/simd/libjpeg_turbo_mmi.elf"
fi
MOCK_CMAKE
chmod +x "$temp/ps2dev/ee/bin/cmake"

run() {
  PS2DEV="$temp/ps2dev" PS2SDK="$temp/ps2sdk" BUILD_JOBS=1 \
    bash "$temp/project/simd/ps2/build-test-elf.sh"
}
# Simulate outputs left behind by the old 16-profile matrix.
printf stale > "$temp/project/libjpeg_turbo_mmi_evenoff_exact.elf"
printf stale > "$temp/project/libjpeg_turbo_mmi_evenon_fpu_vu0_dma.elf"
mkdir -p "$temp/project/build-ps2/profiles/evenoff_exact/simd"
printf stale > "$temp/project/build-ps2/profiles/evenoff_exact/simd/libjpeg_turbo_mmi.elf"
printf stale > "$temp/project/ps2-elf-manifest.csv"

run > "$temp/one.log"
[[ $(find "$temp/project" -maxdepth 1 -type f -name '*.elf' | wc -l) -eq 1 ]]
[[ -s "$temp/project/libjpeg_turbo_mmi.elf" ]]
[[ ! -e "$temp/project/ps2-elf-manifest.csv" ]]
[[ ! -e "$temp/project/libjpeg_turbo_mmi_evenoff_exact.elf" ]]
[[ ! -e "$temp/project/libjpeg_turbo_mmi_evenon_fpu_vu0_dma.elf" ]]
[[ ! -d "$temp/project/build-ps2/profiles" ]]
[[ $(find "$temp/project/build-ps2" -mindepth 1 -maxdepth 1 -type d | wc -l) -eq 1 ]]
output="$temp/project/libjpeg_turbo_mmi.elf"
for switch in \
  'WITH_PS2_MMI_ALL_IN_ONE=ON' \
  'WITH_PS2_EXPERIMENTAL_IDCT_EVEN=OFF' \
  'WITH_PS2_APPROX_FPU_IDCT=OFF' \
  'WITH_PS2_BENCH_APPROX_FPU_IDCT=ON' \
  'WITH_PS2_EXPERIMENTAL_VU0_IDCT=ON' \
  'WITH_PS2_EXPERIMENTAL_VIF0_DMA=ON'; do
  grep -q -- "$switch" "$output"
done
cp "$output" "$temp/expected"
if MOCK_FAIL_BUILD=ON run >/dev/null 2>&1; then
  echo "build failure incorrectly succeeded" >&2
  exit 1
fi
cmp "$output" "$temp/expected"
if MOCK_FAIL_PREFLIGHT=ON run >/dev/null 2>&1; then
  echo "preflight failure incorrectly succeeded" >&2
  exit 1
fi
cmp "$output" "$temp/expected"
if PS2_BUILD_MATRIX=all run >/dev/null 2>&1; then
  echo "obsolete matrix mode incorrectly accepted" >&2
  exit 1
fi
echo "PASS: one root ELF, all contender flags, old cleanup and failure atomicity"
