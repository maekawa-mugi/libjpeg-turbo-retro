#!/usr/bin/env bash
# Host-only integration test: no PS2 compiler or SDK required.
# Fake CMake produces a profile-specific ELF blob, fake preflight checks
# that the matching VU0/DMA flags were passed to the validation stage.
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
case "${PS2_VU0_IDCT:-}" in ON|OFF) ;; *) exit 1;; esac
case "${PS2_VIF0_DMA:-}" in ON|OFF) ;; *) exit 1;; esac
MOCK_PREFLIGHT
cat > "$temp/ps2dev/ee/bin/cmake" <<'MOCK_CMAKE'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == -S ]]; then
  flags=""
  while (($#)); do
    case "$1" in
      -B) directory=$2; shift 2;;
      -DWITH_PS2_*) flags="$flags$1,"; shift;;
      *) shift;;
    esac
  done
  mkdir -p "$directory"
  printf '%s\n' "$flags" > "$directory/mock-config.txt"
else
  [[ "$1" == --build ]]
  mkdir -p "$2/simd"
  cp "$2/mock-config.txt" "$2/simd/libjpeg_turbo_mmi.elf"
fi
MOCK_CMAKE
chmod +x "$temp/ps2dev/ee/bin/cmake"

run() {
  PS2DEV="$temp/ps2dev" PS2SDK="$temp/ps2sdk" BUILD_JOBS=1 \
    bash "$temp/project/simd/ps2/build-test-elf.sh"
}
PS2_BUILD_MATRIX=all run > "$temp/all.log"
[[ $(find "$temp/project" -maxdepth 1 -type f -name '*.elf' | wc -l) -eq 17 ]]
[[ $(wc -l < "$temp/project/ps2-elf-manifest.csv") -eq 17 ]]
[[ $(find "$temp/project/build-ps2/profiles" -mindepth 1 -maxdepth 1 -type d | wc -l) -eq 16 ]]
[[ -s "$temp/project/libjpeg_turbo_mmi_evenoff_exact.elf" ]]
[[ -s "$temp/project/libjpeg_turbo_mmi_evenon_fpu_vu0_dma.elf" ]]
cmp "$temp/project/libjpeg_turbo_mmi.elf" \
    "$temp/project/libjpeg_turbo_mmi_evenoff_exact.elf"
PS2_BUILD_MATRIX=core run > "$temp/core.log"
[[ $(wc -l < "$temp/project/ps2-elf-manifest.csv") -eq 9 ]]
PS2_BUILD_MATRIX=single PS2_VU0_IDCT=ON PS2_IDCT_EVEN=ON run > "$temp/single.log"
[[ $(wc -l < "$temp/project/ps2-elf-manifest.csv") -eq 2 ]]
grep -q '^evenon_exact_vu0,' "$temp/project/ps2-elf-manifest.csv"
if PS2_BUILD_MATRIX=wrong run >/dev/null 2>&1; then
  echo "invalid matrix mode accepted" >&2
  exit 1
fi
if PS2_BUILD_MATRIX=all PS2_ALL_IN_ONE=OFF run >/dev/null 2>&1; then
  echo "legacy-only mode unexpectedly accepted with matrix" >&2
  exit 1
fi
echo "PASS: 16 all, 8 core, single, root ELF/manifest, alias, bad mode"
