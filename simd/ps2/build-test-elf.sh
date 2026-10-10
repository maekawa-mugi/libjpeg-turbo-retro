#!/usr/bin/env bash
# Build the on-screen test suite with an installed PS2SDK and EE toolchain.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
build_jobs=${BUILD_JOBS:-${JOBS:-$(nproc)}}
(( build_jobs >= 1 )) || { echo "BUILD_JOBS must be >= 1" >&2; exit 2; }
: "${PS2DEV:=/usr/local/ps2dev}"
: "${PS2SDK:=$PS2DEV/ps2sdk}"
if [[ ! -f "$PS2SDK/ee/lib/libdebug.a" &&
      -f "$root/build-ps2-sdk/sdk/ee/lib/libcdvd.a" ]]; then
  PS2SDK="$root/build-ps2-sdk/sdk"
fi
export PS2DEV PS2SDK
export PATH="$PS2DEV/ee/bin:$PS2DEV/iop/bin:$PS2DEV/dvp/bin:$PATH"

# An existing SDK source checkout can also supply the needed EE libraries.
# Keep generated files inside the repository, without changing the toolchain.
if [[ ! -f "$PS2SDK/ee/lib/libdebug.a" ||
      ! -f "$PS2SDK/ee/lib/libcdvd.a" ||
      ! -f "$PS2SDK/ee/lib/libpthreadglue.a" ]]; then
  sdk_source=${PS2SDK_SOURCE:-$PS2DEV/ps2dev/build/ps2sdk}
  [[ -f "$sdk_source/Defs.make" ]] || {
    echo "Set PS2SDK to an installed SDK or PS2SDK_SOURCE to an SDK checkout." >&2
    exit 1
  }
  sdk_build="$root/build-ps2-sdk"
  mkdir -p "$sdk_build/src"
  if [[ ! -f "$sdk_build/src/Defs.make" ]]; then
    tar -C "$sdk_source" --exclude=.git -cf - . |
      tar -C "$sdk_build/src" -xf -
  fi
  export PS2SDKSRC="$sdk_build/src"
  export PS2SDK="$sdk_build/sdk"
  mkdir -p "$PS2SDK/ee/include" "$PS2SDK/ee/lib" \
    "$PS2SDK/ee/startup" "$PS2SDK/common/include"
  cp "$PS2SDKSRC"/common/include/*.h "$PS2SDK/common/include/"
  sdk_log="$sdk_build/build.log"
  : > "$sdk_log"
  for part in startup kernel libcglue libpthreadglue debug rpc/cdvd; do
    echo "Building PS2SDK EE $part (log: $sdk_log)"
    make -C "$PS2SDKSRC/ee/$part" -j"$build_jobs" >> "$sdk_log" 2>&1 || {
      tail -60 "$sdk_log" >&2
      exit 1
    }
    # Avoid startup's release target, which overwrites the compiler's crt0.
    if [[ "$part" == startup ]]; then
      cp "$PS2SDKSRC/ee/startup/obj/crt0.o" \
        "$PS2SDKSRC/ee/startup/src/linkfile" "$PS2SDK/ee/startup/"
    else
      make -C "$PS2SDKSRC/ee/$part" release-ee-include release-ee-lib \
        >> "$sdk_log" 2>&1 || { tail -60 "$sdk_log" >&2; exit 1; }
    fi
  done
fi


# Each existing scalar/MMI/table/direct candidate is already built into
# every single-boot ELF. Enumerate the four actual build-time experiment
# switches to get 16 independent, reproducible binary configurations.
# Intermediate CMake files stay in build-ps2; publish only verified ELFs
# and a checksum manifest at the repository root.
mode=${PS2_BUILD_MATRIX:-all}
case "$mode" in
  all|core|single) ;;
  *) echo "PS2_BUILD_MATRIX must be all, core, or single (got: $mode)" >&2; exit 2 ;;
esac
case "${PS2_ALL_IN_ONE:-ON}" in
  OFF|off|FALSE|false|0)
    if [[ "$mode" != single ]]; then
      echo "PS2_ALL_IN_ONE=OFF requires PS2_BUILD_MATRIX=single" >&2
      exit 2
    fi ;;
esac

flag_is_on() {
  case "${1:-OFF}" in
    ON|on|TRUE|true|1) return 0 ;;
    OFF|off|FALSE|false|0) return 1 ;;
    *) echo "Unsupported flag value: $1 (expected ON or OFF)" >&2; exit 2 ;;
  esac
}
profile_name() {
  local even=$1 fpu=$2 vu0=$3 dma=$4
  local even_lower=${even,,}
  local name="even${even_lower}_exact"
  if flag_is_on "$fpu"; then name="even${even_lower}_fpu"; fi
  if flag_is_on "$vu0"; then name="${name}_vu0"; fi
  if flag_is_on "$dma"; then name="${name}_dma"; fi
  printf '%s\n' "$name"
}

build_profile() {
  local even=$1 fpu=$2 vu0=$3 dma=$4
  local profile build_dir compiled published digest staged
  profile=$(profile_name "$even" "$fpu" "$vu0" "$dma")
  build_dir="$root/build-ps2/profiles/$profile"
  compiled="$build_dir/simd/libjpeg_turbo_mmi.elf"
  published="$root/libjpeg_turbo_mmi_${profile}.elf"
  echo "==== PS2 profile $((profile_index + 1))/$profile_total: $profile ===="

  cmake -S "$root" -B "$build_dir" \
    -DCMAKE_TOOLCHAIN_FILE="$root/simd/ps2/toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_SHARED=OFF -DENABLE_STATIC=ON \
    -DWITH_SIMD=ON -DREQUIRE_SIMD=ON -DWITH_TOOLS=OFF \
    -DWITH_TURBOJPEG=OFF -DWITH_PS2_MMI_TESTS=ON \
    -DWITH_PS2_MMI_ALL_IN_ONE="${PS2_ALL_IN_ONE:-ON}" \
    -DWITH_PS2_EXPERIMENTAL_IDCT=ON -DWITH_PS2_EXPERIMENTAL_COLOR=ON \
    -DWITH_PS2_EXPERIMENTAL_QUANTIZE="${PS2_QUANTIZE:-OFF}" \
    -DWITH_PS2_EXPERIMENTAL_IDCT_EVEN="$even" \
    -DWITH_PS2_APPROX_FPU_IDCT="$fpu" \
    -DWITH_PS2_EXPERIMENTAL_VU0_IDCT="$vu0" \
    -DWITH_PS2_EXPERIMENTAL_VIF0_DMA="$dma" \
    -DWITH_PS2_EXPERIMENTAL_COLOR_PMULTH="${PS2_COLOR_PMULTH:-OFF}" \
    -DWITH_PS2_EXPERIMENTAL_COLOR_PMULTH8="${PS2_COLOR_PMULTH8:-OFF}" \
    -DWITH_PS2_EXPERIMENTAL_MERGED="${PS2_MERGED:-OFF}" \
    -DWITH_PS2_EXPERIMENTAL_MERGED_PMULTH="${PS2_MERGED_PMULTH:-OFF}" \
    -DWITH_PS2_EXPERIMENTAL_MERGED_PMULTH8="${PS2_MERGED_PMULTH8:-OFF}" \
    -DWITH_PS2_EXPERIMENTAL_MERGED_ADD_PACK="${PS2_MERGED_ADD_PACK:-OFF}" \
    -DWITH_PS2_EXPERIMENTAL_MERGED_VECTOR_OFFSETS="${PS2_MERGED_VECTOR_OFFSETS:-OFF}"
  cmake --build "$build_dir" -j"$build_jobs" --target ps2_mmi_test_suite
  [[ -s "$compiled" ]] || { echo "Missing compiled ELF: $compiled" >&2; exit 1; }
  if flag_is_on "${PS2_ALL_IN_ONE:-ON}"; then
    PS2_VU0_IDCT="$vu0" PS2_VIF0_DMA="$dma" \
      bash "$root/simd/ps2/preflight-elf.sh" "$compiled"
  fi
  # Do not expose an ELF until link and preflight both succeed.
  staged="${published}.tmp.$$"
  cp "$compiled" "$staged"
  mv -f "$staged" "$published"
  digest=$(sha256sum "$published")
  digest=${digest%% *}
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$profile" "$(basename "$published")" "$even" "$fpu" "$vu0" "$dma" "$digest" >> "$manifest_tmp"
  if [[ "$profile" == evenoff_exact ]]; then baseline_built=1; fi
  echo "ELF: $published"
  profile_index=$((profile_index + 1))
}

profile_index=0
profile_total=1
case "$mode" in
  all) profile_total=16 ;;
  core) profile_total=8 ;;
esac
manifest="$root/ps2-elf-manifest.csv"
manifest_tmp="${manifest}.tmp.$$"
trap 'rm -f "$manifest_tmp"' EXIT
printf 'profile,elf,idct_even,fpu,vu0,vif0_dma,sha256\n' > "$manifest_tmp"

if [[ "$mode" == single ]]; then
  even=OFF
  if flag_is_on "${PS2_IDCT_EVEN:-OFF}"; then even=ON; fi
  build_profile "$even" "${PS2_APPROX_FPU_IDCT:-OFF}" \
                "${PS2_VU0_IDCT:-OFF}" "${PS2_VIF0_DMA:-OFF}"
else
  for even in OFF ON; do
    [[ "$mode" == core && "$even" == ON ]] && continue
    for mask in 0 1 2 3 4 5 6 7; do
      fpu=OFF; vu0=OFF; dma=OFF
      (( (mask & 1) == 0 )) || fpu=ON
      (( (mask & 2) == 0 )) || vu0=ON
      (( (mask & 4) == 0 )) || dma=ON
      build_profile "$even" "$fpu" "$vu0" "$dma"
    done
  done
fi
mv -f "$manifest_tmp" "$manifest"
trap - EXIT
# Backwards-compatible filename for the plain bit-exact build.
if [[ "${baseline_built:-0}" == 1 ]]; then
  cp "$root/libjpeg_turbo_mmi_evenoff_exact.elf" "$root/libjpeg_turbo_mmi.elf"
fi
printf '\nBuilt %u ELF profile(s) in repository root:\n' "$profile_index"
awk -F, 'NR > 1 { printf "  %s\n", $2 }' "$manifest"
echo "Manifest: $manifest"
echo "All-in-one: ${PS2_ALL_IN_ONE:-ON}; detailed CSV on PS2/PCSX2 stdout."
