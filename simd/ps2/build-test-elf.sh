#!/usr/bin/env bash
# Build the on-screen test suite with an installed PS2SDK and EE toolchain.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
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
    make -C "$PS2SDKSRC/ee/$part" -j"${JOBS:-8}" >> "$sdk_log" 2>&1 || {
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

cmake -S "$root" -B "$root/build-ps2" \
  -DCMAKE_TOOLCHAIN_FILE="$root/simd/ps2/toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_SHARED=OFF -DENABLE_STATIC=ON \
  -DWITH_SIMD=ON -DREQUIRE_SIMD=ON -DWITH_TOOLS=OFF \
  -DWITH_TURBOJPEG=OFF -DWITH_PS2_MMI_TESTS=ON \
  -DWITH_PS2_MMI_ALL_IN_ONE="${PS2_ALL_IN_ONE:-ON}" \
  -DWITH_PS2_EXPERIMENTAL_IDCT=ON -DWITH_PS2_EXPERIMENTAL_COLOR=ON \
  -DWITH_PS2_EXPERIMENTAL_QUANTIZE="${PS2_QUANTIZE:-OFF}" \
  -DWITH_PS2_EXPERIMENTAL_IDCT_EVEN="${PS2_IDCT_EVEN:-OFF}" \
  -DWITH_PS2_EXPERIMENTAL_COLOR_PMULTH="${PS2_COLOR_PMULTH:-OFF}" \
  -DWITH_PS2_EXPERIMENTAL_COLOR_PMULTH8="${PS2_COLOR_PMULTH8:-OFF}" \
  -DWITH_PS2_EXPERIMENTAL_MERGED="${PS2_MERGED:-OFF}" \
  -DWITH_PS2_EXPERIMENTAL_MERGED_PMULTH="${PS2_MERGED_PMULTH:-OFF}" \
  -DWITH_PS2_EXPERIMENTAL_MERGED_PMULTH8="${PS2_MERGED_PMULTH8:-OFF}" \
  -DWITH_PS2_EXPERIMENTAL_MERGED_ADD_PACK="${PS2_MERGED_ADD_PACK:-OFF}" \
  -DWITH_PS2_EXPERIMENTAL_MERGED_VECTOR_OFFSETS="${PS2_MERGED_VECTOR_OFFSETS:-OFF}"
cmake --build "$root/build-ps2" -j"${JOBS:-8}" --target ps2_mmi_test_suite
case "${PS2_ALL_IN_ONE:-ON}" in
  OFF|off|FALSE|false|0) ;;
  *) bash "$root/simd/ps2/preflight-elf.sh" \
       "$root/build-ps2/simd/ps2_mmi_test_suite.elf" ;;
esac
echo "ELF: $root/build-ps2/simd/ps2_mmi_test_suite.elf"
echo "All-in-one: ${PS2_ALL_IN_ONE:-ON}; CSV appears on stdout / PCSX2 console."
case "${PS2_ALL_IN_ONE:-ON}" in
  OFF|off|FALSE|false|0)
    echo "Legacy ELF: check the final 7/7 result." ;;
  *)
    echo "One-boot ELF: capture CSV and check the final 9/9 result." ;;
esac
