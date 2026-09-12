#!/usr/bin/env bash
set -euo pipefail

MNN_COMMIT="3db3cc904dfea55286972b472b040ad5525aa083"
MNN_REPO="https://github.com/alibaba/MNN.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ROOT}/.build/mobilei2v-gpu"
MNN_DIR="${WORK}/MNN"
BUILD_DIR="${WORK}/android-arm64"
STAGE_DIR="${ROOT}/app/src/main/jniLibs/arm64-v8a"

NDK_ROOT="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
if [[ -z "${NDK_ROOT}" && -n "${ANDROID_SDK_ROOT:-}" ]]; then
  candidate="$(find "${ANDROID_SDK_ROOT}/ndk" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -V | tail -n1 || true)"
  NDK_ROOT="${candidate}"
fi
if [[ -z "${NDK_ROOT}" || ! -f "${NDK_ROOT}/build/cmake/android.toolchain.cmake" ]]; then
  echo "Android NDK not found; set ANDROID_NDK_HOME" >&2
  exit 2
fi

rm -rf "${WORK}"
mkdir -p "${WORK}" "${STAGE_DIR}"

git clone --filter=blob:none --no-checkout "${MNN_REPO}" "${MNN_DIR}"
git -C "${MNN_DIR}" fetch --depth=1 origin "${MNN_COMMIT}"
git -C "${MNN_DIR}" checkout --detach "${MNN_COMMIT}"
test "$(git -C "${MNN_DIR}" rev-parse HEAD)" = "${MNN_COMMIT}"

cmake \
  -S "${ROOT}/app/src/main/cpp/mobilei2v" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${NDK_ROOT}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DANDROID_STL=c++_static \
  -DMNN_SOURCE_DIR="${MNN_DIR}" \
  -DMNN_OPENCL=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_LOW_MEMORY=ON \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,max-page-size=16384"

cmake --build "${BUILD_DIR}" --target mobilei2v_gpu --parallel "$(nproc)"

LIB="$(find "${BUILD_DIR}" -type f -name 'libmobilei2v_gpu.so' -print -quit)"
if [[ -z "${LIB}" || ! -s "${LIB}" ]]; then
  echo "libmobilei2v_gpu.so was not produced" >&2
  exit 3
fi
cp -f "${LIB}" "${STAGE_DIR}/libmobilei2v_gpu.so"

READELF="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf"
if [[ ! -x "${READELF}" ]]; then
  READELF="${NDK_ROOT}/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-readelf"
fi
"${READELF}" -lW "${STAGE_DIR}/libmobilei2v_gpu.so" | tee "${WORK}/mobilei2v-gpu-elf.txt"
python3 - "${WORK}/mobilei2v-gpu-elf.txt" <<'PY'
import re, sys
lines=open(sys.argv[1], encoding='utf-8').read().splitlines()
aligns=[int(line.split()[-1],16) for line in lines if re.match(r'\s*LOAD\s', line)]
if not aligns or min(aligns) < 0x4000:
    raise SystemExit(f'MobileI2V GPU 16KB ELF gate failed: {aligns}')
print('MobileI2V GPU 16KB ELF gate PASS:', aligns)
PY

sha256sum "${STAGE_DIR}/libmobilei2v_gpu.so"
echo "MOBILEI2V_GPU_RUNTIME_READY=${STAGE_DIR}/libmobilei2v_gpu.so"
