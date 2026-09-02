#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RIFE_COMMIT="a7532fc3f9f8f008cd6eecd6f2ffe2a9698e0cf7"
DEPS="$ROOT/.deps"
SRC="$DEPS/rife-ncnn-vulkan"
BUILD="$DEPS/rife-build"
JNI_DIR="$ROOT/app/src/main/jniLibs/arm64-v8a"
ASSET_DIR="$ROOT/app/src/main/assets/models/rife-v4.6"
MANIFEST_DIR="$ROOT/app/src/main/assets/runtime"

: "${ANDROID_NDK_HOME:?ANDROID_NDK_HOME must point to Android NDK r27+}"

rm -rf "$SRC" "$BUILD"
mkdir -p "$DEPS" "$JNI_DIR" "$ASSET_DIR" "$MANIFEST_DIR"

git clone https://github.com/nihui/rife-ncnn-vulkan.git "$SRC"
git -C "$SRC" checkout "$RIFE_COMMIT"
git -C "$SRC" submodule update --init --recursive

cmake -S "$SRC/src" -B "$BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DANDROID_STL=c++_static \
  -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD" --target rife-ncnn-vulkan --parallel 2

RIFE_BIN="$(find "$BUILD" -type f -name rife-ncnn-vulkan | head -n 1)"
if [[ -z "$RIFE_BIN" || ! -f "$RIFE_BIN" ]]; then
  echo "RIFE executable not found after native build" >&2
  exit 2
fi

cp "$RIFE_BIN" "$JNI_DIR/librife.so"
chmod 755 "$JNI_DIR/librife.so"
cp "$SRC/models/rife-v4.6/flownet.param" "$ASSET_DIR/flownet.param"
cp "$SRC/models/rife-v4.6/flownet.bin" "$ASSET_DIR/flownet.bin"

if [[ $(stat -c%s "$ASSET_DIR/flownet.bin") -lt 10000000 ]]; then
  echo "RIFE model binary is unexpectedly small" >&2
  exit 3
fi
if [[ $(stat -c%s "$JNI_DIR/librife.so") -lt 100000 ]]; then
  echo "RIFE Android executable is unexpectedly small" >&2
  exit 4
fi

{
  echo "upstream=https://github.com/nihui/rife-ncnn-vulkan"
  echo "commit=$RIFE_COMMIT"
  echo "abi=arm64-v8a"
  echo "android_platform=28"
  echo "flexible_page_sizes=ON"
  sha256sum "$JNI_DIR/librife.so" "$ASSET_DIR/flownet.param" "$ASSET_DIR/flownet.bin"
} > "$MANIFEST_DIR/runtime-manifest.txt"

printf '\nRuntime payload ready:\n'
cat "$MANIFEST_DIR/runtime-manifest.txt"
