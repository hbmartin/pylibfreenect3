#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/libfreenect2-metal" >&2
  exit 2
fi

project_dir=$(cd "$(dirname "$0")/.." && pwd)
core_source=$(cd "$1" && pwd)
wheel_prefix=${PYLIBF3_WHEEL_PREFIX:-/tmp/pylibfreenect3-wheel-prefix}
work_dir=$(mktemp -d /tmp/pylibfreenect3-wheel-deps.XXXXXX)

cleanup() {
  rm -rf -- "$work_dir"
}
trap cleanup EXIT

libusb_version=1.0.30
libusb_sha256=fea36f34f9156400209595e300840767ab1a385ede1dc7ee893015aea9c6dbaf
turbojpeg_version=3.2.0
turbojpeg_sha256=6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e

if [[ ${PYLIBF3_REQUIRE_CORE_TAG:-0} == 1 ]]; then
  core_tag=$(git -C "$core_source" describe --tags --exact-match)
  if [[ $core_tag != v0.3.0 ]]; then
    echo "release wheels require libfreenect2-metal tag v0.3.0; found $core_tag" >&2
    exit 1
  fi
fi

download_and_verify() {
  local url=$1
  local output=$2
  local expected=$3
  curl --fail --location --silent --show-error \
    --retry 3 --connect-timeout 30 --max-time 600 \
    "$url" --output "$output"
  local actual
  if command -v shasum >/dev/null 2>&1; then
    actual=$(shasum -a 256 "$output" | awk '{print $1}')
  else
    actual=$(sha256sum "$output" | awk '{print $1}')
  fi
  if [[ $actual != "$expected" ]]; then
    echo "SHA-256 mismatch for $url: expected $expected, found $actual" >&2
    exit 1
  fi
}

mkdir -p "$wheel_prefix" "$work_dir/libusb" "$work_dir/turbojpeg"

libusb_archive="$work_dir/libusb.tar.bz2"
download_and_verify \
  "https://github.com/libusb/libusb/releases/download/v${libusb_version}/libusb-${libusb_version}.tar.bz2" \
  "$libusb_archive" "$libusb_sha256"
tar -xjf "$libusb_archive" -C "$work_dir/libusb" --strip-components=1

libusb_configure=(
  "$work_dir/libusb/configure"
  "--prefix=$wheel_prefix"
  --enable-shared
  --disable-static
  --disable-examples-build
  --disable-tests-build
)
if [[ $(uname -s) == Linux ]]; then
  libusb_configure+=(--disable-udev)
fi
if [[ $(uname -s) == Darwin ]]; then
  export MACOSX_DEPLOYMENT_TARGET=11.0
  export CFLAGS="${CFLAGS:-} -arch arm64 -mmacosx-version-min=11.0"
  export CXXFLAGS="${CXXFLAGS:-} -arch arm64 -mmacosx-version-min=11.0"
  export LDFLAGS="${LDFLAGS:-} -arch arm64 -mmacosx-version-min=11.0"
fi
(cd "$work_dir/libusb" && "${libusb_configure[@]}" && make -j2 && make install)

turbojpeg_archive="$work_dir/libjpeg-turbo.tar.gz"
download_and_verify \
  "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${turbojpeg_version}/libjpeg-turbo-${turbojpeg_version}.tar.gz" \
  "$turbojpeg_archive" "$turbojpeg_sha256"
tar -xzf "$turbojpeg_archive" -C "$work_dir/turbojpeg" --strip-components=1

turbojpeg_options=(
  -S "$work_dir/turbojpeg"
  -B "$work_dir/turbojpeg-build"
  "-DCMAKE_INSTALL_PREFIX=$wheel_prefix"
  -DCMAKE_BUILD_TYPE=Release
  -DENABLE_SHARED=ON
  -DENABLE_STATIC=OFF
  -DWITH_TURBOJPEG=ON
  -DWITH_TOOLS=OFF
  -DWITH_TESTS=OFF
)
if [[ $(uname -s) == Darwin ]]; then
  turbojpeg_options+=(
    -DCMAKE_OSX_ARCHITECTURES=arm64
    -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
  )
fi
cmake "${turbojpeg_options[@]}"
cmake --build "$work_dir/turbojpeg-build" --parallel 2
cmake --install "$work_dir/turbojpeg-build"

export PKG_CONFIG_PATH="$wheel_prefix/lib/pkgconfig:$wheel_prefix/lib64/pkgconfig"
core_options=(
  -S "$core_source"
  -B "$work_dir/core-build"
  "-DCMAKE_INSTALL_PREFIX=$wheel_prefix"
  "-DCMAKE_PREFIX_PATH=$wheel_prefix"
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_EXAMPLES=OFF
  -DBUILD_OPENNI2_DRIVER=OFF
  -DBUILD_STREAMER_RECORDER=OFF
  -DBUILD_MEDIAPIPE_DEMO=OFF
  -DBUILD_TESTING=OFF
  -DENABLE_CUDA=OFF
  -DENABLE_OPENCL=OFF
  -DENABLE_OPENGL=OFF
  -DENABLE_TEGRAJPEG=OFF
  -DENABLE_VAAPI=OFF
)
if [[ $(uname -s) == Darwin ]]; then
  core_options+=(
    -DENABLE_METAL=ON
    -DENABLE_VIDEOTOOLBOX=OFF
    -DCMAKE_OSX_ARCHITECTURES=arm64
    -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
  )
else
  core_options+=(-DENABLE_METAL=OFF)
fi
cmake "${core_options[@]}"
cmake --build "$work_dir/core-build" --parallel 2
cmake --install "$work_dir/core-build"

license_dir="$project_dir/src/pylibfreenect3/licenses"
mkdir -p "$license_dir"
cp "$core_source/APACHE20" "$license_dir/libfreenect2-APACHE20.txt"
cp "$core_source/GPL2" "$license_dir/libfreenect2-GPL2.txt"
cp "$core_source/CONTRIB" "$license_dir/libfreenect2-CONTRIB.txt"
cp "$work_dir/libusb/COPYING" "$license_dir/libusb-COPYING.txt"
cp "$work_dir/turbojpeg/LICENSE.md" "$license_dir/libjpeg-turbo-LICENSE.md"

echo "wheel dependency prefix: $wheel_prefix"
echo "libusb commit: 87a55632db62c9bdc58cd31d3ccfa673f1bb017f"
echo "libjpeg-turbo commit: c85e6b905bf237038faa936dab160ebfc5da0344"
echo "libfreenect2-metal commit: $(git -C "$core_source" rev-parse HEAD)"
