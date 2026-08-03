# Installation

Wheels support GIL-enabled CPython 3.12–3.14 on macOS 11+ arm64 and
manylinux_2_28 x86_64. Declare `pylibfreenect3` in an application's
`pyproject.toml` dependencies and run `uv sync`. A one-off published-wheel
check can use:

```console
uv run --with pylibfreenect3 python -c "import pylibfreenect3"
```

The macOS wheel contains Metal, CPU, dump, TurboJPEG, and libusb support. The
Linux wheel contains CPU, dump, TurboJPEG, and libusb support.

## Building from source

Source builds require a compatible libfreenect2-metal 0.4.x shared library and
API 4. The selected development install must contain the installed vision,
calibration-profile, projective-registration, and recording headers with
matching symbols; configuration fails early when the core is incomplete or
incompatible.
Set `Freenect2_ROOT` when it is not discoverable through its CMake package,
`pkg-config`, or a standard prefix:

```console
Freenect2_ROOT=/opt/freenect2 uv build --wheel
```

On macOS, install the native core from the `hbmartin` Homebrew tap:

```console
brew install hbmartin/tap/libfreenect2-metal
export Freenect2_ROOT="$(brew --prefix libfreenect2-metal)"
uv build --wheel
```

Use `brew install --HEAD hbmartin/tap/libfreenect2-metal` to build against the
core's current development branch.

The build uses scikit-build-core and CMake under `uv build`. It requires a
C++17 compiler, Cython 3.2.8 or newer, and NumPy 2.2 or newer. An sdist does
not require the native core:

```console
uv build --sdist
```
