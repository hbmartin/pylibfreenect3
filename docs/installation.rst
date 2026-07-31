Installation
============

Wheels support GIL-enabled CPython 3.12--3.14 on macOS 11+ arm64 and
manylinux_2_28 x86_64. Declare ``pylibfreenect3`` in an application's
``pyproject.toml`` dependencies and run ``uv sync``. A one-off published-wheel
check can use::

   uv run --with pylibfreenect3 python -c "import pylibfreenect3"

The macOS wheel contains Metal, CPU, dump, TurboJPEG, and libusb support. The
Linux wheel contains CPU, dump, TurboJPEG, and libusb support.

Source builds require a compatible libfreenect2-metal 0.3.x shared library.
Set ``Freenect2_ROOT`` when it is not discoverable through its CMake package,
``pkg-config``, or a standard prefix::

   Freenect2_ROOT=/opt/freenect2 uv build --wheel

The build uses scikit-build-core and CMake under ``uv build``. It requires a
C++17 compiler, Cython >= 3.2.8, and NumPy >= 2.2. An sdist does not require
the native core::

   uv build --sdist
