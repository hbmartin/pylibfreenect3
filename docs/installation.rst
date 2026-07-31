Installation
============

Wheels support GIL-enabled CPython 3.10--3.14 on macOS 11+ arm64 and
manylinux_2_28 x86_64. Install with ``python -m pip install pylibfreenect3``.

The macOS wheel contains Metal, CPU, dump, TurboJPEG, and libusb support. The
Linux wheel contains CPU, dump, TurboJPEG, and libusb support.

Source installations require a compatible libfreenect2-metal 0.3.x shared
library. Set ``LIBFREENECT2_INSTALL_PREFIX`` when it is not discoverable by
``pkg-config`` or a standard prefix::

   LIBFREENECT2_INSTALL_PREFIX=/opt/freenect2 python -m pip install .

The build requires a C++17 compiler, Cython >= 3.2.8, and NumPy >= 2.2.
