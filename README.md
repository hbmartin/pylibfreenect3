# pylibfreenect3

[![PyPI](https://img.shields.io/pypi/v/pylibfreenect3.svg)](https://pypi.org/project/pylibfreenect3/)
[![Python](https://img.shields.io/pypi/pyversions/pylibfreenect3.svg)](https://pypi.org/project/pylibfreenect3/)
[![Build](https://github.com/hbmartin/pylibfreenect3/actions/workflows/wheels.yml/badge.svg)](https://github.com/hbmartin/pylibfreenect3/actions/workflows/wheels.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Modern, typed, ownership-safe Python bindings for Microsoft Kinect v2 through
[`libfreenect2-metal`](https://github.com/hbmartin/libfreenect2-metal).

`pylibfreenect3` 0.3 provides:

- a small high-level `Camera` API for synchronized color, infrared, and depth
  capture;
- zero-copy NumPy views with explicit, safe frame lifetimes;
- CPU and Apple Metal packet-processing backends in prebuilt wheels;
- camera registration, configuration, exposure, and LED controls;
- checksummed recording bundles and deterministic replay; and
- a typed low-level API for applications that need direct device control.

This release targets GIL-enabled CPython 3.10–3.14. It is intentionally a new
API and does **not** provide the old `pylibfreenect2` import name.

## Supported platforms

| Platform | Prebuilt wheel | Included pipelines |
| --- | --- | --- |
| macOS 11+, Apple silicon (`arm64`) | Yes | Metal, CPU, dump |
| Linux `manylinux_2_28`, `x86_64` | Yes | CPU, dump |

The wheels also bundle the compatible core library, libusb, and TurboJPEG.
Windows, Intel macOS, Linux ARM64, free-threaded CPython, and GPU-enabled Linux
wheels are not part of the 0.3 release. Other platforms may work from source if
the core library and a supported packet pipeline can be built there.

Live capture requires a Kinect v2, its power/USB adapter, and a USB 3 connection.
You do not need hardware to use frame allocation, recording validation, or
replay from existing raw captures.

## Installation

Install a wheel from PyPI:

```console
python -m pip install --upgrade pip
python -m pip install pylibfreenect3
```

Confirm that the binding and bundled core load correctly:

```python
import pylibfreenect3 as f3

print("binding:", f3.__version__)
print("core:", f3.core_version(), "API", f3.core_api_version())
print("compiled pipelines:", sorted(f3.compiled_pipelines()))
print("usable pipelines:", sorted(f3.available_pipelines()))
print("connected devices:", f3.Freenect2().enumerate_devices())
```

If your platform has no matching wheel, see [Building from source](#building-from-source).

## Quick start

Open the default device, wait up to two seconds for a synchronized color/depth
pair, and expose both frames as NumPy arrays:

```python
from pylibfreenect3 import Camera

with Camera.open(pipeline="auto", streams=("color", "depth")) as camera:
    with camera.capture(timeout=2.0) as frames:
        color = frames.color.to_numpy()
        depth = frames.depth.to_numpy()

        print("pipeline:", camera.pipeline)
        print("color:", color.shape, color.dtype)
        print("depth:", depth.shape, depth.dtype)
```

A normal decoded capture has these layouts:

| Stream | NumPy shape | dtype | Meaning |
| --- | --- | --- | --- |
| `color` | `(1080, 1920, 4)` | `uint8` | BGRX or RGBX color |
| `ir` | `(424, 512)` | `float32` | Infrared intensity |
| `depth` | `(424, 512)` | `float32` | Depth in millimetres |

See the [OpenCV and registration cookbook](docs/cookbook.rst) for channel
conversion, lossless depth storage, coordinate maps, offline registration,
image flips, and slow-consumer guidance.

Select any non-empty combination of `"color"`, `"ir"`, and `"depth"`. To
open a particular device, pass its zero-based index or serial number:

```python
with Camera.open(device=0, streams=("depth",)) as camera:
    with camera.capture(timeout=2.0) as frames:
        print(frames.depth.sequence)
```

### Continuous capture

Capture iteration is synchronous. A timeout raises `FrameTimeoutError` rather
than ending the iterator:

```python
from pylibfreenect3 import Camera, FrameTimeoutError

with Camera.open(streams=("depth",)) as camera:
    try:
        for frames in camera.frames(timeout=2.0):
            with frames:
                consume(frames.depth.to_numpy())
    except FrameTimeoutError:
        print("No depth frame arrived for two seconds")
```

Break out of the loop whenever your application is finished; leaving the
camera context stops and closes the device.

## Frame ownership and copies

`Camera.capture()` and `Frame.to_numpy()` are zero-copy by default. The returned
array retains the native frame and its owners, so the underlying memory remains
valid for as long as the array is reachable. Releasing a `FrameSet` prevents
new frame lookups, but an already-created array remains valid.

Use the frame set as a context manager for prompt release:

```python
with camera.capture(timeout=2.0) as frames:
    depth_view = frames.depth.to_numpy()  # native-backed view
```

Use `copy=True` on `capture()` when the complete frame set must be independent
of the listener and native capture:

```python
frames = camera.capture(timeout=2.0, copy=True)
try:
    depth = frames.depth.to_numpy()
finally:
    frames.release()
```

Alternatively, `frame.to_numpy(copy=True)` copies only that array. This is
usually the cheapest option when only one stream needs to outlive the capture.

## Registration

`Registration` uses the active device's calibration to align color with depth:

```python
from pylibfreenect3 import Camera, Registration

with Camera.open(streams=("color", "depth")) as camera:
    registration = Registration(
        camera.device.ir_camera_params,
        camera.device.color_camera_params,
    )

    with camera.capture(timeout=2.0) as frames:
        result = registration.apply(frames.color, frames.depth)
        undistorted_depth = result.undistorted.to_numpy()
        registered_color = result.registered.to_numpy()

        x, y, z = registration.point_xyz(result.undistorted, row=212, column=256)
        print(x, y, z)
```

`Registration.apply()` can also produce the 1920×1082 `big_depth` frame and a
512×424 color/depth index map:

```python
result = registration.apply(
    frames.color,
    frames.depth,
    include_big_depth=True,
    include_color_depth_map=True,
)
```

The [cookbook](docs/cookbook.rst) documents the output layouts and shows how
to register saved color and filtered-depth arrays safely.

## Pipelines

Pass `pipeline="auto"` to let the core choose the best usable backend, or name
one explicitly:

```python
with Camera.open(pipeline="metal") as camera:  # Apple silicon wheel
    print(camera.pipeline)

with Camera.open(pipeline="cpu") as camera:    # macOS or Linux wheel
    print(camera.pipeline)
```

Canonical names are `cpu`, `metal`, `opengl`, `opencl`, `opencl_kde`, `cuda`,
`cuda_kde`, and `dump`. The prebuilt-wheel support is summarized above;
additional backends are available only when the linked core was built with
them.

- `compiled_pipelines()` reports what is present in the loaded core.
- `available_pipelines()` reports the subset usable on the current machine.
- An unavailable explicit backend raises `BackendUnavailableError`.
- `dump` emits raw packets and is intended for recording, not decoded arrays.

Pipeline classes are also public for low-level use. A pipeline object becomes
single-use once passed to `open_device()`; create a new instance for each
device:

```python
from pylibfreenect3 import Freenect2, MetalPacketPipeline

context = Freenect2()
pipeline = MetalPacketPipeline()
with context.open_device(pipeline=pipeline) as device:
    print(device.serial_number)
```

## Recording and replay

Recording bundles are directories containing an atomic `manifest.json`,
camera calibration sidecars, and raw JPEG/depth packets. File sizes and SHA-256
checksums are validated when a bundle is opened.

Recording requires the dump pipeline, and the destination must not already
exist:

```python
from pylibfreenect3 import Camera, RecordingWriter

with Camera.open(pipeline="dump", streams=("color", "depth")) as camera:
    with RecordingWriter("capture.f3", camera) as recording:
        recording.capture(100, timeout=2.0)
```

Recording is synchronous by default. For sustained capture, a bounded worker
queue can move checksumming and filesystem writes off the capture loop:

```python
with Camera.open(pipeline="dump", streams=("color", "depth")) as camera:
    with RecordingWriter(
        "capture.f3", camera, queue_size=8, overflow="drop"
    ) as recording:
        recording.capture(1_000, timeout=2.0)
        print(recording.flush())
```

`overflow="block"` applies backpressure when the queue is full;
`overflow="drop"` keeps capture moving and records dropped frame-set counts in
`RecordingWriter.stats` and the final manifest. Recording bundles contain raw
JPEG/depth packets, not an encoded video container.

Replay the bundle through any decoded pipeline available on the machine:

```python
with Camera.open_recording("capture.f3", pipeline="auto") as replay:
    with replay.capture(timeout=5.0) as frames:
        print(frames.color.to_numpy().shape)
        print(frames.depth.to_numpy().shape)
```

`Camera.open_recording(..., streams=(...))` can replay a subset of the streams
stored in a bundle. Loose `.jpg`/`.jpeg` and `.depth` packets can be opened with
`Freenect2Replay`; raw depth packets require an explicit, validated
`ReplayCalibration`.

## Device configuration and logging

Use the low-level API when configuration must be applied before capture starts:

```python
from pylibfreenect3 import DeviceConfig, Freenect2, LoggerLevel, set_global_log_level

set_global_log_level(LoggerLevel.INFO)

context = Freenect2()
with context.open_device(pipeline="cpu") as device:
    device.configuration = DeviceConfig(
        min_depth=0.5,  # metres
        max_depth=4.5,  # metres
        enable_bilateral_filter=True,
        enable_edge_aware_filter=True,
    )
```

`Camera.device` provides the same `Device` object. It also exposes color auto,
semi-auto, and manual exposure controls, typed color settings, camera
parameters, and LED settings. Pass `None` to `set_global_log_level()` to disable
native console logging.

## Low-level API

Use `Freenect2`, `Device`, and `SyncFrameListener` when you need explicit
lifecycle control:

```python
from pylibfreenect3 import Freenect2, FrameType, SyncFrameListener

context = Freenect2()
device = context.open_device(pipeline="cpu")
listener = SyncFrameListener(FrameType.COLOR | FrameType.DEPTH)
device.set_color_listener(listener)
device.set_depth_listener(listener)
device.start()

try:
    with listener.wait(timeout=2.0) as frames:
        print(frames.color.timestamp, frames.depth.sequence)
finally:
    device.close()
```

`Frame`, `FrameSet`, and the parameter dataclasses can also be used without a
physical device, which is useful for tests and replay tooling.

## Exceptions

All library-specific exceptions inherit from `FreenectError`:

| Exception | Raised when |
| --- | --- |
| `BackendUnavailableError` | A requested packet pipeline is not compiled or usable |
| `DeviceOpenError` | A physical or replay device cannot be opened |
| `DeviceStateError` | An operation is invalid for the current lifecycle state |
| `FrameTimeoutError` | No synchronized frame set arrives before the timeout |
| `ReplayError` | Raw packet replay fails |
| `RecordingFormatError` | A bundle is invalid, incomplete, unsafe, or incompatible |

Capture is intentionally synchronous in 0.3. Packet-parser hooks,
decoder-thread Python callbacks, a Python logging callback, and an asyncio
adapter are not exposed because their thread-safety and lifetime semantics
would be misleading.

## Troubleshooting

### No device is found

- Check that the Kinect power adapter is connected and its indicator is lit.
- Connect directly to a USB 3 port rather than through a hub when possible.
- Close other applications that may have claimed the camera.
- On Linux, install an appropriate udev rule for the Kinect USB device and
  reconnect it; running the application as root should not be the permanent
  solution.

### A pipeline is unavailable

Compare `compiled_pipelines()` with `available_pipelines()`. A backend can be
compiled into the core but unusable because the current machine lacks the
required GPU or runtime support. Use `pipeline="auto"` or `pipeline="cpu"` as
a fallback.

### Capture times out

`FrameTimeoutError` means the requested synchronized stream set did not arrive
within the supplied number of seconds. Verify the USB connection, try a longer
timeout, and test a smaller stream set such as `streams=("depth",)`.

### A source build finds the wrong core

Set `LIBFREENECT2_INSTALL_PREFIX` to the exact installation prefix of
`libfreenect2-metal` 0.3.x. The build probe prints the architecture, discovery
source, headers, libraries, runtime/API version, compiled pipelines, and linked
libraries to make mismatches visible.

## Building from source

Source installations link against an already-installed
[`libfreenect2-metal`](https://github.com/hbmartin/libfreenect2-metal) 0.3.x.
The build looks for it in this order:

1. `LIBFREENECT2_INSTALL_PREFIX`
2. `pkg-config` (`freenect2.pc`)
3. `/usr/local`, `/usr`, and `/opt/homebrew`

After installing the core and its native dependencies:

```console
export LIBFREENECT2_INSTALL_PREFIX=/opt/libfreenect2-metal
python -m pip install .
```

The extension uses C++17 and requires Cython 3.2.8 or newer and NumPy 2.2 or
newer. The core headers and runtime must both be version 0.3.x with API 3.

## Development

Create an environment, point the build at a compatible core, and install the
test and documentation dependencies:

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
export LIBFREENECT2_INSTALL_PREFIX=/opt/libfreenect2-metal
python -m pip install -e '.[test,docs]'
```

Run the hardware-free suite and build the documentation with warnings treated
as errors:

```console
python -m pytest -m 'not hardware'
sphinx-build -W --keep-going docs docs/_build/html
```

With a Kinect v2 attached, run the hardware tests explicitly:

```console
python -m pytest tests/test_hardware.py -m hardware
```

See [`examples/`](examples), the [Sphinx documentation](docs/index.rst), and
the [release gates](docs/dev.rst) for more detail. Issues and pull requests are
welcome in the [GitHub repository](https://github.com/hbmartin/pylibfreenect3).

## License

`pylibfreenect3` is distributed under the [MIT License](LICENSE.md). Binary
wheels contain additional bundled components whose notices are included in
the installed package.
