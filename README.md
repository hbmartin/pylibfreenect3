# pylibfreenect3

`pylibfreenect3` 0.3 is the modern, typed Python interface to
[`libfreenect2-metal`](https://github.com/hbmartin/libfreenect2-metal) 0.3. It
supports GIL-enabled CPython 3.10–3.14 and intentionally does not provide a
`pylibfreenect2` import alias.

Binary wheels are self-contained on these targets:

- macOS 11 or newer, arm64: Metal, CPU, dump, and TurboJPEG
- manylinux_2_28, x86_64: CPU, dump, and TurboJPEG

Windows, Intel macOS, Linux arm64, free-threaded CPython, asynchronous capture,
and GPU-enabled Linux wheels are outside the 0.3 release.

## Capture

```python
from pylibfreenect3 import Camera

with Camera.open(pipeline="auto", streams=("color", "depth")) as camera:
    with camera.capture(timeout=2.0, copy=False) as frames:
        depth = frames.depth.to_numpy()
        print(depth.shape, camera.pipeline)
```

`copy=False` is the default. A returned NumPy view keeps its native `Frame`,
`FrameSet`, and listener alive. Releasing the set prevents new lookups but
defers the native release until all borrowed frames and arrays have gone away.
Use `copy=True` when independent storage is preferable; the native capture is
then released before `capture()` returns.

Capture can also be iterated synchronously:

```python
with Camera.open(streams=("depth",)) as camera:
    for frames in camera.frames(timeout=2.0):
        with frames:
            consume(frames.depth.to_numpy())
```

## Pipelines and core identity

```python
import pylibfreenect3 as f3

print(f3.core_version(), f3.core_api_version(), f3.core_build_revision())
print(f3.compiled_pipelines())
print(f3.available_pipelines())
```

Canonical pipeline names are `cpu`, `metal`, `opengl`, `opencl`,
`opencl_kde`, `cuda`, `cuda_kde`, and `dump`. Every backend class is
importable on every platform; constructing one that is not compiled or usable
raises `BackendUnavailableError`. Pipeline objects are single-use after being
passed to `open_device()`. The legacy `gl` and `cl` spellings are accepted only
by the core's `LIBFREENECT2_PIPELINE` environment variable.

## Low-level control and registration

`Freenect2`, `Device`, and `SyncFrameListener` expose explicit device control
with snake-case methods. `DeviceConfig`, `ColorCameraParams`, `IrCameraParams`,
and `LedSettings` validate values before native calls. `Registration` provides
scalar projection, full-frame registration, depth undistortion, XYZ, and
XYZ+RGB queries.

```python
from pylibfreenect3 import Freenect2, FrameType, SyncFrameListener

context = Freenect2()
device = context.open_device(pipeline="cpu")
listener = SyncFrameListener(FrameType.COLOR | FrameType.IR | FrameType.DEPTH)
device.set_color_listener(listener)
device.set_depth_listener(listener)
device.start()
try:
    with listener.wait(timeout=2.0) as frames:
        print(frames.color.timestamp, frames.depth.sequence)
finally:
    device.close()
```

## Recording and replay

Schema-v1 recording bundles are directories with an atomic `manifest.json`,
inline camera parameters, checksummed calibration sidecars, and checksummed raw
JPEG/depth packets. Recording requires the dump pipeline:

```python
from pylibfreenect3 import Camera, RecordingWriter

with Camera.open(pipeline="dump", streams=("color", "depth")) as camera:
    with RecordingWriter("capture.f3", camera) as recording:
        recording.capture(100, timeout=2.0)

with Camera.open_recording("capture.f3", pipeline="metal") as replay:
    with replay.capture(timeout=2.0) as frames:
        print(frames.depth.to_numpy())
```

Loose `.jpg`/`.jpeg` and `.depth` filenames remain supported through
`Freenect2Replay.open_device()`. Raw depth replay requires an explicit,
validated `ReplayCalibration`.

## Errors and thread-safety boundary

The public exception hierarchy is `FreenectError`,
`BackendUnavailableError`, `DeviceOpenError`, `DeviceStateError`,
`FrameTimeoutError`, `ReplayError`, and `RecordingFormatError`.

Reserved packet-processor methods, decoder-thread Python callbacks, and a
Python logging callback are intentionally not exposed. Native logging can be
controlled with `set_global_log_level()`. Capture is synchronous; no asyncio
adapter is included in 0.3.

## Source builds

An sdist links to an installed `libfreenect2-metal` 0.3.x. The build discovers
it in this order: `LIBFREENECT2_INSTALL_PREFIX`, `pkg-config`, then standard
prefixes.

```console
export LIBFREENECT2_INSTALL_PREFIX=/opt/libfreenect2-metal
python -m pip install .
```

The extension is C++17 and requires Cython 3.2.8 or newer and NumPy 2.2 or
newer. On a version or linkage mismatch, the build probe reports the
architecture, header and library locations, runtime API, compiled pipelines,
and linked libraries.

See `docs/` for the API inventory and release gates.
