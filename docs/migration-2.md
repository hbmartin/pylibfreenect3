# Migrating from 1.x to 2.0

Version 2.0 moves the binding to libfreenect2-metal 0.4/API 4. Every extension
and wheel must be rebuilt for that ABI; a 1.x extension cannot be used with the
0.4 shared library.

## Recording is replaced, not upgraded in place

The 1.x Python writer produced a private schema-v1 bundle with a JSON frame
index, hashes, and derived calibration tables. Version 2.0's `RecordingWriter`
owns a Kinect directly and writes the core's canonical manifest-v2 directory:

```python
# 1.x
with Camera.open(pipeline="dump") as camera:
    with RecordingWriter("capture", camera, queue_size=8) as writer:
        writer.capture(300)

# 2.0
with RecordingWriter("capture", queue_capacity=32) as writer:
    writer.capture(depth_frames=300, timeout=20.0)
```

The new writer accepts one incremental bound per `capture()` call:
`depth_frames=`, `color_frames=`, or `duration=`. Its statistics count native
frames and bytes; the old submitted/pending/failed frame-set counters and
block/drop modes no longer exist.

`Camera.open_recording()` now accepts only canonical native directories. It
does not inspect or reinterpret the old Python bundle:

```python
from pylibfreenect3 import Camera, ReplayOptions

with Camera.open_recording(
    "capture",
    replay_options=ReplayOptions(reproduce_timing=True),
) as camera:
    ...
```

The high-level default requests color plus depth. For a recording that contains
only one raw stream, select it explicitly with `streams=("color",)` or
`streams=("depth",)`.

To inspect an old bundle, import its retained reader explicitly, validate it,
and pass its packet paths and `ReplayCalibration` through the loose-file API:

```python
from pylibfreenect3 import lowlevel
from pylibfreenect3.legacy import RecordingBundle

bundle = RecordingBundle.open("old-capture.f3")
device = lowlevel.ReplayContext().open_device(
    bundle.frame_paths(("color", "depth")),
    calibration=bundle.calibration,
    pipeline="cpu",
)
```

## Factory and projective registration are different APIs

`Registration` is unchanged. It uses the Kinect's factory polynomial mapping,
produces registered color and factory coordinate maps, and is constructed from
`IrCameraParams` plus `ColorCameraParams`.

`ProjectiveRegistration` is new. It uses an ordinary projective camera model,
a rigid depth-to-color transform from a native `CalibrationProfile`, and emits
float target-camera depth. Do not replace one with the other solely because
their names are similar.

```python
profile = CalibrationProfile.load("profile.json")
warning = profile.check_device(camera.device)
projective = ProjectiveRegistration(profile)
registered_depth = projective.apply(frames.depth)
```

Depth correction remains disabled unless
`ProjectiveRegistrationOptions(apply_depth_correction=True)` is passed.

## Allocated frames require a format

`Frame.allocate()` no longer creates an `INVALID`-format frame implicitly:

```python
frame = Frame.allocate(
    width=512,
    height=424,
    bytes_per_pixel=4,
    frame_format=FrameFormat.FLOAT,
)
```

`Frame.from_array()` can still infer a supported format from a compatible
NumPy layout.

## New runtime and replay snapshots

`Device.runtime_stats` and `Camera.runtime_stats` return frozen
`DeviceRuntimeStats` values. They remain queryable after `stop()` but not after
the underlying device is closed. A manifest-v2 replay with an attached profile
returns a copied `calibration_profile`; live devices and other replay inputs
return `None`.
