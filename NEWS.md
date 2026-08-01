# Changelog

## 1.0.0

- Requires Python 3.12+ and adopts a curated high-level API with native
  lifecycle primitives under `pylibfreenect3.lowlevel`.
- Replaces the legacy packaging stack with reproducible scikit-build-core/CMake
  builds driven by uv.
- Adds `Stream` and `Pipeline`, mapping-compatible frame sets, NumPy 2 array
  protocol support, frozen value objects, and complete native type boundaries.
- `Camera.capture` and `Camera.frames` now default to `timeout=2.0` seconds
  instead of blocking indefinitely; pass `timeout=None` for the 0.3 behavior.
- Retains libfreenect2-metal 0.3/API 3 and recording schema-v1 compatibility.
- Adds opt-in timestamp alignment, arrival timestamps, delivery/drop
  statistics, pipeline decoder configuration, device state diagnostics, and
  interruptible device waiting.
- Adds native BGR/RGB conversion, OpenCV-compatible IR calibration arrays,
  reusable registration workspaces, forward and reverse maps, batched XYZ,
  and normalized color-landmark lifting without new runtime dependencies.
- Moves the full MediaPipe pose demo from the core repository and adds an
  aligned OpenCV viewer; both examples use a 25 ms threshold and queue capacity
  eight.
- Renames `include_color_depth_map` and `color_depth_map` to
  `include_depth_to_color_map` and `depth_to_color_map`; the old names warn as
  deprecated aliases. The alias covers attribute access only, so
  `RegistrationResult` can no longer be constructed with a `color_depth_map`
  keyword.
- Adds `WorkspaceStateError`, a `DeviceStateError` subclass, for using a
  `RegistrationWorkspace` before `apply()` or without the reverse map it was
  configured for.

## 0.3.0

- Renamed the distribution and import namespace to `pylibfreenect3` with no
  compatibility alias.
- Added safe zero-copy capture ownership, typed value objects, context
  managers, seconds-based timeouts, Pythonic exceptions, and synchronous
  iteration.
- Exposed the complete supported device, calibration, exposure, color,
  pipeline, dump, registration, replay, and logger-utility surfaces.
- Added schema-v1 atomic recording bundles and calibrated CPU/Metal replay.
- Added bounded background recording writes with block/drop overflow policies
  and observable queue statistics.
- Reject native camera resources inherited across `fork()` and document the
  multiprocessing `spawn` boundary.
- `Frame.from_array` now rejects read-only arrays because native frames
  expose writable data; pass a writable copy for mapped or frozen inputs.
- Added ownership stress coverage, hardware RSS soaks, and an OpenCV/offline
  registration cookbook.
- Replaced the legacy build with `pyproject.toml`, Cython 3, C++17,
  cibuildwheel, delocate, and auditwheel release infrastructure.

## 0.2.0

- Historical `pylibfreenect2` release. This namespace is not carried forward.
