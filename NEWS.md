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
