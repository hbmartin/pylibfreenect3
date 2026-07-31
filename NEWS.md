# Changelog

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
- Added ownership stress coverage, hardware RSS soaks, and an OpenCV/offline
  registration cookbook.
- Replaced the legacy build with `pyproject.toml`, Cython 3, C++17,
  cibuildwheel, delocate, and auditwheel release infrastructure.

## 0.2.0

- Historical `pylibfreenect2` release. This namespace is not carried forward.
