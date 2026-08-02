# pylibfreenect3 2.0

Typed, ownership-safe Python bindings for
[libfreenect2-metal](https://github.com/hbmartin/libfreenect2-metal) 0.4.

`pylibfreenect3` provides a small high-level camera API, explicit low-level
lifecycle primitives, safe zero-copy NumPy views, camera and projective
registration, calibration profiles, runtime statistics, and canonical
recording and replay.

## Start here

- [Install pylibfreenect3](installation.md) from a wheel or source.
- Follow the [getting started guide](getstarted.md) for your first capture.
- Use the [OpenCV and registration cookbook](cookbook.md) for image conversion,
  depth storage, registration, and slow-consumer guidance.
- Browse the generated [API reference](api/index.md).
- Maintain the project with the [maintainer guide](maintainers.md).

Live capture requires a Kinect v2, its power/USB adapter, and a USB 3
connection. Hardware is not required to use frame allocation, recording
validation, or replay from existing raw captures.
