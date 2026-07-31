Getting started
===============

The high-level API owns the context, device, listener, and pipeline::

   from pylibfreenect3 import Camera

   with Camera.open(pipeline="auto", streams=("color", "depth")) as camera:
       with camera.capture(timeout=2.0) as frames:
           depth = frames.depth.to_numpy()

The NumPy array is a zero-copy view by default and keeps all required native
state alive. ``copy=True`` creates independent storage and immediately releases
the native capture. Releasing a frame set is idempotent and rejects new frame
lookups.

Use ``Camera.frames()`` for synchronous iteration and ``Camera.open_recording``
for schema-v1 bundles. Async capture and decoder-thread Python callbacks are
not part of 0.3.
