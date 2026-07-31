# Getting started

The high-level API owns the context, device, listener, and pipeline:

```python
from pylibfreenect3 import AlignmentConfig, Camera

with Camera.open(
    pipeline="auto",
    streams=("color", "depth"),
    alignment=AlignmentConfig(max_delta=0.025, queue_capacity=8),
) as camera:
    with camera.capture(timeout=2.0) as frames:
        depth = frames.depth.to_numpy()
        bgr = frames.color.to_color()
        print(frames.alignment_delta_seconds)
```

The NumPy array is a zero-copy view by default and keeps all required native
state alive. `copy=True` creates independent storage and immediately releases
the native capture. Releasing a frame set is idempotent and rejects new frame
lookups.

Use `Camera.frames()` for synchronous iteration and `Camera.open_recording`
for schema-v1 bundles. Async capture and decoder-thread Python callbacks are
not part of 0.3.

Timestamp alignment is opt-in. Omitting `alignment` preserves the legacy
arrival-based pairing behavior and makes `FrameSet.alignment_delta_ticks` and
`Camera.alignment_stats` return `None`. Vision examples explicitly use a 25 ms
threshold and bounded queues of eight frames per stream.

## Processes and threads

Blocking frame waits release the GIL, so ordinary Python threads can continue
while a listener waits for a frame. Native contexts, pipelines, devices,
listeners, frames, and frame sets are bound to the process that created them.
They must not be inherited by a child created with `fork()`; inherited
resources raise `DeviceStateError` before calling into libfreenect2.

When using `multiprocessing`, create and open the camera inside the child
process. Prefer the `spawn` start method when the parent may already have
camera resources or other native threads:

```python
import multiprocessing as mp

from pylibfreenect3 import Camera


def capture_one() -> None:
    with Camera.open(streams=("depth",)) as camera:
        with camera.capture(timeout=2.0) as frames:
            consume(frames.depth.to_numpy(copy=True))


if __name__ == "__main__":
    process = mp.get_context("spawn").Process(target=capture_one)
    process.start()
    process.join()
```
