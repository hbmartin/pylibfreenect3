"""Capture only depth, using independent storage for work after capture."""

from pylibfreenect3 import Camera, FrameTimeoutError

with Camera.open(pipeline="cpu", streams=("depth",)) as camera:
    try:
        frames = camera.capture(timeout=2.0, copy=True)
    except FrameTimeoutError:
        print("Timed out waiting for a depth frame")
    else:
        try:
            depth = frames.depth.to_numpy()
            print(depth.shape, depth.dtype, frames.depth.timestamp)
        finally:
            frames.release()
