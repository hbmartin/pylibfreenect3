"""Capture and register color, IR, and depth frames synchronously."""

from pylibfreenect3 import Camera, FrameTimeoutError, Registration

with Camera.open(pipeline="auto", streams=("color", "ir", "depth")) as camera:
    registration = Registration(
        camera.device.ir_camera_params,
        camera.device.color_camera_params,
    )
    while True:
        try:
            frames = camera.capture(timeout=2.0)
        except FrameTimeoutError:
            print("Timed out waiting for synchronized frames; retrying")
            continue
        with frames:
            result = registration.apply(frames.color, frames.depth)
            print(
                frames.color.sequence,
                frames.ir.to_numpy().shape,
                result.registered.to_numpy().shape,
            )
