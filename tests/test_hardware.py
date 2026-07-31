from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest

from pylibfreenect3 import (
    Camera,
    DeviceConfig,
    FrameFormat,
    FrameType,
    Freenect2,
    LedSettings,
    RecordingWriter,
    Registration,
    SyncFrameListener,
)


@pytest.mark.hardware
@pytest.mark.parametrize("pipeline", ["metal", "auto"])
def test_kinect_capture_100_frames(pipeline: str) -> None:
    sequences: list[int] = []
    with Camera.open(pipeline=pipeline, streams=("color", "ir", "depth")) as camera:
        assert camera.pipeline == "metal"
        registration = Registration(
            camera.device.ir_camera_params,
            camera.device.color_camera_params,
        )
        for index in range(100):
            with camera.capture(timeout=2.0) as frames:
                assert frames.color.type is FrameType.COLOR
                assert frames.ir.type is FrameType.IR
                assert frames.depth.type is FrameType.DEPTH
                assert frames.color.format in (FrameFormat.BGRX, FrameFormat.RGBX)
                assert frames.ir.format is FrameFormat.FLOAT
                assert frames.depth.format is FrameFormat.FLOAT
                assert frames.color.to_numpy().shape == (1080, 1920, 4)
                assert frames.ir.to_numpy().shape == (424, 512)
                assert frames.depth.to_numpy().shape == (424, 512)
                assert frames.color.timestamp > 0
                assert frames.depth.timestamp > 0
                sequences.append(frames.depth.sequence)
                if index == 0:
                    registered = registration.apply(frames.color, frames.depth)
                    assert registered.undistorted.to_numpy().shape == (424, 512)
                    assert registered.registered.to_numpy().shape == (424, 512, 4)
        assert all(right > left for left, right in zip(sequences, sequences[1:]))


@pytest.mark.hardware
def test_cpu_capture_configuration_restart_and_outstanding_array() -> None:
    context = Freenect2()
    device = context.open_device(pipeline="cpu")
    listener = SyncFrameListener(FrameType.COLOR | FrameType.IR | FrameType.DEPTH)
    device.configuration = DeviceConfig(
        min_depth=0.6,
        max_depth=5.0,
        enable_bilateral_filter=True,
        enable_edge_aware_filter=True,
    )
    assert device.configuration.min_depth == 0.6
    device.set_color_auto_exposure(0.0)
    device.set_led_status(LedSettings(0, mode=0, start_level=1000))
    device.set_color_listener(listener)
    device.set_depth_listener(listener)
    device.start()
    first = listener.wait(timeout=2.0)
    borrowed = first.depth
    array = borrowed.to_numpy()
    first.release()
    device.stop()
    assert np.isfinite(array).any()

    device.start()
    with listener.wait(timeout=2.0) as second:
        assert second.depth.sequence >= borrowed.sequence
    device.close()
    assert np.isfinite(array).any()
    del borrowed, array
    gc.collect()


@pytest.mark.hardware
def test_repeated_open_close() -> None:
    for _ in range(5):
        camera = Camera.open(pipeline="metal", streams=("depth",))
        with camera.capture(timeout=2.0) as frames:
            assert frames.depth.to_numpy().shape == (424, 512)
        camera.close()
        camera.close()


@pytest.mark.hardware
def test_dump_recording_and_cpu_metal_replay(tmp_path: Path) -> None:
    recording_path = tmp_path / "capture.f3"
    with Camera.open(pipeline="dump", streams=("color", "ir", "depth")) as camera:
        with RecordingWriter(recording_path, camera) as writer:
            writer.capture(10, timeout=2.0)

    for pipeline in ("cpu", "metal"):
        with Camera.open_recording(recording_path, pipeline=pipeline) as replay:
            with replay.capture(timeout=5.0) as frames:
                assert replay.pipeline == pipeline
                assert frames.color.to_numpy().shape == (1080, 1920, 4)
                assert frames.depth.to_numpy().shape == (424, 512)


_OUTSTANDING_AT_SHUTDOWN: np.ndarray | None = None


@pytest.mark.hardware
def test_interpreter_teardown_with_outstanding_numpy_view() -> None:
    global _OUTSTANDING_AT_SHUTDOWN
    camera = Camera.open(pipeline="cpu", streams=("depth",))
    frames = camera.capture(timeout=2.0)
    _OUTSTANDING_AT_SHUTDOWN = frames.depth.to_numpy()
    frames.release()
    camera.close()
    assert np.isfinite(_OUTSTANDING_AT_SHUTDOWN).any()
