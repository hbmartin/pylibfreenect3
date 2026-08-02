from __future__ import annotations

import gc
import itertools
import os
import resource
import sys
from pathlib import Path

import numpy as np
import pytest

from pylibfreenect3 import (
    AlignmentConfig,
    Camera,
    DeviceConfig,
    FrameFormat,
    FrameType,
    LedSettings,
    RecordingWriter,
    Registration,
    available_pipelines,
)
from pylibfreenect3.lowlevel import Context, FrameListener

AVAILABLE_PIPELINES = available_pipelines()
METAL_UNAVAILABLE = pytest.mark.skipif(
    "metal" not in AVAILABLE_PIPELINES, reason="Metal backend is unavailable"
)


def _positive_environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _maximum_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


@pytest.mark.hardware
@pytest.mark.parametrize(
    "pipeline",
    [pytest.param("metal", marks=METAL_UNAVAILABLE), "auto"],
)
def test_kinect_capture_100_frames(pipeline: str) -> None:
    sequences: list[int] = []
    with Camera.open(
        pipeline=pipeline,
        streams=("color", "ir", "depth"),
        alignment=AlignmentConfig(max_delta=0.025, queue_capacity=8),
    ) as camera:
        if pipeline == "metal":
            assert camera.pipeline == "metal"
        else:
            assert camera.pipeline in AVAILABLE_PIPELINES
        registration = Registration.from_device(camera.device)
        workspace = registration.workspace(
            include_depth_to_color_map=True,
            include_color_to_depth_map=True,
        )
        for index in range(100):
            with camera.capture(timeout=2.0) as frames:
                assert frames.color.frame_type is FrameType.COLOR
                assert frames.ir.frame_type is FrameType.IR
                assert frames.depth.frame_type is FrameType.DEPTH
                assert frames.color.format in (FrameFormat.BGRX, FrameFormat.RGBX)
                assert frames.ir.format is FrameFormat.FLOAT
                assert frames.depth.format is FrameFormat.FLOAT
                assert frames.color.to_numpy().shape == (1080, 1920, 4)
                assert frames.ir.to_numpy().shape == (424, 512)
                assert frames.depth.to_numpy().shape == (424, 512)
                assert frames.color.timestamp > 0
                assert frames.depth.timestamp > 0
                assert frames.color.arrival_timestamp_us > 0
                assert frames.depth.arrival_timestamp_us > 0
                assert frames.alignment_delta_ticks is not None
                assert frames.alignment_delta_ticks <= 200
                sequences.append(frames.depth.sequence)
                if index == 0:
                    registered = workspace.apply(frames.color, frames.depth)
                    assert registered.undistorted.to_numpy().shape == (424, 512)
                    assert registered.registered.to_numpy().shape == (424, 512, 4)
                    lifted = workspace.lift_normalized([[0.5, 0.5]])
                    assert lifted.valid[0]
                    assert np.isfinite(lifted.xyz[0]).all()
        assert all(right > left for left, right in itertools.pairwise(sequences))
        assert camera.alignment_stats is not None
        assert camera.alignment_stats.delivered >= 100
        assert camera.runtime_stats.color.decoded_frames >= 100
        assert camera.runtime_stats.depth.decoded_frames >= 100
        assert camera.runtime_stats.successful_starts == 1


@pytest.mark.hardware
@pytest.mark.parametrize(
    "pipeline",
    [pytest.param("metal", marks=METAL_UNAVAILABLE), "cpu"],
)
def test_capture_memory_soak_reaches_rss_plateau(pipeline: str) -> None:
    frame_count = _positive_environment_integer("PYLIBF3_HARDWARE_SOAK_FRAMES", 900)
    max_growth_mb = _positive_environment_integer(
        "PYLIBF3_HARDWARE_SOAK_MAX_RSS_MB", 128
    )
    warmup_frames = min(90, max(30, frame_count // 10))

    with Camera.open(
        pipeline=pipeline,
        streams=("color", "ir", "depth"),
        alignment=AlignmentConfig(max_delta=0.025, queue_capacity=8),
    ) as camera:
        registration = Registration.from_device(camera.device)
        workspace = registration.workspace(
            include_depth_to_color_map=True,
            include_color_to_depth_map=True,
        )
        color_buffer = np.empty((1080, 1920, 3), dtype=np.uint8)
        buffer_ids = (
            id(workspace.result),
            id(workspace.undistorted),
            id(workspace.registered),
            id(workspace.depth_to_color_map),
            id(workspace.color_to_depth_map),
            id(color_buffer),
        )
        for _ in range(warmup_frames):
            with camera.capture(timeout=2.0) as frames:
                assert np.isfinite(frames.depth.to_numpy()).any()
                workspace.apply(frames.color, frames.depth)
                assert frames.color.to_color(out=color_buffer) is color_buffer
        gc.collect()
        baseline = _maximum_rss_bytes()

        previous_sequence = -1
        for index in range(frame_count):
            with camera.capture(timeout=2.0) as frames:
                depth = frames.depth.to_numpy()
                assert depth.shape == (424, 512)
                assert frames.depth.sequence > previous_sequence
                previous_sequence = frames.depth.sequence
                assert workspace.apply(frames.color, frames.depth) is workspace.result
                assert frames.color.to_color(out=color_buffer) is color_buffer
                assert buffer_ids == (
                    id(workspace.result),
                    id(workspace.undistorted),
                    id(workspace.registered),
                    id(workspace.depth_to_color_map),
                    id(workspace.color_to_depth_map),
                    id(color_buffer),
                )
            del depth
            if index and index % 300 == 0:
                gc.collect()

        gc.collect()
        growth = _maximum_rss_bytes() - baseline

    assert growth <= max_growth_mb * 1024 * 1024, (
        f"{pipeline} capture RSS grew by {growth / (1024 * 1024):.1f} MiB "
        f"over {frame_count} frames"
    )


@pytest.mark.hardware
def test_cpu_capture_configuration_restart_and_outstanding_array() -> None:
    context = Context()
    serial = context.default_device_serial_number()
    assert serial
    assert context.wait_for_device(serial, 2.0)
    device = context.open_device(pipeline="cpu")
    assert device.state.name == "OPEN"
    listener = FrameListener(FrameType.COLOR | FrameType.IR | FrameType.DEPTH)
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
@METAL_UNAVAILABLE
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
    with RecordingWriter(recording_path, streams=("color", "depth")) as writer:
        writer.capture(depth_frames=10, timeout=10.0)
    assert writer.stats.written_depth_frames >= 10
    assert writer.stats.written_frames == (
        writer.stats.written_color_frames + writer.stats.written_depth_frames
    )
    assert writer.stats.written_bytes > 0

    replay_pipelines = ("cpu", "metal") if "metal" in AVAILABLE_PIPELINES else ("cpu",)
    for pipeline in replay_pipelines:
        with (
            Camera.open_recording(recording_path, pipeline=pipeline) as replay,
            replay.capture(timeout=5.0) as frames,
        ):
            assert replay.pipeline == pipeline
            assert frames.color.to_numpy().shape == (1080, 1920, 4)
            assert frames.depth.to_numpy().shape == (424, 512)
            assert replay.runtime_stats.color.decoded_frames >= 1
            assert replay.runtime_stats.depth.decoded_frames >= 1


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
