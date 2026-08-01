"""Hardware-free tests for the MediaPipe pose demo's pure Python logic."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
DEMO_DIRECTORY = PROJECT_DIRECTORY / "examples" / "mediapipe_pose"
sys.path.insert(0, str(DEMO_DIRECTORY))
sys.path.insert(0, str(PROJECT_DIRECTORY / "examples"))

import numpy as np  # noqa: E402
import opencv_viewer  # noqa: E402
import pytest  # noqa: E402
from opencv_viewer import normalize_depth_for_display  # noqa: E402
from pose_demo import video_timestamp_ms  # noqa: E402
from pose_math import (  # noqa: E402
    LANDMARK_INDEX,
    LANDMARK_NAMES,
    EmaLandmarks,
    Measurement,
    angle_degrees,
    build_record,
    compute_measurements,
)


def test_right_and_straight_angles() -> None:
    assert angle_degrees((1, 0, 0), (0, 0, 0), (0, 1, 0)) == pytest.approx(90.0)
    assert angle_degrees((-1, 0, 0), (0, 0, 0), (1, 0, 0)) == pytest.approx(180.0)
    assert angle_degrees((0, 0, 0), (0, 0, 0), (1, 0, 0)) is None


def test_kinect_measurement_wins_without_mixing_sources() -> None:
    kinect = [None] * len(LANDMARK_NAMES)
    model = [None] * len(LANDMARK_NAMES)
    indices = [
        LANDMARK_INDEX[name] for name in ("left_shoulder", "left_elbow", "left_wrist")
    ]
    for points in (kinect, model):
        points[indices[0]] = (1.0, 0.0, 0.0)
        points[indices[1]] = (0.0, 0.0, 0.0)
        points[indices[2]] = (0.0, 1.0, 0.0)

    measurements = compute_measurements(kinect, model)
    assert measurements["left_elbow"].source == "kinect"
    assert measurements["left_elbow"].value == pytest.approx(90.0)

    kinect[indices[2]] = None
    measurements = compute_measurements(kinect, model)
    assert measurements["left_elbow"].source == "model"
    assert measurements["left_elbow"].value == pytest.approx(90.0)

    model[indices[0]] = None
    measurements = compute_measurements(kinect, model)
    assert measurements["left_elbow"].source == "unavailable"
    assert measurements["left_elbow"].value is None


@pytest.mark.parametrize("source", ["kinect", "model"])
def test_trunk_inclination_uses_camera_up_for_both_coordinate_streams(
    source: str,
) -> None:
    shoulders = (LANDMARK_INDEX["left_shoulder"], LANDMARK_INDEX["right_shoulder"])
    hips = (LANDMARK_INDEX["left_hip"], LANDMARK_INDEX["right_hip"])
    kinect = [None] * len(LANDMARK_NAMES)
    model = [None] * len(LANDMARK_NAMES)
    points = kinect if source == "kinect" else model
    for index in shoulders:
        points[index] = (0.0, -1.0, 2.0)
    for index in hips:
        points[index] = (0.0, 0.0, 2.0)

    measurement = compute_measurements(kinect, model)["trunk_inclination"]
    assert measurement.source == source
    assert measurement.value == pytest.approx(0.0)


def test_ema_smooths_and_resets_after_five_missing_frames() -> None:
    smoother = EmaLandmarks(1, alpha=0.35, reset_after=5)
    assert smoother.update([(1.0, 2.0, 3.0)])[0] == (1.0, 2.0, 3.0)
    smoothed = smoother.update([(3.0, 2.0, 1.0)])[0]
    assert smoothed is not None
    assert smoothed[0] == pytest.approx(1.7)
    for _ in range(4):
        assert smoother.update([None])[0] is not None
    assert smoother.update([None])[0] is None


def test_record_is_strict_json_and_preserves_provenance() -> None:
    measurements = {
        "left_elbow": Measurement(90.0, "kinect"),
        "right_elbow": Measurement(None, "unavailable"),
    }
    record = build_record(
        wall_time_utc="2026-07-30T12:00:00+00:00",
        frame={"color_sequence": 10, "synchronization_valid": True},
        landmarks=[{"name": "nose", "kinect_xyz_m": [0.0, 0.0, 1.0]}],
        measurements=measurements,
        fps=29.97,
        pose_detected=True,
    )
    decoded = json.loads(json.dumps(record, allow_nan=False))
    assert decoded["schema_version"] == 1
    assert decoded["frame"]["color_sequence"] == 10
    assert decoded["frame"]["synchronization_valid"]
    assert decoded["landmarks"][0]["name"] == "nose"
    assert decoded["landmarks"][0]["kinect_xyz_m"] == [0.0, 0.0, 1.0]
    assert decoded["measurements"]["left_elbow"]["source"] == "kinect"
    assert decoded["measurements"]["right_elbow"]["value_degrees"] is None
    assert math.isclose(decoded["processing_fps"], 29.97)


def test_non_finite_measurement_and_fps_remain_strict_json() -> None:
    record = build_record(
        wall_time_utc="2026-07-30T12:00:00+00:00",
        frame={"color_sequence": 1, "synchronization_valid": True},
        landmarks=[],
        measurements={"left_elbow": Measurement(float("nan"), "kinect")},
        fps=float("inf"),
        pose_detected=True,
    )
    decoded = json.loads(json.dumps(record, allow_nan=False))
    assert decoded["measurements"]["left_elbow"]["value_degrees"] is None
    assert decoded["processing_fps"] is None


def test_depth_normalization_handles_invalid_and_constant_input() -> None:
    invalid = np.array([[0.0, np.nan, np.inf]], dtype=np.float32)
    np.testing.assert_array_equal(
        normalize_depth_for_display(invalid), np.zeros((1, 3), dtype=np.uint8)
    )
    constant = np.full((2, 2), 1000.0, dtype=np.float32)
    np.testing.assert_array_equal(
        normalize_depth_for_display(constant), np.full((2, 2), 255, dtype=np.uint8)
    )


def test_video_timestamp_uses_arrival_time_and_is_strictly_increasing() -> None:
    assert video_timestamp_ms(1_234_999, -1) == 1234
    assert video_timestamp_ms(1_234_999, 1234) == 1235


def test_opencv_viewer_allows_quit_after_capture_timeout() -> None:
    class TimeoutCamera:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def capture(self, *, timeout):
            raise opencv_viewer.FrameTimeoutError

    cv2 = Mock()
    cv2.waitKey.return_value = ord("q")
    with (
        patch.dict(sys.modules, {"cv2": cv2}),
        patch.object(opencv_viewer.Camera, "open", return_value=TimeoutCamera()),
    ):
        assert opencv_viewer.main([]) == 0

    cv2.waitKey.assert_called_once_with(1)
    cv2.destroyAllWindows.assert_called_once_with()


def test_opencv_viewer_stops_after_repeated_timeouts_and_cleans_up() -> None:
    class TimeoutCamera:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def capture(self, *, timeout):
            self.calls += 1
            raise opencv_viewer.FrameTimeoutError

    camera = TimeoutCamera()
    cv2 = Mock()
    cv2.waitKey.return_value = -1
    with (
        patch.dict(sys.modules, {"cv2": cv2}),
        patch.object(opencv_viewer.Camera, "open", return_value=camera),
        pytest.raises(RuntimeError, match="after 3 timeouts"),
    ):
        opencv_viewer.main([])

    assert camera.calls == 3
    cv2.destroyAllWindows.assert_called_once_with()
