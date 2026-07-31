from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pylibfreenect3 import (
    ColorCameraParams,
    DumpPacketPipeline,
    Frame,
    FrameFormat,
    FrameSet,
    FrameType,
    IrCameraParams,
    RecordingBundle,
    RecordingFormatError,
    RecordingWriter,
)
from pylibfreenect3.types import (
    DEPTH_LOOKUP_TABLE_SIZE,
    DEPTH_TABLE_SIZE,
    P0_TABLES_BYTE_LENGTH,
)


class FakeDumpPipeline(DumpPacketPipeline):
    def __init__(self) -> None:
        pass

    def depth_p0_tables(self) -> np.ndarray:
        return np.zeros(P0_TABLES_BYTE_LENGTH, dtype=np.uint8)

    def depth_x_table(self) -> np.ndarray:
        return np.zeros(DEPTH_TABLE_SIZE, dtype=np.float32)

    def depth_z_table(self) -> np.ndarray:
        return np.zeros(DEPTH_TABLE_SIZE, dtype=np.float32)

    def depth_lookup_table(self) -> np.ndarray:
        return np.zeros(DEPTH_LOOKUP_TABLE_SIZE, dtype=np.int16)


def make_camera() -> SimpleNamespace:
    device = SimpleNamespace(
        pipeline=FakeDumpPipeline(),
        color_camera_params=ColorCameraParams(),
        ir_camera_params=IrCameraParams(),
        serial_number="test-serial",
        firmware_version="test-firmware",
        pipeline_name="dump",
    )
    return SimpleNamespace(device=device, streams=("color", "ir", "depth"))


def make_frames() -> FrameSet:
    color = Frame.from_array(
        np.array([0xFF, 0xD8, 0xFF, 0xD9], dtype=np.uint8),
        frame_type=FrameType.COLOR,
        frame_format=FrameFormat.RAW,
        timestamp=100,
        sequence=1,
    )
    depth = Frame.from_array(
        np.arange(16, dtype=np.uint8),
        frame_type=FrameType.DEPTH,
        frame_format=FrameFormat.RAW,
        timestamp=101,
        sequence=1,
    )
    return FrameSet(copied={FrameType.COLOR: color, FrameType.DEPTH: depth})


def write_bundle(path: Path) -> RecordingBundle:
    with RecordingWriter(path, make_camera()) as writer:
        writer.write(make_frames())
    return RecordingBundle.open(path)


def test_recording_bundle_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "capture"
    bundle = write_bundle(path)
    assert bundle.streams == ("color", "ir", "depth")
    assert bundle.manifest["schema_version"] == 1
    assert [entry["stream"] for entry in bundle.manifest["frame_index"]] == [
        "color",
        "depth",
    ]
    assert len(bundle.frame_paths(("color",))) == 1
    assert len(bundle.frame_paths(("ir",))) == 1
    assert len(bundle.frame_paths(("depth",))) == 1
    assert bundle.calibration.x_table.dtype == np.float32
    assert bundle.calibration.lookup_table.shape == (DEPTH_LOOKUP_TABLE_SIZE,)


def test_recording_integrity_failure_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "capture"
    bundle = write_bundle(path)
    frame_path = Path(bundle.frame_paths(("color",))[0])
    frame_path.write_bytes(b"corrupt")
    with pytest.raises(RecordingFormatError, match="integrity"):
        RecordingBundle.open(path)


def test_recording_path_escape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "capture"
    write_bundle(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["calibration"]["p0_tables"]["path"] = "../outside.bin"
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(RecordingFormatError, match="escapes"):
        RecordingBundle.open(path)


def test_frame_index_path_escape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "capture"
    write_bundle(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    color_entry = next(
        entry for entry in manifest["frame_index"] if entry["stream"] == "color"
    )
    color_entry["path"] = "../outside.jpg"
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(RecordingFormatError, match="escapes"):
        RecordingBundle.open(path)


def test_calibration_sidecar_integrity_failure_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "capture"
    bundle = write_bundle(path)
    x_path = path / bundle.manifest["calibration"]["x_table"]["path"]
    data = bytearray(x_path.read_bytes())
    data[0] ^= 0xFF
    x_path.write_bytes(data)
    with pytest.raises(RecordingFormatError, match="calibration sidecar"):
        RecordingBundle.open(path)


def test_recording_incompatible_core_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "capture"
    write_bundle(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["core"]["api_version"] = 2
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(RecordingFormatError, match="compatible"):
        RecordingBundle.open(path)


def test_recording_requires_frames_for_every_enabled_stream(tmp_path: Path) -> None:
    path = tmp_path / "capture"
    write_bundle(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["frame_index"] = [
        entry for entry in manifest["frame_index"] if entry["stream"] != "color"
    ]
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(RecordingFormatError, match="color"):
        RecordingBundle.open(path)


def test_recording_target_is_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        with RecordingWriter(target, make_camera()):
            pass


def test_recording_exception_removes_partial_bundle(tmp_path: Path) -> None:
    target = tmp_path / "capture"
    with pytest.raises(RuntimeError):
        with RecordingWriter(target, make_camera()):
            raise RuntimeError("capture failed")
    assert not target.exists()
    assert not list(tmp_path.glob(".capture.partial-*"))


def test_replay_calibration_rejects_wrong_shapes_and_dtypes() -> None:
    from pylibfreenect3 import ReplayCalibration

    with pytest.raises(TypeError):
        ReplayCalibration(
            ColorCameraParams(),
            IrCameraParams(),
            np.zeros(P0_TABLES_BYTE_LENGTH, np.int8),
            np.zeros(DEPTH_TABLE_SIZE, np.float32),
            np.zeros(DEPTH_TABLE_SIZE, np.float32),
            np.zeros(DEPTH_LOOKUP_TABLE_SIZE, np.int16),
        )
