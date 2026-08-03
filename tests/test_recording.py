from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import pylibfreenect3.recording as recording_module
from pylibfreenect3 import (
    ColorCameraParams,
    IrCameraParams,
    RecordingError,
    RecordingStats,
    RecordingWriter,
)
from pylibfreenect3.legacy import RecordingBundle
from pylibfreenect3.types import (
    DEPTH_LOOKUP_TABLE_SIZE,
    DEPTH_TABLE_SIZE,
    P0_TABLES_BYTE_LENGTH,
)


def _descriptor(path: Path, root: Path, array: np.ndarray) -> dict[str, object]:
    data = np.ascontiguousarray(array).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": path.relative_to(root).as_posix(),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "byte_length": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def write_legacy_bundle(path: Path) -> RecordingBundle:
    path.mkdir()
    frame_directory = path / "frames"
    calibration_directory = path / "calibration"
    frame_directory.mkdir()
    calibration_directory.mkdir()

    color_data = b"\xff\xd8\xff\xd9"
    depth_data = bytes(range(16))
    color_path = frame_directory / "color_101_1.jpg"
    depth_path = frame_directory / "depth_201_1.depth"
    color_path.write_bytes(color_data)
    depth_path.write_bytes(depth_data)

    calibration = {
        "color": asdict(ColorCameraParams()),
        "ir": asdict(IrCameraParams()),
        "p0_tables": _descriptor(
            calibration_directory / "p0.bin",
            path,
            np.zeros(P0_TABLES_BYTE_LENGTH, np.uint8),
        ),
        "x_table": _descriptor(
            calibration_directory / "x.bin",
            path,
            np.zeros(DEPTH_TABLE_SIZE, np.float32),
        ),
        "z_table": _descriptor(
            calibration_directory / "z.bin",
            path,
            np.zeros(DEPTH_TABLE_SIZE, np.float32),
        ),
        "lookup_table": _descriptor(
            calibration_directory / "lookup.bin",
            path,
            np.zeros(DEPTH_LOOKUP_TABLE_SIZE, np.int16),
        ),
    }
    manifest = {
        "schema_version": 1,
        "core": {"version": "0.3.0", "api_version": 3, "revision": "fixture"},
        "device": {
            "serial": "test-serial",
            "firmware": "test-firmware",
            "pipeline": "dump",
        },
        "enabled_streams": ["color", "ir", "depth"],
        "calibration": calibration,
        "frame_index": [
            {
                "stream": "color",
                "path": color_path.relative_to(path).as_posix(),
                "timestamp": 101,
                "sequence": 1,
                "size": len(color_data),
                "sha256": hashlib.sha256(color_data).hexdigest(),
            },
            {
                "stream": "depth",
                "path": depth_path.relative_to(path).as_posix(),
                "timestamp": 201,
                "sequence": 1,
                "size": len(depth_data),
                "sha256": hashlib.sha256(depth_data).hexdigest(),
            },
        ],
        "recording_stats": {"submitted": 1, "written": 1, "dropped": 0, "failed": 0},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    return RecordingBundle.open(path)


def test_legacy_recording_bundle_remains_readable(tmp_path: Path) -> None:
    bundle = write_legacy_bundle(tmp_path / "capture")
    assert tuple(map(str, bundle.streams)) == ("color", "ir", "depth")
    assert bundle.manifest["schema_version"] == 1
    assert len(bundle.frame_paths(("color",))) == 1
    assert len(bundle.frame_paths(("ir",))) == 1
    assert len(bundle.frame_paths(("depth",))) == 1
    assert bundle.calibration.x_table.dtype == np.float32
    assert bundle.calibration.lookup_table.shape == (DEPTH_LOOKUP_TABLE_SIZE,)


def test_legacy_reader_reuses_frame_integrity_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "capture"
    bundle = write_legacy_bundle(path)
    original_read_bytes = Path.read_bytes

    def reject_second_frame_read(candidate: Path) -> bytes:
        if candidate.parent == path / "frames":
            raise AssertionError(f"frame was hashed more than once: {candidate}")
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", reject_second_frame_read)
    assert bundle.frame_paths(("color", "depth"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest["calibration"]["p0_tables"].update(path="../x"),
            "escapes",
        ),
        (
            lambda manifest: manifest["frame_index"][0].update(path="../x.jpg"),
            "escapes",
        ),
        (lambda manifest: manifest["core"].update(api_version=2), "compatible"),
        (
            lambda manifest: manifest.update(
                frame_index=[
                    entry
                    for entry in manifest["frame_index"]
                    if entry["stream"] != "color"
                ]
            ),
            "color",
        ),
    ],
)
def test_legacy_reader_rejects_invalid_manifests(
    tmp_path: Path, mutation: object, message: str
) -> None:
    path = tmp_path / "capture"
    write_legacy_bundle(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    mutation(manifest)  # type: ignore[operator]
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(ValueError, match=message):
        RecordingBundle.open(path)


def test_legacy_reader_detects_frame_and_calibration_corruption(tmp_path: Path) -> None:
    path = tmp_path / "capture"
    bundle = write_legacy_bundle(path)
    Path(bundle.frame_paths(("color",))[0]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        RecordingBundle.open(path)

    path = tmp_path / "capture-calibration"
    bundle = write_legacy_bundle(path)
    x_path = path / bundle.manifest["calibration"]["x_table"]["path"]
    data = bytearray(x_path.read_bytes())
    data[0] ^= 0xFF
    x_path.write_bytes(data)
    with pytest.raises(ValueError, match="calibration sidecar"):
        RecordingBundle.open(path)


@pytest.mark.parametrize("queue_capacity", [0, -1, 1.5, True])
def test_canonical_recording_queue_capacity_is_validated(
    tmp_path: Path, queue_capacity: object
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RecordingWriter(
            tmp_path / "capture",
            queue_capacity=queue_capacity,  # type: ignore[arg-type]
        )


def test_canonical_recording_streams_and_mismatch_flag_are_validated(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="color and/or depth"):
        RecordingWriter(tmp_path / "capture", streams=("ir",))
    with pytest.raises(TypeError, match="allow_serial_mismatch"):
        RecordingWriter(
            tmp_path / "capture",
            allow_serial_mismatch=1,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="calibration_profile"):
        RecordingWriter(
            tmp_path / "capture",
            calibration_profile=object(),  # type: ignore[arg-type]
        )


def test_canonical_recording_wrapper_owns_and_finalizes_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[object] = []

    class FakeNativeWriter:
        is_open = True
        last_error = ""

        def __init__(self, path: str, queue_capacity: int) -> None:
            events.append(("writer", Path(path), queue_capacity))

        def publish_calibration(self, device: object) -> None:
            events.append(("calibration", device))

        def statistics(self) -> dict[str, int]:
            return {
                "written_frames": 3,
                "written_color_frames": 1,
                "written_depth_frames": 2,
                "dropped_frames": 0,
                "written_bytes": 12,
            }

        def close(self) -> None:
            events.append("writer-close")

    class FakeNativeDevice:
        def set_color_listener(self, listener: object) -> None:
            events.append(("color-listener", listener))

        def set_depth_listener(self, listener: object) -> None:
            events.append(("depth-listener", listener))

    class FakeDevice:
        def __init__(self) -> None:
            self._native = FakeNativeDevice()

        def start(self, *, rgb: bool, depth: bool) -> None:
            events.append(("start", rgb, depth))

        def close(self) -> None:
            events.append("device-close")

    class FakeContext:
        def open_device(self, device: object, *, pipeline: object) -> FakeDevice:
            events.append(("open", device, pipeline))
            return FakeDevice()

    monkeypatch.setattr(
        recording_module._native, "NativeRecordingWriterHandle", FakeNativeWriter
    )
    monkeypatch.setattr(recording_module, "Context", FakeContext)

    with RecordingWriter(tmp_path / "capture", queue_capacity=4) as writer:
        assert writer.stats == RecordingStats(3, 1, 2, 0, 12)

    assert events[-2:] == ["device-close", "writer-close"]
    assert writer.stats == RecordingStats(3, 1, 2, 0, 12)


def test_canonical_capture_requires_one_enabled_incremental_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeNativeWriter:
        is_open = True
        last_error = ""

        def __init__(self, path: str, queue_capacity: int) -> None:
            pass

        def publish_calibration(self, device: object) -> None:
            pass

        def statistics(self) -> dict[str, int]:
            return {
                "written_frames": 0,
                "written_color_frames": 0,
                "written_depth_frames": 0,
                "dropped_frames": 0,
                "written_bytes": 0,
            }

        def close(self) -> None:
            pass

    class FakeNativeDevice:
        def set_depth_listener(self, listener: object) -> None:
            pass

    class FakeDevice:
        _native = FakeNativeDevice()

        def start(self, *, rgb: bool, depth: bool) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeContext:
        def open_device(self, device: object, *, pipeline: object) -> FakeDevice:
            return FakeDevice()

    monkeypatch.setattr(
        recording_module._native, "NativeRecordingWriterHandle", FakeNativeWriter
    )
    monkeypatch.setattr(recording_module, "Context", FakeContext)

    with RecordingWriter(tmp_path / "capture", streams=("depth",)) as writer:
        with pytest.raises(ValueError, match="exactly one"):
            writer.capture()
        with pytest.raises(ValueError, match="exactly one"):
            writer.capture(depth_frames=1, duration=1.0)
        with pytest.raises(ValueError, match="color recording stream"):
            writer.capture(color_frames=1)
        with pytest.raises(ValueError, match="positive integer"):
            writer.capture(depth_frames=0)
        with pytest.raises(ValueError, match="cannot be combined"):
            writer.capture(duration=1.0, timeout=1.0)
        with pytest.raises(ValueError, match="finite and positive"):
            writer.capture(duration=0.0)


def test_exceptional_exit_finalizes_and_preserves_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class FakeNativeWriter:
        is_open = True
        last_error = ""

        def __init__(self, path: str, queue_capacity: int) -> None:
            pass

        def publish_calibration(self, device: object) -> None:
            pass

        def statistics(self) -> dict[str, int]:
            return {
                "written_frames": 0,
                "written_color_frames": 0,
                "written_depth_frames": 0,
                "dropped_frames": 0,
                "written_bytes": 0,
            }

        def close(self) -> None:
            events.append("writer-close")

    class FakeNativeDevice:
        def set_color_listener(self, listener: object) -> None:
            pass

        def set_depth_listener(self, listener: object) -> None:
            pass

    class FakeDevice:
        _native = FakeNativeDevice()

        def start(self, *, rgb: bool, depth: bool) -> None:
            pass

        def close(self) -> None:
            events.append("device-close")

    class FakeContext:
        def open_device(self, device: object, *, pipeline: object) -> FakeDevice:
            return FakeDevice()

    monkeypatch.setattr(
        recording_module._native, "NativeRecordingWriterHandle", FakeNativeWriter
    )
    monkeypatch.setattr(recording_module, "Context", FakeContext)

    with (
        pytest.raises(RuntimeError, match="application failed"),
        RecordingWriter(tmp_path / "capture"),
    ):
        raise RuntimeError("application failed")
    assert events == ["device-close", "writer-close"]


def test_normal_finalization_failure_preserves_incomplete_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "capture"

    class FailingNativeWriter:
        is_open = True
        last_error = "disk full"

        def __init__(self, target: str, queue_capacity: int) -> None:
            Path(target).mkdir()

        def publish_calibration(self, device: object) -> None:
            pass

        def statistics(self) -> dict[str, int]:
            return {
                "written_frames": 0,
                "written_color_frames": 0,
                "written_depth_frames": 0,
                "dropped_frames": 0,
                "written_bytes": 0,
            }

        def close(self) -> None:
            raise RecordingError("disk full")

    class FakeNativeDevice:
        def set_color_listener(self, listener: object) -> None:
            pass

        def set_depth_listener(self, listener: object) -> None:
            pass

    class FakeDevice:
        _native = FakeNativeDevice()

        def start(self, *, rgb: bool, depth: bool) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeContext:
        def open_device(self, device: object, *, pipeline: object) -> FakeDevice:
            return FakeDevice()

    monkeypatch.setattr(
        recording_module._native,
        "NativeRecordingWriterHandle",
        FailingNativeWriter,
    )
    monkeypatch.setattr(recording_module, "Context", FakeContext)

    with pytest.raises(RecordingError, match="disk full"), RecordingWriter(path):
        pass
    assert path.is_dir()


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

    with pytest.raises(ValueError, match="X table must contain exactly"):
        ReplayCalibration(
            ColorCameraParams(),
            IrCameraParams(),
            np.zeros(P0_TABLES_BYTE_LENGTH, np.uint8),
            np.zeros(DEPTH_TABLE_SIZE - 1, np.float32),
            np.zeros(DEPTH_TABLE_SIZE, np.float32),
            np.zeros(DEPTH_LOOKUP_TABLE_SIZE, np.int16),
        )
