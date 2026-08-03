from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from threading import Event

import numpy as np
import pytest

import pylibfreenect3 as f3
from pylibfreenect3.legacy import RecordingBundle
from pylibfreenect3.types import P0_TABLES_BYTE_LENGTH


def write_profile(path: Path, *, serial: str = "test-serial") -> Path:
    camera = {
        "width": 2,
        "height": 2,
        "fx": 1.0,
        "fy": 1.0,
        "cx": 0.5,
        "cy": 0.5,
        "distortion_model": "none",
        "distortion": [],
    }
    profile = {
        "schema": "libfreenect2.calibration-profile",
        "version": 1,
        "device": {"serial": serial, "firmware": "test-firmware"},
        "cameras": {"color": camera, "ir": camera},
        "depth_to_color": {
            "rotation_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation_m": [0.0, 0.0, 0.0],
        },
        "depth_correction": {
            "model": "offset_only",
            "scale": 1.0,
            "offset_mm": 10.0,
            "rmse_mm": 1.5,
        },
        "quality": {
            "color_views": 20,
            "ir_views": 21,
            "stereo_views": 22,
            "depth_views": 23,
            "color_rms_px": 0.2,
            "ir_rms_px": 0.3,
            "held_out_stereo_rms_px": 0.4,
            "depth_rmse_mm": 5.0,
        },
        "provenance": {
            "created_utc": "2026-08-02T00:00:00Z",
            "tool_version": "0.4.0",
            "job_sha256": "ab" * 32,
        },
    }
    path.write_text(json.dumps(profile), "utf-8")
    return path


def write_canonical_recording(
    root: Path,
    *,
    version: int = 2,
    attached_profile: bool = False,
    complete: bool = True,
    arrival_offsets_us: tuple[int, ...] = (0,),
) -> Path:
    (root / "calibration").mkdir(parents=True)
    (root / "frames" / "color").mkdir(parents=True)
    (root / "frames" / "depth").mkdir(parents=True)
    color = b"\xff\xd8\xff\xd9"
    (root / "calibration" / "p0.bin").write_bytes(bytes(P0_TABLES_BYTE_LENGTH))
    color_params = asdict(f3.ColorCameraParams())
    ir_params = asdict(f3.IrCameraParams())
    calibration: dict[str, object] = {
        "color": color_params,
        "ir": ir_params,
        "p0": "calibration/p0.bin",
    }
    if attached_profile:
        write_profile(root / "calibration" / "profile.json")
        calibration["profile"] = "calibration/profile.json"
    manifest = {
        "version": version,
        "device": {"serial": "test-serial", "firmware": "test-firmware"},
        "streams": {
            "color": {"encoding": "jpeg"},
            "depth": {"encoding": "kinect-v2-raw"},
        },
        "calibration": calibration,
        "clocks": {
            "device": "kinect-v2-0.125ms-wrap32",
            "arrival": "monotonic-host-microseconds-relative-to-recording-start",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    journal: list[str] = []
    for index, arrival_offset_us in enumerate(arrival_offsets_us):
        relative = f"frames/color/{index:010d}.jpg"
        (root / relative).write_bytes(color)
        journal.append(
            json.dumps(
                {
                    "index": index,
                    "stream": "color",
                    "path": relative,
                    "byte_count": len(color),
                    "device_timestamp": 345 + index,
                    "sequence": 6 + index,
                    "arrival_offset_us": arrival_offset_us,
                    "exposure": 1.25,
                    "gain": 2.0,
                    "gamma": 1.0,
                }
            )
        )
    (root / "frames.ndjson").write_text("\n".join(journal) + "\n", "utf-8")
    if complete:
        (root / "recording.complete").write_text("complete\n", "utf-8")
    return root


def test_v04_runtime_identity_and_public_exports() -> None:
    assert f3.core_version().startswith("0.4.")
    assert f3.core_api_version() == 4
    assert f3.DistortionModel.NONE == "none"
    assert f3.RegistrationRasterization.FOUR_NEIGHBOR_SPLAT == "four_neighbor_splat"
    assert issubclass(f3.CalibrationError, ValueError)
    assert issubclass(f3.RecordingError, f3.FreenectError)
    assert "RecordingBundle" not in f3.__all__
    with pytest.raises(AttributeError, match=r"legacy\.RecordingBundle"):
        _ = f3.RecordingBundle  # type: ignore[attr-defined]


def test_v04_value_objects_validate_and_remain_immutable() -> None:
    camera = f3.ProjectiveCameraModel(2, 2, 1.0, 1.0, 0.5, 0.5)
    assert camera.rectified() == camera
    scaled = camera.scaled_to(4, 4)
    assert scaled.fx == 2.0
    assert scaled.cx == 1.5
    with pytest.raises(FrozenInstanceError):
        camera.fx = 2.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="coefficient"):
        f3.ProjectiveCameraModel(
            2,
            2,
            1.0,
            1.0,
            0.5,
            0.5,
            f3.DistortionModel.BROWN_CONRADY_5,
            (),
        )
    with pytest.raises(ValueError, match="rigid transform"):
        f3.RigidTransform((1.0,), (0.0, 0.0, 0.0))
    with pytest.raises(TypeError, match="replay options"):
        f3.ReplayOptions(salvage_incomplete=1)  # type: ignore[arg-type]


def test_calibration_profile_round_trip_and_device_binding(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be loaded"):
        f3.CalibrationProfile()
    source = write_profile(tmp_path / "profile.json")
    profile = f3.CalibrationProfile.load(source)
    assert profile.schema_version == 1
    assert profile.serial == "test-serial"
    assert profile.color_camera.width == 2
    assert profile.depth_correction == f3.DepthCorrectionProfile(
        f3.DepthCorrectionModel.OFFSET_ONLY, 1.0, 10.0, 1.5
    )
    assert profile.quality_metrics is not None
    assert profile.quality_metrics.stereo_views == 22
    assert profile.provenance.tool_version == "0.4.0"

    device = type(
        "DeviceIdentity",
        (),
        {"serial_number": "test-serial", "firmware_version": "new-firmware"},
    )()
    assert (
        profile.check_device(device)
        == "calibration profile firmware differs from the device"
    )  # type: ignore[arg-type]
    mismatch = type(
        "DeviceIdentity",
        (),
        {"serial_number": "other", "firmware_version": "test-firmware"},
    )()
    with pytest.raises(f3.CalibrationError, match="serial"):
        profile.check_device(mismatch)  # type: ignore[arg-type]
    assert "explicitly allowed" in (  # type: ignore[arg-type]
        profile.check_device(mismatch, allow_serial_mismatch=True) or ""
    )

    copied = tmp_path / "copy.json"
    profile.save(copied)
    assert f3.CalibrationProfile.load(copied).depth_to_color == profile.depth_to_color

    invalid = json.loads(source.read_text("utf-8"))
    invalid["depth_to_color"]["rotation_row_major"][0] = 2.0
    invalid_path = tmp_path / "invalid-transform.json"
    invalid_path.write_text(json.dumps(invalid), "utf-8")
    with pytest.raises(f3.CalibrationError, match="rotation"):
        f3.CalibrationProfile.load(invalid_path)


def test_projective_registration_is_explicit_and_deterministic(tmp_path: Path) -> None:
    profile = f3.CalibrationProfile.load(write_profile(tmp_path / "profile.json"))
    depth = f3.Frame.from_array(
        np.array([[1000.0, 1100.0], [1200.0, 1300.0]], np.float32),
        frame_type=f3.FrameType.DEPTH,
        frame_format=f3.FrameFormat.FLOAT,
    )
    nearest = f3.ProjectiveRegistration(
        profile,
        options=f3.ProjectiveRegistrationOptions(
            rasterization=f3.RegistrationRasterization.NEAREST
        ),
    )
    first = nearest.apply(depth)
    second = nearest.apply(depth)
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
    np.testing.assert_array_equal(first.to_numpy(), depth.to_numpy())

    splatted = f3.ProjectiveRegistration(profile)
    splat_first = splatted.apply(depth)
    splat_second = splatted.apply(depth)
    np.testing.assert_array_equal(splat_first.to_numpy(), splat_second.to_numpy())
    np.testing.assert_array_equal(
        splat_first.to_numpy(), np.full((2, 2), 1000.0, np.float32)
    )

    corrected = f3.ProjectiveRegistration(
        profile,
        options=f3.ProjectiveRegistrationOptions(
            rasterization=f3.RegistrationRasterization.NEAREST,
            apply_depth_correction=True,
        ),
    ).apply(depth)
    np.testing.assert_array_equal(corrected.to_numpy(), depth.to_numpy() + 10.0)
    limited = f3.ProjectiveRegistration(
        profile,
        options=f3.ProjectiveRegistrationOptions(
            rasterization=f3.RegistrationRasterization.NEAREST,
            min_depth_mm=1050.0,
            max_depth_mm=1250.0,
        ),
    ).apply(depth)
    np.testing.assert_array_equal(
        limited.to_numpy(), np.array([[0.0, 1100.0], [1200.0, 0.0]], np.float32)
    )
    with pytest.raises(ValueError, match="must not alias"):
        nearest.apply(depth, out=depth)
    overlapping_storage = np.zeros((3, 2), np.float32)
    overlapping_depth = f3.Frame.from_array(overlapping_storage[:2])
    overlapping_output = f3.Frame.from_array(overlapping_storage[1:])
    with pytest.raises(ValueError, match="must not alias"):
        nearest.apply(overlapping_depth, out=overlapping_output)

    caller_output = f3.Frame.allocate(2, 2, 4, frame_format=f3.FrameFormat.FLOAT)
    assert nearest.apply(depth, out=caller_output) is caller_output
    np.testing.assert_array_equal(caller_output.to_numpy(), depth.to_numpy())
    with pytest.raises(ValueError, match="target camera"):
        nearest.apply(
            depth,
            out=f3.Frame.allocate(3, 2, 4, frame_format=f3.FrameFormat.FLOAT),
        )
    with pytest.raises(ValueError, match="profile IR camera"):
        nearest.apply(
            f3.Frame.from_array(np.ones((2, 2), np.uint8)),
        )


def test_projective_registration_allows_only_distinct_concurrent_outputs(
    tmp_path: Path,
) -> None:
    profile = f3.CalibrationProfile.load(write_profile(tmp_path / "profile.json"))
    depth = f3.Frame.from_array(np.full((2, 2), 1000.0, np.float32))
    registration = f3.ProjectiveRegistration(
        profile,
        options=f3.ProjectiveRegistrationOptions(
            rasterization=f3.RegistrationRasterization.NEAREST
        ),
    )
    outputs = [
        f3.Frame.allocate(2, 2, 4, frame_format=f3.FrameFormat.FLOAT) for _ in range(4)
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(lambda output: registration.apply(depth, out=output), outputs)
        )
    assert results == outputs
    for output in outputs:
        np.testing.assert_array_equal(output.to_numpy(), depth.to_numpy())

    entered = Event()
    release = Event()

    class BlockingNative:
        def apply(self, depth_frame: object, output_frame: object) -> None:
            entered.set()
            assert release.wait(1.0)

    registration._native = BlockingNative()  # type: ignore[assignment]
    shared = outputs[0]
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(registration.apply, depth, out=shared)
        assert entered.wait(1.0)
        with pytest.raises(ValueError, match="distinct outputs"):
            registration.apply(depth, out=shared)
        release.set()
        assert pending.result() is shared


def test_canonical_recording_replay_stats_and_attached_profile(tmp_path: Path) -> None:
    path = write_canonical_recording(tmp_path / "recording", attached_profile=True)
    context = f3.lowlevel.ReplayContext()
    device = context.open_recording(path, pipeline=f3.Pipeline.DUMP)
    profile = device.calibration_profile
    assert profile is not None
    assert profile.serial == "test-serial"
    listener = f3.lowlevel.FrameListener(f3.FrameType.COLOR)
    device.set_color_listener(listener)
    device.start(rgb=True, depth=False)
    with listener.wait(1.0) as frames:
        assert frames.color.sequence == 6
        assert frames.color.timestamp == 345
        assert frames.color.format is f3.FrameFormat.RAW
    stats = device.runtime_stats
    assert stats.color.decoded_frames == 1
    assert stats.successful_starts == 1
    with pytest.raises(FrozenInstanceError):
        stats.stop_calls = 99  # type: ignore[misc]
    device.stop()
    assert device.runtime_stats.stop_calls == 1
    device.close()
    assert profile.serial == "test-serial"
    with pytest.raises(f3.DeviceStateError, match="closed"):
        _ = device.runtime_stats


@pytest.mark.parametrize("version", [1, 2])
def test_native_replay_accepts_color_only_canonical_versions(
    tmp_path: Path, version: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_canonical_recording(
        tmp_path / f"recording-v{version}", version=version
    )

    def reject_legacy_route(path: object) -> object:
        raise AssertionError(f"canonical replay routed through legacy parser: {path}")

    monkeypatch.setattr(RecordingBundle, "open", reject_legacy_route)
    with f3.Camera.open_recording(
        path,
        pipeline=f3.Pipeline.DUMP,
        streams=(f3.Stream.COLOR,),
    ) as camera:
        with camera.capture(timeout=1.0) as frames:
            assert frames.color.sequence == 6
        assert camera.calibration_profile is None


def test_replay_runtime_snapshots_are_safe_during_capture(tmp_path: Path) -> None:
    path = write_canonical_recording(
        tmp_path / "recording", arrival_offsets_us=tuple(range(20))
    )
    device = f3.lowlevel.ReplayContext().open_recording(path, pipeline=f3.Pipeline.DUMP)
    listener = f3.lowlevel.FrameListener(f3.FrameType.COLOR)
    device.set_color_listener(listener)
    device.start(rgb=True, depth=False)
    with ThreadPoolExecutor(max_workers=4) as executor:
        snapshots = list(executor.map(lambda _: device.runtime_stats, range(100)))
    assert all(snapshot.successful_starts == 1 for snapshot in snapshots)
    device.close()


def test_canonical_replay_salvage_is_explicit(tmp_path: Path) -> None:
    path = write_canonical_recording(tmp_path / "recording", complete=False)
    with pytest.raises(f3.ReplayError):
        f3.lowlevel.ReplayContext().open_recording(path, pipeline=f3.Pipeline.DUMP)
    device = f3.lowlevel.ReplayContext().open_recording(
        path,
        pipeline=f3.Pipeline.DUMP,
        replay_options=f3.ReplayOptions(salvage_incomplete=True),
    )
    assert device.calibration_profile is None
    device.close()


def test_canonical_replay_salvages_a_truncated_final_journal_entry(
    tmp_path: Path,
) -> None:
    path = write_canonical_recording(tmp_path / "recording")
    journal = path / "frames.ndjson"
    journal.write_text(journal.read_text("utf-8") + '{"truncated"', "utf-8")
    with pytest.raises(f3.ReplayError):
        f3.lowlevel.ReplayContext().open_recording(path, pipeline=f3.Pipeline.DUMP)
    device = f3.lowlevel.ReplayContext().open_recording(
        path,
        pipeline=f3.Pipeline.DUMP,
        replay_options=f3.ReplayOptions(salvage_incomplete=True),
    )
    device.close()


def test_canonical_replay_can_reproduce_recorded_timing(tmp_path: Path) -> None:
    path = write_canonical_recording(
        tmp_path / "recording", arrival_offsets_us=(0, 100_000)
    )
    device = f3.lowlevel.ReplayContext().open_recording(
        path,
        pipeline=f3.Pipeline.DUMP,
        replay_options=f3.ReplayOptions(reproduce_timing=True),
    )
    listener = f3.lowlevel.FrameListener(f3.FrameType.COLOR)
    device.set_color_listener(listener)
    start = time.monotonic()
    device.start(rgb=True, depth=False)
    with listener.wait(1.0):
        pass
    with listener.wait(1.0):
        pass
    elapsed = time.monotonic() - start
    device.close()
    assert elapsed >= 0.08
