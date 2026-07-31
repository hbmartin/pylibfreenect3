from __future__ import annotations

import gc
import importlib.util
import weakref

import numpy as np
import pytest

import pylibfreenect3 as f3
from pylibfreenect3 import _native


def test_runtime_identity_and_pipeline_queries() -> None:
    assert f3.core_version().startswith("0.3.")
    assert f3.core_api_version() == 3
    assert f3.core_build_revision()
    assert f3.core_revision() == f3.core_build_revision()
    assert {"cpu", "dump"} <= f3.compiled_pipelines()
    assert f3.available_pipelines() <= f3.compiled_pipelines()
    assert (f3.Freenect2.VENDOR_ID, f3.Freenect2.PRODUCT_ID) == (0x045E, 0x02D8)


def test_backend_classes_are_importable_and_report_availability() -> None:
    classes = {
        "cpu": f3.CpuPacketPipeline,
        "metal": f3.MetalPacketPipeline,
        "opengl": f3.OpenGLPacketPipeline,
        "opencl": f3.OpenCLPacketPipeline,
        "opencl_kde": f3.OpenCLKdePacketPipeline,
        "cuda": f3.CudaPacketPipeline,
        "cuda_kde": f3.CudaKdePacketPipeline,
        "dump": f3.DumpPacketPipeline,
    }
    compiled = f3.compiled_pipelines()
    available = f3.available_pipelines()
    for name, pipeline_type in classes.items():
        if name not in compiled or name not in available:
            with pytest.raises(f3.BackendUnavailableError):
                pipeline_type()
        else:
            pipeline = pipeline_type()
            assert pipeline.name == name
            assert not pipeline.consumed


def test_dump_tables_reject_access_before_device_calibration() -> None:
    with pytest.raises(f3.DeviceStateError, match="not ready"):
        f3.DumpPacketPipeline().depth_p0_tables()


def test_pipeline_is_consumed_exactly_once() -> None:
    pipeline = f3.CpuPacketPipeline()
    replay = f3.Freenect2Replay()
    device = replay.open_device(["missing_color_1_1.jpg"], pipeline=pipeline)
    assert pipeline.consumed
    assert device.pipeline_name == "cpu"
    with pytest.raises(f3.DeviceStateError, match="single-use"):
        replay.open_device(["missing_color_2_2.jpg"], pipeline=pipeline)
    device.close()


def test_consumed_dump_pipeline_rejects_access_after_device_close() -> None:
    pipeline = f3.DumpPacketPipeline()
    device = f3.Freenect2Replay().open_device(
        ["missing_color_1_1.jpg"], pipeline=pipeline
    )
    device.close()
    with pytest.raises(f3.DeviceStateError, match="after the device is closed"):
        pipeline.depth_p0_tables()


def test_replay_lifecycle_is_repeatable_and_access_after_close_fails() -> None:
    replay = f3.Freenect2Replay()
    device = replay.open_device(["missing_color_1_1.jpg"], pipeline="cpu")
    color_params = f3.ColorCameraParams(fx=100.0, fy=101.0, cx=2.0, cy=3.0)
    ir_params = f3.IrCameraParams(fx=200.0, fy=201.0, cx=4.0, cy=5.0)
    device.color_camera_params = color_params
    device.ir_camera_params = ir_params
    assert device.color_camera_params == color_params
    assert device.ir_camera_params == ir_params
    device.configuration = f3.DeviceConfig(min_depth=0.7, max_depth=5.5)
    assert device.configuration == f3.DeviceConfig(min_depth=0.7, max_depth=5.5)
    listener = f3.SyncFrameListener(f3.FrameType.COLOR)
    device.set_color_listener(listener)
    device.start(rgb=True, depth=False)
    with pytest.raises(f3.FrameTimeoutError):
        listener.wait(timeout=0.001)
    device.stop()
    device.stop()
    device.start(rgb=True, depth=False)
    device.stop()
    device.close()
    device.close()
    assert device.is_closed
    with pytest.raises(f3.DeviceStateError):
        _ = device.serial_number
    with pytest.raises(f3.DeviceStateError):
        device.__enter__()


def test_depth_replay_requires_calibration() -> None:
    with pytest.raises(f3.ReplayError, match="calibration"):
        f3.Freenect2Replay().open_device(["depth_packet_1_1.depth"], pipeline="cpu")


def test_logger_utilities_cover_every_level() -> None:
    for level in f3.LoggerLevel:
        assert f3.logger_level_name(level)
        f3.set_global_log_level(level)
        assert f3.global_logger_level() is level
    f3.set_global_log_level(None)
    assert f3.global_logger_level() is None
    f3.set_global_log_level(f3.default_logger_level())


def test_old_import_namespace_is_absent() -> None:
    assert importlib.util.find_spec("pylibfreenect2") is None


@pytest.mark.parametrize(
    ("shape", "dtype", "frame_format"),
    [
        ((19,), np.uint8, f3.FrameFormat.RAW),
        ((3, 4), np.float32, f3.FrameFormat.FLOAT),
        ((3, 4), np.uint8, f3.FrameFormat.GRAY),
        ((3, 4, 4), np.uint8, f3.FrameFormat.BGRX),
        ((3, 4, 4), np.uint8, f3.FrameFormat.RGBX),
    ],
)
def test_frame_formats_and_numpy_source_lifetime(
    shape: tuple[int, ...], dtype: np.dtype, frame_format: f3.FrameFormat
) -> None:
    array = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    source_ref = weakref.ref(array)
    frame = f3.Frame.from_array(
        array,
        frame_type=f3.FrameType.COLOR,
        frame_format=frame_format,
        timestamp=123,
        sequence=9,
        exposure=1.25,
        gain=1.5,
        gamma=2.0,
        status=7,
    )
    view = frame.to_numpy()
    assert np.shares_memory(view, array)
    assert (frame.timestamp, frame.sequence, frame.status) == (123, 9, 7)
    del frame, array
    gc.collect()
    assert source_ref() is not None
    assert view.shape
    copied = view.base.to_numpy(copy=True)
    assert not np.shares_memory(copied, view)
    del view
    gc.collect()
    assert source_ref() is None


def test_mismatched_array_layouts_are_rejected() -> None:
    with pytest.raises(ValueError):
        f3.Frame.from_array(
            np.zeros((2, 2), np.uint8), frame_format=f3.FrameFormat.FLOAT
        )
    with pytest.raises(ValueError):
        f3.Frame.from_array(
            np.zeros((2, 2, 3), np.uint8), frame_format=f3.FrameFormat.BGRX
        )
    with pytest.raises(ValueError):
        f3.Frame.allocate(2, 2, 4, frame_format=f3.FrameFormat.RAW)
    with pytest.raises(ValueError):
        f3.Frame.from_array(np.empty(0, np.uint8), frame_format=f3.FrameFormat.RAW)


def test_native_frame_set_defers_release_until_last_array_dies() -> None:
    native = _native._testing_frame_set()
    assert native.entry_count == 2
    with pytest.raises(KeyError):
        native.get(int(f3.FrameType.IR))
    assert native.entry_count == 2

    frame = native.get(int(f3.FrameType.DEPTH))
    array = frame.to_numpy()
    native.release()
    native.release()
    assert native.is_released
    assert not native.release_complete
    with pytest.raises(f3.DeviceStateError):
        native.get(int(f3.FrameType.DEPTH))

    del frame
    gc.collect()
    assert not native.release_complete
    assert array.shape == (1, 2)
    del array
    gc.collect()
    assert native.release_complete
    assert native.entry_count == 0


def test_frame_set_composite_and_unknown_keys_are_missing() -> None:
    frames = f3.FrameSet(native=_native._testing_frame_set())
    assert f3.FrameType.COLOR | f3.FrameType.DEPTH not in frames
    assert 8 not in frames
    with pytest.raises(KeyError):
        _ = frames[f3.FrameType.COLOR | f3.FrameType.DEPTH]
    with pytest.raises(KeyError):
        _ = frames[8]
    frames.release()


def test_detached_copy_immediately_releases_native_capture() -> None:
    native = _native._testing_frame_set()
    detached = f3.FrameSet(native=native).detached_copy()
    assert native.release_complete
    assert native.entry_count == 0
    assert detached.color.to_numpy().shape == (1, 2, 4)
    assert detached.depth.to_numpy().shape == (1, 2)


def test_copied_frame_set_preserves_metadata_and_release_contract() -> None:
    source = f3.Frame.from_array(
        np.arange(4, dtype=np.float32).reshape(2, 2),
        frame_type=f3.FrameType.DEPTH,
        timestamp=88,
        sequence=4,
        frame_format=f3.FrameFormat.FLOAT,
    )
    frames = f3.FrameSet(copied={f3.FrameType.DEPTH: source})
    detached = frames.detached_copy()
    assert frames.released
    assert detached.depth.timestamp == 88
    assert detached.depth.sequence == 4
    detached.release()
    detached.release()
    with pytest.raises(f3.DeviceStateError):
        _ = detached.depth


def test_typed_values_validate_inputs() -> None:
    assert f3.DeviceConfig().max_depth == 4.5
    with pytest.raises(ValueError):
        f3.DeviceConfig(min_depth=2.0, max_depth=1.0)
    with pytest.raises(ValueError):
        f3.IrCameraParams(fx=float("nan"))
    with pytest.raises(ValueError):
        f3.ColorCameraParams(fy=-1.0)
    with pytest.raises(ValueError):
        f3.LedSettings(0, reserved=1)
    assert [value.value for value in f3.FrameType] == [1, 2, 4]
    assert [value.value for value in f3.FrameFormat] == [0, 1, 2, 4, 5, 6]
    assert [value.value for value in f3.LoggerLevel] == [0, 1, 2, 3, 4]
    assert [value.value for value in f3.ColorSettingCommand] == [
        0,
        1,
        2,
        *range(10, 84),
    ]


def test_registration_overloads_with_synthetic_frames() -> None:
    ir = f3.IrCameraParams(
        fx=365.0,
        fy=365.0,
        cx=256.0,
        cy=212.0,
        k1=0.09,
        k2=-0.27,
        k3=0.10,
    )
    color_params = f3.ColorCameraParams(
        fx=1081.0,
        fy=1081.0,
        cx=959.5,
        cy=539.5,
        shift_d=863.0,
        shift_m=52.0,
        mx_x1y0=1.0,
        my_x0y1=1.0,
    )
    registration = f3.Registration(ir, color_params)
    assert np.isfinite(registration.apply_point(256, 212, 1500.0)).all()
    with pytest.raises(IndexError):
        registration.apply_point(512, 0, 1500.0)

    color = f3.Frame.from_array(
        np.zeros((1080, 1920, 4), np.uint8),
        frame_type=f3.FrameType.COLOR,
        frame_format=f3.FrameFormat.BGRX,
    )
    depth = f3.Frame.from_array(
        np.full((424, 512), 1500.0, np.float32),
        frame_type=f3.FrameType.DEPTH,
        frame_format=f3.FrameFormat.FLOAT,
    )
    result = registration.apply(
        color,
        depth,
        include_big_depth=True,
        include_color_depth_map=True,
    )
    assert result.undistorted.to_numpy().shape == (424, 512)
    assert result.registered.to_numpy().shape == (424, 512, 4)
    assert result.big_depth is not None
    assert result.big_depth.to_numpy().shape == (1082, 1920)
    assert result.color_depth_map is not None
    assert result.color_depth_map.shape == (424, 512)
    assert np.isfinite(registration.point_xyz(result.undistorted, 212, 256)).all()
    assert (
        len(registration.point_xyz_rgb(result.undistorted, result.registered, 212, 256))
        == 6
    )
    assert registration.undistort_depth(depth).to_numpy().shape == (424, 512)
