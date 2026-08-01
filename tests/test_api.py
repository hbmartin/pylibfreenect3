from __future__ import annotations

import gc
import importlib.util
import inspect
import os
import warnings
import weakref
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import pylibfreenect3 as f3
from pylibfreenect3 import _native


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_native_resources_fail_fast_when_inherited_after_fork() -> None:
    listener = f3.lowlevel.FrameListener(f3.FrameType.COLOR)
    context = f3.lowlevel.Context()
    pipeline = f3.lowlevel.CpuPacketPipeline()
    replay = f3.lowlevel.ReplayContext()
    device = replay.open_device(["missing_color_1_1.jpg"], pipeline=pipeline)
    native_frames = _native._testing_frame_set()
    frame = native_frames.get(int(f3.FrameType.DEPTH))
    read_fd, write_fd = os.pipe()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        checks = {
            "context": context.enumerate_devices,
            "device": lambda: device.closed,
            "frame": lambda native_frame=frame: native_frame.width,
            "frame set": lambda: native_frames.contains(int(f3.FrameType.DEPTH)),
            "listener": listener.has_new_frame,
            "pipeline": lambda: pipeline.consumed,
        }
        failures: list[str] = []
        for label, operation in checks.items():
            try:
                operation()
            except f3.DeviceStateError as error:
                if "cannot be used after fork" not in str(error):
                    failures.append(f"{label}: unexpected error: {error}")
            except BaseException as error:
                failures.append(f"{label}: {type(error).__name__}: {error}")
            else:
                failures.append(f"{label}: inherited resource was accepted")
        os.write(write_fd, "\n".join(failures).encode())
        os.close(write_fd)
        os._exit(1 if failures else 0)

    os.close(write_fd)
    try:
        payload = os.read(read_fd, 16_384).decode()
        _, status = os.waitpid(child_pid, 0)
    finally:
        os.close(read_fd)
        del frame
        gc.collect()
        native_frames.release()
        device.close()

    assert os.waitstatus_to_exitcode(status) == 0, payload


def test_runtime_identity_and_pipeline_queries() -> None:
    assert f3.core_version().startswith("0.3.")
    assert f3.core_api_version() == 3
    assert f3.core_build_revision()
    assert {"cpu", "dump"} <= f3.compiled_pipelines()
    assert f3.available_pipelines() <= f3.compiled_pipelines()
    assert (
        f3.lowlevel.Context.VENDOR_ID,
        f3.lowlevel.Context.PRODUCT_ID,
    ) == (0x045E, 0x02D8)


@pytest.mark.parametrize(
    ("symbol", "replacement"),
    [
        ("Freenect2", "lowlevel.Context"),
        ("Freenect2Replay", "lowlevel.ReplayContext"),
        ("SyncFrameListener", "lowlevel.FrameListener"),
        ("Device", "lowlevel.Device"),
        ("STREAM_NAMES", "Stream"),
        ("core_revision", "core_build_revision"),
    ],
)
def test_removed_top_level_symbols_have_actionable_errors(
    symbol: str, replacement: str
) -> None:
    with pytest.raises(AttributeError, match=replacement):
        getattr(f3, symbol)
    assert symbol not in dir(f3)


def test_dir_lists_real_module_attributes() -> None:
    listed = dir(f3)
    assert set(f3.__all__) <= set(listed)
    assert "__version__" in listed
    assert "NoReturn" not in listed
    assert "version" not in listed
    assert "PackageNotFoundError" not in listed
    assert "_MOVED_SYMBOLS" not in listed


def test_backend_classes_are_importable_and_report_availability() -> None:
    classes = {
        "cpu": f3.lowlevel.CpuPacketPipeline,
        "metal": f3.lowlevel.MetalPacketPipeline,
        "opengl": f3.lowlevel.OpenGLPacketPipeline,
        "opencl": f3.lowlevel.OpenCLPacketPipeline,
        "opencl_kde": f3.lowlevel.OpenCLKdePacketPipeline,
        "cuda": f3.lowlevel.CudaPacketPipeline,
        "cuda_kde": f3.lowlevel.CudaKdePacketPipeline,
        "dump": f3.lowlevel.DumpPacketPipeline,
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
        f3.lowlevel.DumpPacketPipeline().depth_p0_tables()


def test_pipeline_is_consumed_exactly_once() -> None:
    pipeline = f3.lowlevel.CpuPacketPipeline()
    replay = f3.lowlevel.ReplayContext()
    device = replay.open_device(["missing_color_1_1.jpg"], pipeline=pipeline)
    assert pipeline.consumed
    assert device.pipeline_name == "cpu"
    with pytest.raises(f3.DeviceStateError, match="single-use"):
        replay.open_device(["missing_color_2_2.jpg"], pipeline=pipeline)
    device.close()


def test_pipeline_enum_and_canonical_strings_are_normalized() -> None:
    replay = f3.lowlevel.ReplayContext()
    enum_device = replay.open_device(
        ["missing_color_1_1.jpg"], pipeline=f3.Pipeline.CPU
    )
    assert enum_device.pipeline_name is f3.Pipeline.CPU
    enum_device.close()

    string_device = replay.open_device(["missing_color_2_2.jpg"], pipeline="CPU")
    assert string_device.pipeline_name is f3.Pipeline.CPU
    string_device.close()

    with pytest.raises(ValueError, match="unknown pipeline"):
        replay.open_device(["missing_color_3_3.jpg"], pipeline="not-a-pipeline")
    with pytest.raises(ValueError, match="streams must contain"):
        f3.Camera.open(streams=("not-a-stream",))


def test_consumed_dump_pipeline_rejects_access_after_device_close() -> None:
    pipeline = f3.lowlevel.DumpPacketPipeline()
    device = f3.lowlevel.ReplayContext().open_device(
        ["missing_color_1_1.jpg"], pipeline=pipeline
    )
    device.close()
    with pytest.raises(f3.DeviceStateError, match="after the device is closed"):
        pipeline.depth_p0_tables()


def test_replay_lifecycle_is_repeatable_and_access_after_close_fails() -> None:
    replay = f3.lowlevel.ReplayContext()
    device = replay.open_device(["missing_color_1_1.jpg"], pipeline="cpu")
    color_params = f3.ColorCameraParams(fx=100.0, fy=101.0, cx=2.0, cy=3.0)
    ir_params = f3.IrCameraParams(fx=200.0, fy=201.0, cx=4.0, cy=5.0)
    device.color_camera_params = color_params
    device.ir_camera_params = ir_params
    assert device.color_camera_params == color_params
    assert device.ir_camera_params == ir_params
    device.configuration = f3.DeviceConfig(min_depth=0.7, max_depth=5.5)
    assert device.configuration == f3.DeviceConfig(min_depth=0.7, max_depth=5.5)
    device.set_color_setting(
        f3.ColorSettingCommand.SET_INTEGRATION_TIME, np.float32(1.25)
    )
    device.set_color_setting(f3.ColorSettingCommand.SET_INTEGRATION_TIME, np.int64(2))
    for invalid_integer in (-1, 2**32):
        with pytest.raises(ValueError, match="fit in uint32"):
            device.set_color_setting(
                f3.ColorSettingCommand.SET_INTEGRATION_TIME, invalid_integer
            )
    for invalid_type in (True, np.bool_(True), "1"):
        with pytest.raises(TypeError, match="integer or floating-point"):
            device.set_color_setting(
                f3.ColorSettingCommand.SET_INTEGRATION_TIME, invalid_type
            )
    listener = f3.lowlevel.FrameListener(f3.FrameType.COLOR)
    device.set_color_listener(listener)
    device.start(rgb=True, depth=False)
    assert device.get_color_setting(f3.ColorSettingCommand.GET_INTEGRATION_TIME) == 0
    with pytest.raises(f3.DeviceStateError, match="cannot change while streaming"):
        device.configuration = f3.DeviceConfig(min_depth=0.9, max_depth=6.0)
    assert device.configuration == f3.DeviceConfig(min_depth=0.7, max_depth=5.5)
    with pytest.raises(f3.FrameTimeoutError):
        listener.wait(timeout=0.001)
    device.stop()
    device.stop()
    device.start(rgb=True, depth=False)
    device.stop()
    device.close()
    device.close()
    assert device.closed
    with pytest.raises(f3.DeviceStateError):
        _ = device.serial_number
    with pytest.raises(f3.DeviceStateError):
        device.__enter__()


def test_native_device_rejects_non_native_listener_addresses() -> None:
    class FakeListener:
        _listener_address = 1

    device = f3.lowlevel.ReplayContext().open_device(
        ["missing_color_1_1.jpg"], pipeline="cpu"
    )
    try:
        with pytest.raises(TypeError, match="native frame listener"):
            device._native.set_color_listener(FakeListener())
        with pytest.raises(TypeError, match="native frame listener"):
            device._native.set_depth_listener(FakeListener())

        sync = f3.lowlevel.FrameListener(f3.FrameType.COLOR | f3.FrameType.DEPTH)
        aligned = f3.lowlevel.AlignedFrameListener(
            f3.FrameType.COLOR | f3.FrameType.DEPTH,
            f3.AlignmentConfig(max_delta=0.025),
        )
        for listener in (sync, aligned):
            device.set_color_listener(listener)
            device.set_depth_listener(listener)
    finally:
        device.close()


def test_depth_replay_requires_calibration() -> None:
    with pytest.raises(f3.ReplayError, match="calibration"):
        f3.lowlevel.ReplayContext().open_device(
            ["depth_packet_1_1.depth"], pipeline="cpu"
        )


def test_logger_utilities_cover_every_level() -> None:
    for level in f3.LoggerLevel:
        assert f3.logger_level_name(level)
        f3.set_global_log_level(level)
        assert f3.global_logger_level() is level
    f3.set_global_log_level(None)
    assert f3.global_logger_level() is None
    f3.set_global_log_level(f3.default_logger_level())


@pytest.mark.skipif(
    "CI" not in os.environ,
    reason="user environments may legitimately have pylibfreenect2 installed",
)
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


def test_frame_numpy_protocol_honors_dtype_and_copy() -> None:
    source = np.arange(4, dtype=np.float32).reshape(2, 2)
    frame = f3.Frame.from_array(source, frame_type=f3.FrameType.DEPTH)
    assert frame.frame_type is f3.FrameType.DEPTH

    view = np.asarray(frame)
    assert np.shares_memory(view, source)
    converted = np.asarray(frame, dtype=np.float64)
    assert converted.dtype == np.float64
    assert not np.shares_memory(converted, source)
    copied = np.array(frame, copy=True)
    assert not np.shares_memory(copied, source)
    with pytest.raises(ValueError, match="dtype conversion"):
        frame.__array__(np.dtype(np.float64), copy=False)


def test_mismatched_array_layouts_are_rejected() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        f3.Frame.allocate(0, 2, 4)
    with pytest.raises(ValueError, match="FLOAT frames require 4"):
        f3.Frame.allocate(2, 2, 1, frame_format=f3.FrameFormat.FLOAT)
    with pytest.raises(ValueError, match="exactly one FrameType"):
        f3.Frame.allocate(
            2,
            2,
            4,
            frame_type=f3.FrameType.COLOR | f3.FrameType.DEPTH,
        )
    with pytest.raises(ValueError, match="array layout does not match FLOAT"):
        f3.Frame.from_array(
            np.zeros((2, 2), np.uint8), frame_format=f3.FrameFormat.FLOAT
        )
    with pytest.raises(ValueError, match="BGRX arrays must have shape"):
        f3.Frame.from_array(
            np.zeros((2, 2, 3), np.uint8), frame_format=f3.FrameFormat.BGRX
        )
    with pytest.raises(ValueError, match="raw frames use width=height=1"):
        f3.Frame.allocate(2, 2, 4, frame_format=f3.FrameFormat.RAW)
    with pytest.raises(ValueError, match="frame arrays must not be empty"):
        f3.Frame.from_array(np.empty(0, np.uint8), frame_format=f3.FrameFormat.RAW)
    read_only = np.zeros((2, 2), np.float32)
    read_only.flags.writeable = False
    with pytest.raises(ValueError, match="frame arrays must be writable"):
        f3.Frame.from_array(read_only, frame_format=f3.FrameFormat.FLOAT)


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


@pytest.mark.parametrize("copy", [False, True])
def test_native_frame_set_release_stress(copy: bool) -> None:
    for _ in range(2_000):
        native = _native._testing_frame_set()
        frame = native.get(int(f3.FrameType.DEPTH))
        array = frame.to_numpy(copy=copy)
        native.release()
        assert array.shape == (1, 2)
        del frame, array, native
    gc.collect()


def test_frame_set_mapping_and_unknown_keys_are_missing() -> None:
    from collections.abc import Mapping

    assert issubclass(f3.FrameSet, Mapping)
    assert tuple(f3.Stream) == (f3.Stream.COLOR, f3.Stream.IR, f3.Stream.DEPTH)
    frames = f3.FrameSet(native=_native._testing_frame_set())
    assert f3.FrameType.COLOR | f3.FrameType.DEPTH not in frames
    assert 8 not in frames
    with pytest.raises(KeyError):
        _ = frames[f3.FrameType.COLOR | f3.FrameType.DEPTH]
    with pytest.raises(KeyError):
        _ = frames[8]
    with pytest.raises(KeyError):
        _ = frames["color"]
    assert frames.get("color") is None
    frames.release()


def test_detached_copy_immediately_releases_native_capture() -> None:
    native = _native._testing_frame_set()
    detached = f3.FrameSet(native=native).detached_copy()
    assert native.release_complete
    assert native.entry_count == 0
    assert detached.color.to_numpy().shape == (1, 2, 4)
    assert detached.depth.to_numpy().shape == (1, 2)


def test_detached_copy_releases_source_when_copying_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = f3.Frame.from_array(
        np.zeros((2, 2), np.float32), frame_type=f3.FrameType.DEPTH
    )
    frames = f3.FrameSet(copied={f3.FrameType.DEPTH: source})

    def fail_copy(*_: object, **__: object) -> f3.Frame:
        raise ValueError("simulated copy failure")

    monkeypatch.setattr(f3.Frame, "from_array", fail_copy)
    with pytest.raises(ValueError, match="simulated copy failure"):
        frames.detached_copy()
    assert frames.released


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
    with pytest.raises(ValueError, match="0 <= min_depth < max_depth"):
        f3.DeviceConfig(min_depth=2.0, max_depth=1.0)
    with pytest.raises(ValueError, match="IR camera parameters must be finite"):
        f3.IrCameraParams(fx=float("nan"))
    with pytest.raises(TypeError, match="bool values"):
        f3.DeviceConfig(enable_bilateral_filter=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="IR focal lengths must be non-negative"):
        f3.IrCameraParams(fx=-1.0)
    with pytest.raises(ValueError, match="color focal lengths must be non-negative"):
        f3.ColorCameraParams(fy=-1.0)
    with pytest.raises(ValueError, match="reserved must be zero"):
        f3.LedSettings(0, reserved=1)
    assert [value.value for value in f3.FrameType] == [1, 2, 4]
    assert [value.value for value in f3.FrameFormat] == [0, 1, 2, 4, 5, 6]
    assert [value.value for value in f3.LoggerLevel] == [0, 1, 2, 3, 4]
    assert f3.Pipeline("opencl_kde") is f3.Pipeline.OPENCL_KDE
    assert f3.Stream("depth") is f3.Stream.DEPTH
    assert [value.value for value in f3.ColorSettingCommand] == [
        0,
        1,
        2,
        *range(10, 84),
    ]


def test_value_objects_are_immutable_and_capture_defaults_to_two_seconds() -> None:
    config = f3.DeviceConfig()
    with pytest.raises(FrozenInstanceError):
        config.max_depth = 9.0  # type: ignore[misc]

    capture_timeout = inspect.signature(f3.Camera.capture).parameters["timeout"]
    frames_timeout = inspect.signature(f3.Camera.frames).parameters["timeout"]
    lowlevel_timeout = inspect.signature(f3.lowlevel.FrameListener.wait).parameters[
        "timeout"
    ]
    assert capture_timeout.default == 2.0
    assert frames_timeout.default == 2.0
    assert lowlevel_timeout.default is None


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
        include_depth_to_color_map=True,
    )
    assert result.undistorted.to_numpy().shape == (424, 512)
    assert result.registered.to_numpy().shape == (424, 512, 4)
    assert result.big_depth is not None
    assert result.big_depth.to_numpy().shape == (1082, 1920)
    assert result.depth_to_color_map is not None
    assert result.depth_to_color_map.shape == (424, 512)
    assert np.isfinite(registration.point_xyz(result.undistorted, 212, 256)).all()
    registered_data = np.zeros((424, 512, 4), np.uint8)
    registered_data[212, 256, :3] = (1, 2, 3)
    registered_bgrx = f3.Frame.from_array(
        registered_data, frame_format=f3.FrameFormat.BGRX
    )
    registered_rgbx = f3.Frame.from_array(
        registered_data.copy(), frame_format=f3.FrameFormat.RGBX
    )
    assert registration.point_xyz_rgb(result.undistorted, registered_bgrx, 212, 256)[
        3:
    ] == (3, 2, 1)
    assert registration.point_xyz_rgb(result.undistorted, registered_rgbx, 212, 256)[
        3:
    ] == (1, 2, 3)
    assert registration.undistort_depth(depth).to_numpy().shape == (424, 512)


def test_native_registration_rejects_none_frames() -> None:
    native = _native.NativeRegistrationHandle(
        f3.IrCameraParams(), f3.ColorCameraParams()
    )
    with pytest.raises(TypeError):
        native.apply(None, None, None, None)
    with pytest.raises(TypeError):
        native.undistort_depth(None, None)
    with pytest.raises(TypeError):
        native.point_xyz(None, 0, 0)
    with pytest.raises(TypeError):
        native.point_xyz_rgb(None, None, 0, 0)


def test_arrival_timestamp_color_conversion_and_detached_copy() -> None:
    bgrx = np.array([[[30, 20, 10, 0], [60, 50, 40, 0]]], dtype=np.uint8)
    frame = f3.Frame.from_array(
        bgrx,
        frame_format=f3.FrameFormat.BGRX,
        arrival_timestamp_us=123_456,
    )
    destination = np.empty((1, 2, 3), dtype=np.uint8)
    assert frame.to_color(out=destination) is destination
    assert destination.tolist() == [[[30, 20, 10], [60, 50, 40]]]
    assert frame.to_color(f3.ColorOrder.RGB).tolist() == [[[10, 20, 30], [40, 50, 60]]]
    assert frame.arrival_timestamp_us == 123_456

    with pytest.raises(TypeError, match="dtype uint8"):
        frame.to_color(out=np.empty((1, 2, 3), dtype=np.float32))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        frame.to_color(out=np.empty((1, 1, 3), dtype=np.uint8))
    noncontiguous = np.empty((1, 2, 6), dtype=np.uint8)[:, :, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        frame.to_color(out=noncontiguous)
    readonly = np.empty((1, 2, 3), dtype=np.uint8)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        frame.to_color(out=readonly)
    with pytest.raises(TypeError, match="NumPy array"):
        frame._native.to_color([[[np.uint8(0), np.uint8(0), np.uint8(0)]]], 0)

    frames = f3.FrameSet(_native._testing_frame_set())
    copied = frames.detached_copy()
    assert copied.color.arrival_timestamp_us == 1000
    assert copied.depth.arrival_timestamp_us == 1100


def test_timestamp_aligned_listener_wraparound_and_stats() -> None:
    listener = f3.lowlevel.AlignedFrameListener(
        f3.FrameType.COLOR | f3.FrameType.DEPTH,
        f3.AlignmentConfig(max_delta=0.001, queue_capacity=8),
    )
    color = f3.Frame.allocate(
        1,
        1,
        4,
        frame_type=f3.FrameType.COLOR,
        frame_format=f3.FrameFormat.BGRX,
        timestamp=2,
        arrival_timestamp_us=10,
    )
    depth = f3.Frame.allocate(
        1,
        1,
        4,
        frame_type=f3.FrameType.DEPTH,
        frame_format=f3.FrameFormat.FLOAT,
        timestamp=2**32 - 2,
        arrival_timestamp_us=11,
    )
    assert listener._native._testing_push(int(f3.FrameType.COLOR), color._native)
    assert listener._native._testing_push(int(f3.FrameType.DEPTH), depth._native)
    with listener.wait(0) as frames:
        assert frames.alignment_delta_ticks == 4
        assert frames.alignment_delta_seconds == 0.0005
    stats = listener.alignment_stats
    assert stats == f3.AlignmentStats(1, 0, 4, 4)
    assert stats.last_delta_seconds == 0.0005
    assert stats.maximum_delta_seconds == 0.0005
    assert stats.last_delta_milliseconds == 0.5
    assert stats.maximum_delta_milliseconds == 0.5


def test_testing_push_copies_array_backed_frames_into_native_storage() -> None:
    listener = _native.NativeAlignedFrameListener(int(f3.FrameType.COLOR), 8, 8)
    source = np.array([[[30, 20, 10, 0]]], dtype=np.uint8)
    source_ref = weakref.ref(source)
    frame = _native.NativeFrame.from_array(
        source,
        int(f3.FrameType.COLOR),
        int(f3.FrameFormat.BGRX),
    )
    assert listener._testing_push(int(f3.FrameType.COLOR), frame)

    source[:] = 255
    frames = listener.wait(0)
    queued = frames.get(int(f3.FrameType.COLOR))
    assert queued.to_numpy(copy=True).tolist() == [[[30, 20, 10, 0]]]

    del source, frame
    gc.collect()
    assert source_ref() is None
    assert queued.to_numpy(copy=True).tolist() == [[[30, 20, 10, 0]]]

    frames.release()
    del queued
    gc.collect()
    assert frames.release_complete


def test_registration_workspace_maps_batch_and_lifting() -> None:
    ir = f3.IrCameraParams(fx=365.0, fy=365.0, cx=256.0, cy=212.0)
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
    workspace = registration.workspace(
        include_big_depth=True,
        include_depth_to_color_map=True,
        include_color_to_depth_map=True,
    )
    assert issubclass(f3.WorkspaceStateError, f3.FreenectError)
    with pytest.raises(f3.WorkspaceStateError, match="has not been applied"):
        workspace.lift_normalized([[0.5, 0.5]])

    color = f3.Frame.from_array(
        np.zeros((1080, 1920, 4), dtype=np.uint8),
        frame_format=f3.FrameFormat.RGBX,
    )
    depth = f3.Frame.from_array(
        np.full((424, 512), 1000.0, dtype=np.float32),
        frame_format=f3.FrameFormat.FLOAT,
    )
    first = workspace.apply(color, depth)
    identities = tuple(
        id(value)
        for value in (
            first,
            first.undistorted,
            first.registered,
            first.big_depth,
            first.depth_to_color_map,
            first.color_to_depth_map,
        )
    )
    second = workspace.apply(color, depth, enable_filter=False)
    assert identities == tuple(
        id(value)
        for value in (
            second,
            second.undistorted,
            second.registered,
            second.big_depth,
            second.depth_to_color_map,
            second.color_to_depth_map,
        )
    )
    assert second.registered.format is f3.FrameFormat.RGBX
    assert second.color_to_depth_map is not None
    assert second.color_to_depth_map.shape == (1080, 1920)

    center_index = 212 * 512 + 256
    workspace.undistorted.to_numpy().fill(0)
    workspace.undistorted.to_numpy()[212, 256] = 1000.0
    workspace.color_to_depth_map.fill(-1)
    workspace.color_to_depth_map[540, 960] = center_index
    lifted = workspace.lift_normalized(np.array([[0.5, 0.5], [-0.1, 0.5]]))
    assert lifted.xyz.shape == (2, 3)
    assert lifted.valid.tolist() == [True, False]
    assert lifted.depth_pixels.tolist() == [[212, 256], [-1, -1]]
    assert np.isfinite(lifted.xyz[0]).all()
    assert lifted.xyz[0, 2] == pytest.approx(1.0)

    xyz = registration.points_xyz(
        workspace.undistorted, np.array([[212, 256], [0, 0]], dtype=np.int64)
    )
    assert xyz.shape == (2, 3)
    assert np.isfinite(xyz[0]).all()
    with pytest.raises(IndexError):
        registration.points_xyz(workspace.undistorted, [[424, 0]])

    with pytest.warns(DeprecationWarning, match="include_color_depth_map"):
        deprecated = registration.apply(color, depth, include_color_depth_map=True)
    assert deprecated.depth_to_color_map is not None
    with pytest.warns(DeprecationWarning, match="color_depth_map"):
        assert deprecated.color_depth_map is deprecated.depth_to_color_map
    with pytest.raises(ValueError, match="conflicting"):
        registration.apply(
            color,
            depth,
            include_depth_to_color_map=True,
            include_color_depth_map=False,
        )

    plain = registration.workspace()
    plain.apply(color, depth)
    with pytest.raises(f3.WorkspaceStateError, match="include_color_to_depth_map"):
        plain.lift_normalized([[0.5, 0.5]])


def test_pipeline_config_device_diagnostics_and_calibration_matrices() -> None:
    config = f3.PacketPipelineConfig(
        rgb_decoder=f3.RgbDecoder.TURBOJPEG, allow_fallback=False
    )
    pipeline = f3.lowlevel.CpuPacketPipeline(config=config)
    with pytest.raises(ValueError, match="preconstructed"):
        f3.lowlevel.ReplayContext().open_device(
            ["missing_color_1_1.jpg"], pipeline=pipeline, pipeline_config=config
        )
    with pytest.raises(ValueError, match="without a pipeline"):
        f3.lowlevel.ReplayContext().open_device(
            ["missing_color_1_1.jpg"], pipeline=None, pipeline_config=config
        )
    with pytest.raises(ValueError, match="without a pipeline"):
        f3.lowlevel.Context().open_device(pipeline=None, pipeline_config=config)
    assert set(_native.RGB_DECODER_VALUES) == {member.value for member in f3.RgbDecoder}
    device = f3.lowlevel.ReplayContext().open_device(
        ["missing_color_1_1.jpg"], pipeline="cpu", pipeline_config=config
    )
    assert device.state is f3.DeviceState.OPEN
    assert device.last_error == ""
    device.close()
    assert device.state is f3.DeviceState.CLOSED

    ir = f3.IrCameraParams(
        fx=2.0, fy=3.0, cx=4.0, cy=5.0, k1=1.0, k2=2.0, k3=3.0, p1=4.0, p2=5.0
    )
    assert ir.camera_matrix().dtype == np.float64
    assert ir.camera_matrix().tolist() == [
        [2.0, 0.0, 4.0],
        [0.0, 3.0, 5.0],
        [0.0, 0.0, 1.0],
    ]
    assert ir.distortion_coefficients().tolist() == [1.0, 2.0, 4.0, 5.0, 3.0]
    assert not f3.lowlevel.Context().wait_for_device("not-present", 0)
