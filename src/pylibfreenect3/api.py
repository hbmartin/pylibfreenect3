from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from math import isfinite
from os import PathLike, fspath
from typing import ClassVar, cast, overload

import numpy as np
import numpy.typing as npt

from . import _native
from .errors import DeviceOpenError, DeviceStateError, ReplayError
from .types import (
    ColorCameraParams,
    ColorSettingCommand,
    DeviceConfig,
    FrameFormat,
    FrameType,
    IrCameraParams,
    LedSettings,
    LoggerLevel,
    Pipeline,
    ReplayCalibration,
    Stream,
    _dataclass_from_mapping,
)

__all__ = [
    "Camera",
    "Context",
    "CpuPacketPipeline",
    "CudaKdePacketPipeline",
    "CudaPacketPipeline",
    "Device",
    "DumpPacketPipeline",
    "Frame",
    "FrameListener",
    "FrameSet",
    "MetalPacketPipeline",
    "OpenCLKdePacketPipeline",
    "OpenCLPacketPipeline",
    "OpenGLPacketPipeline",
    "PacketPipeline",
    "Registration",
    "RegistrationResult",
    "ReplayContext",
    "available_pipelines",
    "compiled_pipelines",
    "core_api_version",
    "core_build_revision",
    "core_version",
    "default_logger_level",
    "global_logger_level",
    "logger_level_name",
    "set_global_log_level",
]

_STREAM_TYPES: Mapping[Stream, FrameType] = {
    Stream.COLOR: FrameType.COLOR,
    Stream.IR: FrameType.IR,
    Stream.DEPTH: FrameType.DEPTH,
}

type _FrameArray = npt.NDArray[np.uint8] | npt.NDArray[np.float32]


def core_version() -> str:
    return _native.core_version()


def core_api_version() -> int:
    return _native.core_api_version()


def core_build_revision() -> str:
    """Return the source revision embedded in the loaded core library."""
    return _native.core_revision()


def compiled_pipelines() -> frozenset[Pipeline]:
    return frozenset(Pipeline(name) for name in _native.compiled_pipelines())


def available_pipelines() -> frozenset[Pipeline]:
    return frozenset(Pipeline(name) for name in _native.available_pipelines())


def default_logger_level() -> LoggerLevel:
    return LoggerLevel(_native.default_logger_level())


def logger_level_name(level: LoggerLevel) -> str:
    return _native.logger_level_name(int(level)) or level.name.title()


def global_logger_level() -> LoggerLevel | None:
    level = _native.global_logger_level()
    return None if level is None else LoggerLevel(level)


def set_global_log_level(level: LoggerLevel | None) -> None:
    """Install a native console logger, or disable native logging with ``None``."""
    _native.set_global_log_level(None if level is None else int(level))


class PacketPipeline:
    name: ClassVar[Pipeline]

    def __init__(self, device_id: int = -1) -> None:
        self._native = _native.NativePipeline(self.name.value, device_id)

    @property
    def consumed(self) -> bool:
        return self._native.is_consumed


class CpuPacketPipeline(PacketPipeline):
    name = Pipeline.CPU


class MetalPacketPipeline(PacketPipeline):
    name = Pipeline.METAL


class OpenGLPacketPipeline(PacketPipeline):
    name = Pipeline.OPENGL


class OpenCLPacketPipeline(PacketPipeline):
    name = Pipeline.OPENCL


class OpenCLKdePacketPipeline(PacketPipeline):
    name = Pipeline.OPENCL_KDE


class CudaPacketPipeline(PacketPipeline):
    name = Pipeline.CUDA


class CudaKdePacketPipeline(PacketPipeline):
    name = Pipeline.CUDA_KDE


class DumpPacketPipeline(PacketPipeline):
    name = Pipeline.DUMP

    def depth_p0_tables(self) -> npt.NDArray[np.uint8]:
        return self._native.depth_p0_tables()

    def depth_x_table(self) -> npt.NDArray[np.float32]:
        return self._native.depth_x_table()

    def depth_z_table(self) -> npt.NDArray[np.float32]:
        return self._native.depth_z_table()

    def depth_lookup_table(self) -> npt.NDArray[np.int16]:
        return self._native.depth_lookup_table()


_PIPELINE_TYPES: Mapping[Pipeline, type[PacketPipeline]] = {
    cls.name: cls
    for cls in (
        CpuPacketPipeline,
        MetalPacketPipeline,
        OpenGLPacketPipeline,
        OpenCLPacketPipeline,
        OpenCLKdePacketPipeline,
        CudaPacketPipeline,
        CudaKdePacketPipeline,
        DumpPacketPipeline,
    )
}


def _coerce_pipeline(
    value: str | Pipeline | PacketPipeline | None,
) -> PacketPipeline | None:
    if value is None:
        return None
    if isinstance(value, PacketPipeline):
        return value
    try:
        name = Pipeline(str(value).lower())
        return None if name is Pipeline.AUTO else _PIPELINE_TYPES[name]()
    except (KeyError, ValueError) as error:
        raise ValueError(f"unknown pipeline: {value!r}") from error


class Frame:
    def __init__(self, native: _native.NativeFrame) -> None:
        self._native = native

    @classmethod
    def allocate(
        cls,
        width: int,
        height: int,
        bytes_per_pixel: int,
        *,
        frame_type: FrameType | None = None,
        frame_format: FrameFormat = FrameFormat.INVALID,
        timestamp: int = 0,
        sequence: int = 0,
        exposure: float = 0.0,
        gain: float = 0.0,
        gamma: float = 0.0,
        status: int = 0,
    ) -> Frame:
        if width <= 0 or height <= 0 or bytes_per_pixel <= 0:
            raise ValueError("frame dimensions and bytes_per_pixel must be positive")
        expected_bpp = {
            FrameFormat.FLOAT: 4,
            FrameFormat.GRAY: 1,
            FrameFormat.BGRX: 4,
            FrameFormat.RGBX: 4,
        }.get(frame_format)
        if expected_bpp is not None and bytes_per_pixel != expected_bpp:
            raise ValueError(
                f"{frame_format.name} frames require {expected_bpp} bytes per pixel"
            )
        if frame_format is FrameFormat.RAW and (width != 1 or height != 1):
            raise ValueError(
                "raw frames use width=height=1 and bytes_per_pixel as buffer size"
            )
        if frame_type is not None and frame_type not in tuple(FrameType):
            raise ValueError("a frame has exactly one FrameType")
        return cls(
            _native.NativeFrame.allocate(
                width,
                height,
                bytes_per_pixel,
                -1 if frame_type is None else int(frame_type),
                int(frame_format),
                timestamp,
                sequence,
                exposure,
                gain,
                gamma,
                status,
            )
        )

    @classmethod
    def from_array(
        cls,
        array: _FrameArray,
        *,
        frame_type: FrameType | None = None,
        frame_format: FrameFormat | None = None,
        timestamp: int = 0,
        sequence: int = 0,
        exposure: float = 0.0,
        gain: float = 0.0,
        gamma: float = 0.0,
        status: int = 0,
    ) -> Frame:
        value = cast(_FrameArray, np.asarray(array))
        if value.size == 0 or any(dimension == 0 for dimension in value.shape):
            raise ValueError("frame arrays must not be empty")
        if frame_type is not None and frame_type not in tuple(FrameType):
            raise ValueError("a frame has exactly one FrameType")
        if frame_format is None:
            if value.dtype == np.float32 and value.ndim == 2:
                frame_format = FrameFormat.FLOAT
            elif value.dtype == np.uint8 and value.ndim == 2:
                frame_format = FrameFormat.GRAY
            elif value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 4:
                frame_format = FrameFormat.BGRX
            elif value.dtype == np.uint8 and value.ndim == 1:
                frame_format = FrameFormat.RAW
            else:
                raise ValueError(
                    "cannot infer a libfreenect2 frame format from this array"
                )
        expected = {
            FrameFormat.RAW: (np.dtype(np.uint8), 1),
            FrameFormat.FLOAT: (np.dtype(np.float32), 2),
            FrameFormat.GRAY: (np.dtype(np.uint8), 2),
            FrameFormat.BGRX: (np.dtype(np.uint8), 3),
            FrameFormat.RGBX: (np.dtype(np.uint8), 3),
        }.get(frame_format)
        if expected is None:
            raise ValueError(f"cannot create an array-backed {frame_format.name} frame")
        if value.dtype != expected[0] or value.ndim != expected[1]:
            raise ValueError(f"array layout does not match {frame_format.name}")
        if (
            frame_format in (FrameFormat.BGRX, FrameFormat.RGBX)
            and value.shape[-1] != 4
        ):
            raise ValueError(
                f"{frame_format.name} arrays must have shape (height, width, 4)"
            )
        return cls(
            _native.NativeFrame.from_array(
                value,
                -1 if frame_type is None else int(frame_type),
                int(frame_format),
                timestamp,
                sequence,
                exposure,
                gain,
                gamma,
                status,
            )
        )

    @property
    def width(self) -> int:
        return self._native.width

    @property
    def height(self) -> int:
        return self._native.height

    @property
    def bytes_per_pixel(self) -> int:
        return self._native.bytes_per_pixel

    @property
    def timestamp(self) -> int:
        return self._native.timestamp

    @property
    def sequence(self) -> int:
        return self._native.sequence

    @property
    def exposure(self) -> float:
        return self._native.exposure

    @property
    def gain(self) -> float:
        return self._native.gain

    @property
    def gamma(self) -> float:
        return self._native.gamma

    @property
    def status(self) -> int:
        return self._native.status

    @property
    def format(self) -> FrameFormat:
        return FrameFormat(self._native.format)

    @property
    def frame_type(self) -> FrameType | None:
        return None if self._native.type < 0 else FrameType(self._native.type)

    def to_numpy(self, *, copy: bool = False) -> _FrameArray:
        return self._native.to_numpy(copy)

    @overload
    def __array__(
        self, dtype: None = None, copy: bool | None = None
    ) -> _FrameArray: ...

    @overload
    def __array__[ScalarT: np.generic](
        self, dtype: np.dtype[ScalarT], copy: bool | None = None
    ) -> npt.NDArray[ScalarT]: ...

    def __array__(
        self,
        dtype: np.dtype[np.generic] | None = None,
        copy: bool | None = None,
    ) -> npt.NDArray[np.generic]:
        array = self.to_numpy(copy=copy is True)
        if dtype is None or array.dtype == dtype:
            return array
        if copy is False:
            raise ValueError("a dtype conversion requires copy=True or copy=None")
        return array.astype(dtype, copy=True)


class FrameSet(Mapping[FrameType, Frame]):
    def __init__(
        self,
        native: _native.NativeFrameSet | None = None,
        copied: Mapping[FrameType, Frame] | None = None,
    ) -> None:
        self._native = native
        self._copied = dict(copied or {})
        self._released = False

    def __enter__(self) -> FrameSet:
        if self._released:
            raise DeviceStateError("frame set has already been released")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def __getitem__(self, key: FrameType) -> Frame:
        if self._released:
            raise DeviceStateError("frame set has already been released")
        frame_type = FrameType(key)
        if frame_type not in tuple(FrameType):
            raise KeyError(key)
        if self._native is not None:
            return Frame(self._native.get(int(frame_type)))
        try:
            return self._copied[frame_type]
        except KeyError as error:
            raise KeyError(key) from error

    def __contains__(self, key: object) -> bool:
        if self._released:
            return False
        try:
            frame_type = FrameType(key)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        if frame_type not in tuple(FrameType):
            return False
        if self._native is not None:
            return bool(self._native.contains(int(frame_type)))
        return frame_type in self._copied

    def __iter__(self) -> Iterator[FrameType]:
        if self._released:
            return iter(())
        return (frame_type for frame_type in FrameType if frame_type in self)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @property
    def color(self) -> Frame:
        return self[FrameType.COLOR]

    @property
    def ir(self) -> Frame:
        return self[FrameType.IR]

    @property
    def depth(self) -> Frame:
        return self[FrameType.DEPTH]

    def detached_copy(self) -> FrameSet:
        copied: dict[FrameType, Frame] = {}
        try:
            for frame_type in FrameType:
                if frame_type in self:
                    source = self[frame_type]
                    copied[frame_type] = Frame.from_array(
                        source.to_numpy(copy=True),
                        frame_type=frame_type,
                        frame_format=source.format,
                        timestamp=source.timestamp,
                        sequence=source.sequence,
                        exposure=source.exposure,
                        gain=source.gain,
                        gamma=source.gamma,
                        status=source.status,
                    )
        finally:
            self.release()
        return FrameSet(copied=copied)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._native is not None:
            self._native.release()
        else:
            self._copied.clear()

    @property
    def released(self) -> bool:
        return self._released


class FrameListener:
    def __init__(self, frame_types: FrameType) -> None:
        self.frame_types: FrameType = FrameType(frame_types)
        if not self.frame_types or int(self.frame_types) & ~int(
            FrameType.COLOR | FrameType.IR | FrameType.DEPTH
        ):
            raise ValueError("frame_types must select color, IR, and/or depth")
        self._native = _native.NativeSyncFrameListener(int(self.frame_types))

    def has_new_frame(self) -> bool:
        return self._native.has_new_frame()

    def wait(self, timeout: float | None = None) -> FrameSet:
        return FrameSet(self._native.wait(timeout))


class Device:
    def __init__(
        self,
        native: _native.NativeDeviceHandle,
        pipeline: PacketPipeline | None = None,
    ) -> None:
        self._native = native
        self.pipeline = pipeline
        self._configuration = DeviceConfig()

    def __enter__(self) -> Device:
        if self.closed:
            raise DeviceStateError("device is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def serial_number(self) -> str:
        return self._native.serial_number

    @property
    def firmware_version(self) -> str:
        return self._native.firmware_version

    @property
    def pipeline_name(self) -> Pipeline:
        return Pipeline(self._native.pipeline_name)

    @property
    def color_camera_params(self) -> ColorCameraParams:
        return _dataclass_from_mapping(
            ColorCameraParams, self._native.color_camera_params()
        )

    @color_camera_params.setter
    def color_camera_params(self, value: ColorCameraParams) -> None:
        self._native.set_color_camera_params(value)

    @property
    def ir_camera_params(self) -> IrCameraParams:
        return _dataclass_from_mapping(IrCameraParams, self._native.ir_camera_params())

    @ir_camera_params.setter
    def ir_camera_params(self, value: IrCameraParams) -> None:
        self._native.set_ir_camera_params(value)

    def set_configuration(self, config: DeviceConfig) -> None:
        self._native.set_configuration(config)
        self._configuration = config

    @property
    def configuration(self) -> DeviceConfig:
        """Last configuration applied through this object.

        The device offers no read-back; before the first assignment this
        reports the library defaults, which the core is assumed to share.
        """
        if self.closed:
            raise DeviceStateError("device is closed")
        return self._configuration

    @configuration.setter
    def configuration(self, value: DeviceConfig) -> None:
        self.set_configuration(value)

    def set_color_listener(self, listener: FrameListener) -> None:
        self._native.set_color_listener(listener._native)

    def set_depth_listener(self, listener: FrameListener) -> None:
        self._native.set_depth_listener(listener._native)

    def set_color_auto_exposure(self, compensation: float = 0.0) -> None:
        self._native.set_color_auto_exposure(compensation)

    def set_color_semi_auto_exposure(self, exposure_time_ms: float) -> None:
        self._native.set_color_semi_auto_exposure(exposure_time_ms)

    def set_color_manual_exposure(
        self, integration_time_ms: float, analog_gain: float
    ) -> None:
        self._native.set_color_manual_exposure(integration_time_ms, analog_gain)

    def set_color_setting(
        self, command: ColorSettingCommand, value: int | float
    ) -> None:
        self._native.set_color_setting(int(command), value)

    def get_color_setting(
        self, command: ColorSettingCommand, *, as_float: bool = False
    ) -> int | float:
        return self._native.get_color_setting(int(command), as_float)

    def set_led_status(self, settings: LedSettings) -> None:
        self._native.set_led_status(settings)

    def start(self, *, rgb: bool = True, depth: bool = True) -> None:
        self._native.start(rgb, depth)

    def stop(self) -> None:
        self._native.stop()

    def close(self) -> None:
        self._native.close()

    @property
    def running(self) -> bool:
        return self._native.is_running

    @property
    def closed(self) -> bool:
        return self._native.is_closed


class Context:
    VENDOR_ID = 0x045E
    PRODUCT_ID = 0x02D8
    PREVIEW_PRODUCT_ID = 0x02C4

    def __init__(self) -> None:
        self._native = _native.NativeFreenect2Context()

    def enumerate_devices(self) -> int:
        return self._native.enumerate_devices()

    def device_serial_number(self, index: int) -> str:
        return self._native.device_serial_number(index)

    def default_device_serial_number(self) -> str:
        return self._native.default_device_serial_number()

    def open_device(
        self,
        name: str | int | None = None,
        *,
        pipeline: str | Pipeline | PacketPipeline | None = Pipeline.AUTO,
    ) -> Device:
        selected = _coerce_pipeline(pipeline)
        return Device(
            self._native.open_device(
                name, None if selected is None else selected._native
            ),
            selected,
        )


class ReplayContext:
    def __init__(self) -> None:
        self._native = _native.NativeReplayContext()

    def open_device(
        self,
        filenames: str | PathLike[str] | Iterable[str | PathLike[str]],
        *,
        calibration: ReplayCalibration | None = None,
        pipeline: str | Pipeline | PacketPipeline | None = Pipeline.AUTO,
    ) -> Device:
        if isinstance(filenames, (str, PathLike)):
            paths = [fspath(filenames)]
        else:
            paths = [fspath(filename) for filename in filenames]
        if calibration is None and any(
            path.lower().endswith(".depth") for path in paths
        ):
            raise ReplayError("depth replay requires explicit calibration")
        selected = _coerce_pipeline(pipeline)
        try:
            native = self._native.open_device(
                paths,
                calibration,
                None if selected is None else selected._native,
            )
        except DeviceOpenError as error:
            raise ReplayError("libfreenect2 could not open the replay input") from error
        return Device(native, selected)


@dataclass(slots=True)
class RegistrationResult:
    undistorted: Frame
    registered: Frame
    big_depth: Frame | None = None
    color_depth_map: npt.NDArray[np.int32] | None = None


class Registration:
    def __init__(
        self, ir_params: IrCameraParams, color_params: ColorCameraParams
    ) -> None:
        self._native = _native.NativeRegistrationHandle(ir_params, color_params)

    def apply_point(self, dx: int, dy: int, depth_mm: float) -> tuple[float, float]:
        self._validate_point(dy, dx)
        if not isfinite(depth_mm) or depth_mm <= 0:
            raise ValueError("depth_mm must be finite and positive")
        return self._native.apply_point(dx, dy, depth_mm)

    def apply(
        self,
        color: Frame,
        depth: Frame,
        *,
        enable_filter: bool = True,
        include_big_depth: bool = False,
        include_color_depth_map: bool = False,
    ) -> RegistrationResult:
        self._validate_color(color)
        self._validate_depth(depth)
        undistorted = Frame.allocate(512, 424, 4, frame_format=FrameFormat.FLOAT)
        registered = Frame.allocate(512, 424, 4, frame_format=FrameFormat.BGRX)
        big_depth = (
            Frame.allocate(1920, 1082, 4, frame_format=FrameFormat.FLOAT)
            if include_big_depth
            else None
        )
        mapping = (
            np.empty((424, 512), dtype=np.int32) if include_color_depth_map else None
        )
        self._native.apply(
            color._native,
            depth._native,
            undistorted._native,
            registered._native,
            enable_filter,
            None if big_depth is None else big_depth._native,
            mapping,
        )
        return RegistrationResult(undistorted, registered, big_depth, mapping)

    def undistort_depth(self, depth: Frame) -> Frame:
        self._validate_depth(depth)
        undistorted = Frame.allocate(512, 424, 4, frame_format=FrameFormat.FLOAT)
        self._native.undistort_depth(depth._native, undistorted._native)
        return undistorted

    def point_xyz(
        self, undistorted: Frame, row: int, column: int
    ) -> tuple[float, float, float]:
        self._validate_depth(undistorted)
        self._validate_point(row, column)
        return self._native.point_xyz(undistorted._native, row, column)

    def point_xyz_rgb(
        self, undistorted: Frame, registered: Frame, row: int, column: int
    ) -> tuple[float, float, float, int, int, int]:
        self._validate_depth(undistorted)
        if (
            registered.width != 512
            or registered.height != 424
            or registered.format
            not in (
                FrameFormat.BGRX,
                FrameFormat.RGBX,
            )
        ):
            raise ValueError("registered must be a 512x424 BGRX/RGBX frame")
        self._validate_point(row, column)
        return self._native.point_xyz_rgb(
            undistorted._native, registered._native, row, column
        )

    @staticmethod
    def _validate_color(frame: Frame) -> None:
        if (
            frame.width != 1920
            or frame.height != 1080
            or frame.format
            not in (
                FrameFormat.BGRX,
                FrameFormat.RGBX,
            )
        ):
            raise ValueError("color must be a 1920x1080 BGRX/RGBX frame")

    @staticmethod
    def _validate_depth(frame: Frame) -> None:
        if (
            frame.width != 512
            or frame.height != 424
            or frame.format is not FrameFormat.FLOAT
        ):
            raise ValueError("depth must be a 512x424 float frame")

    @staticmethod
    def _validate_point(row: int, column: int) -> None:
        if not 0 <= row < 424 or not 0 <= column < 512:
            raise IndexError("depth coordinates are outside the 512x424 frame")


class Camera:
    def __init__(
        self,
        context: Context | ReplayContext,
        device: Device,
        listener: FrameListener,
        streams: tuple[Stream, ...],
    ) -> None:
        self.context: Context | ReplayContext = context
        self.device: Device = device
        self.listener: FrameListener = listener
        self.streams: tuple[Stream, ...] = streams
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        device: str | int | None = None,
        pipeline: str | Pipeline | PacketPipeline | None = Pipeline.AUTO,
        streams: Iterable[str | Stream] = (Stream.COLOR, Stream.DEPTH),
    ) -> Camera:
        names = cls._normalize_streams(streams)
        context = Context()
        opened = context.open_device(device, pipeline=pipeline)
        return cls._start(context, opened, names)

    @classmethod
    def open_recording(
        cls,
        path: str | PathLike[str],
        *,
        pipeline: str | Pipeline | PacketPipeline | None = Pipeline.AUTO,
        streams: Iterable[str | Stream] | None = None,
    ) -> Camera:
        from .recording import RecordingBundle

        bundle = RecordingBundle.open(path)
        names = cls._normalize_streams(bundle.streams if streams is None else streams)
        context = ReplayContext()
        opened = context.open_device(
            bundle.frame_paths(names),
            calibration=bundle.calibration,
            pipeline=pipeline,
        )
        return cls._start(context, opened, names)

    @staticmethod
    def _normalize_streams(streams: Iterable[str | Stream]) -> tuple[Stream, ...]:
        try:
            names = tuple(
                dict.fromkeys(Stream(str(stream).lower()) for stream in streams)
            )
        except ValueError as error:
            raise ValueError("streams must contain color, ir, and/or depth") from error
        if not names:
            raise ValueError("streams must contain color, ir, and/or depth")
        return names

    @classmethod
    def _start(
        cls,
        context: Context | ReplayContext,
        device: Device,
        streams: tuple[Stream, ...],
    ) -> Camera:
        mask = FrameType(0)
        for stream in streams:
            mask |= _STREAM_TYPES[stream]
        listener = FrameListener(mask)
        try:
            if Stream.COLOR in streams:
                device.set_color_listener(listener)
            if Stream.IR in streams or Stream.DEPTH in streams:
                device.set_depth_listener(listener)
            device.start(
                rgb=Stream.COLOR in streams,
                depth=(Stream.IR in streams or Stream.DEPTH in streams),
            )
        except Exception:
            device.close()
            raise
        return cls(context, device, listener, streams)

    def __enter__(self) -> Camera:
        if self._closed:
            raise DeviceStateError("camera is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def pipeline(self) -> Pipeline:
        return self.device.pipeline_name

    def capture(self, *, timeout: float | None = 2.0, copy: bool = False) -> FrameSet:
        if self._closed:
            raise DeviceStateError("camera is closed")
        frames = self.listener.wait(timeout)
        return frames.detached_copy() if copy else frames

    def frames(
        self, *, timeout: float | None = 2.0, copy: bool = False
    ) -> Iterator[FrameSet]:
        while not self._closed:
            yield self.capture(timeout=timeout, copy=copy)

    def __iter__(self) -> Iterator[FrameSet]:
        return self.frames()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.device.close()
