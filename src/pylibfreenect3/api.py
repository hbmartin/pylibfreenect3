from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from math import isfinite
from os import PathLike, fspath
from typing import ClassVar, cast, overload
from warnings import warn

import numpy as np
import numpy.typing as npt

from . import _native
from .errors import (
    DeviceOpenError,
    DeviceStateError,
    ReplayError,
    WorkspaceStateError,
)
from .types import (
    _TIMESTAMP_TICK_SECONDS,
    AlignmentConfig,
    AlignmentStats,
    ColorCameraParams,
    ColorOrder,
    ColorSettingCommand,
    DepthSearchOptions,
    DeviceConfig,
    DeviceState,
    FrameFormat,
    FrameType,
    IrCameraParams,
    LedSettings,
    LoggerLevel,
    PacketPipelineConfig,
    Pipeline,
    ReplayCalibration,
    Stream,
    _dataclass_from_mapping,
)

__all__ = [
    "AlignedFrameListener",
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
    "LandmarkLiftResult",
    "MetalPacketPipeline",
    "OpenCLKdePacketPipeline",
    "OpenCLPacketPipeline",
    "OpenGLPacketPipeline",
    "PacketPipeline",
    "Registration",
    "RegistrationResult",
    "RegistrationWorkspace",
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

_DEFAULT_ALIGNMENT_CONFIG = AlignmentConfig()
_DEFAULT_DEPTH_SEARCH_OPTIONS = DepthSearchOptions()


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

    def __init__(
        self,
        device_id: int = -1,
        *,
        config: PacketPipelineConfig | None = None,
    ) -> None:
        selected = PacketPipelineConfig() if config is None else config
        decoder = _native.RGB_DECODER_VALUES[selected.rgb_decoder.value]
        self.config = selected
        self._native = _native.NativePipeline(
            self.name.value,
            device_id,
            decoder,
            selected.vaapi_device,
            selected.allow_fallback,
        )

    @property
    def consumed(self) -> bool:
        return self._native.is_consumed


class _DefaultPacketPipeline(PacketPipeline):
    name = Pipeline.AUTO


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
    config: PacketPipelineConfig | None = None,
) -> PacketPipeline | None:
    if value is None:
        if config is not None:
            raise ValueError("pipeline_config cannot be supplied without a pipeline")
        return None
    if isinstance(value, PacketPipeline):
        if config is not None:
            raise ValueError(
                "pipeline_config cannot be supplied with a preconstructed "
                "PacketPipeline"
            )
        return value
    try:
        name = Pipeline(str(value).lower())
        if name is Pipeline.AUTO:
            return _DefaultPacketPipeline(config=config)
        return _PIPELINE_TYPES[name](config=config)
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
        arrival_timestamp_us: int = 0,
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
                arrival_timestamp_us,
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
        arrival_timestamp_us: int = 0,
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
                arrival_timestamp_us,
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
    def arrival_timestamp_us(self) -> int:
        return self._native.arrival_timestamp_us

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

    def to_color(
        self,
        order: ColorOrder = ColorOrder.BGR,
        *,
        out: npt.NDArray[np.uint8] | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Convert packed BGRX/RGBX data to contiguous BGR or RGB."""
        selected = ColorOrder(order)
        if self.format not in (FrameFormat.BGRX, FrameFormat.RGBX):
            raise ValueError("to_color() requires a BGRX or RGBX frame")
        if out is None:
            destination = np.empty((self.height, self.width, 3), dtype=np.uint8)
        else:
            if not isinstance(out, np.ndarray):
                raise TypeError("out must be a NumPy array")
            destination = out
            if destination.dtype != np.dtype(np.uint8):
                raise TypeError("out must use dtype uint8")
            if destination.shape != (self.height, self.width, 3):
                raise ValueError(
                    f"out must have shape ({self.height}, {self.width}, 3)"
                )
            if not destination.flags.writeable:
                raise ValueError("out must be writable")
            if not destination.flags.c_contiguous:
                raise ValueError("out must be C-contiguous")
        self._native.to_color(destination, 0 if selected is ColorOrder.BGR else 1)
        return destination

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
        alignment_delta_ticks: int | None = None,
    ) -> None:
        self._native = native
        self._copied = dict(copied or {})
        self._released = False
        self._alignment_delta_ticks = (
            native.delta_ticks if native is not None else alignment_delta_ticks
        )

    def __enter__(self) -> FrameSet:
        if self._released:
            raise DeviceStateError("frame set has already been released")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def __getitem__(self, key: FrameType) -> Frame:
        if self._released:
            raise DeviceStateError("frame set has already been released")
        # Invalid keys (including the removed 0.3 string keys) must raise
        # KeyError so the Mapping mixins, e.g. get(), handle them.
        try:
            frame_type = FrameType(key)
        except (TypeError, ValueError) as error:
            raise KeyError(key) from error
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
                        arrival_timestamp_us=source.arrival_timestamp_us,
                        sequence=source.sequence,
                        exposure=source.exposure,
                        gain=source.gain,
                        gamma=source.gamma,
                        status=source.status,
                    )
        finally:
            self.release()
        return FrameSet(
            copied=copied, alignment_delta_ticks=self._alignment_delta_ticks
        )

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

    @property
    def alignment_delta_ticks(self) -> int | None:
        return self._alignment_delta_ticks

    @property
    def alignment_delta_seconds(self) -> float | None:
        if self._alignment_delta_ticks is None:
            return None
        return self._alignment_delta_ticks * _TIMESTAMP_TICK_SECONDS


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


class AlignedFrameListener:
    """Deliver frame sets whose device timestamps agree within ``max_delta``.

    The listener is designed for a single consuming thread. With concurrent
    ``wait()`` callers, ``FrameSet.alignment_delta_ticks`` and
    ``alignment_stats`` reflect the most recent delivery, which may belong to
    a set consumed by another thread.
    """

    def __init__(
        self,
        frame_types: FrameType,
        config: AlignmentConfig = _DEFAULT_ALIGNMENT_CONFIG,
    ) -> None:
        self.frame_types: FrameType = FrameType(frame_types)
        if not self.frame_types or int(self.frame_types) & ~int(
            FrameType.COLOR | FrameType.IR | FrameType.DEPTH
        ):
            raise ValueError("frame_types must select color, IR, and/or depth")
        self.config = config
        self._native = _native.NativeAlignedFrameListener(
            int(self.frame_types), config.max_delta_ticks, config.queue_capacity
        )

    def has_new_frame(self) -> bool:
        return self._native.has_new_frame()

    def wait(self, timeout: float | None = None) -> FrameSet:
        return FrameSet(self._native.wait(timeout))

    @property
    def alignment_stats(self) -> AlignmentStats:
        return _dataclass_from_mapping(AlignmentStats, self._native.statistics())


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

    def set_color_listener(
        self, listener: FrameListener | AlignedFrameListener
    ) -> None:
        self._native.set_color_listener(listener._native)

    def set_depth_listener(
        self, listener: FrameListener | AlignedFrameListener
    ) -> None:
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

    @property
    def state(self) -> DeviceState:
        return DeviceState(self._native.state)

    @property
    def last_error(self) -> str:
        return self._native.last_error


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

    def wait_for_device(
        self, serial: str, timeout: float, poll_interval: float = 0.25
    ) -> bool:
        return self._native.wait_for_device(serial, timeout, poll_interval)

    def open_device(
        self,
        name: str | int | None = None,
        *,
        pipeline: str | Pipeline | PacketPipeline | None = Pipeline.AUTO,
        pipeline_config: PacketPipelineConfig | None = None,
    ) -> Device:
        selected = _coerce_pipeline(pipeline, pipeline_config)
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
        pipeline_config: PacketPipelineConfig | None = None,
    ) -> Device:
        if isinstance(filenames, str | PathLike):
            paths = [fspath(filenames)]
        else:
            paths = [fspath(filename) for filename in filenames]
        if calibration is None and any(
            path.lower().endswith(".depth") for path in paths
        ):
            raise ReplayError("depth replay requires explicit calibration")
        selected = _coerce_pipeline(pipeline, pipeline_config)
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
    depth_to_color_map: npt.NDArray[np.int32] | None = None
    color_to_depth_map: npt.NDArray[np.int32] | None = None

    @property
    def color_depth_map(self) -> npt.NDArray[np.int32] | None:
        warn(
            "color_depth_map is deprecated; use depth_to_color_map",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.depth_to_color_map


@dataclass(slots=True)
class LandmarkLiftResult:
    xyz: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]
    depth_pixels: npt.NDArray[np.int32]


class Registration:
    def __init__(
        self, ir_params: IrCameraParams, color_params: ColorCameraParams
    ) -> None:
        self._native = _native.NativeRegistrationHandle(ir_params, color_params)

    @classmethod
    def from_device(cls, device: Device) -> Registration:
        return cls(device.ir_camera_params, device.color_camera_params)

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
        include_depth_to_color_map: bool | None = None,
        include_color_to_depth_map: bool = False,
        include_color_depth_map: bool | None = None,
    ) -> RegistrationResult:
        include_forward = self._resolve_map_alias(
            include_depth_to_color_map, include_color_depth_map
        )
        self._validate_color(color)
        self._validate_depth(depth)
        undistorted = Frame.allocate(512, 424, 4, frame_format=FrameFormat.FLOAT)
        registered = Frame.allocate(512, 424, 4, frame_format=color.format)
        big_depth = (
            Frame.allocate(1920, 1082, 4, frame_format=FrameFormat.FLOAT)
            if include_big_depth
            else None
        )
        internal_mapping = (
            np.empty((424, 512), dtype=np.int32)
            if include_forward or include_color_to_depth_map
            else None
        )
        self._native.apply(
            color._native,
            depth._native,
            undistorted._native,
            registered._native,
            enable_filter,
            None if big_depth is None else big_depth._native,
            internal_mapping,
        )
        reverse = None
        if include_color_to_depth_map:
            if internal_mapping is None:  # pragma: no cover - construction invariant
                raise RuntimeError("depth-to-color workspace was not allocated")
            reverse = np.empty((1080, 1920), dtype=np.int32)
            self._native.build_color_to_depth_map(
                undistorted._native, internal_mapping, reverse
            )
        return RegistrationResult(
            undistorted,
            registered,
            big_depth,
            internal_mapping if include_forward else None,
            reverse,
        )

    def workspace(
        self,
        *,
        include_big_depth: bool = False,
        include_depth_to_color_map: bool | None = None,
        include_color_to_depth_map: bool = False,
        include_color_depth_map: bool | None = None,
    ) -> RegistrationWorkspace:
        include_forward = self._resolve_map_alias(
            include_depth_to_color_map, include_color_depth_map
        )
        return RegistrationWorkspace(
            self,
            include_big_depth=include_big_depth,
            include_depth_to_color_map=include_forward,
            include_color_to_depth_map=include_color_to_depth_map,
        )

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

    def points_xyz(
        self, undistorted: Frame, pixels: npt.ArrayLike
    ) -> npt.NDArray[np.float32]:
        self._validate_depth(undistorted)
        values = np.asarray(pixels)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("pixels must have shape (N, 2)")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("pixels must contain integer row/column coordinates")
        if values.size and (
            np.any(values[:, 0] < 0)
            or np.any(values[:, 0] >= 424)
            or np.any(values[:, 1] < 0)
            or np.any(values[:, 1] >= 512)
        ):
            raise IndexError("depth coordinates are outside the 512x424 frame")
        coordinates = np.ascontiguousarray(values, dtype=np.int32)
        return self._native.points_xyz(undistorted._native, coordinates)

    @staticmethod
    def _resolve_map_alias(canonical: bool | None, deprecated: bool | None) -> bool:
        if canonical is not None and deprecated is not None and canonical != deprecated:
            raise ValueError(
                "conflicting include_depth_to_color_map and "
                "include_color_depth_map values"
            )
        if deprecated is not None:
            warn(
                "include_color_depth_map is deprecated; use include_depth_to_color_map",
                DeprecationWarning,
                stacklevel=3,
            )
        return bool(canonical if canonical is not None else deprecated)

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


class RegistrationWorkspace:
    """Reusable registration buffers whose contents are overwritten by apply()."""

    def __init__(
        self,
        registration: Registration,
        *,
        include_big_depth: bool,
        include_depth_to_color_map: bool,
        include_color_to_depth_map: bool,
    ) -> None:
        self.registration: Registration = registration
        self.undistorted: Frame = Frame.allocate(
            512, 424, 4, frame_format=FrameFormat.FLOAT
        )
        # The core assigns the actual packed color order on every apply.
        self.registered: Frame = Frame.allocate(
            512, 424, 4, frame_format=FrameFormat.BGRX
        )
        self.big_depth: Frame | None = (
            Frame.allocate(1920, 1082, 4, frame_format=FrameFormat.FLOAT)
            if include_big_depth
            else None
        )
        self._internal_depth_to_color_map: npt.NDArray[np.int32] | None = (
            np.empty((424, 512), dtype=np.int32)
            if include_depth_to_color_map or include_color_to_depth_map
            else None
        )
        self.depth_to_color_map: npt.NDArray[np.int32] | None = (
            self._internal_depth_to_color_map if include_depth_to_color_map else None
        )
        self.color_to_depth_map: npt.NDArray[np.int32] | None = (
            np.empty((1080, 1920), dtype=np.int32)
            if include_color_to_depth_map
            else None
        )
        self.result: RegistrationResult = RegistrationResult(
            self.undistorted,
            self.registered,
            self.big_depth,
            self.depth_to_color_map,
            self.color_to_depth_map,
        )
        self._has_applied = False

    def apply(
        self, color: Frame, depth: Frame, *, enable_filter: bool = True
    ) -> RegistrationResult:
        self.registration._validate_color(color)
        self.registration._validate_depth(depth)
        self.registration._native.apply(
            color._native,
            depth._native,
            self.undistorted._native,
            self.registered._native,
            enable_filter,
            None if self.big_depth is None else self.big_depth._native,
            self._internal_depth_to_color_map,
        )
        if self.color_to_depth_map is not None:
            if self._internal_depth_to_color_map is None:  # pragma: no cover
                raise RuntimeError("depth-to-color workspace was not allocated")
            self.registration._native.build_color_to_depth_map(
                self.undistorted._native,
                self._internal_depth_to_color_map,
                self.color_to_depth_map,
            )
        self._has_applied = True
        return self.result

    def lift_normalized(
        self,
        xy: npt.ArrayLike,
        options: DepthSearchOptions = _DEFAULT_DEPTH_SEARCH_OPTIONS,
    ) -> LandmarkLiftResult:
        if not self._has_applied:
            raise WorkspaceStateError("registration workspace has not been applied")
        if self.color_to_depth_map is None:
            raise WorkspaceStateError(
                "workspace was created without include_color_to_depth_map=True"
            )
        values = np.asarray(xy)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2)")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("xy must contain numeric normalized coordinates")
        points = np.ascontiguousarray(values, dtype=np.float32)
        xyz, valid, indices = self.registration._native.lift_normalized(
            self.undistorted._native,
            self.color_to_depth_map,
            points,
            options.primary_radius,
            options.fallback_radius,
            options.cluster_span_mm,
        )
        linear = indices
        depth_pixels = np.full((linear.shape[0], 2), -1, dtype=np.int32)
        present = linear >= 0
        depth_pixels[present, 0] = linear[present] // 512
        depth_pixels[present, 1] = linear[present] % 512
        return LandmarkLiftResult(
            xyz,
            valid,
            depth_pixels,
        )


class Camera:
    def __init__(
        self,
        context: Context | ReplayContext,
        device: Device,
        listener: FrameListener | AlignedFrameListener,
        streams: tuple[Stream, ...],
    ) -> None:
        self.context: Context | ReplayContext = context
        self.device: Device = device
        self.listener: FrameListener | AlignedFrameListener = listener
        self.streams: tuple[Stream, ...] = streams
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        device: str | int | None = None,
        pipeline: str | Pipeline | PacketPipeline | None = Pipeline.AUTO,
        pipeline_config: PacketPipelineConfig | None = None,
        alignment: AlignmentConfig | None = None,
        streams: Iterable[str | Stream] = (Stream.COLOR, Stream.DEPTH),
    ) -> Camera:
        names = cls._normalize_streams(streams)
        context = Context()
        opened = context.open_device(
            device, pipeline=pipeline, pipeline_config=pipeline_config
        )
        return cls._start(context, opened, names, alignment)

    @classmethod
    def open_recording(
        cls,
        path: str | PathLike[str],
        *,
        pipeline: str | Pipeline | PacketPipeline | None = Pipeline.AUTO,
        pipeline_config: PacketPipelineConfig | None = None,
        alignment: AlignmentConfig | None = None,
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
            pipeline_config=pipeline_config,
        )
        return cls._start(context, opened, names, alignment)

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
        alignment: AlignmentConfig | None,
    ) -> Camera:
        mask = FrameType(0)
        for stream in streams:
            mask |= _STREAM_TYPES[stream]
        listener: FrameListener | AlignedFrameListener = (
            FrameListener(mask)
            if alignment is None
            else AlignedFrameListener(mask, alignment)
        )
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

    @property
    def alignment_stats(self) -> AlignmentStats | None:
        if isinstance(self.listener, AlignedFrameListener):
            return self.listener.alignment_stats
        return None

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
