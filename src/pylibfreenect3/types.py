from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import IntEnum, IntFlag, StrEnum
from math import isfinite
from typing import Any, cast

import numpy as np
import numpy.typing as npt

__all__ = [
    "AlignmentConfig",
    "AlignmentStats",
    "CalibrationProvenance",
    "CalibrationQualityMetrics",
    "ColorCameraParams",
    "ColorOrder",
    "ColorSettingCommand",
    "DepthCorrectionModel",
    "DepthCorrectionProfile",
    "DepthSearchOptions",
    "DeviceConfig",
    "DeviceRuntimeStats",
    "DeviceState",
    "DistortionModel",
    "FrameFormat",
    "FrameType",
    "IrCameraParams",
    "LedSettings",
    "LoggerLevel",
    "PacketPipelineConfig",
    "Pipeline",
    "ProjectiveCameraModel",
    "ProjectiveRegistrationOptions",
    "RegistrationRasterization",
    "ReplayCalibration",
    "ReplayOptions",
    "RgbDecoder",
    "RigidTransform",
    "Stream",
    "StreamRuntimeStats",
]


DEPTH_TABLE_SIZE = 512 * 424
DEPTH_LOOKUP_TABLE_SIZE = 2048
P0_TABLES_BYTE_LENGTH = 32 + 3 * (2 + DEPTH_TABLE_SIZE * 2 + 2)
_TIMESTAMP_TICK_SECONDS = 0.000125


class FrameType(IntFlag):
    COLOR = 1
    IR = 2
    DEPTH = 4


class Stream(StrEnum):
    COLOR = "color"
    IR = "ir"
    DEPTH = "depth"


class Pipeline(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    METAL = "metal"
    OPENGL = "opengl"
    OPENCL = "opencl"
    OPENCL_KDE = "opencl_kde"
    CUDA = "cuda"
    CUDA_KDE = "cuda_kde"
    DUMP = "dump"


class RgbDecoder(StrEnum):
    AUTO = "auto"
    TURBOJPEG = "turbojpeg"
    VIDEOTOOLBOX = "videotoolbox"
    VAAPI = "vaapi"
    TEGRAJPEG = "tegrajpeg"


class ColorOrder(StrEnum):
    BGR = "bgr"
    RGB = "rgb"


class DeviceState(IntEnum):
    CREATED = 0
    OPEN = 1
    STREAMING = 2
    DISCONNECTED = 3
    ERROR = 4
    CLOSED = 5


class FrameFormat(IntEnum):
    INVALID = 0
    RAW = 1
    FLOAT = 2
    BGRX = 4
    RGBX = 5
    GRAY = 6


class LoggerLevel(IntEnum):
    NONE = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4


class DistortionModel(StrEnum):
    NONE = "none"
    BROWN_CONRADY_5 = "brown_conrady_5"
    RATIONAL_8 = "rational_8"


class DepthCorrectionModel(StrEnum):
    OFFSET_ONLY = "offset_only"
    LINEAR = "linear"


class RegistrationRasterization(StrEnum):
    NEAREST = "nearest"
    FOUR_NEIGHBOR_SPLAT = "four_neighbor_splat"


@dataclass(frozen=True, slots=True)
class StreamRuntimeStats:
    decoded_frames: int
    status_error_frames: int
    sequence_gaps: int
    last_sequence: int
    last_device_timestamp: int
    last_arrival_timestamp_us: int


@dataclass(frozen=True, slots=True)
class DeviceRuntimeStats:
    color: StreamRuntimeStats
    ir: StreamRuntimeStats
    depth: StreamRuntimeStats
    start_attempts: int
    successful_starts: int
    stop_calls: int
    disconnect_events: int
    transfer_stall_events: int


@dataclass(frozen=True, slots=True)
class ProjectiveCameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: DistortionModel = DistortionModel.NONE
    distortion: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "distortion_model", DistortionModel(self.distortion_model)
        )
        object.__setattr__(self, "distortion", tuple(self.distortion))
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or not 1 <= self.width <= 16_384
            or not 1 <= self.height <= 16_384
        ):
            raise ValueError("projective camera resolution is invalid")
        if not all(isfinite(value) for value in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("projective camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("projective camera focal lengths must be positive")
        expected = {
            DistortionModel.NONE: 0,
            DistortionModel.BROWN_CONRADY_5: 5,
            DistortionModel.RATIONAL_8: 8,
        }[self.distortion_model]
        if len(self.distortion) != expected or not all(
            isfinite(value) for value in self.distortion
        ):
            raise ValueError(
                "distortion coefficients must be finite and match the distortion model"
            )

    def scaled_to(self, width: int, height: int) -> ProjectiveCameraModel:
        from . import _native

        return _dataclass_from_mapping(
            ProjectiveCameraModel,
            _native.scale_projective_camera_model(self, width, height),
        )

    def rectified(self) -> ProjectiveCameraModel:
        from . import _native

        return _dataclass_from_mapping(
            ProjectiveCameraModel,
            _native.rectify_projective_camera_model(self),
        )


@dataclass(frozen=True, slots=True)
class RigidTransform:
    rotation: tuple[float, ...]
    translation_m: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", tuple(self.rotation))
        object.__setattr__(self, "translation_m", tuple(self.translation_m))
        if len(self.rotation) != 9 or len(self.translation_m) != 3:
            raise ValueError(
                "rigid transform must contain a 3x3 rotation and translation"
            )
        if not all(isfinite(value) for value in (*self.rotation, *self.translation_m)):
            raise ValueError("rigid transform values must be finite")


@dataclass(frozen=True, slots=True)
class DepthCorrectionProfile:
    model: DepthCorrectionModel
    scale: float
    offset_mm: float
    rmse_mm: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", DepthCorrectionModel(self.model))
        if not all(
            isfinite(value) for value in (self.scale, self.offset_mm, self.rmse_mm)
        ):
            raise ValueError("depth correction values must be finite")
        if self.scale <= 0 or self.rmse_mm < 0:
            raise ValueError(
                "depth correction scale must be positive and RMSE non-negative"
            )


@dataclass(frozen=True, slots=True)
class CalibrationQualityMetrics:
    color_views: int
    ir_views: int
    stereo_views: int
    depth_views: int
    color_rms_px: float
    ir_rms_px: float
    held_out_stereo_rms_px: float
    depth_rmse_mm: float

    def __post_init__(self) -> None:
        counts = (self.color_views, self.ir_views, self.stereo_views, self.depth_views)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("calibration view counts must be non-negative integers")
        errors = (
            self.color_rms_px,
            self.ir_rms_px,
            self.held_out_stereo_rms_px,
            self.depth_rmse_mm,
        )
        if any(not isfinite(value) or value < 0 for value in errors):
            raise ValueError(
                "calibration error metrics must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    created_utc: str
    tool_version: str
    job_sha256: str


@dataclass(frozen=True, slots=True)
class ProjectiveRegistrationOptions:
    rasterization: RegistrationRasterization = (
        RegistrationRasterization.FOUR_NEIGHBOR_SPLAT
    )
    min_depth_mm: float = 500.0
    max_depth_mm: float = 4500.0
    apply_depth_correction: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rasterization", RegistrationRasterization(self.rasterization)
        )
        if (
            not isfinite(self.min_depth_mm)
            or not isfinite(self.max_depth_mm)
            or self.min_depth_mm <= 0
            or self.max_depth_mm <= self.min_depth_mm
        ):
            raise ValueError("projective registration depth range is invalid")
        if not isinstance(self.apply_depth_correction, bool):
            raise TypeError("apply_depth_correction must be bool")


@dataclass(frozen=True, slots=True)
class ReplayOptions:
    salvage_incomplete: bool = False
    reproduce_timing: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.salvage_incomplete, bool) or not isinstance(
            self.reproduce_timing, bool
        ):
            raise TypeError("replay options must be bool values")


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    max_delta: float = 0.025
    queue_capacity: int = 8

    def __post_init__(self) -> None:
        if not isfinite(self.max_delta) or self.max_delta < 0:
            raise ValueError("max_delta must be finite and non-negative")
        if self.max_delta > 0xFFFFFFFF * _TIMESTAMP_TICK_SECONDS:
            raise ValueError("max_delta exceeds the native timestamp range")
        if (
            not isinstance(self.queue_capacity, int)
            or isinstance(self.queue_capacity, bool)
            or self.queue_capacity <= 0
        ):
            raise ValueError("queue_capacity must be a positive integer")

    @property
    def max_delta_ticks(self) -> int:
        return round(self.max_delta / _TIMESTAMP_TICK_SECONDS)


@dataclass(frozen=True, slots=True)
class AlignmentStats:
    delivered: int
    dropped: int
    last_delta_ticks: int
    maximum_delta_ticks: int

    @property
    def last_delta_seconds(self) -> float:
        return self.last_delta_ticks * _TIMESTAMP_TICK_SECONDS

    @property
    def maximum_delta_seconds(self) -> float:
        return self.maximum_delta_ticks * _TIMESTAMP_TICK_SECONDS

    @property
    def last_delta_milliseconds(self) -> float:
        return self.last_delta_seconds * 1000.0

    @property
    def maximum_delta_milliseconds(self) -> float:
        return self.maximum_delta_seconds * 1000.0


@dataclass(frozen=True, slots=True)
class PacketPipelineConfig:
    rgb_decoder: RgbDecoder = RgbDecoder.AUTO
    vaapi_device: str | None = None
    allow_fallback: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "rgb_decoder", RgbDecoder(self.rgb_decoder))
        if self.vaapi_device is not None and not isinstance(self.vaapi_device, str):
            raise TypeError("vaapi_device must be a string or None")
        if not isinstance(self.allow_fallback, bool):
            raise TypeError("allow_fallback must be bool")


@dataclass(frozen=True, slots=True)
class DepthSearchOptions:
    primary_radius: int = 8
    fallback_radius: int = 20
    cluster_span_mm: float = 150.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.primary_radius, int)
            or isinstance(self.primary_radius, bool)
            or self.primary_radius < 0
        ):
            raise ValueError("primary_radius must be a non-negative integer")
        if (
            not isinstance(self.fallback_radius, int)
            or isinstance(self.fallback_radius, bool)
            or self.fallback_radius < self.primary_radius
        ):
            raise ValueError("fallback_radius must be at least primary_radius")
        if not isfinite(self.cluster_span_mm) or self.cluster_span_mm < 0:
            raise ValueError("cluster_span_mm must be finite and non-negative")


class ColorSettingCommand(IntEnum):
    SET_EXPOSURE_MODE = 0
    SET_INTEGRATION_TIME = 1
    GET_INTEGRATION_TIME = 2
    SET_WHITE_BALANCE_MODE = 10
    SET_RED_CHANNEL_GAIN = 11
    SET_GREEN_CHANNEL_GAIN = 12
    SET_BLUE_CHANNEL_GAIN = 13
    GET_RED_CHANNEL_GAIN = 14
    GET_GREEN_CHANNEL_GAIN = 15
    GET_BLUE_CHANNEL_GAIN = 16
    SET_EXPOSURE_TIME_MS = 17
    GET_EXPOSURE_TIME_MS = 18
    SET_DIGITAL_GAIN = 19
    GET_DIGITAL_GAIN = 20
    SET_ANALOG_GAIN = 21
    GET_ANALOG_GAIN = 22
    SET_EXPOSURE_COMPENSATION = 23
    GET_EXPOSURE_COMPENSATION = 24
    SET_ACS = 25
    GET_ACS = 26
    SET_EXPOSURE_METERING_MODE = 27
    SET_EXPOSURE_METERING_ZONES = 28
    SET_EXPOSURE_METERING_ZONE_0_WEIGHT = 29
    SET_EXPOSURE_METERING_ZONE_1_WEIGHT = 30
    SET_EXPOSURE_METERING_ZONE_2_WEIGHT = 31
    SET_EXPOSURE_METERING_ZONE_3_WEIGHT = 32
    SET_EXPOSURE_METERING_ZONE_4_WEIGHT = 33
    SET_EXPOSURE_METERING_ZONE_5_WEIGHT = 34
    SET_EXPOSURE_METERING_ZONE_6_WEIGHT = 35
    SET_EXPOSURE_METERING_ZONE_7_WEIGHT = 36
    SET_EXPOSURE_METERING_ZONE_8_WEIGHT = 37
    SET_EXPOSURE_METERING_ZONE_9_WEIGHT = 38
    SET_EXPOSURE_METERING_ZONE_10_WEIGHT = 39
    SET_EXPOSURE_METERING_ZONE_11_WEIGHT = 40
    SET_EXPOSURE_METERING_ZONE_12_WEIGHT = 41
    SET_EXPOSURE_METERING_ZONE_13_WEIGHT = 42
    SET_EXPOSURE_METERING_ZONE_14_WEIGHT = 43
    SET_EXPOSURE_METERING_ZONE_15_WEIGHT = 44
    SET_EXPOSURE_METERING_ZONE_16_WEIGHT = 45
    SET_EXPOSURE_METERING_ZONE_17_WEIGHT = 46
    SET_EXPOSURE_METERING_ZONE_18_WEIGHT = 47
    SET_EXPOSURE_METERING_ZONE_19_WEIGHT = 48
    SET_EXPOSURE_METERING_ZONE_20_WEIGHT = 49
    SET_EXPOSURE_METERING_ZONE_21_WEIGHT = 50
    SET_EXPOSURE_METERING_ZONE_22_WEIGHT = 51
    SET_EXPOSURE_METERING_ZONE_23_WEIGHT = 52
    SET_EXPOSURE_METERING_ZONE_24_WEIGHT = 53
    SET_EXPOSURE_METERING_ZONE_25_WEIGHT = 54
    SET_EXPOSURE_METERING_ZONE_26_WEIGHT = 55
    SET_EXPOSURE_METERING_ZONE_27_WEIGHT = 56
    SET_EXPOSURE_METERING_ZONE_28_WEIGHT = 57
    SET_EXPOSURE_METERING_ZONE_29_WEIGHT = 58
    SET_EXPOSURE_METERING_ZONE_30_WEIGHT = 59
    SET_EXPOSURE_METERING_ZONE_31_WEIGHT = 60
    SET_EXPOSURE_METERING_ZONE_32_WEIGHT = 61
    SET_EXPOSURE_METERING_ZONE_33_WEIGHT = 62
    SET_EXPOSURE_METERING_ZONE_34_WEIGHT = 63
    SET_EXPOSURE_METERING_ZONE_35_WEIGHT = 64
    SET_EXPOSURE_METERING_ZONE_36_WEIGHT = 65
    SET_EXPOSURE_METERING_ZONE_37_WEIGHT = 66
    SET_EXPOSURE_METERING_ZONE_38_WEIGHT = 67
    SET_EXPOSURE_METERING_ZONE_39_WEIGHT = 68
    SET_EXPOSURE_METERING_ZONE_40_WEIGHT = 69
    SET_EXPOSURE_METERING_ZONE_41_WEIGHT = 70
    SET_EXPOSURE_METERING_ZONE_42_WEIGHT = 71
    SET_EXPOSURE_METERING_ZONE_43_WEIGHT = 72
    SET_EXPOSURE_METERING_ZONE_44_WEIGHT = 73
    SET_EXPOSURE_METERING_ZONE_45_WEIGHT = 74
    SET_EXPOSURE_METERING_ZONE_46_WEIGHT = 75
    SET_EXPOSURE_METERING_ZONE_47_WEIGHT = 76
    SET_MAX_ANALOG_GAIN_CAP = 77
    SET_MAX_DIGITAL_GAIN_CAP = 78
    SET_FLICKER_FREE_FREQUENCY = 79
    GET_EXPOSURE_MODE = 80
    GET_WHITE_BALANCE_MODE = 81
    SET_FRAME_RATE = 82
    GET_FRAME_RATE = 83


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    min_depth: float = 0.5
    max_depth: float = 4.5
    enable_bilateral_filter: bool = True
    enable_edge_aware_filter: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.min_depth) or not isfinite(self.max_depth):
            raise ValueError("depth range must be finite")
        if self.min_depth < 0 or self.max_depth <= self.min_depth:
            raise ValueError("depth range must satisfy 0 <= min_depth < max_depth")
        if not isinstance(self.enable_bilateral_filter, bool) or not isinstance(
            self.enable_edge_aware_filter, bool
        ):
            raise TypeError("depth filter settings must be bool values")


@dataclass(frozen=True, slots=True)
class IrCameraParams:
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def __post_init__(self) -> None:
        if not all(isfinite(getattr(self, field.name)) for field in fields(self)):
            raise ValueError("IR camera parameters must be finite")
        if self.fx < 0 or self.fy < 0:
            raise ValueError("IR focal lengths must be non-negative")

    def camera_matrix(self) -> npt.NDArray[np.float64]:
        """Return the OpenCV-compatible 3x3 intrinsic matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def distortion_coefficients(self) -> npt.NDArray[np.float64]:
        """Return OpenCV distortion coefficients in k1, k2, p1, p2, k3 order."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ColorCameraParams:
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    shift_d: float = 0.0
    shift_m: float = 0.0
    mx_x3y0: float = 0.0
    mx_x0y3: float = 0.0
    mx_x2y1: float = 0.0
    mx_x1y2: float = 0.0
    mx_x2y0: float = 0.0
    mx_x0y2: float = 0.0
    mx_x1y1: float = 0.0
    mx_x1y0: float = 0.0
    mx_x0y1: float = 0.0
    mx_x0y0: float = 0.0
    my_x3y0: float = 0.0
    my_x0y3: float = 0.0
    my_x2y1: float = 0.0
    my_x1y2: float = 0.0
    my_x2y0: float = 0.0
    my_x0y2: float = 0.0
    my_x1y1: float = 0.0
    my_x1y0: float = 0.0
    my_x0y1: float = 0.0
    my_x0y0: float = 0.0

    def __post_init__(self) -> None:
        if not all(isfinite(getattr(self, field.name)) for field in fields(self)):
            raise ValueError("color camera parameters must be finite")
        if self.fx < 0 or self.fy < 0:
            raise ValueError("color focal lengths must be non-negative")


@dataclass(frozen=True, slots=True)
class LedSettings:
    led_id: int
    mode: int = 0
    start_level: int = 0
    stop_level: int = 0
    interval_ms: int = 0
    reserved: int = 0

    def __post_init__(self) -> None:
        if self.led_id not in (0, 1):
            raise ValueError("led_id must be 0 or 1")
        if self.mode not in (0, 1):
            raise ValueError("mode must be 0 (constant) or 1 (blink)")
        if not 0 <= self.start_level <= 1000 or not 0 <= self.stop_level <= 1000:
            raise ValueError("LED levels must be in [0, 1000]")
        if self.interval_ms < 0:
            raise ValueError("interval_ms must be non-negative")
        if self.interval_ms > 0xFFFFFFFF:
            raise ValueError("interval_ms exceeds the native uint32 range")
        if self.reserved != 0:
            raise ValueError("reserved must be zero")


@dataclass(slots=True)
class ReplayCalibration:
    color: ColorCameraParams
    ir: IrCameraParams
    p0_tables: npt.NDArray[np.uint8]
    x_table: npt.NDArray[np.float32]
    z_table: npt.NDArray[np.float32]
    lookup_table: npt.NDArray[np.int16]

    def __post_init__(self) -> None:
        self.p0_tables = cast(
            npt.NDArray[np.uint8],
            self._array(self.p0_tables, np.uint8, P0_TABLES_BYTE_LENGTH, "P0 tables"),
        )
        self.x_table = cast(
            npt.NDArray[np.float32],
            self._array(self.x_table, np.float32, DEPTH_TABLE_SIZE, "X table"),
        )
        self.z_table = cast(
            npt.NDArray[np.float32],
            self._array(self.z_table, np.float32, DEPTH_TABLE_SIZE, "Z table"),
        )
        self.lookup_table = cast(
            npt.NDArray[np.int16],
            self._array(
                self.lookup_table,
                np.int16,
                DEPTH_LOOKUP_TABLE_SIZE,
                "lookup table",
            ),
        )

    @staticmethod
    def _array(value: Any, dtype: Any, size: int, label: str) -> np.ndarray[Any, Any]:
        array = np.asarray(value)
        if array.dtype != np.dtype(dtype):
            raise TypeError(f"{label} must use {np.dtype(dtype)}")
        if array.size != size:
            raise ValueError(f"{label} must contain exactly {size} values")
        return np.ascontiguousarray(array).reshape(-1)


def _dataclass_from_mapping[DataclassT](
    cls: type[DataclassT], values: Mapping[str, Any]
) -> DataclassT:
    dataclass_fields = fields(cast(Any, cls))
    return cls(**{field.name: values[field.name] for field in dataclass_fields})
