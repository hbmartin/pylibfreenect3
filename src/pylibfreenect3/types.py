from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import IntEnum, IntFlag, StrEnum
from math import isfinite
from typing import Any, cast

import numpy as np
import numpy.typing as npt

__all__ = [
    "ColorCameraParams",
    "ColorSettingCommand",
    "DeviceConfig",
    "FrameFormat",
    "FrameType",
    "IrCameraParams",
    "LedSettings",
    "LoggerLevel",
    "Pipeline",
    "ReplayCalibration",
    "Stream",
]


DEPTH_TABLE_SIZE = 512 * 424
DEPTH_LOOKUP_TABLE_SIZE = 2048
P0_TABLES_BYTE_LENGTH = 32 + 3 * (2 + DEPTH_TABLE_SIZE * 2 + 2)


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
