from importlib.metadata import PackageNotFoundError, version
from typing import NoReturn

try:
    __version__ = version("pylibfreenect3")
except PackageNotFoundError:
    __version__ = "1.0.0.dev0"

from . import lowlevel
from .api import (
    Camera,
    Frame,
    FrameSet,
    Registration,
    RegistrationResult,
    available_pipelines,
    compiled_pipelines,
    core_api_version,
    core_build_revision,
    core_version,
    default_logger_level,
    global_logger_level,
    logger_level_name,
    set_global_log_level,
)
from .errors import (
    BackendUnavailableError,
    DeviceOpenError,
    DeviceStateError,
    FrameTimeoutError,
    FreenectError,
    RecordingFormatError,
    ReplayError,
)
from .recording import RecordingBundle, RecordingStats, RecordingWriter
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
)

__all__ = [
    "BackendUnavailableError",
    "Camera",
    "ColorCameraParams",
    "ColorSettingCommand",
    "DeviceConfig",
    "DeviceOpenError",
    "DeviceStateError",
    "Frame",
    "FrameFormat",
    "FrameSet",
    "FrameTimeoutError",
    "FrameType",
    "FreenectError",
    "IrCameraParams",
    "LedSettings",
    "LoggerLevel",
    "Pipeline",
    "RecordingBundle",
    "RecordingFormatError",
    "RecordingStats",
    "RecordingWriter",
    "Registration",
    "RegistrationResult",
    "ReplayCalibration",
    "ReplayError",
    "Stream",
    "available_pipelines",
    "compiled_pipelines",
    "core_api_version",
    "core_build_revision",
    "core_version",
    "default_logger_level",
    "global_logger_level",
    "logger_level_name",
    "lowlevel",
    "set_global_log_level",
]

_MOVED_SYMBOLS = {
    "Freenect2": "pylibfreenect3.lowlevel.Context",
    "Freenect2Replay": "pylibfreenect3.lowlevel.ReplayContext",
    "SyncFrameListener": "pylibfreenect3.lowlevel.FrameListener",
    "Device": "pylibfreenect3.lowlevel.Device",
    "PacketPipeline": "pylibfreenect3.lowlevel.PacketPipeline",
    "CpuPacketPipeline": "pylibfreenect3.lowlevel.CpuPacketPipeline",
    "MetalPacketPipeline": "pylibfreenect3.lowlevel.MetalPacketPipeline",
    "OpenGLPacketPipeline": "pylibfreenect3.lowlevel.OpenGLPacketPipeline",
    "OpenCLPacketPipeline": "pylibfreenect3.lowlevel.OpenCLPacketPipeline",
    "OpenCLKdePacketPipeline": "pylibfreenect3.lowlevel.OpenCLKdePacketPipeline",
    "CudaPacketPipeline": "pylibfreenect3.lowlevel.CudaPacketPipeline",
    "CudaKdePacketPipeline": "pylibfreenect3.lowlevel.CudaKdePacketPipeline",
    "DumpPacketPipeline": "pylibfreenect3.lowlevel.DumpPacketPipeline",
    "STREAM_NAMES": "pylibfreenect3.Stream",
    "core_revision": "pylibfreenect3.core_build_revision",
}


def __getattr__(name: str) -> NoReturn:
    replacement = _MOVED_SYMBOLS.get(name)
    if replacement is not None:
        raise AttributeError(
            f"pylibfreenect3.{name} was removed in 1.0; use {replacement} instead"
        )
    raise AttributeError(f"module 'pylibfreenect3' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
