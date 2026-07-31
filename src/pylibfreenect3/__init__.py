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
    LandmarkLiftResult,
    Registration,
    RegistrationResult,
    RegistrationWorkspace,
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
    RgbDecoder,
    Stream,
)

del PackageNotFoundError, version

__all__ = [
    "AlignmentConfig",
    "AlignmentStats",
    "BackendUnavailableError",
    "Camera",
    "ColorCameraParams",
    "ColorOrder",
    "ColorSettingCommand",
    "DepthSearchOptions",
    "DeviceConfig",
    "DeviceOpenError",
    "DeviceState",
    "DeviceStateError",
    "Frame",
    "FrameFormat",
    "FrameSet",
    "FrameTimeoutError",
    "FrameType",
    "FreenectError",
    "IrCameraParams",
    "LandmarkLiftResult",
    "LedSettings",
    "LoggerLevel",
    "PacketPipelineConfig",
    "Pipeline",
    "RecordingBundle",
    "RecordingFormatError",
    "RecordingStats",
    "RecordingWriter",
    "Registration",
    "RegistrationResult",
    "RegistrationWorkspace",
    "ReplayCalibration",
    "ReplayError",
    "RgbDecoder",
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
    # List the curated API plus real module attributes such as __version__
    # and the submodules, without typing helpers or private globals.
    public = {name for name in globals() if not name.startswith("_")} - {"NoReturn"}
    return sorted(public | set(__all__) | {"__version__"})
