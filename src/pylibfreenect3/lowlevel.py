"""Explicit libfreenect2 lifecycle primitives.

Most applications should use :class:`pylibfreenect3.Camera`. This module is
for callers that need to attach listeners, select packet-pipeline objects, or
manage native device lifecycles directly.
"""

from .api import (
    Context,
    CpuPacketPipeline,
    CudaKdePacketPipeline,
    CudaPacketPipeline,
    Device,
    DumpPacketPipeline,
    FrameListener,
    MetalPacketPipeline,
    OpenCLKdePacketPipeline,
    OpenCLPacketPipeline,
    OpenGLPacketPipeline,
    PacketPipeline,
    ReplayContext,
)

__all__ = [
    "Context",
    "CpuPacketPipeline",
    "CudaKdePacketPipeline",
    "CudaPacketPipeline",
    "Device",
    "DumpPacketPipeline",
    "FrameListener",
    "MetalPacketPipeline",
    "OpenCLKdePacketPipeline",
    "OpenCLPacketPipeline",
    "OpenGLPacketPipeline",
    "PacketPipeline",
    "ReplayContext",
]
