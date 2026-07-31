Public API
==========

Capture, replay, and registration
---------------------------------

.. autosummary::
   :toctree: generated

   pylibfreenect3.Camera
   pylibfreenect3.RecordingWriter
   pylibfreenect3.RecordingStats
   pylibfreenect3.RecordingBundle
   pylibfreenect3.Frame
   pylibfreenect3.FrameSet
   pylibfreenect3.Registration
   pylibfreenect3.RegistrationResult

Pipelines
---------

Every backend class is importable on every platform. Constructing an
uncompiled or runtime-unusable backend raises ``BackendUnavailableError``.

.. autosummary::
   :toctree: generated

   pylibfreenect3.lowlevel.Context
   pylibfreenect3.lowlevel.ReplayContext
   pylibfreenect3.lowlevel.Device
   pylibfreenect3.lowlevel.FrameListener
   pylibfreenect3.lowlevel.PacketPipeline
   pylibfreenect3.lowlevel.CpuPacketPipeline
   pylibfreenect3.lowlevel.MetalPacketPipeline
   pylibfreenect3.lowlevel.OpenGLPacketPipeline
   pylibfreenect3.lowlevel.OpenCLPacketPipeline
   pylibfreenect3.lowlevel.OpenCLKdePacketPipeline
   pylibfreenect3.lowlevel.CudaPacketPipeline
   pylibfreenect3.lowlevel.CudaKdePacketPipeline
   pylibfreenect3.lowlevel.DumpPacketPipeline

Typed values
------------

.. autosummary::
   :toctree: generated

   pylibfreenect3.FrameType
   pylibfreenect3.FrameFormat
   pylibfreenect3.Stream
   pylibfreenect3.Pipeline
   pylibfreenect3.LoggerLevel
   pylibfreenect3.ColorSettingCommand
   pylibfreenect3.DeviceConfig
   pylibfreenect3.ColorCameraParams
   pylibfreenect3.IrCameraParams
   pylibfreenect3.LedSettings
   pylibfreenect3.ReplayCalibration

Core and logging queries
------------------------

.. autosummary::
   :toctree: generated

   pylibfreenect3.core_version
   pylibfreenect3.core_api_version
   pylibfreenect3.core_build_revision
   pylibfreenect3.compiled_pipelines
   pylibfreenect3.available_pipelines
   pylibfreenect3.default_logger_level
   pylibfreenect3.global_logger_level
   pylibfreenect3.logger_level_name
   pylibfreenect3.set_global_log_level

``Stream`` and ``Pipeline`` are string enums. Canonical string values remain
accepted at API boundaries and are normalized to their enum member.

Exceptions
----------

.. autosummary::
   :toctree: generated

   pylibfreenect3.FreenectError
   pylibfreenect3.BackendUnavailableError
   pylibfreenect3.DeviceOpenError
   pylibfreenect3.DeviceStateError
   pylibfreenect3.FrameTimeoutError
   pylibfreenect3.ReplayError
   pylibfreenect3.RecordingFormatError

Intentional exclusions
----------------------

Packet parser/processor hooks, decoder-thread callbacks into Python, and a
Python logger callback are excluded for thread safety. Native console logging
is available through ``set_global_log_level``.
