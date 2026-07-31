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
   pylibfreenect3.Freenect2
   pylibfreenect3.Freenect2Replay
   pylibfreenect3.Device
   pylibfreenect3.SyncFrameListener
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

   pylibfreenect3.PacketPipeline
   pylibfreenect3.CpuPacketPipeline
   pylibfreenect3.MetalPacketPipeline
   pylibfreenect3.OpenGLPacketPipeline
   pylibfreenect3.OpenCLPacketPipeline
   pylibfreenect3.OpenCLKdePacketPipeline
   pylibfreenect3.CudaPacketPipeline
   pylibfreenect3.CudaKdePacketPipeline
   pylibfreenect3.DumpPacketPipeline

Typed values
------------

.. autosummary::
   :toctree: generated

   pylibfreenect3.FrameType
   pylibfreenect3.FrameFormat
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

Stream names
------------

``STREAM_NAMES`` is the immutable mapping from public stream names to
``FrameType`` values.

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
