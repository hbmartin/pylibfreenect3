class FreenectError(RuntimeError):
    """Base exception raised by pylibfreenect3."""


class BackendUnavailableError(FreenectError):
    """Requested packet pipeline is not compiled or usable."""


class DeviceOpenError(FreenectError):
    """A physical or replay device could not be opened."""


class DeviceStateError(FreenectError):
    """An operation is invalid for the current device/frame state."""


class WorkspaceStateError(FreenectError):
    """A registration workspace was used before the required apply() call."""


class FrameTimeoutError(FreenectError, TimeoutError):
    """No synchronized frame set arrived before the timeout."""


class ReplayError(FreenectError):
    """Raw frame replay failed."""


class RecordingFormatError(ReplayError, ValueError):
    """A recording bundle is incomplete, unsafe, or incompatible."""
