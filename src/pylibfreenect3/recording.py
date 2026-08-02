from __future__ import annotations

import math
import os
import shutil
import time
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from warnings import warn

from . import _native
from .api import Context, Device
from .calibration import CalibrationProfile
from .errors import DeviceStateError, FrameTimeoutError, RecordingError
from .types import Pipeline, Stream, _dataclass_from_mapping

__all__ = ["RecordingStats", "RecordingWriter"]


@dataclass(frozen=True, slots=True)
class RecordingStats:
    """Frame-oriented snapshot from the native canonical recording writer."""

    written_frames: int
    written_color_frames: int
    written_depth_frames: int
    dropped_frames: int
    written_bytes: int


class RecordingWriter:
    """Own a live Kinect and write a canonical native recording directory."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        device: str | int | None = None,
        streams: Iterable[str | Stream] = (Stream.COLOR, Stream.DEPTH),
        queue_capacity: int = 32,
        calibration_profile: CalibrationProfile | None = None,
        allow_serial_mismatch: bool = False,
    ) -> None:
        if (
            isinstance(queue_capacity, bool)
            or not isinstance(queue_capacity, int)
            or queue_capacity <= 0
        ):
            raise ValueError("queue_capacity must be a positive integer")
        if not isinstance(allow_serial_mismatch, bool):
            raise TypeError("allow_serial_mismatch must be bool")
        if calibration_profile is not None and not isinstance(
            calibration_profile, CalibrationProfile
        ):
            raise TypeError("calibration_profile must be a CalibrationProfile")
        try:
            selected = tuple(
                dict.fromkeys(Stream(str(stream).lower()) for stream in streams)
            )
        except ValueError as error:
            raise ValueError(
                "recording streams must contain color and/or depth"
            ) from error
        if not selected or any(
            stream not in (Stream.COLOR, Stream.DEPTH) for stream in selected
        ):
            raise ValueError("recording streams must contain color and/or depth")

        self.path: Path = Path(path)
        self.device_selector = device
        self.streams = selected
        self.queue_capacity = queue_capacity
        self.calibration_profile = calibration_profile
        self.allow_serial_mismatch = allow_serial_mismatch
        self._context: Context | None = None
        self._device: Device | None = None
        self._native: _native.NativeRecordingWriterHandle | None = None
        self._closed = False
        self._entered = False
        self._final_stats = RecordingStats(0, 0, 0, 0, 0)

    def __enter__(self) -> RecordingWriter:
        if self._entered:
            raise DeviceStateError("recording writer cannot be entered more than once")
        if self.path.exists():
            raise FileExistsError(f"recording target already exists: {self.path}")
        self._entered = True
        try:
            native = _native.NativeRecordingWriterHandle(
                os.fspath(self.path), self.queue_capacity
            )
            self._native = native
            context = Context()
            self._context = context
            opened = context.open_device(
                self.device_selector,
                pipeline=Pipeline.DUMP,
            )
            self._device = opened
            if Stream.COLOR in self.streams:
                opened._native.set_color_listener(native)
            if Stream.DEPTH in self.streams:
                opened._native.set_depth_listener(native)
            opened.start(
                rgb=Stream.COLOR in self.streams,
                depth=Stream.DEPTH in self.streams,
            )
            native.publish_calibration(opened._native)
            if self.calibration_profile is not None:
                warning = self.calibration_profile.check_device(
                    opened,
                    allow_serial_mismatch=self.allow_serial_mismatch,
                )
                if warning is not None:
                    warn(warning, UserWarning, stacklevel=2)
                native.set_calibration_profile(
                    self.calibration_profile._native,
                    self.allow_serial_mismatch,
                )
        except BaseException:
            self._cleanup_failed_entry()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if isinstance(exc, BaseException):
                exc.add_note(f"recording cleanup also failed: {cleanup_error}")
                return
            raise

    def _cleanup_failed_entry(self) -> None:
        if self._device is not None:
            with suppress(BaseException):
                self._device.close()
        if self._native is not None:
            with suppress(BaseException):
                self._native.close()
        self._device = None
        self._context = None
        self._native = None
        self._closed = True
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)

    def _require_open(self) -> _native.NativeRecordingWriterHandle:
        if self._closed or self._native is None or self._device is None:
            raise DeviceStateError("recording writer is not open")
        return self._native

    @property
    def stats(self) -> RecordingStats:
        if self._native is None:
            return self._final_stats
        return _dataclass_from_mapping(RecordingStats, self._native.statistics())

    def capture(
        self,
        *,
        depth_frames: int | None = None,
        color_frames: int | None = None,
        duration: float | None = None,
        timeout: float | None = None,
    ) -> RecordingStats:
        native = self._require_open()
        bounds = (depth_frames, color_frames, duration)
        if sum(value is not None for value in bounds) != 1:
            raise ValueError(
                "capture requires exactly one of depth_frames, color_frames, or duration"
            )
        if depth_frames is not None:
            self._validate_frame_bound(depth_frames, Stream.DEPTH, "depth_frames")
        if color_frames is not None:
            self._validate_frame_bound(color_frames, Stream.COLOR, "color_frames")
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and non-negative")

        baseline = self.stats
        deadline: float | None
        if duration is not None:
            if timeout is not None:
                raise ValueError("timeout cannot be combined with a duration bound")
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("duration must be finite and positive")
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                self._check_native_open(native)
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
            return self.stats

        counter = (
            "written_depth_frames"
            if depth_frames is not None
            else "written_color_frames"
        )
        requested = depth_frames if depth_frames is not None else color_frames
        if requested is None:  # pragma: no cover - guarded above
            raise AssertionError("missing frame bound")
        target = getattr(baseline, counter) + requested
        deadline = None if timeout is None else time.monotonic() + timeout
        while getattr(self.stats, counter) < target:
            self._check_native_open(native)
            if deadline is not None and time.monotonic() >= deadline:
                raise FrameTimeoutError(
                    f"timed out waiting for {requested} additional {counter}"
                )
            time.sleep(0.01)
        return self.stats

    def _validate_frame_bound(self, count: int, stream: Stream, label: str) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{label} must be a positive integer")
        if stream not in self.streams:
            raise ValueError(f"{label} requires the {stream.value} recording stream")

    @staticmethod
    def _check_native_open(native: _native.NativeRecordingWriterHandle) -> None:
        if not native.is_open:
            raise RecordingError(native.last_error or "recording writer stopped")

    def close(self) -> None:
        if self._closed:
            return
        if not self._entered or self._native is None:
            raise DeviceStateError("recording writer was never opened")
        device_error: BaseException | None = None
        if self._device is not None:
            try:
                self._device.close()
            except BaseException as error:
                device_error = error
        try:
            self._native.close()
            self._final_stats = _dataclass_from_mapping(
                RecordingStats, self._native.statistics()
            )
        finally:
            self._device = None
            self._context = None
            self._closed = True
        if device_error is not None:
            raise RecordingError(
                f"recording finalized, but device shutdown failed: {device_error}"
            ) from device_error
