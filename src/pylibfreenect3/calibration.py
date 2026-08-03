from __future__ import annotations

from os import PathLike, fspath
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _native as _native_module
from .types import (
    CalibrationProvenance,
    CalibrationQualityMetrics,
    DepthCorrectionProfile,
    FrameFormat,
    ProjectiveCameraModel,
    ProjectiveRegistrationOptions,
    RigidTransform,
    _dataclass_from_mapping,
)

if TYPE_CHECKING:
    from .api import Device, Frame

__all__ = ["CalibrationProfile", "ProjectiveRegistration"]

_DEFAULT_PROJECTIVE_OPTIONS = ProjectiveRegistrationOptions()


class CalibrationProfile:
    """Validated, read-only native calibration profile."""

    _native: _native_module.NativeCalibrationProfileHandle

    def __init__(self) -> None:
        raise TypeError(
            "CalibrationProfile objects must be loaded or obtained from a recording"
        )

    @classmethod
    def _from_native(
        cls, native: _native_module.NativeCalibrationProfileHandle
    ) -> CalibrationProfile:
        profile = object.__new__(cls)
        profile._native = native
        return profile

    @classmethod
    def load(cls, path: str | PathLike[str]) -> CalibrationProfile:
        return cls._from_native(
            _native_module.NativeCalibrationProfileHandle.load(fspath(path))
        )

    def save(self, path: str | PathLike[str]) -> None:
        self._native.save(fspath(path))

    def check_device(
        self, device: Device, allow_serial_mismatch: bool = False
    ) -> str | None:
        if not isinstance(allow_serial_mismatch, bool):
            raise TypeError("allow_serial_mismatch must be bool")
        return self._native.check_device(
            device.serial_number,
            device.firmware_version,
            allow_serial_mismatch,
        )

    @property
    def schema_version(self) -> int:
        return self._native.schema_version

    @property
    def serial(self) -> str:
        return self._native.serial

    @property
    def firmware(self) -> str:
        return self._native.firmware

    @property
    def color_camera(self) -> ProjectiveCameraModel:
        return _dataclass_from_mapping(
            ProjectiveCameraModel, self._native.color_camera()
        )

    @property
    def ir_camera(self) -> ProjectiveCameraModel:
        return _dataclass_from_mapping(ProjectiveCameraModel, self._native.ir_camera())

    @property
    def depth_to_color(self) -> RigidTransform:
        return _dataclass_from_mapping(RigidTransform, self._native.depth_to_color())

    @property
    def depth_correction(self) -> DepthCorrectionProfile | None:
        values = self._native.depth_correction()
        return (
            None
            if values is None
            else _dataclass_from_mapping(DepthCorrectionProfile, values)
        )

    @property
    def quality_metrics(self) -> CalibrationQualityMetrics | None:
        values = self._native.quality_metrics()
        return (
            None
            if values is None
            else _dataclass_from_mapping(CalibrationQualityMetrics, values)
        )

    @property
    def provenance(self) -> CalibrationProvenance:
        return _dataclass_from_mapping(CalibrationProvenance, self._native.provenance())


class ProjectiveRegistration:
    """Conventional profile-based depth registration."""

    def __init__(
        self,
        profile: CalibrationProfile,
        target: ProjectiveCameraModel | None = None,
        options: ProjectiveRegistrationOptions = _DEFAULT_PROJECTIVE_OPTIONS,
    ) -> None:
        selected_target = profile.color_camera if target is None else target
        self._native = _native_module.NativeProjectiveRegistrationHandle(
            profile._native, selected_target, options
        )
        self._target: ProjectiveCameraModel = _dataclass_from_mapping(
            ProjectiveCameraModel, self._native.target_camera()
        )
        self._options: ProjectiveRegistrationOptions = _dataclass_from_mapping(
            ProjectiveRegistrationOptions, self._native.options()
        )
        self._source: ProjectiveCameraModel = profile.ir_camera
        self._active_outputs: list[np.ndarray[Any, Any]] = []
        self._active_outputs_lock = Lock()

    @property
    def target(self) -> ProjectiveCameraModel:
        return self._target

    @property
    def options(self) -> ProjectiveRegistrationOptions:
        return self._options

    @property
    def source(self) -> ProjectiveCameraModel:
        return self._source

    def apply(self, depth: Frame, *, out: Frame | None = None) -> Frame:
        from .api import Frame

        if (
            depth.width != self.source.width
            or depth.height != self.source.height
            or depth.bytes_per_pixel != 4
            or depth.format is not FrameFormat.FLOAT
        ):
            raise ValueError(
                "depth must be a float frame matching the profile IR camera"
            )
        result = (
            Frame.allocate(
                self.target.width,
                self.target.height,
                4,
                frame_format=FrameFormat.FLOAT,
            )
            if out is None
            else out
        )
        if (
            result.width != self.target.width
            or result.height != self.target.height
            or result.bytes_per_pixel != 4
            or result.format is not FrameFormat.FLOAT
        ):
            raise ValueError("out must be a float frame matching the target camera")
        depth_array = depth.to_numpy()
        result_array = result.to_numpy()
        if np.shares_memory(depth_array, result_array):
            raise ValueError("projective registration input and output must not alias")
        with self._active_outputs_lock:
            if any(
                np.shares_memory(result_array, active)
                for active in self._active_outputs
            ):
                raise ValueError(
                    "concurrent projective registration calls require distinct outputs"
                )
            self._active_outputs.append(result_array)
        try:
            self._native.apply(depth._native, result._native)
        finally:
            with self._active_outputs_lock:
                for index, active in enumerate(self._active_outputs):
                    if active is result_array:
                        del self._active_outputs[index]
                        break
        return result
