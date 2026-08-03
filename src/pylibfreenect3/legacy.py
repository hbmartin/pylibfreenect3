"""Reader for pylibfreenect3 1.x Python-specific recording bundles."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np

from .errors import RecordingFormatError
from .types import (
    ColorCameraParams,
    IrCameraParams,
    ReplayCalibration,
    Stream,
)

__all__ = ["RecordingBundle"]

SCHEMA_VERSION = 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if not relative or relative_path.is_absolute():
        raise RecordingFormatError(f"recording path must be relative: {relative!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise RecordingFormatError(
            f"recording path escapes its bundle: {relative!r}"
        ) from error
    return candidate


class RecordingBundle:
    """Validated reader for the non-canonical bundle written by version 1.x."""

    def __init__(
        self, root: Path, manifest: dict[str, Any], calibration: ReplayCalibration
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.calibration = calibration
        self.streams: tuple[Stream, ...] = tuple(
            Stream(str(value)) for value in manifest["enabled_streams"]
        )
        self._verified: set[Path] = set()

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> RecordingBundle:
        root = Path(path).resolve()
        try:
            loaded: Any = json.loads((root / "manifest.json").read_text("utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise RecordingFormatError(
                f"cannot read recording manifest at {root}"
            ) from error
        if not isinstance(loaded, dict):
            raise RecordingFormatError("recording manifest must be a JSON object")
        manifest = cast(dict[str, Any], loaded)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise RecordingFormatError(
                f"unsupported recording schema: {manifest.get('schema_version')!r}"
            )
        core_value = manifest.get("core")
        if not isinstance(core_value, dict):
            raise RecordingFormatError(
                "recording was not produced by a compatible 0.3 core"
            )
        core = cast(dict[str, Any], core_value)
        if (
            not str(core.get("version", "")).startswith("0.3.")
            or core.get("api_version") != 3
            or not isinstance(core.get("revision"), str)
        ):
            raise RecordingFormatError(
                "recording was not produced by a compatible 0.3 core"
            )
        device_value = manifest.get("device")
        if not isinstance(device_value, dict):
            raise RecordingFormatError("recording device metadata is incomplete")
        device = cast(dict[str, Any], device_value)
        if any(
            not isinstance(device.get(key), str)
            for key in ("serial", "firmware", "pipeline")
        ):
            raise RecordingFormatError("recording device metadata is incomplete")
        if (
            not isinstance(manifest.get("frame_index"), list)
            or not manifest["frame_index"]
        ):
            raise RecordingFormatError("recording contains no indexed frames")
        streams_value = manifest.get("enabled_streams")
        if not isinstance(streams_value, list):
            raise RecordingFormatError("recording has an invalid stream list")
        streams = cast(list[Any], streams_value)  # type: ignore[redundant-cast]
        if (
            not streams
            or len(streams) != len(set(streams))
            or any(stream not in Stream for stream in streams)
        ):
            raise RecordingFormatError("recording has an invalid stream list")
        calibration_data = manifest.get("calibration")
        if not isinstance(calibration_data, dict):
            raise RecordingFormatError("recording calibration is missing")
        calibration_data = cast(dict[str, Any], calibration_data)
        try:
            calibration = ReplayCalibration(
                color=ColorCameraParams(**calibration_data["color"]),
                ir=IrCameraParams(**calibration_data["ir"]),
                p0_tables=cls._read_array(
                    root, calibration_data["p0_tables"], np.uint8
                ),
                x_table=cls._read_array(root, calibration_data["x_table"], np.float32),
                z_table=cls._read_array(root, calibration_data["z_table"], np.float32),
                lookup_table=cls._read_array(
                    root, calibration_data["lookup_table"], np.int16
                ),
            )
        except RecordingFormatError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise RecordingFormatError(
                "recording calibration metadata is invalid"
            ) from error

        bundle = cls(root, manifest, calibration)
        bundle.frame_paths(bundle.streams)
        return bundle

    @staticmethod
    def _read_array(
        root: Path, descriptor: dict[str, Any], expected_dtype: Any
    ) -> np.ndarray[Any, Any]:
        try:
            path = _safe_child(root, descriptor["path"])
            data = path.read_bytes()
            if (
                len(data) != int(descriptor["byte_length"])
                or _sha256(data) != descriptor["sha256"]
            ):
                raise RecordingFormatError(
                    f"calibration sidecar failed integrity check: {path}"
                )
            dtype = np.dtype(descriptor["dtype"])
            if dtype != np.dtype(expected_dtype):
                raise RecordingFormatError(
                    f"calibration sidecar has the wrong dtype: {path}"
                )
            shape = tuple(int(value) for value in descriptor["shape"])
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if expected != len(data):
                raise RecordingFormatError(
                    f"calibration sidecar shape is inconsistent: {path}"
                )
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        except (KeyError, OSError, TypeError, ValueError) as error:
            if isinstance(error, RecordingFormatError):
                raise
            raise RecordingFormatError(
                "invalid calibration sidecar descriptor"
            ) from error

    def frame_paths(self, streams: Iterable[str | Stream]) -> list[str]:
        try:
            selected = {Stream(str(stream).lower()) for stream in streams}
        except ValueError as error:
            raise RecordingFormatError(
                "requested replay streams are invalid"
            ) from error
        if not selected:
            raise RecordingFormatError("requested replay streams are invalid")
        if not selected <= set(self.streams):
            raise RecordingFormatError("requested replay streams were not recorded")
        file_streams: set[str] = set()
        if Stream.COLOR in selected:
            file_streams.add("color")
        if selected & {Stream.IR, Stream.DEPTH}:
            file_streams.add("depth")
        recorded_file_streams: set[str] = set()
        if Stream.COLOR in self.streams:
            recorded_file_streams.add("color")
        if set(self.streams) & {Stream.IR, Stream.DEPTH}:
            recorded_file_streams.add("depth")
        paths: list[str] = []
        found_streams: set[str] = set()
        seen: set[Path] = set()
        for entry in self.manifest["frame_index"]:
            try:
                stream = entry["stream"]
                if stream not in ("color", "depth"):
                    raise RecordingFormatError("frame index contains an invalid stream")
                if stream not in recorded_file_streams:
                    raise RecordingFormatError(
                        "frame index contains a stream not enabled by the manifest"
                    )
                path = _safe_child(self.root, entry["path"])
                if path in seen:
                    raise RecordingFormatError(f"frame index repeats a path: {path}")
                seen.add(path)
                if path not in self._verified:
                    data = path.read_bytes()
                    if (
                        len(data) != int(entry["size"])
                        or _sha256(data) != entry["sha256"]
                    ):
                        raise RecordingFormatError(
                            f"frame failed integrity check: {path}"
                        )
                    self._verified.add(path)
                if not isinstance(entry["timestamp"], int) or not isinstance(
                    entry["sequence"], int
                ):
                    raise RecordingFormatError(
                        "frame timestamp/sequence must be integers"
                    )
                expected_suffix = ".jpg" if stream == "color" else ".depth"
                if path.suffix != expected_suffix:
                    raise RecordingFormatError(
                        f"frame has an incompatible suffix: {path}"
                    )
                if stream in file_streams:
                    paths.append(str(path))
                    found_streams.add(stream)
            except (KeyError, OSError, TypeError, ValueError) as error:
                if isinstance(error, RecordingFormatError):
                    raise
                raise RecordingFormatError("invalid frame index entry") from error
        missing_streams = file_streams - found_streams
        if missing_streams:
            missing = ", ".join(sorted(missing_streams))
            raise RecordingFormatError(
                f"recording contains no frames for the selected streams: {missing}"
            )
        return paths
