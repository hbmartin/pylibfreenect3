from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .api import Camera, DumpPacketPipeline, FrameFormat, FrameSet, FrameType
from .errors import DeviceStateError, RecordingFormatError
from .types import ColorCameraParams, IrCameraParams, ReplayCalibration


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


class RecordingWriter:
    """Atomically writes raw dump frames and calibration as schema-v1 bundle."""

    def __init__(self, path: str | os.PathLike[str], camera: Camera) -> None:
        self.path = Path(path)
        self.camera = camera
        self._working: Path | None = None
        self._manifest: dict[str, Any] | None = None
        self._closed = False

    def __enter__(self) -> RecordingWriter:
        if self.path.exists():
            raise FileExistsError(f"recording target already exists: {self.path}")
        pipeline = self.camera.device.pipeline
        if not isinstance(pipeline, DumpPacketPipeline):
            raise DeviceStateError(
                "RecordingWriter requires Camera.open(pipeline='dump')"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._working = Path(
            tempfile.mkdtemp(prefix=f".{self.path.name}.partial-", dir=self.path.parent)
        )
        (self._working / "calibration").mkdir()
        (self._working / "frames").mkdir()

        calibration = {
            "color": asdict(self.camera.device.color_camera_params),
            "ir": asdict(self.camera.device.ir_camera_params),
            "p0_tables": self._write_array(
                "calibration/p0.bin", pipeline.depth_p0_tables()
            ),
            "x_table": self._write_array("calibration/x.bin", pipeline.depth_x_table()),
            "z_table": self._write_array("calibration/z.bin", pipeline.depth_z_table()),
            "lookup_table": self._write_array(
                "calibration/lookup.bin", pipeline.depth_lookup_table()
            ),
        }
        from .api import core_api_version, core_revision, core_version

        self._manifest = {
            "schema_version": SCHEMA_VERSION,
            "core": {
                "version": core_version(),
                "api_version": core_api_version(),
                "revision": core_revision(),
            },
            "device": {
                "serial": self.camera.device.serial_number,
                "firmware": self.camera.device.firmware_version,
                "pipeline": self.camera.device.pipeline_name,
            },
            "enabled_streams": list(self.camera.streams),
            "calibration": calibration,
            "frame_index": [],
        }
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def _write_array(
        self, relative: str, value: np.ndarray[Any, Any]
    ) -> dict[str, Any]:
        if self._working is None:
            raise DeviceStateError("recording writer is not open")
        array = np.ascontiguousarray(value)
        data = array.tobytes(order="C")
        _safe_child(self._working, relative).write_bytes(data)
        return {
            "path": relative,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "byte_length": len(data),
            "sha256": _sha256(data),
        }

    def write(self, frames: FrameSet) -> None:
        if self._closed or self._working is None or self._manifest is None:
            raise DeviceStateError("recording writer is not open")
        entries: list[tuple[str, FrameType]] = []
        if "color" in self.camera.streams and FrameType.COLOR in frames:
            entries.append(("color", FrameType.COLOR))
        if "ir" in self.camera.streams or "depth" in self.camera.streams:
            if FrameType.DEPTH in frames:
                entries.append(("depth", FrameType.DEPTH))
            elif FrameType.IR in frames:
                entries.append(("depth", FrameType.IR))
        for stream, frame_type in entries:
            frame = frames[frame_type]
            if frame.format is not FrameFormat.RAW:
                raise RecordingFormatError(
                    f"dump recording expected raw {stream} frame, got {frame.format.name}"
                )
            if stream == "color":
                suffix = "jpg"
            else:
                suffix = "depth"
            data = frame.to_numpy(copy=True).tobytes(order="C")
            relative = f"frames/{stream}_{frame.timestamp}_{frame.sequence}.{suffix}"
            destination = _safe_child(self._working, relative)
            if destination.exists():
                raise RecordingFormatError(
                    f"recording frame path is duplicated: {relative}"
                )
            destination.write_bytes(data)
            self._manifest["frame_index"].append(
                {
                    "stream": stream,
                    "path": relative,
                    "timestamp": frame.timestamp,
                    "sequence": frame.sequence,
                    "size": len(data),
                    "sha256": _sha256(data),
                }
            )

    def capture(self, count: int, *, timeout: float | None = 2.0) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        for _ in range(count):
            with self.camera.capture(timeout=timeout) as frames:
                self.write(frames)

    def close(self) -> None:
        if self._closed:
            return
        if self._working is None or self._manifest is None:
            raise DeviceStateError("recording writer was never opened")
        try:
            manifest_data = (
                json.dumps(
                    self._manifest, indent=2, sort_keys=True, separators=(",", ": ")
                ).encode("utf-8")
                + b"\n"
            )
            (self._working / "manifest.json").write_bytes(manifest_data)
            if self.path.exists():
                raise FileExistsError(
                    f"recording target appeared while writing: {self.path}"
                )
            os.replace(self._working, self.path)
        except Exception:
            self.abort()
            raise
        self._working = None
        self._manifest = None
        self._closed = True

    def abort(self) -> None:
        """Discard an incomplete bundle owned by this writer."""
        if self._working is not None:
            shutil.rmtree(self._working, ignore_errors=True)
        self._working = None
        self._manifest = None
        self._closed = True


class RecordingBundle:
    def __init__(
        self, root: Path, manifest: dict[str, Any], calibration: ReplayCalibration
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.calibration = calibration
        self.streams = tuple(str(value) for value in manifest["enabled_streams"])

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> RecordingBundle:
        root = Path(path).resolve()
        try:
            manifest = json.loads((root / "manifest.json").read_text("utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise RecordingFormatError(
                f"cannot read recording manifest at {root}"
            ) from error
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise RecordingFormatError(
                f"unsupported recording schema: {manifest.get('schema_version')!r}"
            )
        core = manifest.get("core")
        if (
            not isinstance(core, dict)
            or not str(core.get("version", "")).startswith("0.3.")
            or core.get("api_version") != 3
            or not isinstance(core.get("revision"), str)
        ):
            raise RecordingFormatError(
                "recording was not produced by a compatible 0.3 core"
            )
        device = manifest.get("device")
        if not isinstance(device, dict) or any(
            not isinstance(device.get(key), str)
            for key in ("serial", "firmware", "pipeline")
        ):
            raise RecordingFormatError("recording device metadata is incomplete")
        if (
            not isinstance(manifest.get("frame_index"), list)
            or not manifest["frame_index"]
        ):
            raise RecordingFormatError("recording contains no indexed frames")
        streams = manifest.get("enabled_streams")
        if (
            not isinstance(streams, list)
            or not streams
            or len(streams) != len(set(streams))
            or any(stream not in FrameSet._NAMES for stream in streams)
        ):
            raise RecordingFormatError("recording has an invalid stream list")
        calibration_data = manifest.get("calibration")
        if not isinstance(calibration_data, dict):
            raise RecordingFormatError("recording calibration is missing")
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

    def frame_paths(self, streams: Iterable[str]) -> list[str]:
        selected = set(streams)
        if not selected or any(stream not in FrameSet._NAMES for stream in selected):
            raise RecordingFormatError("requested replay streams are invalid")
        if not selected <= set(self.streams):
            raise RecordingFormatError("requested replay streams were not recorded")
        file_streams: set[str] = set()
        if "color" in selected:
            file_streams.add("color")
        if selected & {"ir", "depth"}:
            file_streams.add("depth")
        recorded_file_streams = set()
        if "color" in self.streams:
            recorded_file_streams.add("color")
        if set(self.streams) & {"ir", "depth"}:
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
                data = path.read_bytes()
                if len(data) != int(entry["size"]) or _sha256(data) != entry["sha256"]:
                    raise RecordingFormatError(f"frame failed integrity check: {path}")
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
