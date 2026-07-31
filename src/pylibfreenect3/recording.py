from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Full, Queue
from threading import Lock, Thread
from typing import Any, Literal

import numpy as np

from .api import (
    Camera,
    FrameSet,
)
from .errors import DeviceStateError, RecordingFormatError
from .lowlevel import DumpPacketPipeline
from .types import (
    ColorCameraParams,
    FrameFormat,
    FrameType,
    IrCameraParams,
    ReplayCalibration,
    Stream,
)

__all__ = ["RecordingBundle", "RecordingStats", "RecordingWriter"]

SCHEMA_VERSION = 1
_STOP_WRITER = object()


@dataclass(frozen=True, slots=True)
class RecordingStats:
    """Snapshot of frame-set throughput for a recording writer.

    ``failed`` counts frame sets that were not persisted because a background
    write failed; after the first failure every subsequent frame set is
    counted here without a write attempt.
    """

    submitted: int
    written: int
    dropped: int
    failed: int

    @property
    def pending(self) -> int:
        return max(0, self.submitted - self.written - self.dropped - self.failed)


@dataclass(frozen=True, slots=True)
class _RecordingPacket:
    stream: str
    relative: str
    timestamp: int
    sequence: int
    data: bytes


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
    """Atomically writes raw dump frames and calibration as schema-v1 bundle.

    A writer expects a single producer thread: ``write``, ``flush``, and
    ``close`` must not be called concurrently. Only the internal worker
    thread touches the filesystem in parallel with the producer.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        camera: Camera,
        *,
        queue_size: int = 0,
        overflow: Literal["block", "drop"] = "block",
    ) -> None:
        if (
            isinstance(queue_size, bool)
            or not isinstance(queue_size, int)
            or queue_size < 0
        ):
            raise ValueError("queue_size must be a non-negative integer")
        if overflow not in ("block", "drop"):
            raise ValueError("overflow must be 'block' or 'drop'")
        if queue_size == 0 and overflow != "block":
            raise ValueError("overflow='drop' requires queue_size greater than zero")
        self.path: Path = Path(path)
        self.camera: Camera = camera
        self.queue_size: int = queue_size
        self.overflow: Literal["block", "drop"] = overflow
        self._working: Path | None = None
        self._manifest: dict[str, Any] | None = None
        self._closed = False
        self._queue: Queue[tuple[_RecordingPacket, ...] | object] | None = None
        self._worker: Thread | None = None
        self._worker_error: BaseException | None = None
        self._stats_lock = Lock()
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._failed = 0
        self._scheduled_paths: set[str] = set()

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
        try:
            (self._working / "calibration").mkdir()
            (self._working / "frames").mkdir()

            calibration = {
                "color": asdict(self.camera.device.color_camera_params),
                "ir": asdict(self.camera.device.ir_camera_params),
                "p0_tables": self._write_array(
                    "calibration/p0.bin", pipeline.depth_p0_tables()
                ),
                "x_table": self._write_array(
                    "calibration/x.bin", pipeline.depth_x_table()
                ),
                "z_table": self._write_array(
                    "calibration/z.bin", pipeline.depth_z_table()
                ),
                "lookup_table": self._write_array(
                    "calibration/lookup.bin", pipeline.depth_lookup_table()
                ),
            }
            from .api import core_api_version, core_build_revision, core_version

            self._manifest = {
                "schema_version": SCHEMA_VERSION,
                "core": {
                    "version": core_version(),
                    "api_version": core_api_version(),
                    "revision": core_build_revision(),
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
            if self.queue_size:
                self._queue = Queue(maxsize=self.queue_size)
                worker = Thread(
                    target=self._worker_main,
                    name="pylibfreenect3-recording-writer",
                    daemon=True,
                )
                worker.start()
                self._worker = worker
        except BaseException:
            self.abort()
            raise
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

    @property
    def stats(self) -> RecordingStats:
        with self._stats_lock:
            return RecordingStats(
                submitted=self._submitted,
                written=self._written,
                dropped=self._dropped,
                failed=self._failed,
            )

    def _packets(self, frames: FrameSet) -> tuple[_RecordingPacket, ...]:
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
        packets: list[_RecordingPacket] = []
        for stream, frame_type in entries:
            frame = frames[frame_type]
            if frame.format is not FrameFormat.RAW:
                raise RecordingFormatError(
                    f"dump recording expected raw {stream} frame, "
                    f"got {frame.format.name}"
                )
            suffix = "jpg" if stream == "color" else "depth"
            data = frame.to_numpy().tobytes(order="C")
            relative = f"frames/{stream}_{frame.timestamp}_{frame.sequence}.{suffix}"
            if (
                relative in self._scheduled_paths
                or _safe_child(self._working, relative).exists()
            ):
                raise RecordingFormatError(
                    f"recording frame path is duplicated: {relative}"
                )
            packets.append(
                _RecordingPacket(
                    stream=stream,
                    relative=relative,
                    timestamp=frame.timestamp,
                    sequence=frame.sequence,
                    data=data,
                )
            )
        return tuple(packets)

    def _write_packets(self, packets: tuple[_RecordingPacket, ...]) -> None:
        if self._working is None or self._manifest is None:
            raise DeviceStateError("recording writer is not open")
        for packet in packets:
            destination = _safe_child(self._working, packet.relative)
            if destination.exists():
                raise RecordingFormatError(
                    f"recording frame path is duplicated: {packet.relative}"
                )
            destination.write_bytes(packet.data)
            self._manifest["frame_index"].append(
                {
                    "stream": packet.stream,
                    "path": packet.relative,
                    "timestamp": packet.timestamp,
                    "sequence": packet.sequence,
                    "size": len(packet.data),
                    "sha256": _sha256(packet.data),
                }
            )

    def _worker_main(self) -> None:
        if self._queue is None:
            return
        while True:
            item = self._queue.get()
            try:
                if item is _STOP_WRITER:
                    return
                with self._stats_lock:
                    previous_error = self._worker_error
                if previous_error is not None:
                    with self._stats_lock:
                        self._failed += 1
                    continue
                try:
                    self._write_packets(item)  # type: ignore[arg-type]
                except BaseException as error:
                    with self._stats_lock:
                        if self._worker_error is None:
                            self._worker_error = error
                        self._failed += 1
                else:
                    with self._stats_lock:
                        self._written += 1
            finally:
                self._queue.task_done()

    def _raise_worker_error(self) -> None:
        with self._stats_lock:
            error = self._worker_error
        if error is not None:
            raise RecordingFormatError("background recording write failed") from error

    def write(self, frames: FrameSet) -> bool:
        """Submit one frame set, returning false only when a full queue drops it."""
        self._raise_worker_error()
        packets = self._packets(frames)
        with self._stats_lock:
            self._submitted += 1
        if self._queue is None:
            try:
                self._write_packets(packets)
            except BaseException:
                with self._stats_lock:
                    self._failed += 1
                raise
            with self._stats_lock:
                self._written += 1
            self._scheduled_paths.update(packet.relative for packet in packets)
            return True

        if self.overflow == "drop":
            try:
                self._queue.put_nowait(packets)
            except Full:
                with self._stats_lock:
                    self._dropped += 1
                return False
        else:
            self._queue.put(packets)
        self._scheduled_paths.update(packet.relative for packet in packets)
        return True

    def flush(self) -> RecordingStats:
        """Wait for queued writes and surface any background failure."""
        if self._closed or self._working is None:
            raise DeviceStateError("recording writer is not open")
        if self._queue is not None:
            self._queue.join()
        self._raise_worker_error()
        return self.stats

    def capture(self, count: int, *, timeout: float | None = 2.0) -> RecordingStats:
        if count < 0:
            raise ValueError("count must be non-negative")
        for _ in range(count):
            with self.camera.capture(timeout=timeout) as frames:
                self.write(frames)
        return self.stats

    def _stop_worker(self) -> None:
        if self._queue is None or self._worker is None:
            return
        # The worker exits only after consuming the sentinel, so a put cannot
        # block for long while the worker is alive; the liveness check keeps
        # this loop from hanging on a full queue if the worker ever gains an
        # early exit path.
        while self._worker.is_alive():
            try:
                self._queue.put(_STOP_WRITER, timeout=1.0)
                break
            except Full:
                continue
        self._worker.join()
        self._worker = None
        self._queue = None

    def close(self) -> None:
        if self._closed:
            return
        if self._working is None or self._manifest is None:
            raise DeviceStateError("recording writer was never opened")
        try:
            self.flush()
            self._stop_worker()
            final_stats = self.stats
            self._manifest["recording_stats"] = {
                "submitted": final_stats.submitted,
                "written": final_stats.written,
                "dropped": final_stats.dropped,
                "failed": final_stats.failed,
            }
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
        self._stop_worker()
        if self._working is not None:
            shutil.rmtree(self._working, ignore_errors=True)
        self._working = None
        self._manifest = None
        self._closed = True


class RecordingBundle:
    def __init__(
        self, root: Path, manifest: dict[str, Any], calibration: ReplayCalibration
    ) -> None:
        self.root: Path = root
        self.manifest: dict[str, Any] = manifest
        self.calibration: ReplayCalibration = calibration
        self.streams: tuple[Stream, ...] = tuple(
            Stream(str(value)) for value in manifest["enabled_streams"]
        )
        self._verified: set[Path] = set()

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
            or any(stream not in Stream for stream in streams)
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
            file_streams.add(Stream.COLOR)
        if selected & {Stream.IR, Stream.DEPTH}:
            file_streams.add("depth")
        recorded_file_streams = set()
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
