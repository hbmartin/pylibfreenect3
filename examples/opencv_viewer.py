#!/usr/bin/env python3
"""Display timestamp-aligned Kinect v2 color and depth with OpenCV."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pylibfreenect3 import AlignmentConfig, Camera, FrameTimeoutError, Pipeline

_MAX_CONSECUTIVE_TIMEOUTS = 3


def normalize_depth_for_display(
    depth_mm: npt.NDArray[np.float32],
) -> npt.NDArray[np.uint8]:
    """Safely scale positive finite depth into an 8-bit display image."""
    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    output = np.zeros(depth_mm.shape, dtype=np.uint8)
    if not np.any(valid):
        return output
    lower, upper = np.percentile(depth_mm[valid], (1.0, 99.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        output[valid] = 255
        return output
    scaled = np.clip((depth_mm[valid] - lower) / (upper - lower), 0.0, 1.0)
    output[valid] = np.asarray(np.rint((1.0 - scaled) * 255.0), dtype=np.uint8)
    return output


def _device(value: str) -> str | int:
    return int(value) if value.isdecimal() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=_device, help="device index or serial")
    parser.add_argument(
        "--pipeline",
        choices=tuple(value.value for value in Pipeline),
        default=Pipeline.AUTO.value,
    )
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="capture timeout in seconds"
    )
    parser.add_argument("--max-delta-ms", type=float, default=25.0)
    parser.add_argument("--queue-capacity", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.timeout < 0:
        raise ValueError("timeout must be non-negative")

    import cv2

    alignment = AlignmentConfig(
        max_delta=arguments.max_delta_ms / 1000.0,
        queue_capacity=arguments.queue_capacity,
    )
    bgr = np.empty((1080, 1920, 3), dtype=np.uint8)
    timeout_count = 0
    try:
        with Camera.open(
            device=arguments.device,
            pipeline=arguments.pipeline,
            alignment=alignment,
        ) as camera:
            while True:
                try:
                    with camera.capture(timeout=arguments.timeout) as frames:
                        frames.color.to_color(out=bgr)
                        depth_view = normalize_depth_for_display(
                            frames.depth.to_numpy()
                        )
                        depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_TURBO)
                        cv2.imshow("Kinect v2 color", cv2.resize(bgr, (960, 540)))
                        cv2.imshow("Kinect v2 depth", depth_view)
                    timeout_count = 0
                except FrameTimeoutError:
                    timeout_count += 1
                    if timeout_count >= _MAX_CONSECUTIVE_TIMEOUTS:
                        raise RuntimeError(
                            f"no frames received after {timeout_count} timeouts; "
                            "check the Kinect connection"
                        ) from None
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
