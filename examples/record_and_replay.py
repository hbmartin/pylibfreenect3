"""Record canonical raw packets, then replay one decoded frame."""

from __future__ import annotations

import argparse
from pathlib import Path

from pylibfreenect3 import Camera, RecordingWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--depth-frames", type=int, default=100)
    parser.add_argument("--pipeline", default="auto")
    arguments = parser.parse_args()

    with RecordingWriter(arguments.path) as writer:
        stats = writer.capture(depth_frames=arguments.depth_frames, timeout=30.0)
    print("recorded:", stats)

    with (
        Camera.open_recording(arguments.path, pipeline=arguments.pipeline) as camera,
        camera.capture(timeout=5.0) as frames,
    ):
        print("color:", frames.color.to_numpy().shape)
        print("depth:", frames.depth.to_numpy().shape)


if __name__ == "__main__":
    main()
