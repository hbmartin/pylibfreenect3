"""Register live depth with a canonical conventional calibration profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from pylibfreenect3 import CalibrationProfile, Camera, ProjectiveRegistration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    arguments = parser.parse_args()

    profile = CalibrationProfile.load(arguments.profile)
    registration = ProjectiveRegistration(profile)
    with Camera.open(streams=("depth",)) as camera:
        warning = profile.check_device(camera.device)
        if warning:
            print("warning:", warning)
        with camera.capture(timeout=2.0) as frames:
            target = registration.apply(frames.depth)
            print(target.to_numpy().shape, target.to_numpy().dtype)


if __name__ == "__main__":
    main()
