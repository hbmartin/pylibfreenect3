# MediaPipe Kinect exercise-pose demo

This example runs MediaPipe Pose Landmarker on the Kinect v2 color
stream and uses libfreenect2 registration to attach camera-relative metric XYZ
coordinates to visible landmarks. The window shows the RGB skeleton, measured
front/side projections, and exercise joint angles. It does not classify posture
or provide medical or safety guidance.

## Set up

The prebuilt MediaPipe wheel officially targets Python 3.12. On Apple Silicon,
use a native arm64 Python and a working `pylibfreenect3` installation.

```sh
examples/mediapipe_pose/setup_demo.sh
```

`setup_demo.sh` creates `examples/mediapipe_pose/.venv`, installs
`mediapipe==0.10.35`, `numpy==2.5.1`, and
`opencv-contrib-python==4.13.0.92`, and downloads revision 1 of the official
Full FP16 model. The download is accepted only when its SHA-256 is
`5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`.

## Run

```sh
examples/mediapipe_pose/.venv/bin/python examples/mediapipe_pose/pose_demo.py
```

Capture, registration, color conversion, and metric lifting use the
`pylibfreenect3` API directly. Useful options include:

```text
--pipeline auto|cpu|metal|opengl|opencl|opencl_kde|cuda|cuda_kde
--serial KINECT_SERIAL
--model /path/to/another_pose_landmarker.task
--output captures/my_session
--visibility 0.6
--presence 0.6
--depth-radius 8
--depth-fallback-radius 20
--cluster-span-mm 150
--timeout-ms 10000
--max-delta-ms 25
--queue-capacity 8
```

The `dump` pipeline is intentionally excluded because this demo requires
decoded color and depth frames.

Controls are `Q` or Escape to quit, Space to pause, `R` to toggle JSONL session
recording, and `S` to save an annotated PNG plus matching JSON record. Recording
is opt-in. Outputs default to an ignored `captures/mediapipe_pose_<timestamp>`
directory.

Green landmarks have Kinect-measured XYZ. Orange landmarks are model-only, and
gray landmarks are below the confidence threshold. An angle is labeled
`kinect` only when all required joints have measured depth. Otherwise it is
computed entirely from MediaPipe world landmarks and labeled `model`; the demo
never mixes the coordinate systems within one angle.

For full-body exercise measurements, keep one person centered with their head
and feet visible. Kinect RGB and depth exposure are not simultaneous; metric
capture is timestamp-aligned and rejected when the device timestamps differ by
more than 25 ms by default.

## Tests

The pure Python logic does not import MediaPipe or require a Kinect:

```sh
python -m pytest tests/test_examples.py
```

Native reverse-map and depth-selection tests are part of the regular Python and
core CTest suites.
