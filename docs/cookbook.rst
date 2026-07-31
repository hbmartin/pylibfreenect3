OpenCV and registration cookbook
================================

This page collects the conversions that are easy to get subtly wrong when
moving between Kinect frames, NumPy, OpenCV, and saved data.

Color frames and OpenCV
-----------------------

A decoded color frame is a ``uint8`` array with shape ``(1080, 1920, 4)``.
The fourth byte is padding, not meaningful alpha. Inspect the frame format
before selecting OpenCV's channel conversion::

   import cv2
   import numpy as np
   from pylibfreenect3 import Camera, FrameFormat

   with Camera.open(streams=("color",)) as camera:
       with camera.capture(timeout=2.0) as frames:
           pixels = frames.color.to_numpy()
           if frames.color.format is FrameFormat.BGRX:
               bgr = np.ascontiguousarray(pixels[..., :3])
           elif frames.color.format is FrameFormat.RGBX:
               bgr = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
           else:
               raise ValueError(f"unexpected color format: {frames.color.format}")
           cv2.imshow("color", bgr)
           cv2.waitKey(1)

Do not unconditionally use ``COLOR_RGBA2BGR``: a BGRX source is already in
OpenCV's channel order. To flip either a color or depth image horizontally,
use ``np.ascontiguousarray(image[:, ::-1])`` or ``cv2.flip(image, 1)``. Flipping
changes image coordinates, so apply it consistently to every stream and any
saved calibration-dependent coordinates.

Depth units, display, and storage
---------------------------------

Decoded depth is a ``float32`` array with shape ``(424, 512)``. Values are
millimetres; zero and non-finite values mean that no valid measurement is
available. Scaling to 8-bit is suitable only for display::

   import cv2
   import numpy as np

   depth_mm = frames.depth.to_numpy()
   display = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
   display = np.clip(display / 4500.0, 0.0, 1.0)
   cv2.imshow("depth", display)

An 8-bit video cannot recover the original millimetre values. Use ``np.save``
for processed depth arrays, or a ``RecordingWriter`` bundle for checksummed raw
packets and calibration. Convert to metres with ``depth_m = depth_mm / 1000.0``.

The optional registration ``big_depth`` output is also ``float32`` depth in
millimetres. Calling ``to_numpy()`` with a ``uint8`` conversion would discard
nearly all of its measurement precision; normalize a separate copy for display
instead.

Registration outputs
--------------------

One registration call can produce four related outputs::

   from pylibfreenect3 import Camera, Registration

   with Camera.open(streams=("color", "depth")) as camera:
       registration = Registration(
           camera.device.ir_camera_params,
           camera.device.color_camera_params,
       )
       with camera.capture(timeout=2.0) as frames:
           result = registration.apply(
               frames.color,
               frames.depth,
               include_big_depth=True,
               include_color_depth_map=True,
           )

The outputs have these meanings:

.. list-table:: Registration output layouts
   :header-rows: 1

   * - Output
     - Shape and dtype
     - Meaning
   * - ``undistorted``
     - ``(424, 512) float32``
     - Depth in the undistorted depth grid
   * - ``registered``
     - ``(424, 512, 4) uint8``
     - Color sampled for each depth pixel
   * - ``big_depth``
     - ``(1082, 1920) float32``
     - Depth sampled in the color grid
   * - ``color_depth_map``
     - ``(424, 512) int32``
     - Flattened color index per depth pixel

``big_depth`` has a padding row at the top and bottom. Use
``result.big_depth.to_numpy()[1:-1]`` to obtain the ``(1080, 1920)`` color-grid
depth image. A ``color_depth_map`` value of ``-1`` means there is no mapping;
otherwise ``divmod(index, 1920)`` returns the corresponding color row and
column.

``Registration.point_xyz`` accepts ``row, column`` depth coordinates and
returns XYZ in metres. The binding checks the 424-by-512 bounds before calling
the native core.

Offline and filtered-depth registration
---------------------------------------

``Frame.from_array`` makes offline registration explicit and keeps its source
array alive. Color must be contiguous ``uint8`` BGRX/RGBX and depth must be
contiguous ``float32`` millimetres::

   import cv2
   import numpy as np
   from pylibfreenect3 import Frame, FrameFormat, FrameType, Registration

   color_bgr = cv2.imread("color.png", cv2.IMREAD_COLOR)
   color_bgrx = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2BGRA)
   depth_mm = np.load("depth.npy").astype(np.float32, copy=False)

   # Apply filtering in millimetres, then restore a contiguous float32 layout.
   filtered_depth = np.ascontiguousarray(filter_depth(depth_mm), dtype=np.float32)

   color = Frame.from_array(
       np.ascontiguousarray(color_bgrx),
       frame_type=FrameType.COLOR,
       frame_format=FrameFormat.BGRX,
   )
   depth = Frame.from_array(
       filtered_depth,
       frame_type=FrameType.DEPTH,
       frame_format=FrameFormat.FLOAT,
   )
   registration = Registration(saved_ir_params, saved_color_params)
   result = registration.apply(color, depth, include_big_depth=True)

The saved camera parameters must belong to the Kinect that produced the
frames. Recording bundles preserve those parameters and the raw replay
calibration automatically; arbitrary image files do not.

Frame rate and slow consumers
-----------------------------

Use blocking ``capture()`` or ``SyncFrameListener.wait()`` and compare frame
``sequence`` and ``timestamp`` values. Busy-polling ``has_new_frame()`` can make
an application loop appear to run faster than the camera and wastes CPU.

The Kinect depth stream is not made faster by selecting a GPU pipeline; the
pipeline reduces processing latency. ``ColorSettingCommand.SET_FRAME_RATE`` is
a color-camera control, not a general depth-stream throttle, and support for a
requested value depends on the device firmware.

If a consumer needs a lower sampling rate, receive and promptly release every
frame set while processing selected sequences. Video encoding and multi-camera
disk writes should not run serially in the capture loop. ``RecordingWriter``
can use a bounded background queue and reports whether frame sets were dropped.

Backend failures
----------------

``compiled_pipelines()`` reports code present in the core and
``available_pipelines()`` reports backends that passed construction-time
checks. A driver can still fail once packet processing begins. If OpenCL or
CUDA reports packet-buffer or initialization errors, reproduce with the same
backend in the native core, then retry ``pipeline="cpu"``. Do not silently
reinterpret a native backend failure as malformed NumPy data.
