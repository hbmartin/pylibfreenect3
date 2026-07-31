Development and release gates
=============================

Run hardware-free tests with::

   pytest -m "not hardware"

Core tests cover runtime/API identity, canonical factories, environment aliases,
dump selection, replay filename parsing, stream filtering, calibration, and
repeatable lifecycle. Binding tests cover every enum and frame layout, typed
values, missing keys, source-backed frames, deletion orders, pipeline
consumption, registration validation, dump table copies, bundles, and loose
replay.

Release candidates additionally run ownership and replay under ASan/UBSan. On
a real Kinect, every macOS wheel must capture at least 100 color/IR/depth frames
with explicit Metal and automatic selection, then exercise registration,
configuration/exposure, repeated open/close, outstanding arrays, recording, and
Metal replay.

Stable publication requires both repositories to be tagged ``v0.3.0``. Each
repaired wheel is installed in a clean environment and inspected with
``otool``/delocate or ``ldd``/auditwheel. Only bundled project dependencies and
platform-provided libraries may remain.
