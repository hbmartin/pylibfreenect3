# Development and release gates

Repository owners should also follow the [maintainer guide](maintainers.md) for
compatibility reviews, dependency updates, CI operation, and publication.

Run hardware-free tests with:

```console
uv sync
uv run pytest -m "not hardware"
```

Run the complete fast quality gate with:

```console
uv run ruff check .
uv run ruff format --check .
uv run cython-lint --max-line-length 120 src/pylibfreenect3/_native.pyx
uv run mypy
uv run pyright --verifytypes pylibfreenect3 --ignoreexternal
```

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

The hardware lane also runs a 900-frame memory soak on both Metal and CPU. It
warms each pipeline before sampling process high-water RSS and rejects growth
above 128 MiB. `PYLIBF3_HARDWARE_SOAK_FRAMES` and
`PYLIBF3_HARDWARE_SOAK_MAX_RSS_MB` can raise those release-gate thresholds for
longer local qualification runs.

Stable publication requires this repository to be tagged `v1.0.0` while the
core remains coordinated at `v0.3.0`. Each repaired wheel is installed in a
clean environment and inspected with `otool`/delocate or `ldd`/auditwheel. Only
bundled project dependencies and platform-provided libraries may remain.
Releases use `uv build`, provenance attestation through `actions/attest`, and
Trusted Publishing through `uv publish`. The GitHub provenance step is
separate because uv does not currently generate PEP 740 attestations.

## Documentation

Build the site locally with warnings treated as errors:

```console
uv sync --only-group docs --no-install-project
uv run --no-sync zensical build --clean --strict
```

Pushes to `master` publish the generated `site` directory to GitHub Pages
through the Documentation workflow.
