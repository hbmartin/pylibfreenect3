# Maintainer guide

This guide covers repository stewardship, compatibility reviews, dependency
updates, CI operation, and releases. Contributor-facing commands are summarized
in [Development and release gates](dev.md); this page explains the decisions and
manual checks that remain the maintainer's responsibility.

## Maintenance model

`pylibfreenect3` is a Python package around a native C++ library. A published
wheel combines code and policy from three places:

1. this repository's Python, Cython, CMake, and packaging configuration;
2. a coordinated `libfreenect2-metal` revision; and
3. pinned libusb and libjpeg-turbo source releases.

Treat changes to any of those inputs as release changes. In particular, review
native ownership, ABI compatibility, bundled-library licenses, and wheel repair
output rather than relying only on the Python tests.

The main maintenance surfaces are:

| Concern | Source of truth |
| --- | --- |
| Distribution version and supported Python metadata | `pyproject.toml` |
| Source-checkout fallback version | `src/pylibfreenect3/__init__.py` |
| User-visible changes | `NEWS.md` |
| Public package API | `pylibfreenect3.__all__` and `pylibfreenect3.lowlevel.__all__` |
| Native declarations and typing | `_native.pyx`, `libfreenect2.pxd`, and `_native.pyi` |
| Core discovery and compatibility checks | `cmake/FindFreenect2.cmake` |
| Wheel platforms and Python versions | `[tool.cibuildwheel]` in `pyproject.toml` |
| CI core revision and release core tag | `.github/workflows/wheels.yml` |
| Bundled dependency versions, hashes, and core build flags | `tools/build_wheel_dependencies.sh` |
| Required wheel contents and dependency closure | `tools/verify_wheel.py` |
| Recording compatibility | Native core loader plus `pylibfreenect3.legacy` for 1.x bundles |
| Documentation navigation and build | `zensical.toml` and `docs/` |

Keep duplicated values coordinated. A search for the old version, core tag,
core commit, or dependency version before opening a release pull request is a
cheap way to find stale declarations.

## Preparing a change

Open pull requests against the default branch, `master`. Pull requests run the
complete hardware-free workflow, including repaired wheels, an sdist round
trip, sanitizer tests, prerelease/free-threaded smoke tests, and a strict
documentation build. Hardware qualification and publication run only for tags.

For native development, install a compatible core and point the build at its
prefix:

```console
export Freenect2_ROOT=/absolute/path/to/libfreenect2-metal-prefix
uv sync --locked
```

Use the repository's supported uv 0.12.x series. If dependency metadata is
intentionally changed, update `uv.lock`, then prove it is current:

```console
uv lock
uv sync --locked
```

Do not hand-edit `uv.lock`. Documentation can be built without compiling the
package or installing the native core:

```console
uv sync --only-group docs --no-install-project --locked
uv run --no-sync zensical build --clean --strict
```

### Local quality gate

Run the same fast gate as CI before requesting review:

```console
uv run ruff check .
uv run ruff format --check .
uv run cython-lint --max-line-length 120 src/pylibfreenect3/_native.pyx
uv run mypy
uv run pyright --verifytypes pylibfreenect3 --ignoreexternal
uv run pytest -m "not hardware" \
  --cov=pylibfreenect3 --cov-report=term-missing --cov-fail-under=80
uv build --sdist
uv run twine check dist/*
```

The first `uv sync` compiles the extension against `Freenect2_ROOT`. Re-run it
after changing Cython, CMake, build dependencies, or the selected core.

With an attached Kinect v2, run the hardware suite explicitly:

```console
uv run pytest tests/test_hardware.py -m hardware -q
```

The release runner uses 900-frame Metal and CPU memory soaks with a 128 MiB RSS
growth ceiling. Longer local qualification may raise, but should not lower,
those defaults with `PYLIBF3_HARDWARE_SOAK_FRAMES` and
`PYLIBF3_HARDWARE_SOAK_MAX_RSS_MB`.

## Reviewing compatibility-sensitive changes

### Public API and types

For a public API addition or change:

- keep the high-level API curated in `pylibfreenect3.__all__` and put direct
  device primitives in `pylibfreenect3.lowlevel`;
- update runtime annotations, dataclasses/enums, Cython declarations, and
  `_native.pyi` together where applicable;
- test accepted strings and enums, error types, invalid inputs, and repeated
  lifecycle operations;
- add a `NEWS.md` entry for user-visible behavior; and
- build the docs. The API reference is generated statically from both
  `__all__` lists, so a missing or unresolvable export fails the strict build.

Avoid exposing callbacks that execute Python on native decoder threads unless
their threading, shutdown, and ownership contracts are designed and tested.
Optional integrations such as OpenCV and MediaPipe belong in examples or
separate dependency groups, not the base runtime dependency set.

### Ownership and native boundaries

Any change involving frames, NumPy views, listeners, devices, pipelines, or
registration workspaces needs tests for deletion order and repeated cleanup.
Preserve these contracts:

- zero-copy arrays retain the native storage and its owners;
- `release()` and `close()` remain repeatable;
- copied captures detach promptly from listener-owned storage;
- pipeline objects are consumed only once; and
- native camera resources inherited through `fork()` fail safely.

Run the ASan/UBSan lane before merging ownership changes. It is part of the
ordinary pull-request workflow, but its focused command and environment live in
the `sanitizers` job in `.github/workflows/wheels.yml`.

### Core and recording compatibility

The current binding requires libfreenect2 0.4.x and API 4, including its
vision, calibration-profile, projective-registration, runtime-statistics, and
canonical recording surfaces. Changing that contract affects more than the
build:

- update the core version, runtime API, and symbol probe in
  `cmake/FindFreenect2.cmake`;
- update the default CI core commit and release tag in `wheels.yml`;
- update the release-tag check in `build_wheel_dependencies.sh`;
- audit native declarations and every linked symbol;
- preserve the explicit legacy reader and retain tests for old supported
  recordings; and
- update platform/support documentation and `NEWS.md`.

Do not parse or relax canonical recording validation in Python. Native core
versions 1 and 2 are authoritative. If the canonical schema changes, update
the coordinated core first; retain the isolated legacy Python-bundle reader
only for the former pylibfreenect3 1.x format.

## Understanding CI

The `2.0 quality and release artifacts` workflow has three triggers:

- every pull request;
- manual dispatch, optionally with a `libfreenect2-metal` ref; and
- any pushed `v*` tag.

Its jobs are deliberately independent gates:

| Job | What it proves |
| --- | --- |
| `quality` | Locked environment, lint, formatting, types, 80% branch coverage, and sdist metadata |
| `wheels` | CPython 3.12-3.14 repaired wheels for macOS arm64 and manylinux x86_64, plus clean-environment tests |
| `sdist` | The sdist builds without an installed core and can later build a working wheel |
| `sanitizers` | Core unit tests and binding ownership/replay tests pass under ASan and UBSan |
| `prerelease` | The source builds and core API smoke tests pass on CPython 3.15 prerelease |
| `free_threaded` | CPython 3.14t imports safely by enabling its GIL fallback |
| `docs` | The Zensical site builds with warnings as errors |
| `hardware` | Every macOS release wheel passes capture, lifecycle, registration, recording/replay, and memory-soak tests on a Kinect v2 |
| `publish` | Artifacts are attested and uploaded with PyPI Trusted Publishing |

The hardware and publish jobs run only for tags. Publication waits for every
other release job. The self-hosted hardware runner must have the labels
`macOS`, `ARM64`, and `kinect-v2`, and must have exclusive access to a working
camera for up to two hours.

A manual dispatch is the safest rehearsal for a core update. Pass the intended
release core tag as `core_ref`; it exercises all hardware-free jobs but cannot
publish and does not run the tag-only hardware lane.

Documentation has a second workflow. A push to `master` builds `site/` and
deploys it to GitHub Pages. A release tag alone does not deploy the site.

## Updating dependencies and support policy

### Python and CI dependencies

Change requirements in `pyproject.toml`, then update and commit `uv.lock`.
Prefer a focused update when only one package is intended to move:

```console
uv lock --upgrade-package PACKAGE
uv sync --locked
```

`cibuildwheel` is pinned both in the `build` dependency group and in the wheel
job command. Keep those pins equal. The uv version is repeated in both workflow
files. GitHub Actions are pinned by commit SHA; update the SHA and its version
comment together after reviewing the upstream release.

### Bundled native dependencies

libusb and libjpeg-turbo versions and SHA-256 values live in
`tools/build_wheel_dependencies.sh`. For either update:

1. obtain the archive from the project's canonical release location and
   independently verify its checksum;
2. review build-system and minimum-platform changes;
3. replace the packaged license snapshot under
   `src/pylibfreenect3/licenses/` when upstream changed it;
4. build repaired wheels on both supported operating systems; and
5. run `tools/verify_wheel.py` to inspect bundled binaries, licenses, and
   forbidden host paths.

The dependency builder intentionally compares source licenses with the
packaged copies and fails if they differ. Do not bypass that check.

### Python or platform support

Adding or removing a supported Python/platform target requires a coordinated
edit to:

- `requires-python`, classifiers, and cibuildwheel build/skip/architecture
  settings in `pyproject.toml`;
- the wheel matrix, prerelease/free-threaded lanes, and hardware test loop in
  `.github/workflows/wheels.yml`;
- repair and dependency-closure checks;
- `README.md` and `docs/installation.md`; and
- the changelog.

Do not claim support based on a source-build smoke test. A supported wheel
target needs repaired artifacts, isolated installation tests, and—where live
capture is supported—hardware qualification.

## Release procedure

Pushing a matching version tag is the publication action. Do not push it until
the release commit, native core tag, hardware runner, and PyPI configuration
are ready.

### 1. Prepare the release pull request

Choose the release version and update the project metadata:

```console
release_version=X.Y.Z
uv version "$release_version"
```

`uv version` updates `pyproject.toml` and re-locks the project. Also:

- change the source-checkout fallback in `src/pylibfreenect3/__init__.py` to the
  corresponding `.dev0` value;
- add the release section to `NEWS.md`;
- update release-specific statements in the README and docs;
- confirm Python classifiers, wheel targets, and dependency pins; and
- search for the previous package version and coordinated core tag.

For the current compatibility line, a release tag resolves the core as
`v0.4.0`, and `build_wheel_dependencies.sh` enforces that exact tag. If the
coordinated core release changes, make that a reviewed code change—do not use a
different manual `core_ref` as a substitute for updating the release policy.

Run the local quality and docs gates, then merge only after the pull-request
workflow is green.

### 2. Check release infrastructure

Before tagging, confirm:

- the release commit is on `master` and the working tree is clean;
- the coordinated core tag exists and points to the reviewed core commit;
- the self-hosted Kinect runner is online and the camera is available;
- the GitHub `pypi` environment and PyPI Trusted Publisher still authorize
  this repository/workflow; and
- GitHub Pages is healthy for the merged documentation.

Optionally run a manual workflow dispatch with the release core tag and inspect
the produced wheel and sdist artifacts before creating the package tag.

### 3. Tag and monitor

Create a `v`-prefixed tag whose value exactly matches `pyproject.toml`:

```console
release_version=$(uv version --short)
git tag -a "v${release_version}" -m "pylibfreenect3 ${release_version}"
git push origin "v${release_version}"
```

The quality job rejects a mismatched tag. Monitor every job through `publish`;
do not treat successful wheel builds as a completed release. The workflow
attests all wheels and the sdist before `uv publish` uploads them to PyPI.

The workflow does not create a GitHub Release. If the project uses GitHub
Releases for that version, create it after PyPI publication from the immutable
tag and use the corresponding `NEWS.md` section as the basis for its notes.

### 4. Verify the published release

On each supported platform, install from the public index into an isolated
environment rather than reusing a checkout:

```console
release_version=$(uv version --short)
uv run --isolated --no-project \
  --with "pylibfreenect3==${release_version}" \
  python -c "import pylibfreenect3 as f3; print(f3.__version__, f3.core_version(), f3.compiled_pipelines())"
```

Confirm that PyPI shows the sdist and all six expected wheels: CPython
3.12-3.14 for macOS arm64 and manylinux x86_64. Check the provenance display,
then perform a basic live capture on the supported hardware platform. Confirm
the documentation reflects the released version.

## Failed or broken releases

If a tag workflow fails before publication, fix the cause on a pull request and
rerun the hardware-free workflow. Do not casually move a public tag. Recreate
an unpublished tag only after confirming that no file reached PyPI and after
coordinating with the other maintainers; if publication may have started, use
a new patch version.

PyPI artifacts are immutable. If a published release is broken:

1. stop recommending the version and record the impact;
2. yank it through PyPI when installation should be discouraged;
3. fix forward with a patch release rather than attempting to overwrite files;
4. add regression coverage, including hardware or sanitizer coverage when
   relevant; and
5. explain the failure and replacement in `NEWS.md` and the GitHub Release.

If only documentation is wrong, merge the correction to `master`; the Pages
workflow deploys it without a package release.

## Periodic maintenance

At least once per release cycle:

- rehearse the release workflow with a manual dispatch;
- verify the self-hosted runner labels, uv installation, free disk space, USB
  access, and camera exclusivity;
- review Python prerelease and cibuildwheel support for the next interpreter;
- review pinned Actions, uv, cibuildwheel, libusb, and libjpeg-turbo versions;
- build the docs strictly and check external project links;
- test reading an older schema-v1 recording; and
- inspect recent deprecations and ensure removal plans are documented before
  changing compatibility behavior.

Keep this guide synchronized with workflow changes. A release procedure that
does not match the automation is itself a release defect.
