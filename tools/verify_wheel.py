#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import subprocess
import tempfile
import zipfile
from pathlib import Path


FORBIDDEN_PREFIXES = ("/opt/homebrew", "/usr/local", "/private/tmp", "/tmp/")
LICENSE_NAMES = {
    "libfreenect2-APACHE20.txt",
    "libfreenect2-CONTRIB.txt",
    "libfreenect2-GPL2.txt",
    "libusb-COPYING.txt",
    "libjpeg-turbo-LICENSE.md",
}


def run(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"{' '.join(command)} failed with {result.returncode}\n{result.stdout}{result.stderr}"
        )
    return result.stdout + result.stderr


def verify_macos(binary: Path) -> None:
    output = run(["otool", "-L", str(binary)])
    lines = output.splitlines()[1:]
    if binary.suffix == ".dylib":
        install_name = lines.pop(0).strip().split(" ", 1)[0]
        if install_name.startswith(FORBIDDEN_PREFIXES):
            raise RuntimeError(f"host install name in {binary}: {install_name}")
        if not install_name.startswith(("@", "/DLC/")):
            raise RuntimeError(f"unexpected install name in {binary}: {install_name}")
    for line in lines:
        dependency = line.strip().split(" ", 1)[0]
        if dependency.startswith(FORBIDDEN_PREFIXES):
            raise RuntimeError(f"unbundled macOS dependency in {binary}: {dependency}")
        if not dependency.startswith(("@", "/usr/lib/", "/System/Library/")):
            raise RuntimeError(f"unexpected macOS dependency in {binary}: {dependency}")


def verify_linux(binary: Path) -> None:
    output = run(["ldd", str(binary)])
    if "not found" in output:
        raise RuntimeError(f"unresolved Linux dependency in {binary}:\n{output}")
    for forbidden in FORBIDDEN_PREFIXES:
        if forbidden in output:
            raise RuntimeError(
                f"host dependency {forbidden} leaked into {binary}:\n{output}"
            )


def verify(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        packaged_licenses = {Path(name).name for name in names if "/licenses/" in name}
        missing = LICENSE_NAMES - packaged_licenses
        if missing:
            raise RuntimeError(
                f"wheel is missing dependency licenses: {sorted(missing)}"
            )
        binary_names = {Path(name).name for name in names}
        required_binaries = {
            "libfreenect2": any(
                name.startswith("libfreenect2") and (".dylib" in name or ".so" in name)
                for name in binary_names
            ),
            "libusb": any(
                name.startswith("libusb") and (".dylib" in name or ".so" in name)
                for name in binary_names
            ),
            "TurboJPEG": any(
                name.startswith("libturbojpeg") and (".dylib" in name or ".so" in name)
                for name in binary_names
            ),
        }
        missing_binaries = [
            dependency
            for dependency, present in required_binaries.items()
            if not present
        ]
        if missing_binaries:
            raise RuntimeError(
                f"wheel is missing bundled dependencies: {missing_binaries}"
            )
        with tempfile.TemporaryDirectory(
            prefix="pylibfreenect3-wheel-check-"
        ) as directory:
            archive.extractall(directory)
            binaries = [
                path
                for path in Path(directory).rglob("*")
                if path.suffix in (".so", ".dylib") or ".so." in path.name
            ]
            if not binaries:
                raise RuntimeError("wheel contains no native binaries")
            for binary in binaries:
                if platform.system() == "Darwin":
                    verify_macos(binary)
                elif platform.system() == "Linux":
                    verify_linux(binary)
                else:
                    raise RuntimeError(
                        f"unsupported verification host: {platform.system()}"
                    )

    if platform.system() == "Darwin":
        print(run(["delocate-listdeps", "--all", str(wheel)]))
    else:
        print(run(["auditwheel", "show", str(wheel)]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    arguments = parser.parse_args()
    for wheel in arguments.wheels:
        verify(wheel.resolve())
        print(f"verified {wheel}")


if __name__ == "__main__":
    main()
