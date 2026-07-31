from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


def _prefix_layout(prefix: Path) -> tuple[list[str], list[str]] | None:
    include = prefix / "include"
    config = include / "libfreenect2" / "config.h"
    libraries = [path for path in (prefix / "lib", prefix / "lib64") if path.is_dir()]
    if config.is_file() and any(
        any(directory.glob(pattern))
        for directory in libraries
        for pattern in ("libfreenect2.so*", "libfreenect2.dylib", "freenect2.lib")
    ):
        return [str(include)], [str(path) for path in libraries]
    return None


def _pkg_config_layout() -> tuple[list[str], list[str]] | None:
    try:
        result = subprocess.run(
            ["pkg-config", "--cflags", "--libs-only-L", "freenect2"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    includes: list[str] = []
    libraries: list[str] = []
    for token in shlex.split(result.stdout):
        if token.startswith("-I"):
            includes.append(token[2:])
        elif token.startswith("-L"):
            libraries.append(token[2:])

    def variable_path(name: str) -> list[str]:
        queried = subprocess.run(
            ["pkg-config", f"--variable={name}", "freenect2"],
            check=False,
            text=True,
            capture_output=True,
        )
        value = queried.stdout.strip()
        return [value] if queried.returncode == 0 and value else []

    if not includes:
        includes = variable_path("includedir")
    if not libraries:
        libraries = variable_path("libdir")
    return includes, libraries


def _linked_libraries(executable: Path) -> str:
    command = (
        ["otool", "-L", str(executable)]
        if platform.system() == "Darwin"
        else ["ldd", str(executable)]
    )
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return f"{command[0]} is unavailable"
    return (result.stdout + result.stderr).strip()


def _core_library(libraries: list[str]) -> Path:
    preferred = ("libfreenect2.dylib", "libfreenect2.so", "freenect2.dll")
    for directory in map(Path, libraries):
        for name in preferred:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        for pattern in ("libfreenect2.dylib", "libfreenect2.so.*", "freenect2.dll"):
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
    raise RuntimeError(f"no libfreenect2 runtime was found in {libraries}")


def _probe_core(includes: list[str], libraries: list[str]) -> dict[str, str]:
    source = r"""
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/packet_pipeline.h>
int main() {
  std::cout << "runtime_version=" << libfreenect2::getVersion() << "\n";
  std::cout << "runtime_api=" << libfreenect2::getApiVersion() << "\n";
  std::cout << "build_revision=" << libfreenect2::getBuildRevision() << "\n";
  const std::vector<std::string> names = libfreenect2::getCompiledPacketPipelines();
  std::cout << "compiled_pipelines=";
  for(std::size_t i = 0; i < names.size(); ++i) {
    if(i) std::cout << ',';
    std::cout << names[i];
  }
  std::cout << "\n";
  return 0;
}
"""
    with tempfile.TemporaryDirectory(prefix="pylibfreenect3-probe-") as temporary:
        directory = Path(temporary)
        source_path = directory / "probe.cpp"
        executable = directory / "probe"
        source_path.write_text(source, "utf-8")
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        command = [
            *compiler,
            *shlex.split(os.environ.get("CPPFLAGS", "")),
            *shlex.split(os.environ.get("CXXFLAGS", "")),
            "-std=c++17",
            str(source_path),
            "-o",
            str(executable),
        ]
        command.extend(f"-I{value}" for value in includes)
        command.extend(f"-L{value}" for value in libraries)
        command.append("-lfreenect2")
        command.extend(f"-Wl,-rpath,{value}" for value in libraries)
        command.extend(shlex.split(os.environ.get("LDFLAGS", "")))
        try:
            core_links = _linked_libraries(_core_library(libraries))
        except RuntimeError as error:
            core_links = f"core will be resolved by the platform linker: {error}"
        compiled = subprocess.run(command, text=True, capture_output=True, check=False)
        if compiled.returncode:
            raise RuntimeError(
                "failed to compile the libfreenect2 runtime probe:\n"
                + compiled.stdout
                + compiled.stderr
                + "\ncore linked libraries:\n"
                + core_links
            )
        environment = os.environ.copy()
        variable = (
            "DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH"
        )
        environment[variable] = os.pathsep.join(
            [*libraries, environment.get(variable, "")]
        ).rstrip(os.pathsep)
        probed = subprocess.run(
            [str(executable)],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        linked = _linked_libraries(executable) + "\ncore runtime:\n" + core_links
        if probed.returncode:
            raise RuntimeError(
                "failed to execute the libfreenect2 runtime probe:\n"
                + probed.stdout
                + probed.stderr
                + "\nlinked libraries:\n"
                + linked
            )
        values = dict(
            line.split("=", 1) for line in probed.stdout.splitlines() if "=" in line
        )
        values["linked_libraries"] = linked
        return values


def discover_core() -> tuple[list[str], list[str]]:
    candidates: list[tuple[str, tuple[list[str], list[str]] | None]] = []
    explicit = os.environ.get("LIBFREENECT2_INSTALL_PREFIX")
    if explicit:
        resolved = Path(explicit).expanduser().resolve()
        layout = _prefix_layout(resolved)
        if layout is None:
            raise RuntimeError(
                "LIBFREENECT2_INSTALL_PREFIX does not contain a usable libfreenect2 "
                f"installation: {resolved}\n  architecture: {platform.machine()}"
            )
        candidates.append(("LIBFREENECT2_INSTALL_PREFIX", layout))
    candidates.append(("pkg-config", _pkg_config_layout()))
    for prefix in (Path("/usr/local"), Path("/usr"), Path("/opt/homebrew")):
        candidates.append((str(prefix), _prefix_layout(prefix)))

    attempted: list[str] = []
    rejections: list[str] = []
    for source_name, layout in candidates:
        attempted.append(source_name)
        if layout is None:
            continue
        includes, libraries = layout
        configs = [
            Path(include) / "libfreenect2" / "config.h"
            for include in includes
            if (Path(include) / "libfreenect2" / "config.h").is_file()
        ]
        source_rejections: list[str] = []
        if includes and not configs:
            source_rejections.append(
                f"{source_name}: no libfreenect2/config.h under {includes}"
            )
        for config in configs or ([None] if not includes else []):
            header_version = "compiler-default search path"
            if config is not None:
                contents = config.read_text("utf-8")
                match = re.search(
                    r'^#define\s+LIBFREENECT2_VERSION\s+"([^"]+)"',
                    contents,
                    re.MULTILINE,
                )
                if match is None or not match.group(1).startswith("0.3."):
                    found = "unknown" if match is None else match.group(1)
                    source_rejections.append(
                        f"{source_name}: header version {found} at {config}; "
                        "requires 0.3.x"
                    )
                    continue
                header_version = f"{match.group(1)} at {config}"
            try:
                runtime = _probe_core(includes, libraries)
            except RuntimeError as error:
                source_rejections.append(
                    f"{source_name}: headers={includes}, libraries={libraries}: {error}"
                )
                continue
            if (
                not runtime.get("runtime_version", "").startswith("0.3.")
                or runtime.get("runtime_api") != "3"
            ):
                source_rejections.append(
                    f"{source_name}: header={header_version}, "
                    f"runtime={runtime.get('runtime_version', 'unknown')}, "
                    f"API={runtime.get('runtime_api', 'unknown')}"
                )
                continue
            print(
                "pylibfreenect3 build configuration:\n"
                f"  architecture: {platform.machine()}\n"
                f"  discovery source: {source_name}\n"
                f"  headers: {includes}\n"
                f"  libraries: {libraries}\n"
                f"  header version: {header_version}\n"
                f"  runtime version: {runtime['runtime_version']}\n"
                f"  runtime ABI/API: {runtime['runtime_api']}\n"
                f"  build revision: {runtime['build_revision']}\n"
                f"  compiled pipelines: {runtime['compiled_pipelines']}\n"
                f"  linked libraries:\n{runtime['linked_libraries']}"
            )
            return includes, libraries
        if source_name == "LIBFREENECT2_INSTALL_PREFIX" and source_rejections:
            raise RuntimeError("\n".join(source_rejections))
        rejections.extend(source_rejections)
    rejection_details = (
        "\n  rejected candidates:\n  - " + "\n  - ".join(rejections)
        if rejections
        else ""
    )
    raise RuntimeError(
        "libfreenect2 0.3.x was not found. Set LIBFREENECT2_INSTALL_PREFIX, install "
        "freenect2.pc, or install the core under a standard prefix.\n"
        f"  architecture: {platform.machine()}\n"
        f"  attempted: {', '.join(attempted)}"
        f"{rejection_details}"
    )


def _needs_native_core() -> bool:
    """Whether this invocation compiles the extension.

    Metadata-only commands (sdist, egg_info, ...) must not require an
    installed libfreenect2; unknown commands default to probing so a missing
    core fails loudly rather than producing a wheel without the extension.
    """
    arguments = set(sys.argv[1:])
    native_commands = {
        "bdist",
        "bdist_wheel",
        "build",
        "build_ext",
        "develop",
        "editable_wheel",
        "install",
    }
    metadata_only_commands = {"check", "clean", "dist_info", "egg_info", "sdist"}
    if arguments & native_commands:
        return True
    return not arguments & metadata_only_commands


def _native_extensions() -> list:
    include_dirs, library_dirs = discover_core()
    extra_compile_args = ["-std=c++17"]
    extra_link_args: list[str] = []
    if platform.system() == "Darwin":
        extra_compile_args.append("-stdlib=libc++")
        if library_dirs:
            extra_link_args.append(f"-Wl,-rpath,{library_dirs[0]}")
        extra_link_args.append("-Wl,-rpath,@loader_path/.dylibs")
    elif platform.system() == "Linux":
        if library_dirs:
            extra_link_args.append(f"-Wl,-rpath,{library_dirs[0]}")
        extra_link_args.append("-Wl,-rpath,$ORIGIN/../pylibfreenect3.libs")

    extension = Extension(
        "pylibfreenect3._native",
        ["src/pylibfreenect3/_native.pyx"],
        include_dirs=[numpy.get_include(), *include_dirs],
        library_dirs=library_dirs,
        libraries=["freenect2"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION")],
    )
    return cythonize([extension], compiler_directives={"language_level": 3})


setup(ext_modules=_native_extensions() if _needs_native_core() else [])
