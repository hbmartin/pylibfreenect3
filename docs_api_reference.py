"""Zensical macros for API reference pages."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable


class MacroEnvironment(Protocol):
    """Subset of the Zensical macro environment used here."""

    def macro(self, fn: Callable[..., str]) -> Callable[..., str]:
        """Register a macro function."""


PACKAGE_ROOT = Path(__file__).parent / "src"
SUPPLEMENTAL_MODULES = ("pylibfreenect3.lowlevel",)


def _module_path(module_name: str) -> Path:
    """Return the source path for a Python module in the src tree."""
    relative = Path(*module_name.split("."))
    package = PACKAGE_ROOT / relative / "__init__.py"
    if package.is_file():
        return package
    module = PACKAGE_ROOT / relative.with_suffix(".py")
    if module.is_file():
        return module
    msg = f"cannot find Python source for {module_name!r}"
    raise FileNotFoundError(msg)


def _string_sequence(node: ast.AST, *, description: str) -> list[str]:
    """Return strings from a literal list or tuple AST node."""
    if not isinstance(node, ast.List | ast.Tuple) or not all(
        isinstance(item, ast.Constant) and isinstance(item.value, str)
        for item in node.elts
    ):
        msg = f"{description} must be a literal list or tuple of strings"
        raise TypeError(msg)
    return [item.value for item in node.elts]  # type: ignore[union-attr]


def _module_exports(module_name: str) -> list[str]:
    """Return canonical identifiers exported by a source module."""
    path = _module_path(module_name)
    tree = ast.parse(path.read_text(), filename=str(path))
    exports: list[str] | None = None
    origins: dict[str, str] = {}
    definitions: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            exports = _string_sequence(
                node.value,
                description=f"{module_name}.__all__",
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = module_name.split(".")
                if path.name != "__init__.py":
                    base_parts = base_parts[:-1]
                base_parts = base_parts[: len(base_parts) - node.level + 1]
                if node.module:
                    base_parts.extend(node.module.split("."))
                origin_module = ".".join(base_parts)
            else:
                origin_module = node.module or ""
            for alias in node.names:
                local_name = alias.asname or alias.name
                origins[local_name] = ".".join(
                    part for part in (origin_module, alias.name) if part
                )
        elif isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            definitions.add(node.name)

    if exports is None:
        msg = f"{module_name} must define __all__ as a literal list or tuple"
        raise TypeError(msg)

    identifiers: list[str] = []
    for name in exports:
        if name in origins:
            identifiers.append(origins[name])
        elif name in definitions:
            identifiers.append(f"{module_name}.{name}")
        else:
            msg = f"cannot resolve {name!r} exported by {module_name}"
            raise ValueError(msg)
    return identifiers


def _mkdocstrings_directive(identifier: str) -> str:
    """Return a mkdocstrings directive for one API object."""
    return "\n".join(
        (
            f"::: {identifier}",
            "    options:",
            "      show_root_heading: true",
            "",
        )
    )


def _reference_markdown(package_name: str) -> str:
    """Return mkdocstrings directives for a package's public API."""
    identifiers = _module_exports(package_name)
    for supplemental in SUPPLEMENTAL_MODULES:
        identifiers.extend(_module_exports(supplemental))

    seen: set[str] = set()
    directives: list[str] = []
    for identifier in identifiers:
        if identifier in seen or identifier == f"{package_name}.lowlevel":
            continue
        seen.add(identifier)
        directives.append(_mkdocstrings_directive(identifier))
    return "\n".join(directives)


def define_env(env: MacroEnvironment) -> None:
    """Register documentation macros with Zensical."""

    @env.macro
    def public_api_reference(package_name: str = "pylibfreenect3") -> str:
        """Return generated public API reference directives."""
        return _reference_markdown(package_name=package_name)
