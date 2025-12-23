#!/usr/bin/env python3
"""
Extract third-party Python dependencies from the repo and update requirements files.

Usage:
  python scripts/extract_requirements.py [--dry-run] [--include-notebooks]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
import re

EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "site-packages",
}

IMPORT_MAP = {
    "cv2": "opencv-python-headless",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "skimage": "scikit-image",
    "huggingface_hub": "huggingface-hub",
}

REQUIREMENTS_FILES = {
    "runtime": "requirements.txt",
    "dev": "requirements-dev.txt",
}


def _stdlib_modules() -> set[str]:
    stdlib = set(sys.builtin_module_names)
    if hasattr(sys, "stdlib_module_names"):
        stdlib.update(sys.stdlib_module_names)
    return stdlib


def _canonical_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _iter_py_files(root: Path, include_notebooks: bool) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir():
            if path.name in EXCLUDE_DIRS:
                continue
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix == ".py":
            files.append(path)
        elif include_notebooks and path.suffix == ".ipynb":
            files.append(path)
    return files


def _parse_imports_from_source(source: str) -> set[str]:
    imports: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def _parse_notebook_imports(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    imports: set[str] = set()
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        imports.update(_parse_imports_from_source(source))
    return imports


def _parse_requirements_file(
    path: Path,
    seen: set[Path] | None = None,
    include_references: bool = False,
) -> dict[str, str]:
    if seen is None:
        seen = set()
    if path in seen or not path.exists():
        return {}
    seen.add(path)

    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            if include_references:
                ref_path = (path.parent / line[3:].strip()).resolve()
                parsed.update(_parse_requirements_file(ref_path, seen, include_references))
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)", line)
        if not match:
            continue
        name = match.group(1)
        canonical = _canonical_name(name)
        parsed.setdefault(canonical, line)
    return parsed


def _internal_top_level_modules(repo_root: Path) -> set[str]:
    internal: set[str] = {"backend", "scripts", "tests"}
    for item in repo_root.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_dir():
            internal.add(item.name)
        elif item.is_file() and item.suffix == ".py":
            internal.add(item.stem)
    return internal


def _map_import(name: str) -> str:
    return IMPORT_MAP.get(name, name)


def _collect_dependencies(
    repo_root: Path, include_notebooks: bool
) -> tuple[set[str], set[str], int]:
    stdlib = _stdlib_modules()
    internal = _internal_top_level_modules(repo_root)

    runtime_imports: set[str] = set()
    dev_imports: set[str] = set()

    scan_paths = [repo_root / "backend", repo_root / "tests", repo_root / "scripts"]
    files_scanned = 0

    for base in scan_paths:
        if not base.exists():
            continue
        for path in _iter_py_files(base, include_notebooks):
            files_scanned += 1
            if path.suffix == ".ipynb":
                imports = _parse_notebook_imports(path)
            else:
                imports = _parse_imports_from_source(path.read_text(encoding="utf-8"))

            if base.name == "backend":
                runtime_imports.update(imports)
            else:
                dev_imports.update(imports)

    def filter_imports(imports: set[str]) -> set[str]:
        filtered: set[str] = set()
        for name in imports:
            if name in stdlib:
                continue
            if name in internal:
                continue
            mapped = _map_import(name)
            filtered.add(mapped)
        return filtered

    runtime = filter_imports(runtime_imports)
    dev = filter_imports(dev_imports)
    dev_only = dev - runtime

    return runtime, dev_only, files_scanned


def _write_requirements(
    path: Path,
    detected: set[str],
    existing: dict[str, str],
    dry_run: bool,
    exclude: set[str] | None = None,
) -> None:
    existing_keys = set(existing.keys())
    detected_canon = {_canonical_name(name): name for name in detected}

    merged_lines: list[str] = []
    for canonical, line in existing.items():
        if exclude and canonical in exclude:
            continue
        merged_lines.append(line)

    missing = sorted(
        [name for canon, name in detected_canon.items() if canon not in existing_keys],
        key=str.lower,
    )
    merged_lines.extend(missing)

    content = "\n".join(merged_lines) + "\n"
    if dry_run:
        print(f"\n[{path}]\n{content}")
        return

    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument(
        "--include-notebooks",
        action="store_true",
        help="Include notebooks/ in the scan",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runtime, dev, files_scanned = _collect_dependencies(
        repo_root, include_notebooks=args.include_notebooks
    )

    runtime_existing = _parse_requirements_file(
        repo_root / REQUIREMENTS_FILES["runtime"], include_references=False
    )
    dev_existing = _parse_requirements_file(
        repo_root / REQUIREMENTS_FILES["dev"], include_references=False
    )

    print(f"Files scanned: {files_scanned}")
    print("Runtime deps:")
    for name in sorted(runtime, key=str.lower):
        print(f"- {name}")
    print("Dev deps:")
    for name in sorted(dev, key=str.lower):
        print(f"- {name}")

    dev_only_canon = {_canonical_name(name) for name in dev}
    dev_existing_only = {key for key in dev_existing if key not in {_canonical_name(n) for n in runtime}}
    runtime_exclude = dev_only_canon | dev_existing_only

    _write_requirements(
        repo_root / REQUIREMENTS_FILES["runtime"],
        runtime,
        runtime_existing,
        args.dry_run,
        exclude=runtime_exclude,
    )
    _write_requirements(
        repo_root / REQUIREMENTS_FILES["dev"],
        dev,
        dev_existing,
        args.dry_run,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
