#!/usr/bin/env python3
"""Find local names that are too sentence-like.

This checker is intentionally conservative:

- it reports definitions and file stems, not imports or ordinary uses;
- it skips generated/cache/build/vendor paths;
- it skips Rust trait methods where the method name is imposed by a trait impl;
- it skips Rust module loader names, which must match filenames;
- it skips explicitly marked Python overrides on imported external bases;
- it includes public definitions unless `--private-only` is supplied;
- it rejects trailing overload numbers on public definitions while allowing
  numeric notation such as `f32`, `int4`, `l2`, `dist2`, and `e2m1`;
- it treats snake_case names with more than one underscore as violations;
- it treats overlong names as violations even when a sentence was collapsed into
  one token.

The goal is to catch names such as long test labels and helper names that use a
function name as a stand-in for a semantic sentence.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SKIP_PARTS = {
    ".git",
    ".home",
    ".pixi",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "buck-out",
    "dist",
    "target",
}

SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
}

FILE_SUFFIXES = {
    ".bzl",
    ".cl",
    ".ipynb",
    ".json",
    ".md",
    ".metal",
    ".py",
    ".rs",
    ".toml",
    ".yaml",
    ".yml",
}

RUST_DEF = re.compile(
    r"""
    ^\s*
    (?P<vis>pub|pub\([^)]*\))?\s*
    (?:(?:async|unsafe|const|extern\s+"[^"]+")\s+)*
    (?P<kind>fn|struct|enum|trait|type|const|static|mod|macro_rules!)\s*!?\s*
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)

RUST_IMPL = re.compile(r"^\s*impl(?:<[^>]+>)?\s+(?P<head>.+?)\s*\{")
RUST_TRAIT_IMPL = re.compile(r"\bfor\b")

NUMERIC_NOTATION = re.compile(
    r"(?:"
    r"(?:bf16|f\d+|i\d+|u\d+)"
    r"|(?:^|_)(?:dist2|e\d+m\d+|fp\d+|int\d+|l2|uint\d+)"
    r"|splitmix64|mix64"
    r")$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    language: str
    kind: str
    name: str
    underscores: int
    reason: str


def repo_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    except Exception:
        return Path.cwd()
    return Path(out)


def skipped(path: Path) -> bool:
    if path.suffix in SKIP_SUFFIXES:
        return True
    return any(part in SKIP_PARTS for part in path.parts)


def candidate_roots(root: Path, paths: list[str]) -> Iterable[Path]:
    if paths:
        candidates = [Path(path) for path in paths]
    else:
        candidates = [root]
    for candidate in candidates:
        yield candidate if candidate.is_absolute() else root / candidate


def source_files(root: Path, paths: list[str]) -> Iterable[Path]:
    for path in candidate_roots(root, paths):
        yield from path_sources(path)


def path_sources(path: Path) -> Iterable[Path]:
    if skipped(path):
        return
    if path.is_file():
        if path.suffix in {".rs", ".py"}:
            yield path
        return
    if not path.is_dir():
        return
    for child in path.rglob("*"):
        if child.is_file() and child.suffix in {".rs", ".py"} and not skipped(child):
            yield child


def file_targets(root: Path, paths: list[str]) -> Iterable[Path]:
    for path in candidate_roots(root, paths):
        yield from path_files(path)


def path_files(path: Path) -> Iterable[Path]:
    if skipped(path):
        return
    if path.is_file():
        if path.suffix in FILE_SUFFIXES or path.name in {
            "BUCK",
            "Cargo.toml",
            "MODULE.bazel",
            "README.md",
        }:
            yield path
        return
    if not path.is_dir():
        return
    for child in path.rglob("*"):
        if child.is_file() and not skipped(child):
            if child.suffix in FILE_SUFFIXES or child.name in {
                "BUCK",
                "Cargo.toml",
                "MODULE.bazel",
                "README.md",
            }:
                yield child


def overload_number(name: str) -> bool:
    return bool(re.search(r"\d+$", name)) and not NUMERIC_NOTATION.search(name)


def bad_name(name: str, max_underscores: int, check_number: bool = False) -> bool:
    if name.startswith("__") and name.endswith("__"):
        return False
    return (
        name.count("_") > max_underscores
        or len(name) > 24
        or (check_number and overload_number(name))
    )


def reason(
    name: str, kind: str, max_underscores: int, check_number: bool = False
) -> str:
    reasons = []
    if name.count("_") > max_underscores:
        reasons.append(f"more than {max_underscores} underscore")
    if len(name) > 24:
        reasons.append("more than 24 characters")
    if check_number and overload_number(name):
        reasons.append("trailing overload number")
    joined = " and ".join(reasons)
    return f"local {kind} name has {joined}"


def file_findings(root: Path, paths: list[str], max_underscores: int) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(set(file_targets(root, paths))):
        stem = path.stem
        if stem == "__init__":
            continue
        if bad_name(stem, max_underscores):
            findings.append(
                Finding(
                    path=str(path.relative_to(root)),
                    line=1,
                    language="file",
                    kind="file",
                    name=stem,
                    underscores=stem.count("_"),
                    reason=reason(stem, "file", max_underscores),
                )
            )
    return findings


def rust_findings(
    path: Path, root: Path, max_underscores: int, include_public: bool
) -> list[Finding]:
    findings: list[Finding] = []
    trait_depths: list[int] = []
    depth = 0
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        return [
            Finding(
                path=str(path.relative_to(root)),
                line=1,
                language="rust",
                kind="decode",
                name="<non-utf8>",
                underscores=0,
                reason=str(error),
            )
        ]
    for line_no, line in enumerate(text.splitlines(), 1):
        impl = RUST_IMPL.match(line)
        if impl and RUST_TRAIT_IMPL.search(impl.group("head")):
            trait_depths.append(depth + line.count("{") - line.count("}"))

        in_trait_impl = any(depth >= item for item in trait_depths)
        match = RUST_DEF.match(line)
        if match:
            kind = match.group("kind")
            name = match.group("name")
            visibility = match.group("vis")
            is_public = visibility is not None
            is_external = visibility == "pub"
            if (
                kind != "mod"
                and not (in_trait_impl and kind == "fn")
                and (include_public or not is_public)
                and bad_name(name, max_underscores, is_external)
            ):
                findings.append(
                    Finding(
                        path=str(path.relative_to(root)),
                        line=line_no,
                        language="rust",
                        kind=kind,
                        name=name,
                        underscores=name.count("_"),
                        reason=reason(name, kind, max_underscores, is_external),
                    )
                )

        depth += line.count("{") - line.count("}")
        trait_depths = [item for item in trait_depths if depth >= item]
    return findings


def _overrides(tree: ast.AST) -> set[int]:
    """External interfaces own their method names, not ENNX's naming policy."""
    imports, markers = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level:
            if node.module and node.module.split(".")[0] != "ennx":
                imports.update(alias.asname or alias.name for alias in node.names)
            if node.module in {"typing", "typing_extensions"}:
                markers.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "override"
                )
        elif isinstance(node, ast.Import):
            imports.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
    methods = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [ast.unparse(base).split(".")[0] for base in node.bases]
        if not any(base in imports for base in bases):
            continue
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                isinstance(marker, ast.Name) and marker.id in markers
                for marker in member.decorator_list
            ):
                methods.add(id(member))
    return methods


def python_findings(
    path: Path, root: Path, max_underscores: int, include_public: bool
) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        return [
            Finding(
                path=str(path.relative_to(root)),
                line=1,
                language="python",
                kind="decode",
                name="<non-utf8>",
                underscores=0,
                reason=str(error),
            )
        ]
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as error:
        return [
            Finding(
                path=str(path.relative_to(root)),
                line=error.lineno or 1,
                language="python",
                kind="syntax",
                name="<parse-error>",
                underscores=0,
                reason=str(error),
            )
        ]

    findings: list[Finding] = []
    relative_parts = path.relative_to(root).parts
    check_public_number = bool(relative_parts and relative_parts[0] == "src")
    overrides = _overrides(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if id(node) in overrides:
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "fn"
            name = node.name
            is_public = not name.startswith("_")
            if (include_public or not is_public) and bad_name(
                name, max_underscores, is_public and check_public_number
            ):
                findings.append(
                    Finding(
                        path=str(path.relative_to(root)),
                        line=node.lineno,
                        language="python",
                        kind=kind,
                        name=name,
                        underscores=name.count("_"),
                        reason=reason(
                            name,
                            kind,
                            max_underscores,
                            is_public and check_public_number,
                        ),
                    )
                )
    return findings


def scan(
    root: Path, paths: list[str], max_underscores: int, include_public: bool
) -> list[Finding]:
    findings: list[Finding] = file_findings(root, paths, max_underscores)
    for path in sorted(set(source_files(root, paths))):
        if path.suffix == ".rs":
            findings.extend(rust_findings(path, root, max_underscores, include_public))
        elif path.suffix == ".py":
            findings.extend(
                python_findings(path, root, max_underscores, include_public)
            )
    return sorted(findings, key=lambda item: (item.path, item.line, item.name))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="files or directories to scan")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--private-only",
        action="store_true",
        help="skip public definitions and report only private/local definitions",
    )
    parser.add_argument("--max-underscores", type=int, default=1)
    args = parser.parse_args()

    root = repo_root()
    findings = scan(root, args.paths, args.max_underscores, not args.private_only)
    if args.json:
        print(json.dumps([asdict(item) for item in findings], indent=2))
    else:
        for item in findings:
            print(
                f"{item.path}:{item.line}: {item.language} {item.kind} "
                f"{item.name} ({item.underscores} underscores): {item.reason}"
            )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
