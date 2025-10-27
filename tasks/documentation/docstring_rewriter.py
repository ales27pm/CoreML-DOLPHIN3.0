"""Google-style docstring rewriter CLI.

This module exposes a command-line entry point that rewrites module, class, and
function docstrings into the Google style guide format. Source files are parsed
into an abstract syntax tree, but rather than serialising the full module the
tool surgically replaces only the docstring literals in the original source.
The non-docstring content—including comments, blank lines, and formatting—thus
remains untouched.
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

LOGGER = logging.getLogger("docstring_rewriter")


@dataclass
class RewriteReport:
    """Summary of rewrites applied to a single Python source file."""

    path: Path
    module_updated: bool
    classes_updated: int
    functions_updated: int

    @property
    def changes(self) -> int:
        """Return the total number of docstrings rewritten."""

        return int(self.module_updated) + self.classes_updated + self.functions_updated


@dataclass
class TextEdit:
    """Representation of a textual replacement."""

    start: int
    end: int
    replacement: str


class DocstringPlanner(ast.NodeVisitor):
    """Plan docstring rewrites without reserialising full modules."""

    def __init__(self, source: str, line_offsets: List[int]):
        self.source = source
        self.line_offsets = line_offsets
        self.rewrites: List[TextEdit] = []
        self.module_updated = False
        self.classes_updated = 0
        self.functions_updated = 0

    # -- Node visitors -----------------------------------------------------
    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802 (ast API)
        docstring = build_module_docstring(node)
        if docstring and self._plan_docstring(node, node.body, docstring, ""):
            self.module_updated = True
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        docstring = build_class_docstring(node)
        if docstring and self._plan_docstring(
            node, node.body, docstring, self._indent_for(node)
        ):
            self.classes_updated += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        docstring = build_function_docstring(node)
        if docstring and self._plan_docstring(
            node, node.body, docstring, self._indent_for(node)
        ):
            self.functions_updated += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        docstring = build_function_docstring(node)
        if docstring and self._plan_docstring(
            node, node.body, docstring, self._indent_for(node)
        ):
            self.functions_updated += 1
        self.generic_visit(node)

    # -- Planning helpers --------------------------------------------------
    def _plan_docstring(
        self,
        node: ast.AST,
        body: List[ast.stmt],
        value: str,
        indent: str,
    ) -> bool:
        if body:
            current = _string_literal(body[0])
            if current is not None:
                if current == value:
                    return False
                start = self._to_offset(body[0].lineno, body[0].col_offset)
                end = self._to_offset(body[0].end_lineno, body[0].end_col_offset)
                replacement = format_docstring_literal(value, indent)
                self.rewrites.append(
                    TextEdit(start=start, end=end, replacement=replacement)
                )
                return True

        insertion_offset = self._insertion_offset(node, body, indent)
        replacement = self._insertion_text(value, indent, body)
        self.rewrites.append(
            TextEdit(
                start=insertion_offset, end=insertion_offset, replacement=replacement
            )
        )
        return True

    def _indent_for(self, node: ast.AST) -> str:
        if isinstance(node, ast.Module):
            return ""
        body: List[ast.stmt] = getattr(node, "body", []) or []
        if body:
            col = getattr(body[0], "col_offset", 0)
        else:
            col = getattr(node, "col_offset", 0) + 4
        return " " * col

    def _to_offset(self, lineno: int | None, col: int | None) -> int:
        if lineno is None or col is None:
            return len(self.source)
        if lineno - 1 < 0:
            return 0
        if lineno - 1 >= len(self.line_offsets):
            return len(self.source)
        return self.line_offsets[lineno - 1] + col

    def _module_insert_offset(self) -> int:
        position = 0
        for index, line in enumerate(self.source.splitlines(keepends=True)):
            stripped = line.lstrip()
            if index == 0 and line.startswith("#!"):
                position += len(line)
                continue
            if index < 2 and stripped.startswith("#") and "coding" in stripped:
                position += len(line)
                continue
            if stripped.startswith("#"):
                position += len(line)
                continue
            if not stripped.strip():
                position += len(line)
                continue
            break
        return position

    def _insertion_offset(
        self, node: ast.AST, body: List[ast.stmt], indent: str
    ) -> int:
        if isinstance(node, ast.Module):
            return self._module_insert_offset()
        if body:
            target = body[0]
            return self._to_offset(target.lineno, target.col_offset)
        return self._to_offset(
            getattr(node, "end_lineno", None), getattr(node, "end_col_offset", None)
        )

    def _insertion_text(self, value: str, indent: str, body: List[ast.stmt]) -> str:
        literal = format_docstring_literal(value, indent)
        if body:
            if not indent:
                return literal + "\n\n"
            return literal + "\n"
        return literal + "\n"


def _string_literal(node: ast.stmt) -> str | None:
    if isinstance(node, ast.Expr):
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def _is_string_expr(node: ast.stmt) -> bool:
    return _string_literal(node) is not None


def build_module_docstring(module: ast.Module) -> str:
    """Create a Google-style module docstring."""

    summary = "Module docstring." if module.body else "Module."  # fallback
    return summary


def build_class_docstring(node: ast.ClassDef) -> str:
    """Create a Google-style class docstring."""

    summary = build_summary(node.name, "class")
    bases = [ast.unparse(base) for base in node.bases] if node.bases else []
    sections: List[str] = [summary]
    if bases:
        bases_line = ", ".join(bases)
        sections.append(f"Bases:\n    {bases_line}")
    sections.append("Attributes:\n    None")
    return "\n\n".join(sections)


def build_function_docstring(node: ast.AST) -> str:
    """Create a Google-style function or coroutine docstring."""

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    role = "coroutine" if isinstance(node, ast.AsyncFunctionDef) else "function"
    summary = build_summary(node.name, role)
    params = list(iter_parameters(node))
    args_block = format_args_block(params)
    returns_block = format_returns_block(node.returns)
    sections = [summary, args_block, returns_block]
    return "\n\n".join(sections)


def format_args_block(params: List[ParameterDoc]) -> str:
    """Render the Args section."""

    if not params:
        return "Args:\n    None"
    lines = [param.to_doc_line() for param in params]
    return "Args:\n" + "\n".join(f"    {line}" for line in lines)


def format_returns_block(annotation: ast.expr | None) -> str:
    """Render the Returns section."""

    if annotation is None:
        return "Returns:\n    None"
    return_type = ast.unparse(annotation)
    if return_type in {"None", "NoReturn"}:
        return "Returns:\n    None"
    return f"Returns:\n    {return_type}: Return value."


@dataclass
class ParameterDoc:
    """Parameter documentation payload."""

    name: str
    annotation: str | None
    kind: str

    def to_doc_line(self) -> str:
        """Convert the parameter descriptor into a Google-style line."""

        annotation = self.annotation or "Any"
        description = f"{humanize_name(self.name)} parameter."
        if self.kind == "positional-only":
            description += " Positional only."
        elif self.kind == "keyword-only":
            description += " Keyword only."
        elif self.kind == "var-positional":
            description += " Variable positional argument."
        elif self.kind == "var-keyword":
            description += " Variable keyword arguments."
        return f"{self.name} ({annotation}): {description}"


def iter_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterable[ParameterDoc]:
    """Yield metadata for function parameters."""

    skip = {"self", "cls"}
    for arg in node.args.posonlyargs:
        if arg.arg not in skip:
            yield ParameterDoc(
                arg.arg, annotation_to_str(arg.annotation), "positional-only"
            )
    for arg in node.args.args:
        if arg.arg not in skip:
            yield ParameterDoc(arg.arg, annotation_to_str(arg.annotation), "positional")
    if node.args.vararg is not None:
        yield ParameterDoc(
            node.args.vararg.arg,
            annotation_to_str(node.args.vararg.annotation),
            "var-positional",
        )
    for arg in node.args.kwonlyargs:
        yield ParameterDoc(arg.arg, annotation_to_str(arg.annotation), "keyword-only")
    if node.args.kwarg is not None:
        yield ParameterDoc(
            node.args.kwarg.arg,
            annotation_to_str(node.args.kwarg.annotation),
            "var-keyword",
        )


def annotation_to_str(annotation: ast.expr | None) -> str | None:
    """Return the string representation of an annotation if present."""

    if annotation is None:
        return None
    return ast.unparse(annotation)


def humanize_name(name: str) -> str:
    """Convert a snake_case identifier into a capitalised phrase."""

    words = re.split(r"[_\-]+", name)
    cleaned = " ".join(word for word in words if word)
    if not cleaned:
        cleaned = name
    return cleaned[0].upper() + cleaned[1:]


def build_summary(name: str, role: str | None = None) -> str:
    """Create a sentence-style summary for an identifier and optional role."""

    base = humanize_name(name)
    if role:
        if not base.lower().endswith(role):
            return f"{base} {role}."
        return f"{base}."
    return f"{base}."


def format_docstring_literal(value: str, indent: str) -> str:
    """Return a triple-quoted literal for ``value`` respecting indentation."""

    escaped = value.replace('"""', '\\"""')
    if "\n" in escaped:
        return f'{indent}"""\n{escaped}\n{indent}"""'
    return f'{indent}"""{escaped}"""'


def compute_line_offsets(source: str) -> List[int]:
    """Return start offsets for each 1-indexed source line."""

    offsets: List[int] = []
    position = 0
    for line in source.splitlines(keepends=True):
        offsets.append(position)
        position += len(line)
    return offsets


# -- CLI plumbing -------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Rewrite docstrings into Google style")
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Files or directories to rewrite"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logging and only emit warnings",
    )
    return parser.parse_args(argv)


def expand_targets(paths: Sequence[Path]) -> List[Path]:
    """Expand file and directory inputs into a unique, sorted file list."""

    files: set[Path] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if path.is_dir():
            ignored = {
                "venv",
                ".venv",
                "node_modules",
                ".git",
                "build",
                "dist",
                ".tox",
                ".mypy_cache",
                ".pytest_cache",
            }
            for candidate in path.rglob("*.py"):
                if any(part in ignored for part in candidate.parts):
                    continue
                if candidate.is_file():
                    files.add(candidate)
        else:
            if path.suffix != ".py":
                raise ValueError(f"Unsupported file extension for {path}")
            files.add(path)
    return sorted(files)


def rewrite_file(path: Path) -> RewriteReport:
    """Rewrite docstrings within a single Python file."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    planner = DocstringPlanner(source=source, line_offsets=compute_line_offsets(source))
    planner.visit(tree)

    rewritten = source
    if planner.rewrites:
        for edit in sorted(planner.rewrites, key=lambda item: item.start, reverse=True):
            rewritten = (
                rewritten[: edit.start] + edit.replacement + rewritten[edit.end :]
            )
        path.write_text(rewritten, encoding="utf-8")
    return RewriteReport(
        path=path,
        module_updated=planner.module_updated,
        classes_updated=planner.classes_updated,
        functions_updated=planner.functions_updated,
    )


def configure_logging(quiet: bool) -> None:
    """Configure logging for CLI usage."""

    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    LOGGER.propagate = False
    LOGGER.setLevel(logging.WARNING if quiet else logging.INFO)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the CLI."""

    args = parse_args(argv)
    configure_logging(args.quiet)
    try:
        targets = expand_targets(args.paths)
    except (FileNotFoundError, ValueError) as error:
        LOGGER.exception("Failed to expand targets: %s", error)
        return 1

    reports: List[RewriteReport] = []
    for target in targets:
        LOGGER.info("Rewriting docstrings in %s", target)
        try:
            reports.append(rewrite_file(target))
        except SyntaxError as error:
            LOGGER.exception("Failed to parse %s: %s", target, error)
            return 1

    total_changes = sum(report.changes for report in reports)
    LOGGER.info("Updated %d docstrings across %d files", total_changes, len(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
