"""Google-style docstring rewriter CLI.

This module exposes a command-line entry point that rewrites module, class, and
function docstrings into the Google style guide format. It operates by parsing
Python source files into an abstract syntax tree, transforming docstring
literals, and emitting the updated source via ``ast.unparse``. The tool accepts
individual files or directories (recursively) and logs a concise summary of
changes performed.
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


class GoogleDocstringTransformer(ast.NodeTransformer):
    """AST transformer that injects Google-style docstrings."""

    def __init__(self) -> None:
        self.module_updated = False
        self.classes_updated = 0
        self.functions_updated = 0

    # -- Node visitors -----------------------------------------------------
    def visit_Module(self, node: ast.Module) -> ast.AST:  # noqa: N802 (ast API)
        self.generic_visit(node)
        docstring = build_module_docstring(node)
        if docstring:
            if update_docstring(node.body, docstring):
                self.module_updated = True
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        docstring = build_class_docstring(node)
        if docstring and update_docstring(node.body, docstring):
            self.classes_updated += 1
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        docstring = build_function_docstring(node)
        if docstring and update_docstring(node.body, docstring):
            self.functions_updated += 1
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        docstring = build_function_docstring(node)
        if docstring and update_docstring(node.body, docstring):
            self.functions_updated += 1
        return node


# -- Formatting helpers -------------------------------------------------------

def update_docstring(body: List[ast.stmt], value: str) -> bool:
    """Insert or replace the first statement with a string constant."""

    doc_expr = ast.Expr(value=ast.Constant(value=value))
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if body[0].value.value == value:
            return False
        body[0] = doc_expr
        return True
    body.insert(0, doc_expr)
    return True


def build_module_docstring(module: ast.Module) -> str:
    """Create a Google-style module docstring."""

    summary = "Module docstring." if module.body else "Module."  # fallback
    return f"{summary}\n\nReturns:\n    None"


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
    summary = build_summary(node.name, "function")
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


def iter_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterable[ParameterDoc]:
    """Yield metadata for function parameters."""

    skip = {"self", "cls"}
    for arg in node.args.posonlyargs:
        if arg.arg not in skip:
            yield ParameterDoc(arg.arg, annotation_to_str(arg.annotation), "positional-only")
    for arg in node.args.args:
        if arg.arg not in skip:
            yield ParameterDoc(arg.arg, annotation_to_str(arg.annotation), "positional")
    if node.args.vararg is not None:
        yield ParameterDoc(node.args.vararg.arg, annotation_to_str(node.args.vararg.annotation), "var-positional")
    for arg in node.args.kwonlyargs:
        yield ParameterDoc(arg.arg, annotation_to_str(arg.annotation), "keyword-only")
    if node.args.kwarg is not None:
        yield ParameterDoc(node.args.kwarg.arg, annotation_to_str(node.args.kwarg.annotation), "var-keyword")


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


# -- CLI plumbing -------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Rewrite docstrings into Google style")
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to rewrite")
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
            for candidate in path.rglob("*.py"):
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
    transformer = GoogleDocstringTransformer()
    transformer.visit(tree)
    ast.fix_missing_locations(tree)
    rewritten = ast.unparse(tree)
    if not rewritten.endswith("\n"):
        rewritten += "\n"
    path.write_text(rewritten, encoding="utf-8")
    return RewriteReport(
        path=path,
        module_updated=transformer.module_updated,
        classes_updated=transformer.classes_updated,
        functions_updated=transformer.functions_updated,
    )


def configure_logging(quiet: bool) -> None:
    """Configure logging for CLI usage."""

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.WARNING if quiet else logging.INFO)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the CLI."""

    args = parse_args(argv)
    configure_logging(args.quiet)
    try:
        targets = expand_targets(args.paths)
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1

    reports: List[RewriteReport] = []
    for target in targets:
        LOGGER.info("Rewriting docstrings in %s", target)
        try:
            reports.append(rewrite_file(target))
        except SyntaxError as error:
            LOGGER.error("Failed to parse %s: %s", target, error)
            return 1

    total_changes = sum(report.changes for report in reports)
    LOGGER.info("Updated %d docstrings across %d files", total_changes, len(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
