from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tasks.documentation.docstring_rewriter import main, rewrite_file


def test_rewrite_inserts_function_docstring(tmp_path: Path) -> None:
    source = textwrap.dedent(
        """
        def sample_function(foo: int, bar: str) -> str:
            return str(foo) + bar
        """
    )
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")

    report = rewrite_file(path)

    output = path.read_text(encoding="utf-8")
    assert "Sample function." in output
    assert "Args:\n    foo (int): Foo parameter." in output
    assert "Returns:\n    str: Return value." in output
    assert report.functions_updated == 1


def test_cli_reports_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"
    exit_code = main([str(missing)])
    assert exit_code == 1


def test_rewrite_handles_async_and_class(tmp_path: Path) -> None:
    source = textwrap.dedent(
        """
        class sample_manager:
            async def fetch(self, *, limit: int = 10) -> None:
                return None
        """
    )
    path = tmp_path / "async_module.py"
    path.write_text(source, encoding="utf-8")

    report = rewrite_file(path)
    contents = path.read_text(encoding="utf-8")

    assert "Sample manager class." in contents
    assert "Fetch function." in contents
    assert "limit (int): Limit parameter. Keyword only." in contents
    assert report.classes_updated == 1
    assert report.functions_updated == 1
