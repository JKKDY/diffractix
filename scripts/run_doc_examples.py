"""Run the Python code blocks of the documentation pages.

Each page runs in its own Python process, so a page may choose its backend. Its
``python`` blocks run in order in one namespace, so a page can build on its earlier
blocks. A ``text`` block titled ``Output`` must match what the preceding
Python block prints. The API pages and ``docs/agents/`` are skipped.

    python scripts/run_doc_examples.py [page.md ...]
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FENCE = re.compile(r"^```(\w+)([^\n]*)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def pages() -> list[Path]:
    skipped = (DOCS / "agents", DOCS / "api")
    return sorted(
        page for page in DOCS.rglob("*.md")
        if not any(page.is_relative_to(directory) for directory in skipped)
    )


def run_page(page: Path) -> list[str]:
    text = page.read_text(encoding="utf-8")
    name = page.relative_to(ROOT).as_posix()
    namespace = {"__name__": "__docs__"}
    problems = []
    printed = None
    for match in FENCE.finditer(text):
        language, info, code = match.groups()
        line = text.count("\n", 0, match.start()) + 1
        if language == "python":
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    exec(compile(code, f"{name}:{line}", "exec"), namespace)
            except Exception:
                problems.append(f"{name}:{line}: block failed\n{traceback.format_exc()}")
                return problems
            printed = output.getvalue()
        elif language == "text" and 'title="Output"' in info:
            if printed is None or printed.strip() != code.strip():
                problems.append(
                    f"{name}:{line}: output differs\n--- expected\n{code.strip()}\n"
                    f"--- printed\n{(printed or '').strip()}"
                )
    return problems


def main(arguments: list[str]) -> int:
    if arguments[:1] == ["--page"]:
        problems = run_page(Path(arguments[1]))
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    selected = [Path(argument).resolve() for argument in arguments] or pages()
    failed = 0
    for page in selected:
        process = subprocess.run([sys.executable, __file__, "--page", str(page)])
        failed += process.returncode != 0
    print(f"{len(selected)} pages checked, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
