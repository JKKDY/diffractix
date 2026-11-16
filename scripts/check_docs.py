"""Check the built documentation site (run after ``zensical build``).

Fails if a Sphinx role is left unrendered, a cross-reference link points at a missing
anchor, or a public object is missing from the API pages.

    python scripts/check_docs.py [site_dir]
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"^::: ([\w.]+)", re.MULTILINE)
LINK = re.compile(r'<a class="autorefs[^"]*"[^>]*?href="([^"]*)"')
ANCHOR = re.compile(r'\sid="([^"]+)"')

# Exported names that are not classes or functions with their own documentation.
NOT_DOCUMENTED = {"units", "AMBIENT_N", "WAVELENGTH"}


def check_site(site: Path) -> list[str]:
    pages = {page.resolve(): page.read_text(encoding="utf-8") for page in site.rglob("index.html")}
    anchors = {page: set(ANCHOR.findall(text)) for page, text in pages.items()}
    problems = []
    links = 0
    for page, text in pages.items():
        name = page.relative_to(site.resolve()).parent.as_posix() or "."
        if re.search(r":(?:meth|func|class|attr|mod|obj):", text):
            problems.append(f"{name}: unrendered Sphinx role")
        for href in LINK.findall(text):
            links += 1
            href = html.unescape(href)
            path, _, fragment = href.partition("#")
            target = page if not path else (page.parent / path).resolve()
            if target.is_dir():
                target = target / "index.html"
            if target not in anchors or (fragment and unquote(fragment) not in anchors[target]):
                problems.append(f"{name}: broken link {href}")
    print(f"{len(pages)} pages, {links} cross-reference links")
    return problems


def check_coverage() -> list[str]:
    import diffractix

    documented = {
        path.rsplit(".", 1)[-1]
        for page in (ROOT / "docs" / "api").glob("*.md")
        for path in BLOCK.findall(page.read_text(encoding="utf-8"))
    }
    missing = sorted(set(diffractix.__all__) - documented - NOT_DOCUMENTED)
    return [f"not on any API page: diffractix.{name}" for name in missing]


def main() -> int:
    site = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "site"
    problems = check_site(site) + check_coverage()
    for problem in problems:
        print("FAIL", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
