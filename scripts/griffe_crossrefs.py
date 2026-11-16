"""Griffe extension: link names in docstrings to the API reference pages.

Docstrings refer to other objects with plain backticks (`Solver.target`) and,
for methods, with a Sphinx role (:meth:`build`). mkdocstrings renders neither as
a link. This extension rewrites them while the docs are built, so the docstrings
themselves stay readable in ``help()``:

* a name that resolves to an object documented on a page under ``docs/api`` becomes a
  link; unqualified names such as `build` are looked up in the enclosing class first;
* the role wrapper is dropped, and a name that does not resolve stays inline code.

Enabled in ``zensical.toml`` through the Python handler's ``extensions`` option.
"""

from __future__ import annotations

import re
from pathlib import Path

import griffe

PACKAGE = "diffractix"
API_PAGES = Path(__file__).resolve().parents[1] / "docs" / "api"

BLOCK = re.compile(r"^::: ([\w.]+)", re.MULTILINE)
CODE = re.compile(
    r"(?<!`)(?::(?:meth|func|class|attr|mod|obj):)?`([A-Za-z_][\w.]*(?:\(\))?)`(?!`)"
)


class DocstringLinks(griffe.Extension):
    """Turn backticked names and Sphinx roles in docstrings into API links."""

    def on_package(self, *, pkg, loader, **kwargs):
        if pkg.name != PACKAGE:
            return
        rendered, roots = self._rendered(pkg)
        for obj in self._walk(pkg):
            if obj.docstring is not None:
                obj.docstring.value = self._rewrite(obj.docstring.value, obj, pkg, rendered, roots)

    @staticmethod
    def _lookup(base, dotted):
        try:
            target = base[dotted]
            return target.final_target if target.is_alias else target
        except Exception:  # unresolved names and aliases to other packages are simply not linked
            return None

    def _walk(self, obj):
        yield obj
        if obj.is_class or obj.is_module:
            for member in obj.members.values():
                if not member.is_alias:
                    yield from self._walk(member)

    def _rendered(self, pkg):
        """Canonical paths of the objects on the API pages, and those objects by short name."""
        rendered, roots, nested = set(), {}, {}

        def collect(obj):
            rendered.add(obj.canonical_path)
            if obj.is_class or obj.is_module:
                for name, member in obj.members.items():
                    # Same filter as mkdocstrings: private names are hidden, dunders are shown.
                    private = name.startswith("_") and not name.startswith("__")
                    if not member.is_alias and not private and member.has_docstring:
                        collect(member)
                        if member.is_class:
                            nested.setdefault(member.name, []).append(member)

        for page in API_PAGES.glob("*.md"):
            for path in BLOCK.findall(page.read_text(encoding="utf-8")):
                dotted = path.removeprefix(PACKAGE).lstrip(".")
                target = self._lookup(pkg, dotted) if dotted else pkg
                if target is not None:
                    collect(target)
                    roots[target.name] = target
        for name, members in nested.items():  # nested classes by short name, if unambiguous
            if len(members) == 1:
                roots.setdefault(name, members[0])
        return rendered, roots

    def _resolve(self, name, obj, pkg, rendered, roots):
        scope = obj
        while scope is not None and not scope.is_class:
            scope = scope.parent
        head, _, rest = name.partition(".")
        bases = [(scope, name), (pkg, name)]
        if head in roots:  # a documented class that is not exported at the top level
            bases.append((roots[head], rest))
        for base, dotted in bases:
            if base is None:
                continue
            target = base if not dotted else self._lookup(base, dotted)
            if target is not None and target.canonical_path in rendered and target.canonical_path != obj.canonical_path:
                return target.canonical_path
        return None

    def _rewrite(self, text, obj, pkg, rendered, roots):
        def replace(match):
            shown = match.group(1)
            target = self._resolve(shown.removesuffix("()"), obj, pkg, rendered, roots)
            return f"[`{shown}`][{target}]" if target else f"`{shown}`"

        lines = []
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith((">>>", "...")):
                lines.append(line)  # doctest input is code, not prose
            else:
                lines.append(CODE.sub(replace, line))
        return "".join(lines)
