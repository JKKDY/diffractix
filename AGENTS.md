# AGENTS.md

Context for coding agents working on this repository. Humans: see `README.md`.

## What this is

Diffractix is a Python library for differentiable paraxial optics and inverse design:
build an optical system from elements, simulate rays or Gaussian beams with ABCD
matrices, and optimize design parameters with gradients from JAX or Autograd.

## Layout (`src/diffractix/`)

- `graph/`: expression graph (`Parameter`, `Symbol`, `Node`), compilation to programs.
- `core/`: `OpticalBase` (parameters, labels, constraints) and `units` (unit constants).
- `elements/`, `composites/`: optical elements and composites built from them.
- `states/`: light states (`Ray`, `RayBundle`, `GaussianBeam`), used as inputs and in results.
- `system/`: `System`, placement and the build into a `Simulation`.
- `simulation/`: compiled `Simulation`, results, and the symbolic result.
- `solver/`: `Solver`, `Solution`, objectives, constraints; `optimizers/` wraps SciPy, IPOPT, NLopt.
- `plotting/`, `backend/` (numerical backend selection), `units.py` (re-exports `core.units`).

Outside `src/`: `docs/` holds the published site (Zensical, configured in
`zensical.toml`; `docs/api/*.md` are `::: diffractix.X` blocks rendered from the
docstrings) and, unpublished, `docs/agents/`. `scripts/` has `run_examples.py`,
`griffe_crossrefs.py` (links names in docstrings to the API pages) and `check_docs.py`.

## Commands

The `.venv` is managed by uv and has no pip: install with
`uv pip install --python .venv/Scripts/python.exe <package>`.

```bash
DIFFRACTIX_BACKEND=autograd .venv/Scripts/python.exe -m pytest -q   # also run with jax
.venv/Scripts/python.exe -m ruff check .
MPLBACKEND=Agg .venv/Scripts/python.exe scripts/run_examples.py --backend autograd
```

Documentation (needs `uv pip install --python .venv/Scripts/python.exe -e ".[docs]"`):

```bash
.venv/Scripts/python.exe -m zensical build --clean --strict
.venv/Scripts/python.exe scripts/check_docs.py   # links, unrendered roles, API coverage
.venv/Scripts/python.exe -m zensical serve       # preview at http://localhost:8000
```

The full suite on both backends is slow on the maintainer's laptop: run the tests
affected by a change, and the full suite only before finishing larger work.

## Conventions

- Values are SI internally; examples write them with `diffractix.units` (`50 * mm`).
- NumPy-style docstrings on the public API (what `diffractix.__all__` exports),
  following `docs/agents/docstring-style.md`.
- Lint only (ruff `F`, `E9`); no formatter. Keep the existing style, including
  aligned blocks. Do not run `ruff check --fix` blindly: it deletes imports that are
  only re-exported.
- Keep each file's existing line endings (most are CRLF, some tests are LF); check
  with Python byte counts, not `grep`, and never leave a file mixed.
- Docstrings refer to other objects with plain backticks (`Solver.target`); a method
  may also use `:meth:`. The docs build turns names that are documented on an API page
  into links, so a new public class or function goes on a page under `docs/api/`
  (`check_docs.py` fails if an exported name is missing).
- Change tests, examples and docs in the same step as the code they cover.
- Library (`src/`) changes stay small and are agreed with the maintainer first.
- Keep the code and API clean; do not overengineer. Before adding machinery for a
  fix or review finding (class-level registries, metaprogramming such as
  `__init_subclass__` or wrapped properties, special-case fields, lone module-level
  helpers), say that it would overcomplicate things and propose the simple option:
  an explicit line where it is needed, a sentence in the docs, or leaving it.
  Explicit beats implicit: `self.require(self.n > 0)` in a class's own
  `__post_init__`, not a declarative attribute interpreted by a base class.

## More context

- `docs/agents/`: notes written for agents, not published. `diffractix-llm-guide.md`
  (API usage), `system-simulation-result.md` and `graph-reference.md` (how the
  pieces fit), `testing-specifications/` (what each test area must cover),
  `future-ideas.md` (post-0.1.0 ideas, not to be built unasked).
- `.github/workflows/`: `ci.yml` (lint, tests on Linux and Windows for Python
  3.11-3.14 on both backends with coverage to Codecov, an oldest-dependencies job,
  wheel checks), `docs.yml` (builds the site on every push and PR, publishes it to
  GitHub Pages from `main`) and `release.yml` (tag-triggered PyPI release that
  reruns CI).
- `pyproject.toml`: dependencies and their minimums, extras, tool settings
  (pytest, ruff, coverage).
