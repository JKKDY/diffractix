# AGENTS.md

Context for coding agents working on this repository. Humans: see `README.md` and
`DEVELOPMENT.md`.

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

## Commands

The `.venv` is managed by uv and has no pip: install with
`uv pip install --python .venv/Scripts/python.exe <package>`.

```bash
DIFFRACTIX_BACKEND=autograd .venv/Scripts/python.exe -m pytest -q   # also run with jax
.venv/Scripts/python.exe -m ruff check .
MPLBACKEND=Agg .venv/Scripts/python.exe scripts/run_examples.py --backend autograd
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
- Files use CRLF line endings; preserve them when editing.
- Change tests, examples and docs in the same step as the code they cover.
- Library (`src/`) changes stay small and are agreed with the maintainer first.

## More context

- `docs/agents/`: notes written for agents, not published. `diffractix-llm-guide.md`
  (API usage), `system-simulation-result.md` and `graph-reference.md` (how the
  pieces fit), `testing-specifications/` (what each test area must cover),
  `future-ideas.md` (post-0.0.1 ideas, not to be built unasked).
- `DEVELOPMENT.md`: local checks, CI and the release process.
- `ARCHITECTURE.md`, `USAGE.md`, `THEORY.md` predate recent API changes; check them
  against the code before relying on them.
