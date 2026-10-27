# Development and releases

## Local checks

Use Python 3.11, 3.12, or 3.13 and a virtual environment. From the repository root:

```console
python -m pip install -e ".[dev,jax]"
python -m pip install build twine
python -m pytest tests
python scripts/run_examples.py --backend autograd
python scripts/run_examples.py --backend jax
python -m build
python -m twine check --strict dist/*
```

Run the complete suite with each numerical backend. In PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
$env:DIFFRACTIX_BACKEND = "autograd"
python -m pytest tests
$env:DIFFRACTIX_BACKEND = "jax"
python -m pytest tests
```

In Bash use `MPLBACKEND=Agg DIFFRACTIX_BACKEND=autograd python -m pytest tests`
(and repeat with `jax`). An Autograd-only environment needs just `.[dev]`.
The example runner starts a fresh process for every non-underscore-prefixed
`examples/*.py`, forces Agg, verifies the selected backend, and rejects failed or
infeasible optimization results. All six current examples are standalone programs.
Underscore-prefixed files are reserved for future helper modules.

For a clean wheel check, create a separate environment, install the wheel without
extras, and run the smoke test with `-I` so the checkout and `PYTHONPATH` cannot
provide the imported package:

```powershell
python -m venv scratch/wheel-check
.\scratch\wheel-check\Scripts\python -m pip install (Get-ChildItem dist\*.whl).FullName
$env:DIFFRACTIX_BACKEND = "autograd"
.\scratch\wheel-check\Scripts\python -I scripts\smoke_wheel.py
```

On POSIX use `scratch/wheel-check/bin/python` instead. The smoke test checks the import
location, public exports, README-style forward propagation and inverse design,
and plotting. Matplotlib is a required dependency, not an extra.

## CI and release gates

`ci.yml` runs on pushes to `main`, pull requests, manual dispatch, and reusable
workflow calls. Every Ubuntu/Windows × Python 3.11/3.12/3.13 × Autograd/JAX cell
installs the package, verifies backend selection, runs **all** tests, and runs
**all** standalone examples. Windows JAX CPU support is provided upstream but
is labelled experimental; no matrix cells are excluded or allowed to fail.

Autograd jobs intentionally do not install JAX: their JAX-specific tests skip,
while the JAX jobs execute those tests and cross-backend parity checks. IPOPT and
NLopt are genuinely optional. Their real integration tests use import-time skips
when absent; adapter tests still run. Install `.[dev,all-solvers]` to run those
optional integrations locally. They are not prerequisites for the standard SciPy
examples or the required release matrix.

Separate jobs build an sdist and wheel, check both with Twine, and install the
wheel with its normal dependencies into a clean environment. The existing AGPLv3
`LICENSE` is shipped verbatim; no author identity or only/or-later SPDX qualifier
has been inferred.

On a published GitHub release, `release.yml` checks that the tag is exactly
`v<project.version>` and points to the triggering commit. It calls the same CI
workflow using that immutable commit SHA. Publication needs **both** the tag
check and the entire reusable validation workflow to succeed, including every
matrix cell, the build, and wheel verification. Only then does a separate job
download the same validated `distributions` artifact and publish it. It does
not check out code or rebuild the package. Normal pushes/PRs cannot publish.

## One-time GitHub/PyPI setup

1. Push the workflow files and confirm CI succeeds. Configure desired branch/tag
   protection rules for `main` and release tags such as `v*`.
2. Create the GitHub environment **`pypi`** in this repository. Restrict its
   deployment tags to `v*`; add required reviewers if desired.
3. On PyPI, register a Trusted Publisher (a pending publisher for the first
   release): project **`diffractix`**, owner **`JKKDY`**, repository **`diffractix`**,
   workflow filename **`release.yml`**, environment **`pypi`**. Verify project-name
   availability and repository ownership. No API token or GitHub secret is needed.
4. Review the existing license and release metadata before the first public
   release. Build/inspect the artifacts locally and ensure the full CI is green.
5. With `project.version = "0.0.1"`, create tag **`v0.0.1`** on the intended
   release commit, push it, and publish a GitHub release for that tag. Publishing
   the GitHub release starts fresh validation; a prior CI run is not substituted.
6. Approve the `pypi` environment deployment if configured. Any failed validation
   blocks publication. Inspect the Actions run and the uploaded PyPI files.

After publication, verify in another empty environment:

```console
python -m pip install diffractix==0.0.1
python -c "import diffractix, matplotlib; from importlib.metadata import version; print(version('diffractix'), diffractix.__file__)"
```

The workflows themselves cannot create the GitHub environment, register the PyPI
publisher, or reserve the project name. Never upload a different build manually
to bypass a failed validation run.

References: [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/),
[pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/),
[GitHub reusable workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows),
[JAX installation](https://docs.jax.dev/en/latest/installation.html).
