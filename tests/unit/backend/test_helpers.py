"""Example diagnostics must not interfere with the numerical JSON protocol."""
import pytest

from tests.helpers import backend


def test_example_setup_routes_prints_to_stderr_and_stops_before_solve(tmp_path, monkeypatch, capsys):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "noisy.py").write_text(
        'import sys\n'
        'print("setup diagnostic")\n'
        'print("stderr diagnostic", file=sys.stderr)\n'
        'value = 42\n'
        'solver.solve()\n'
        'raise AssertionError("must not run optimization or later statements")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(backend, "ROOT", tmp_path)
    monkeypatch.setitem(backend.sys.modules, "_backend_test_noisy", None)

    namespace = backend.example_before_solve("noisy")

    assert namespace["value"] == 42
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "setup diagnostic\nstderr diagnostic\n"


def test_example_setup_errors_are_not_swallowed_by_output_capture(tmp_path, monkeypatch, capsys):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "broken.py").write_text(
        'print("before failure")\n'
        'raise ValueError("invalid example setup")\n'
        'solver.solve()\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(backend, "ROOT", tmp_path)
    monkeypatch.setitem(backend.sys.modules, "_backend_test_broken", None)

    with pytest.raises(ValueError, match="invalid example setup"):
        backend.example_before_solve("broken")

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "before failure\n"
