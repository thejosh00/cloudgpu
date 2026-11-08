"""The remote `recover` command must not hide failed app recoveries (plan item 0.5)."""

from __future__ import annotations

import sys

import pytest

from cloudgpu.remote import __main__ as remote_main
from cloudgpu.remote import recovery
from cloudgpu.remote.state import State


def _fake_installer(name, calls, error=None):
    class Fake:
        def __init__(self, state):
            self.state = state

        def recover(self):
            calls.append(name)
            if error is not None:
                raise RuntimeError(error)

    return Fake


def _recover(monkeypatch, persistent_dir, installers, recorded):
    """Run the remote `recover` through main(); return (exit code, stderr+stdout, calls)."""
    state = State(persistent_dir)
    for app in recorded:
        state.set_app(app, {"status": "installed"})
    path_calls = []
    monkeypatch.setattr(recovery, "setup_path", lambda st: path_calls.append(st))
    monkeypatch.setattr(remote_main, "APP_INSTALLERS", installers)
    monkeypatch.setattr(sys, "argv", ["remote", "--persistent-dir", persistent_dir, "recover"])
    code = 0
    try:
        remote_main.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return code, path_calls


def test_one_failure_keeps_going_and_exits_nonzero(monkeypatch, tmp_persistent_dir, capsys):
    """Catches: a failed app recovery swallowed, so `recover` exits 0 and `up` says the box is up."""
    calls = []
    installers = {
        "comfyui": _fake_installer("comfyui", calls, error="venv missing"),
        "ai-toolkit": _fake_installer("ai-toolkit", calls),
    }
    code, path_calls = _recover(monkeypatch, tmp_persistent_dir, installers,
                                ["comfyui", "ai-toolkit"])
    assert calls == ["comfyui", "ai-toolkit"], "the second app must still be recovered"
    assert len(path_calls) == 1, "setup_path must still run"
    assert code != 0


def test_failure_named_with_error_and_no_complete(monkeypatch, tmp_persistent_dir, capsys):
    """Catches: a failure summary that omits the app or its error, or still says 'Recovery complete!'."""
    calls = []
    installers = {
        "comfyui": _fake_installer("comfyui", calls, error="venv missing"),
        "ai-toolkit": _fake_installer("ai-toolkit", calls),
    }
    code, _ = _recover(monkeypatch, tmp_persistent_dir, installers, ["comfyui", "ai-toolkit"])
    out = capsys.readouterr()
    text = out.err + out.out
    assert code != 0
    assert "Recovery complete!" not in text
    summary = text[text.index("Recovery failed"):]
    assert "comfyui: venv missing" in summary
    assert "ai-toolkit" not in summary


def test_two_failures_both_named(monkeypatch, tmp_persistent_dir, capsys):
    """Catches: a summary that names only the first (or last) failed app."""
    calls = []
    installers = {
        "comfyui": _fake_installer("comfyui", calls, error="venv missing"),
        "ai-toolkit": _fake_installer("ai-toolkit", calls, error="clone failed"),
    }
    code, path_calls = _recover(monkeypatch, tmp_persistent_dir, installers,
                                ["comfyui", "ai-toolkit"])
    out = capsys.readouterr()
    text = out.err + out.out
    assert code != 0
    assert calls == ["comfyui", "ai-toolkit"]
    assert len(path_calls) == 1
    summary = text[text.index("Recovery failed"):]
    assert "comfyui: venv missing" in summary
    assert "ai-toolkit: clone failed" in summary
    assert "Recovery complete!" not in text


def test_all_recover_exits_zero(monkeypatch, tmp_persistent_dir, capsys):
    """Catches: the fix making a fully successful recovery exit non-zero or drop its message."""
    calls = []
    installers = {
        "comfyui": _fake_installer("comfyui", calls),
        "ai-toolkit": _fake_installer("ai-toolkit", calls),
    }
    code, path_calls = _recover(monkeypatch, tmp_persistent_dir, installers,
                                ["comfyui", "ai-toolkit"])
    text = "".join(capsys.readouterr())
    assert code == 0
    assert calls == ["comfyui", "ai-toolkit"]
    assert len(path_calls) == 1
    assert "Recovery complete!" in text
    assert "Recovery failed" not in text


def test_nothing_recorded_exits_zero(monkeypatch, tmp_persistent_dir, capsys):
    """Catches: an empty state.json treated as a failure."""
    code, _ = _recover(monkeypatch, tmp_persistent_dir, {}, [])
    assert code == 0
    assert "nothing to recover" in "".join(capsys.readouterr())


def test_app_without_installer_stays_a_warning(monkeypatch, tmp_persistent_dir, capsys):
    """Catches: folding the 'no installer' skip into the failures (out of scope for 0.5)."""
    calls = []
    installers = {"comfyui": _fake_installer("comfyui", calls)}
    code, _ = _recover(monkeypatch, tmp_persistent_dir, installers, ["comfyui", "mystery"])
    text = "".join(capsys.readouterr())
    assert code == 0
    assert "Skipping mystery: no installer available" in text
    assert "Recovery complete!" in text

