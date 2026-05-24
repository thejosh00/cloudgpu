"""Tests for scripts/comfylib.py (the vendored ComfyUI provision helper)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cloudgpu.templates import comfylib


@pytest.fixture
def cl():
    return comfylib


class TestPaths:
    def test_comfy_dir_from_apps(self, cl, monkeypatch, tmp_path):
        monkeypatch.delenv("COMFYUI_DIR", raising=False)
        monkeypatch.setenv("CLOUDGPU_APPS_DIR", str(tmp_path / "apps"))
        assert cl.comfy_dir() == tmp_path / "apps" / "comfyui"

    def test_comfyui_dir_override_wins(self, cl, monkeypatch, tmp_path):
        monkeypatch.setenv("COMFYUI_DIR", str(tmp_path / "custom"))
        assert cl.comfy_dir() == tmp_path / "custom"

    def test_model_subdirs_created(self, cl, monkeypatch, tmp_path):
        monkeypatch.delenv("COMFYUI_DIR", raising=False)
        monkeypatch.setenv("CLOUDGPU_APPS_DIR", str(tmp_path))
        ckpt = cl.checkpoints()
        assert ckpt == tmp_path / "comfyui" / "models" / "checkpoints"
        assert ckpt.is_dir()
        assert cl.loras().name == "loras"

    def test_missing_apps_dir_exits(self, cl, monkeypatch):
        monkeypatch.delenv("COMFYUI_DIR", raising=False)
        monkeypatch.delenv("CLOUDGPU_APPS_DIR", raising=False)
        with pytest.raises(SystemExit):
            cl.comfy_dir()


class TestDownload:
    def test_skips_existing(self, cl, monkeypatch, tmp_path):
        dest = tmp_path / "m.safetensors"
        dest.write_text("already here")
        monkeypatch.setattr(cl.subprocess, "run", lambda *a, **k: pytest.fail("should not download"))
        assert cl.download("http://x/m", dest) == dest

    def test_token_via_stdin_not_argv(self, cl, monkeypatch, tmp_path):
        dest = tmp_path / "m.bin"
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["input"] = kwargs.get("input")
            # simulate curl writing the -o target (the .part file)
            Path(cmd[cmd.index("-o") + 1]).write_text("data")
            return None

        monkeypatch.setattr(cl.subprocess, "run", fake_run)
        cl.download("http://x/m", dest, headers={"Authorization": "Bearer SECRET"})

        assert "SECRET" not in " ".join(captured["cmd"])      # not on the command line
        assert 'Bearer SECRET' in captured["input"]           # passed via stdin config
        assert dest.exists() and not dest.with_name("m.bin.part").exists()


class TestCivitai:
    def test_requires_token(self, cl, monkeypatch, tmp_path):
        monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
        with pytest.raises(SystemExit):
            cl.civitai(123, tmp_path / "x.safetensors")

    def test_builds_url_and_auth_header(self, cl, monkeypatch, tmp_path):
        monkeypatch.setenv("CIVITAI_TOKEN", "TOK")
        captured = {}
        monkeypatch.setattr(cl, "download",
                            lambda url, dest, *, headers=None: captured.update(url=url, headers=headers) or dest)
        cl.civitai(128713, tmp_path / "a.safetensors")
        assert captured["url"] == "https://civitai.com/api/download/models/128713"
        assert captured["headers"] == {"Authorization": "Bearer TOK"}
