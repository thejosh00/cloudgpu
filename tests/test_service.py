"""Tests for the remote systemd service helpers (no real sudo/systemctl)."""

from __future__ import annotations

from cloudgpu.remote import service

SPEC = {
    "name": "comfyui",
    "exec_start": "/lambda/nfs/p/cloudgpu/bin/comfyui",
    "workdir": "/lambda/nfs/p/apps/comfyui",
    "port": 8188,
}


class TestUnitText:
    def test_contains_key_directives(self):
        unit = service.unit_text(SPEC)
        assert "Description=cloudgpu comfyui" in unit
        assert "ExecStart=/lambda/nfs/p/cloudgpu/bin/comfyui" in unit
        assert "WorkingDirectory=/lambda/nfs/p/apps/comfyui" in unit
        assert "Restart=always" in unit
        assert "WantedBy=multi-user.target" in unit
        assert "User=ubuntu" in unit


class TestInstallService:
    def test_writes_unit_and_enables(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(service, "run", lambda cmd, **k: calls.append(cmd))
        # write to a temp path we can inspect (avoid /tmp clutter is fine, but check cmds)
        service.install_service(SPEC)

        # install the unit, reload, enable --now
        assert any(c[:2] == ["sudo", "install"] and c[-1] == "/etc/systemd/system/comfyui.service" for c in calls)
        assert ["sudo", "systemctl", "daemon-reload"] in calls
        assert ["sudo", "systemctl", "enable", "--now", "comfyui"] in calls


class TestRestartAndActive:
    def test_restart(self, monkeypatch):
        calls = []
        monkeypatch.setattr(service, "run", lambda cmd, **k: calls.append(cmd))
        service.restart_service("comfyui")
        assert ["sudo", "systemctl", "restart", "comfyui"] in calls

    def test_active_parses_output(self, monkeypatch):
        class R:
            stdout = "active\n"
        monkeypatch.setattr(service, "run", lambda cmd, **k: R())
        assert service.service_active("comfyui") == "active"

    def test_active_is_best_effort(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("no systemctl")
        monkeypatch.setattr(service, "run", boom)
        assert service.service_active("comfyui") == "unknown"
