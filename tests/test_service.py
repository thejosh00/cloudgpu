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
        assert "User=cloudgpu-app" in unit


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


class TestSandboxedAppUnit:
    """R3: app services run unprivileged + sandboxed, never as sudo-capable ubuntu."""

    @staticmethod
    def _directive(unit: str, key: str) -> str:
        values = [line.split("=", 1)[1] for line in unit.splitlines() if line.startswith(f"{key}=")]
        assert len(values) == 1, f"expected exactly one {key}= line, got {values}"
        return values[0]

    def _comfyui_unit(self, tmp_path):
        from cloudgpu.remote.apps.comfyui import ComfyUIInstaller
        from cloudgpu.remote.state import State

        p = str(tmp_path)
        return p, service.unit_text(ComfyUIInstaller(State(p)).service_spec())

    def test_comfyui_runs_as_unprivileged_user(self, tmp_path):
        _, unit = self._comfyui_unit(tmp_path)
        assert self._directive(unit, "User") == "cloudgpu-app"
        assert self._directive(unit, "Group") == "cloudgpu-app"
        assert "ubuntu" not in unit

    def test_comfyui_sandbox_directives(self, tmp_path):
        _, unit = self._comfyui_unit(tmp_path)
        assert self._directive(unit, "NoNewPrivileges") == "yes"
        assert self._directive(unit, "ProtectSystem") == "strict"
        assert self._directive(unit, "ProtectHome") in ("yes", "tmpfs", "read-only")
        assert self._directive(unit, "PrivateTmp") == "yes"

    def test_comfyui_read_write_paths_are_only_its_own_dirs(self, tmp_path):
        p, unit = self._comfyui_unit(tmp_path)
        assert set(self._directive(unit, "ReadWritePaths").split()) == {
            f"{p}/apps/comfyui",
            f"{p}/venvs/comfyui",
            service.APP_HOME,
        }

    def test_home_is_owned_app_dir_not_ubuntu(self, tmp_path):
        _, unit = self._comfyui_unit(tmp_path)
        assert self._directive(unit, "Environment") == f"HOME={service.APP_HOME}"

    def test_every_installed_app_gets_the_sandbox(self, tmp_path):
        from cloudgpu.remote.apps.aitoolkit import AIToolkitInstaller
        from cloudgpu.remote.state import State

        p = str(tmp_path)
        unit = service.unit_text(AIToolkitInstaller(State(p)).service_spec())
        assert self._directive(unit, "User") == "cloudgpu-app"
        assert self._directive(unit, "NoNewPrivileges") == "yes"
        assert self._directive(unit, "ProtectSystem") == "strict"
        assert set(self._directive(unit, "ReadWritePaths").split()) == {
            f"{p}/apps/ai-toolkit", f"{p}/venvs/ai-toolkit", service.APP_HOME,
        }

    def test_rw_paths_default_to_workdir(self):
        unit = service.unit_text(SPEC)
        assert self._directive(unit, "ReadWritePaths") == f"{SPEC['workdir']} {service.APP_HOME}"


class TestAppUserAndMigration:
    def test_creates_user_without_shell_or_sudo(self, monkeypatch):
        calls = []
        monkeypatch.setattr(service, "run", lambda cmd, **k: calls.append(cmd))

        def missing(name):
            raise KeyError(name)
        monkeypatch.setattr(service.pwd, "getpwnam", missing)
        service.ensure_app_user()
        useradd = next(c for c in calls if c[:2] == ["sudo", "useradd"])
        assert "--system" in useradd and useradd[-1] == "cloudgpu-app"
        assert useradd[useradd.index("--shell") + 1] == "/usr/sbin/nologin"
        assert not any("sudo" in part for c in calls for part in c[2:] if part != "sudo")
        assert ["sudo", "install", "-d", "-o", "cloudgpu-app", "-g", "cloudgpu-app",
                "-m", "750", service.APP_HOME] in calls

    def test_existing_user_not_recreated(self, monkeypatch):
        calls = []
        monkeypatch.setattr(service, "run", lambda cmd, **k: calls.append(cmd))
        monkeypatch.setattr(service.pwd, "getpwnam", lambda name: object())
        service.ensure_app_user()
        assert not any(c[:2] == ["sudo", "useradd"] for c in calls)

    def test_migration_shares_dirs_via_group_without_following_symlinks(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(service, "run", lambda cmd, **k: calls.append(cmd))
        d = str(tmp_path)
        service.migrate_ownership([d, str(tmp_path / "missing")])
        assert ["sudo", "chgrp", "-hR", "cloudgpu-app", d] in calls
        assert ["sudo", "chmod", "-R", "g+rwX", d] in calls
        assert ["sudo", "find", d, "-type", "d", "-exec", "chmod", "g+s", "{}", "+"] in calls
        assert not any(str(tmp_path / "missing") in c for c in calls)  # skips absent dirs

    def test_migration_is_idempotent(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(service, "run", lambda cmd, **k: calls.append(cmd))
        service.migrate_ownership([str(tmp_path)])
        first = list(calls)
        service.migrate_ownership([str(tmp_path)])
        assert calls == first + first

    def test_install_creates_user_and_migrates_before_starting(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(service, "run", lambda cmd, **k: calls.append(cmd))
        monkeypatch.setattr(service.pwd, "getpwnam", lambda name: object())
        spec = {**SPEC, "workdir": str(tmp_path), "rw_paths": [str(tmp_path)]}
        service.install_service(spec)
        chgrp = calls.index(["sudo", "chgrp", "-hR", "cloudgpu-app", str(tmp_path)])
        start = calls.index(["sudo", "systemctl", "enable", "--now", "comfyui"])
        assert chgrp < start
