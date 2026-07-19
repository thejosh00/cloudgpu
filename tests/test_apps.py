"""Tests for the local app registry (cloudgpu.local.apps)."""

from __future__ import annotations

from cloudgpu.local import apps


def test_available_apps_includes_comfyui():
    assert "comfyui" in apps.AVAILABLE_APPS


def test_available_apps_includes_aitoolkit():
    assert "ai-toolkit" in apps.AVAILABLE_APPS


def test_app_ports():
    assert apps.app_ports(["comfyui"]) == [8188]
    assert apps.app_ports(["ai-toolkit"]) == [8675]
    assert apps.app_ports(["comfyui", "ai-toolkit"]) == [8188, 8675]
    assert apps.app_ports([]) == []
    assert apps.app_ports(["unknown"]) == []


def test_app_ports_deduped_in_order(monkeypatch):
    from cloudgpu.local.apps import AppSpec
    monkeypatch.setattr(apps, "APPS", {
        "a": AppSpec("a", ports=(8188, 7000)),
        "b": AppSpec("b", ports=(7000, 9000)),
    })
    assert apps.app_ports(["a", "b"]) == [8188, 7000, 9000]


def test_service_apps():
    assert apps.service_apps(["comfyui"]) == ["comfyui"]
    assert apps.service_apps(["ai-toolkit"]) == ["ai-toolkit"]
    assert apps.service_apps([]) == []
    assert apps.service_apps(["unknown"]) == []


def test_scaffold_apps_comfyui(tmp_path):
    written = apps.scaffold_apps(tmp_path, ["comfyui"])
    assert set(written) == {"comfylib.py", "provision.py"}
    assert (tmp_path / "comfylib.py").exists()
    assert (tmp_path / "provision.py").exists()


def test_scaffold_apps_bare(tmp_path):
    assert apps.scaffold_apps(tmp_path, []) == []
    assert list(tmp_path.iterdir()) == []


def test_scaffold_apps_aitoolkit_writes_nothing(tmp_path):
    assert apps.scaffold_apps(tmp_path, ["ai-toolkit"]) == []
    assert list(tmp_path.iterdir()) == []


def test_scaffold_does_not_clobber_provision(tmp_path):
    (tmp_path / "provision.py").write_text("# my edits\n")
    written = apps.scaffold_apps(tmp_path, ["comfyui"])
    assert "provision.py" not in written           # preserved
    assert (tmp_path / "provision.py").read_text() == "# my edits\n"
    assert (tmp_path / "comfylib.py").exists()
