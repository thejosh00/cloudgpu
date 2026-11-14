"""A config can list profiles, and `up` always provisions them (plan item 2.1).

No real API, instance or key. The `up` tests run the real CLI in a subprocess against the fake
Lambda with ssh and rsync shimmed; `down` and `status` run in-process with the API and ssh mocked.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import config, configs, lambda_api
from tests.fake_lambda import FakeLambda, run_cloudgpu

BASE = 'gpu = ["gh200"]\nssh_key = "mini"\n'
AB = 'profiles = ["a", "b"]\n'
COMFY = {"comfyui": {"status": "installed", "version": "1", "app_dir": "/lambda/nfs/box-fs/apps/comfyui"}}
MARKED = {"comfyui": {**COMFY["comfyui"], "reprovision": True}}


def _write_config(base, name, body):
    d = base / "configs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.toml").write_text(body)
    return d / f"{name}.toml"


def _profile(workdir, name, apps=()):
    d = workdir / "profiles" / name
    d.mkdir(parents=True)
    (d / "cloudgpu.toml").write_text(f"apps = {json.dumps(list(apps))}\n")
    (d / "provision.py").write_text("print('provisioned')\n")


def _provisions(result):
    """The profile names whose provision entry point ran, in order."""
    return [c["command"].split("/cloudgpu/provision/")[1].split(" ")[0] for c in result.shim_calls
            if c["tool"] == "ssh" and "provision.py" in c["command"]]


def _clears(result):
    return [c["command"].split("--app ")[1] for c in result.shim_calls
            if c["tool"] == "ssh" and " clear-reprovision " in c["command"]]


def _up(fake, tmp_path, *args, config_extra=AB, state=None, shim_env=None):
    result = run_cloudgpu(fake, tmp_path, ["up", "--config", "box", *args],
                          config_extra=config_extra, state=state, shim_env=shim_env)
    assert "Traceback" not in result.output, result.output
    return result


# --- (1) load_config: the shape of `profiles` -----------------------------------------------------


class TestLoadConfigProfiles:
    def test_no_profiles_key_gives_an_empty_list(self, tmp_path, monkeypatch):
        """Catches: a config written before this item failing to load, or `up` reading a missing key."""
        _write_config(tmp_path, "c", BASE)
        monkeypatch.chdir(tmp_path)
        assert configs.load_config("c")["profiles"] == []

    def test_a_list_of_names_is_kept_in_order(self, tmp_path, monkeypatch):
        """Catches: the list dropped, sorted or reshaped on load."""
        _write_config(tmp_path, "c", BASE + 'profiles = ["qwen-image", "krea2", "./elsewhere/p"]\n')
        monkeypatch.chdir(tmp_path)
        assert configs.load_config("c")["profiles"] == ["qwen-image", "krea2", "./elsewhere/p"]

    def test_loading_does_not_need_the_profiles_to_exist(self, tmp_path, monkeypatch):
        """Catches: load_config checking profile directories, which would break `down` and `status`."""
        _write_config(tmp_path, "c", BASE + 'profiles = ["not-on-disk"]\n')
        monkeypatch.chdir(tmp_path)
        assert configs.load_config("c")["profiles"] == ["not-on-disk"]

    @pytest.mark.parametrize("value", ["3", '"a"', '["a", 3]', '["a", ""]', '["a", "  "]', '[["a"]]', "true"])
    def test_a_wrong_shape_is_refused_naming_file_and_key(self, tmp_path, monkeypatch, value):
        """Catches: `profiles = "ab"` read as profiles a and b, or a number reaching `up`."""
        path = _write_config(tmp_path, "c", BASE + f"profiles = {value}\n")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(config.ConfigError) as e:
            configs.load_config("c")
        assert "'profiles'" in str(e.value)
        assert str(path.resolve()) in str(e.value)

    def test_one_load_cannot_change_the_default_for_the_next(self, tmp_path, monkeypatch):
        """Catches: every config sharing one default list, so editing one's changes them all."""
        _write_config(tmp_path, "c", BASE)
        monkeypatch.chdir(tmp_path)
        configs.load_config("c")["profiles"].append("leaked")
        assert configs.load_config("c")["profiles"] == []


# --- (2) `up -C c` provisions the config's profiles with no -P -----------------------------------


def test_fresh_launch_provisions_the_configs_profiles_in_order_and_records_them(tmp_path):
    """Catches: `up -C box` after a termination bringing back a machine with nothing provisioned."""
    _profile(tmp_path, "a")
    _profile(tmp_path, "b")
    with FakeLambda() as fake:
        result = _up(fake, tmp_path)
        launches = len(fake.launches)
    assert result.returncode == 0, result.output
    assert launches == 1
    assert _provisions(result) == ["a", "b"], result.output
    assert result.state["provisioned_profiles"] == ["a", "b"]


def test_reused_instance_provisions_the_configs_profiles(tmp_path):
    """Catches: the config's profiles applied only when `up` launches, not when it reuses."""
    _profile(tmp_path, "a")
    _profile(tmp_path, "b")
    with FakeLambda() as fake:
        tracked = fake.add_instance()
        result = _up(fake, tmp_path, state={"instance_id": tracked})
        launches = len(fake.launches)
    assert result.returncode == 0, result.output
    assert launches == 0 and "Reusing" in result.output, result.output
    assert _provisions(result) == ["a", "b"], result.output
    assert result.state["provisioned_profiles"] == ["a", "b"]


def test_adopted_instance_provisions_the_configs_profiles(tmp_path):
    """Catches: the config's profiles skipped when `up` adopts an instance it had lost track of."""
    _profile(tmp_path, "a")
    _profile(tmp_path, "b")
    with FakeLambda() as fake:
        fake.add_instance()
        result = _up(fake, tmp_path)
        launches = len(fake.launches)
    assert result.returncode == 0, result.output
    assert launches == 0 and "Adopting" in result.output, result.output
    assert _provisions(result) == ["a", "b"], result.output
    assert result.state["provisioned_profiles"] == ["a", "b"]


# --- (3) -P adds to the config's list, each profile once -----------------------------------------


def test_dash_p_adds_a_profile_after_the_configs(tmp_path):
    """Catches: -P replacing the config's profiles, or running before them."""
    for name in ("a", "b", "x"):
        _profile(tmp_path, name)
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "-P", "x")
    assert result.returncode == 0, result.output
    assert _provisions(result) == ["a", "b", "x"], result.output
    assert result.state["provisioned_profiles"] == ["a", "b", "x"]


@pytest.mark.parametrize("flag", ["a", "./profiles/a"])
def test_dash_p_naming_a_config_profile_provisions_it_once(tmp_path, flag):
    """Catches: `up -C box -P a` provisioning a twice, by name or by a path to the same folder."""
    _profile(tmp_path, "a")
    _profile(tmp_path, "b")
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "-P", flag)
    assert result.returncode == 0, result.output
    assert _provisions(result) == ["a", "b"], result.output
    assert result.state["provisioned_profiles"] == ["a", "b"]


def test_a_profile_listed_twice_in_the_config_is_provisioned_once(tmp_path):
    """Catches: a repeated name in the config's list running that profile's provisioning twice."""
    _profile(tmp_path, "a")
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, config_extra='profiles = ["a", "a"]\n')
    assert result.returncode == 0, result.output
    assert _provisions(result) == ["a"], result.output


# --- (4) a re-provision mark and a config-listed profile -----------------------------------------


@pytest.mark.parametrize("recorded", [["a"], ["a", "b"], []])
def test_a_marked_app_does_not_provision_a_config_profile_twice(tmp_path, recorded):
    """Catches: a venv rebuild adding the recorded copy of a to the config's a, so it runs twice."""
    _profile(tmp_path, "a", ["comfyui"])
    _profile(tmp_path, "b")
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, state={"provisioned_profiles": recorded},
                     shim_env={"FAKE_SHIM_APPS": json.dumps(COMFY),
                               "FAKE_SHIM_APPS_AFTER_RECOVER": json.dumps(MARKED)})
    assert result.returncode == 0, result.output
    assert _provisions(result) == ["a", "b"], result.output
    assert _clears(result) == ["comfyui"], result.output
    assert result.state["provisioned_profiles"] == ["a", "b"]


# --- (5) a config profile that can't be loaded stops `up` before it does anything ----------------


@pytest.mark.parametrize("state", [None, {"instance_id": "fake-instance-1", "provisioned_profiles": ["a"]}])
def test_a_missing_config_profile_stops_up_before_anything_happens(tmp_path, state):
    """Catches: a typo in the config's list launching an instance that bills with nothing on it."""
    _profile(tmp_path, "a")
    with FakeLambda() as fake:
        fake.add_instance()
        result = _up(fake, tmp_path, config_extra='profiles = ["a", "typo"]\n', state=state)
        requests, alive = list(fake.requests), fake.alive()

    assert result.returncode != 0, result.output
    said = " ".join(result.output.split())
    assert "'typo'" in said, result.output
    assert "box.toml" in said, result.output
    assert requests == [], "no Lambda call may be made"
    assert alive == ["fake-instance-1"]
    assert result.shim_calls == [], "nothing may be run over ssh"
    assert not (tmp_path / "configs" / "box.up.lock").exists(), "the lock must not be taken"
    if state is None:
        assert result.state_bytes is None, "no state file may be written"
    else:
        assert result.state_bytes == json.dumps(state).encode()


def test_a_config_profile_that_does_not_parse_stops_up_the_same_way(tmp_path):
    """Catches: only a missing folder being checked early, a broken cloudgpu.toml found after launch."""
    _profile(tmp_path, "a")
    (tmp_path / "profiles" / "a" / "cloudgpu.toml").write_text("apps = [\n")
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, config_extra='profiles = ["a"]\n')
        requests = list(fake.requests)
    assert result.returncode != 0, result.output
    assert "'a'" in result.output and "box.toml" in " ".join(result.output.split()), result.output
    assert requests == []
    assert not (tmp_path / "configs" / "box.up.lock").exists()


def test_a_bad_dash_p_still_fails_as_before(tmp_path):
    """Catches: the new check changing how a bad -P is refused."""
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "-P", "nope", config_extra="")
        requests = list(fake.requests)
    assert result.returncode == 2, result.output
    assert "No cloudgpu.toml" in result.output
    assert requests == []
    assert not (tmp_path / "configs" / "box.up.lock").exists()


# --- (6) `down` and `status` on a config whose profile is missing --------------------------------


def _repo_with_missing_profile(tmp_path, monkeypatch):
    _write_config(tmp_path, "c", BASE + 'filesystem = "prof"\nprofiles = ["typo"]\n')
    monkeypatch.chdir(tmp_path)
    configs.save_state("c", {
        "instance_id": "i-1", "ip": "1.2.3.4", "host": "ubuntu@1.2.3.4",
        "filesystem": "prof", "persistent_dir": "/lambda/nfs/prof",
    })


def test_down_still_works_when_a_config_profile_is_missing(tmp_path, monkeypatch):
    """Catches: a typo in `profiles` making it impossible to terminate the instance."""
    _repo_with_missing_profile(tmp_path, monkeypatch)
    terminated = []
    monkeypatch.setattr(lambda_api, "get_instance", lambda i: {"status": "active", "ip": "1.2.3.4"})
    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: terminated.append(ids))

    result = CliRunner().invoke(cli.down, ["--yes", "-C", "c"])
    assert result.exit_code == 0, result.output
    assert terminated == [["i-1"]]
    assert configs.load_state("c") == {}


def test_status_still_works_when_a_config_profile_is_missing(tmp_path, monkeypatch):
    """Catches: a typo in `profiles` making `status` refuse to look at the instance."""
    _repo_with_missing_profile(tmp_path, monkeypatch)
    monkeypatch.setattr(cli.sync, "sync_remote", lambda *a, **k: None)
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: cli.ssh.SSHResult(0, json.dumps({"apps": COMFY}), ""))

    result = CliRunner().invoke(cli.status, ["-C", "c"])
    assert result.exit_code == 0, result.output
    assert "comfyui" in result.output


# --- (7) a config with no `profiles` key behaves as before ---------------------------------------


def test_bare_up_without_the_key_provisions_nothing(tmp_path):
    """Catches: a config that lists no profiles starting to provision recorded ones on every `up`."""
    _profile(tmp_path, "a", ["comfyui"])
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, config_extra="", state={"provisioned_profiles": ["a"]},
                     shim_env={"FAKE_SHIM_APPS": json.dumps(COMFY)})
    assert result.returncode == 0, result.output
    assert _provisions(result) == []
    assert result.state["provisioned_profiles"] == ["a"]


def test_bare_up_without_the_key_still_obeys_a_mark(tmp_path):
    """Catches: the re-provision of a rebuilt app's profiles lost for configs that list none."""
    _profile(tmp_path, "a", ["comfyui"])
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, config_extra="", state={"provisioned_profiles": ["a"]},
                     shim_env={"FAKE_SHIM_APPS": json.dumps(COMFY),
                               "FAKE_SHIM_APPS_AFTER_RECOVER": json.dumps(MARKED)})
    assert result.returncode == 0, result.output
    assert _provisions(result) == ["a"]
    assert _clears(result) == ["comfyui"]


def test_a_profile_dropped_from_the_config_stays_recorded(tmp_path):
    """Catches: `up` forgetting a provisioned profile because the config no longer lists it."""
    _profile(tmp_path, "a")
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, config_extra='profiles = ["a"]\n',
                     state={"provisioned_profiles": ["old"]})
    assert result.returncode == 0, result.output
    assert _provisions(result) == ["a"], result.output
    assert result.state["provisioned_profiles"] == ["old", "a"]


def test_provision_still_requires_dash_p(tmp_path, monkeypatch):
    """Catches: `cloudgpu provision -C c` quietly starting to provision the config's list."""
    _write_config(tmp_path, "c", BASE + AB)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.ssh, "ssh_run", lambda *a, **k: pytest.fail("must not reach the instance"))
    result = CliRunner().invoke(cli.provision, ["-C", "c"])
    assert result.exit_code == 2, result.output
    assert "--profile" in result.output


# --- (8) the scaffold documents the key -----------------------------------------------------------


def test_scaffolded_config_shows_a_commented_profiles_line_and_loads(tmp_path):
    """Catches: `cloudgpu init --config` writing a config that never mentions `profiles`."""
    path = configs.scaffold("c", ssh_key="mini", gpu=["gh200"], configs_dir=tmp_path)
    lines = [line for line in path.read_text().splitlines() if line.startswith("# profiles = [")]
    assert len(lines) == 1, path.read_text()
    assert configs.load_config(str(path))["profiles"] == []
    # Uncommented, the example line is a valid value.
    path.write_text(path.read_text().replace("# profiles = ", "profiles = "))
    assert configs.load_config(str(path))["profiles"]
