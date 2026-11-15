"""Plan item 3.1: each app's default sync paths, the `[sync]` table, and what a config syncs.

Nothing here copies anything: no rclone, no network, no instance.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cloudgpu.local import apps, config, configs, profiles
from cloudgpu.remote.apps import aitoolkit, comfyui

BASE = 'gpu = ["gh200"]\nssh_key = "mini"\n'
COMFYUI_PATHS = ["apps/comfyui/output", "apps/comfyui/input", "apps/comfyui/user"]
AITOOLKIT_PATHS = ["apps/ai-toolkit/datasets", "apps/ai-toolkit/output"]


def _config(tmp_path, body=""):
    path = tmp_path / "configs" / "small.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BASE + body)
    return path


def _profile(tmp_path, body="", name="p"):
    d = tmp_path / "profiles" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "cloudgpu.toml").write_text('apps = ["comfyui"]\n' + body)
    return d


def _cfg(tmp_path, body=""):
    return configs.load_config(str(_config(tmp_path, body)))


# --- (1) (2) the config's [sync] table --------------------------------------


def test_config_without_sync_loads_with_sync_off(tmp_path):
    c = _cfg(tmp_path)
    assert c["sync"] is None
    assert c["gpu"] == ["gh200"] and c["ssh_key"] == "mini" and c["poll_seconds"] == 20


def test_config_with_only_remote_gets_the_defaults(tmp_path):
    c = _cfg(tmp_path, '[sync]\nremote = "b2:my-cloudgpu-bucket/small"\n')
    assert c["sync"] == {
        "remote": "b2:my-cloudgpu-bucket/small", "interval_minutes": 5, "extra_paths": []}


def test_config_full_table_loads_as_written(tmp_path):
    c = _cfg(tmp_path, '[sync]\nremote = "b2:bucket"\ninterval_minutes = 15\n'
                       'extra_paths = ["notebooks", "apps/comfyui/custom_nodes/x/data"]\n')
    assert c["sync"] == {
        "remote": "b2:bucket", "interval_minutes": 15,
        "extra_paths": ["notebooks", "apps/comfyui/custom_nodes/x/data"]}


def test_default_extra_paths_are_not_shared_between_configs(tmp_path):
    a = _cfg(tmp_path / "a", '[sync]\nremote = "b2:bucket"\n')
    a["sync"]["extra_paths"].append("x")
    assert _cfg(tmp_path / "b", '[sync]\nremote = "b2:bucket"\n')["sync"]["extra_paths"] == []


# --- (3) each bad [sync] is a ConfigError naming the file and the key -------

_REMOTE = 'remote = "b2:bucket"\n'


@pytest.mark.parametrize("body, key", [
    ("interval_minutes = 5\n", "remote"),                    # no remote
    ('remote = ""\n', "remote"),
    ('remote = "my-cloudgpu-bucket"\n', "remote"),           # no ':'
    ('remote = ":bucket"\n', "remote"),                      # no remote name
    ('remote = "b2:"\n', "remote"),                          # no bucket
    ("remote = 7\n", "remote"),
    (_REMOTE + "interval_minutes = 0\n", "interval_minutes"),
    (_REMOTE + "interval_minutes = 2.5\n", "interval_minutes"),
    (_REMOTE + 'interval_minutes = "5"\n', "interval_minutes"),
    (_REMOTE + "interval_minutes = true\n", "interval_minutes"),
    (_REMOTE + 'extra_paths = "notebooks"\n', "extra_paths"),
    (_REMOTE + "extra_paths = [3]\n", "extra_paths"),
    (_REMOTE + 'extra_paths = [""]\n', "extra_paths"),
    (_REMOTE + 'extra_paths = ["/etc"]\n', "extra_paths"),
    (_REMOTE + 'extra_paths = ["../x"]\n', "extra_paths"),
    (_REMOTE + 'extra_paths = ["a/../../x"]\n', "extra_paths"),
    (_REMOTE + 'extra_paths = ["~/x"]\n', "extra_paths"),
    (_REMOTE + 'extra_paths = ["."]\n', "extra_paths"),
    (_REMOTE + "interval_minute = 9\n", "interval_minute"),  # typo: unknown key
])
def test_bad_config_sync_names_file_and_key(tmp_path, body, key):
    path = _config(tmp_path, "[sync]\n" + body)
    with pytest.raises(config.ConfigError) as e:
        configs.load_config(str(path))
    assert str(path) in str(e.value)
    assert f"'{key}'" in str(e.value)


def test_sync_that_is_not_a_table_is_refused(tmp_path):
    path = tmp_path / "configs" / "small.toml"
    path.parent.mkdir(parents=True)
    path.write_text('sync = "b2:bucket"\n' + BASE)
    with pytest.raises(config.ConfigError, match="'sync'") as e:
        configs.load_config(str(path))
    assert str(path) in str(e.value)


# --- (4) a profile's [sync] ---------------------------------------------------


def test_profile_extra_paths_load(tmp_path):
    p = profiles.load_profile(_profile(tmp_path, '[sync]\nextra_paths = ["notebooks"]\n'))
    assert p["sync"] == {"extra_paths": ["notebooks"]}
    assert p["apps"] == ["comfyui"]


def test_profile_without_sync_loads_as_before(tmp_path):
    p = profiles.load_profile(_profile(tmp_path))
    assert p["sync"] == {"extra_paths": []}
    assert p["apps"] == ["comfyui"] and p["ports"] == [] and p["provision_timeout"] == 3600


@pytest.mark.parametrize("key, line", [
    ("remote", 'remote = "b2:bucket"'),
    ("interval_minutes", "interval_minutes = 5"),
])
def test_profile_cannot_set_remote_or_interval(tmp_path, key, line):
    d = _profile(tmp_path, f"[sync]\n{line}\n")
    with pytest.raises(config.ConfigError) as e:
        profiles.load_profile(d)
    message = str(e.value)
    assert str(d / "cloudgpu.toml") in message and f"'{key}'" in message
    assert "config" in message  # says where it belongs


@pytest.mark.parametrize("value", ['"notebooks"', "[3]", '["/etc"]', '["../x"]', '["~/x"]'])
def test_profile_extra_paths_follow_the_configs_rules(tmp_path, value):
    d = _profile(tmp_path, f"[sync]\nextra_paths = {value}\n")
    with pytest.raises(config.ConfigError) as e:
        profiles.load_profile(d)
    assert str(d / "cloudgpu.toml") in str(e.value) and "'extra_paths'" in str(e.value)


# --- (5) what a config syncs --------------------------------------------------

_ON = '[sync]\nremote = "b2:bucket"\n'


def test_sync_paths_for_comfyui(tmp_path):
    assert apps.sync_paths(_cfg(tmp_path, _ON), ["comfyui"], []) == COMFYUI_PATHS


def test_sync_paths_for_both_apps_in_app_order(tmp_path):
    cfg = _cfg(tmp_path, _ON)
    assert apps.sync_paths(cfg, ["comfyui", "ai-toolkit"], []) == COMFYUI_PATHS + AITOOLKIT_PATHS
    assert apps.sync_paths(cfg, ["ai-toolkit", "comfyui"], []) == AITOOLKIT_PATHS + COMFYUI_PATHS


def test_sync_paths_profile_extras_then_config_extras(tmp_path):
    cfg = _cfg(tmp_path, _ON + 'extra_paths = ["from-config"]\n')
    one = profiles.load_profile(_profile(tmp_path, '[sync]\nextra_paths = ["one"]\n', "one"))
    two = profiles.load_profile(_profile(tmp_path, '[sync]\nextra_paths = ["two"]\n', "two"))
    plain = profiles.load_profile(_profile(tmp_path, "", "plain"))
    assert apps.sync_paths(cfg, ["comfyui"], [one, plain, two]) == [
        *COMFYUI_PATHS, "one", "two", "from-config"]


def test_sync_paths_lists_a_path_named_twice_once(tmp_path):
    cfg = _cfg(tmp_path, _ON + 'extra_paths = ["shared", "apps/comfyui/output/", "./shared"]\n')
    p = profiles.load_profile(_profile(tmp_path, '[sync]\nextra_paths = ["shared"]\n'))
    assert apps.sync_paths(cfg, ["comfyui", "comfyui"], [p, p]) == [*COMFYUI_PATHS, "shared"]


def test_sync_paths_drops_a_path_inside_another(tmp_path):
    # Later path inside an earlier one: dropped. A sibling with the same prefix is not "inside".
    cfg = _cfg(tmp_path, _ON + 'extra_paths = ["apps/comfyui/output/keep", "apps/comfyui/output2"]\n')
    assert apps.sync_paths(cfg, ["comfyui"], []) == [*COMFYUI_PATHS, "apps/comfyui/output2"]
    # Later path that contains earlier ones: it replaces them.
    cfg = _cfg(tmp_path, _ON + 'extra_paths = ["apps/comfyui"]\n')
    assert apps.sync_paths(cfg, ["comfyui", "ai-toolkit"], []) == [*AITOOLKIT_PATHS, "apps/comfyui"]


def test_sync_paths_skips_an_unknown_app(tmp_path):
    assert apps.sync_paths(_cfg(tmp_path, _ON), ["nope", "comfyui"], []) == COMFYUI_PATHS
    assert apps.sync_paths(_cfg(tmp_path, _ON), ["nope"], []) == []


def test_sync_paths_is_empty_without_sync(tmp_path):
    p = profiles.load_profile(_profile(tmp_path, '[sync]\nextra_paths = ["notebooks"]\n'))
    assert apps.sync_paths(_cfg(tmp_path), ["comfyui", "ai-toolkit"], [p]) == []


# --- (6) one declaration, read from both sides --------------------------------


def test_declared_paths_are_the_remote_modules_own():
    assert apps.APPS["comfyui"].sync_paths is comfyui.COMFYUI_SYNC_PATHS
    assert apps.APPS["ai-toolkit"].sync_paths is aitoolkit.AITOOLKIT_SYNC_PATHS
    assert list(comfyui.COMFYUI_SYNC_PATHS) == COMFYUI_PATHS
    assert list(aitoolkit.AITOOLKIT_SYNC_PATHS) == AITOOLKIT_PATHS


def test_declared_paths_sit_under_each_apps_own_dir(tmp_persistent_dir):
    # The installer's app_dir is where the app really lives; a renamed dir must not leave the
    # declared paths pointing at nothing.
    from cloudgpu.remote.__main__ import APP_INSTALLERS
    from cloudgpu.remote.state import State

    state = State(str(tmp_persistent_dir))
    for name, installer in APP_INSTALLERS.items():
        app_dir = Path(installer(state).app_dir).relative_to(state.persistent_dir).as_posix()
        declared = apps.APPS[name].sync_paths
        assert declared and all(p.startswith(app_dir + "/") for p in declared), name


# --- (7) the scaffolds --------------------------------------------------------


def _uncomment_sync(text):
    """The scaffold with its commented [sync] table switched on."""
    head, table = text.split("# [sync]\n")
    return head + "[sync]\n" + "".join(line[2:] + "\n" for line in table.splitlines())


def test_config_scaffold_shows_a_commented_b2_sync_and_still_loads(tmp_path):
    path = configs.scaffold("small", ssh_key="mini", gpu=["gh200"], configs_dir=tmp_path)
    text = path.read_text()
    for line in ("# [sync]\n", '# remote = "b2:', "# interval_minutes = 5", "# extra_paths = []"):
        assert line in text
    assert configs.load_config(str(path))["sync"] is None


def test_config_scaffold_with_sync_uncommented_keeps_the_rest_top_level(tmp_path):
    path = configs.scaffold("small", ssh_key="mini", gpu=["gh200", "a100"], configs_dir=tmp_path,
                            filesystem="fs1")
    path.write_text(_uncomment_sync(path.read_text()))
    c = configs.load_config(str(path))
    assert c["sync"] == {
        "remote": "b2:my-cloudgpu-bucket/small", "interval_minutes": 5, "extra_paths": []}
    assert c["gpu"] == ["gh200", "a100"] and c["ssh_key"] == "mini" and c["filesystem"] == "fs1"
    assert c["poll_seconds"] == 20 and c["max_hours"] == 12


def test_profile_scaffold_shows_commented_extra_paths_and_still_loads(tmp_path):
    d = tmp_path / "new"
    path = profiles.scaffold(d, apps=["comfyui"])
    text = path.read_text()
    assert "# [sync]\n" in text and "# extra_paths = [" in text
    assert profiles.load_profile(d)["sync"] == {"extra_paths": []}

    path.write_text(_uncomment_sync(text))
    p = profiles.load_profile(d)
    assert p["sync"] == {"extra_paths": ["notebooks"]}
    assert p["apps"] == ["comfyui"] and p["ports"] == [] and p["provision_timeout"] == 3600


# --- (8) the person's own configs and profiles still load ---------------------


def _checkouts():
    """This checkout, and the main one when this is a worktree (configs/ and profiles/ are
    untracked, so only the main checkout has the person's)."""
    here = Path(__file__).resolve().parents[1]
    roots = [here]
    try:
        common = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--git-common-dir"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        common = ""
    main = (here / common).resolve().parent if common else here  # relative when not a worktree
    if main not in roots:
        roots.append(main)
    return roots


def test_own_configs_and_profiles_still_load():
    config_files = [f for root in _checkouts() for f in sorted(root.glob("configs/*.toml"))]
    profile_files = [f for root in _checkouts() for f in sorted(root.glob("profiles/*/cloudgpu.toml"))]
    if not config_files and not profile_files:
        pytest.skip("no configs/ or profiles/ in this checkout or the main one")
    for f in config_files:
        assert configs.load_config(str(f))["sync"] is None, f
    for f in profile_files:
        assert profiles.load_profile(f.parent)["sync"] == {"extra_paths": []}, f
