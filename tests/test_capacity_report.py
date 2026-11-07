"""Tests for `cloudgpu capacity report`: the go/no-go on region failover is read off its output."""

from __future__ import annotations

import json
import urllib.request

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import capacity, lambda_api

FILESYSTEMS = [{"name": "prof", "region": {"name": "us-east-3", "description": "Washington DC"}}]


def _sample(types, timestamp="2026-11-01T00:00:00Z"):
    return json.dumps({"timestamp": timestamp, "types": types})


# One sample in each bucket, for gpu = ["gh200"] with home us-east-3, plus a second "elsewhere".
FOUR_BUCKETS = [
    _sample({"gpu_1x_gh200": ["us-east-3", "us-west-2"]}, "2026-11-01T00:00:00Z"),
    _sample({"gpu_1x_gh200": ["us-west-2"]}, "2026-11-01T00:05:00Z"),
    _sample({"gpu_1x_gh200": ["us-west-2", "us-south-1"]}, "2026-11-01T00:10:00Z"),
    _sample({"gpu_1x_gh200": []}, "2026-11-01T00:15:00Z"),
    _sample({"gpu_8x_h100": ["us-east-3"]}, "2026-11-01T00:20:00Z"),
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """HOME and the working directory in tmp, no key, and any real API call is a failure."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(lambda_api.API_KEY_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("the report tried to reach the network"))
    (tmp_path / "configs").mkdir()
    return tmp_path


def _config(env, name="c", gpu='["gh200"]', extra='filesystem = "prof"'):
    (env / "configs" / f"{name}.toml").write_text(f'gpu = {gpu}\nssh_key = "k"\n{extra}\n')


def _log(env, lines, end="\n"):
    path = env / ".local" / "state" / "cloudgpu" / "capacity.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + end)
    return path


def _filesystems(monkeypatch, value=FILESYSTEMS):
    monkeypatch.setattr(lambda_api, "list_filesystems", lambda: value)


def _report():
    return CliRunner().invoke(cli.cli, ["capacity", "report"])


def _assert_clean_failure(result):
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "%" not in result.output


def _config_block(output, name):
    """The report's lines for one config, so a figure can't be matched from another config."""
    blocks = output.split("\n\n")
    return next(b for b in blocks if b.startswith(f"{name}: "))


def test_counts_and_percentages_for_all_four_buckets(env, monkeypatch):
    _config(env)
    _log(env, FOUR_BUCKETS)
    _filesystems(monkeypatch)
    result = _report()
    assert result.exit_code == 0, result.output
    assert "Home region had capacity:        1 of 5 (20.0%)" in result.output
    assert "Only other regions had capacity: 2 of 5 (40.0%)" in result.output
    assert "No region had capacity:          1 of 5 (20.0%)" in result.output
    assert "Types not listed in the sample:  1 of 5 (20.0%)" in result.output


def test_a_sample_with_capacity_in_home_and_elsewhere_counts_once_as_home(env, monkeypatch):
    _config(env, gpu='["a100"]')
    _log(env, [_sample({"gpu_1x_a100": ["us-west-2"], "gpu_1x_a100_sxm4": ["us-east-3"]})])
    _filesystems(monkeypatch)
    output = _report().output
    assert "Home region had capacity:        1 of 1 (100.0%)" in output
    assert "Only other regions had capacity: 0 of 1 (0.0%)" in output


def test_not_listed_samples_stay_in_the_denominator(env, monkeypatch):
    """Dropping them would inflate the go/no-go number when most of the data is missing."""
    _config(env)
    _log(env, [_sample({"gpu_1x_gh200": ["us-west-2"]})] + [_sample({"other": []})] * 9)
    _filesystems(monkeypatch)
    output = _report().output
    assert "Only other regions had capacity: 1 of 10 (10.0%)" in output
    assert "Types not listed in the sample:  9 of 10 (90.0%)" in output


def test_a_percentage_just_under_the_threshold_does_not_round_up_to_it():
    assert capacity.percent(624, 2500) == "24.9%"
    assert capacity.percent(2, 3) == "66.6%"
    assert capacity.percent(5, 5) == "100.0%"


def test_the_report_gives_no_verdict(env, monkeypatch):
    _config(env)
    _log(env, FOUR_BUCKETS)
    _filesystems(monkeypatch)
    output = _report().output.lower()
    assert "go/no-go reads this" in output
    assert "25" not in output
    for verdict in ("result: go", "no-go:", "verdict", "build failover"):
        assert verdict not in output


def test_an_alias_in_gpu_matches_its_type_names(env, monkeypatch):
    _config(env, gpu='["a100"]')
    _log(env, [_sample({"gpu_1x_a100_sxm4": ["us-east-3"]})])
    _filesystems(monkeypatch)
    output = _report().output
    assert "Home region had capacity:        1 of 1 (100.0%)" in output
    assert "gpu a100 = gpu_1x_a100, gpu_1x_a100_sxm4" in output


def test_region_in_the_config_wins_over_the_filesystem(env, monkeypatch):
    _config(env, extra='filesystem = "prof"\nregion = "us-west-2"')
    _log(env, [_sample({"gpu_1x_gh200": ["us-west-2"]})])
    monkeypatch.setattr(lambda_api, "list_filesystems",
                        lambda: pytest.fail("a config with a region needs no API call"))
    result = _report()
    assert result.exit_code == 0, result.output
    assert "Home region: us-west-2 (set in the config)" in result.output
    assert "Home region had capacity:        1 of 1 (100.0%)" in result.output


def test_home_region_comes_from_the_filesystem(env, monkeypatch):
    _config(env)
    _log(env, [_sample({"gpu_1x_gh200": ["us-east-3"]})])
    _filesystems(monkeypatch)
    assert "Home region: us-east-3 (its filesystem 'prof')" in _report().output


def test_filesystems_are_looked_up_once_however_many_configs(env, monkeypatch):
    _config(env, "a")
    _config(env, "b")
    _log(env, FOUR_BUCKETS)
    calls = []
    monkeypatch.setattr(lambda_api, "list_filesystems", lambda: calls.append(1) or FILESYSTEMS)
    assert _report().exit_code == 0
    assert len(calls) == 1


def test_regions_ranked_by_how_often_they_had_capacity(env, monkeypatch):
    _config(env)
    _log(env, FOUR_BUCKETS)
    _filesystems(monkeypatch)
    output = _report().output
    assert "us-west-2: 3 of 5 (60.0%)" in output
    assert "us-east-3: 1 of 5 (20.0%)" in output
    assert "us-south-1: 1 of 5 (20.0%)" in output
    assert output.index("us-west-2: 3") < output.index("us-east-3: 1") < output.index("us-south-1: 1")


def test_a_region_counts_once_per_sample_across_a_configs_types(env, monkeypatch):
    _config(env, gpu='["a100"]')
    _log(env, [_sample({"gpu_1x_a100": ["us-west-2"], "gpu_1x_a100_sxm4": ["us-west-2"]})])
    _filesystems(monkeypatch)
    assert "us-west-2: 1 of 1 (100.0%)" in _report().output


def test_regions_ignore_types_the_config_does_not_use(env, monkeypatch):
    _config(env)
    _log(env, [_sample({"gpu_1x_gh200": [], "gpu_8x_h100": ["us-west-2"]})])
    _filesystems(monkeypatch)
    output = _report().output
    assert "No region had capacity:          1 of 1 (100.0%)" in output
    assert "us-west-2" not in output


@pytest.mark.parametrize("bad", [
    "not json",
    '{"timestamp": "2026-11-01T00:00:00Z", "ty',  # a write cut short
    _sample({"gpu_1x_gh200": ["us-west-2"]})[:-3] + _sample({"gpu_1x_gh200": ["us-west-2"]}),  # joined
    "[]",
    '"a string"',
    '{"types": {"gpu_1x_gh200": ["us-west-2"]}}',  # no timestamp
    '{"timestamp": "2026-11-01T00:00:00Z"}',  # no types
    '{"timestamp": 5, "types": {"gpu_1x_gh200": ["us-west-2"]}}',
    '{"timestamp": "yesterday", "types": {"gpu_1x_gh200": ["us-west-2"]}}',
    '{"timestamp": "2026-11-01T00:00:00Z", "types": []}',
    '{"timestamp": "2026-11-01T00:00:00Z", "types": {}}',
    '{"timestamp": "2026-11-01T00:00:00Z", "types": {"gpu_1x_gh200": "us-west-2"}}',
    '{"timestamp": "2026-11-01T00:00:00Z", "types": {"gpu_1x_gh200": [5]}}',
    "",
])
def test_a_bad_line_is_skipped_counted_and_in_no_bucket(env, monkeypatch, bad):
    _config(env)
    _log(env, [_sample({"gpu_1x_gh200": ["us-east-3"]}), bad])
    _filesystems(monkeypatch)
    result = _report()
    assert result.exit_code == 0, result.output
    assert "Samples: 1," in result.output
    assert "Skipped: 1 unreadable lines" in result.output
    assert "Home region had capacity:        1 of 1 (100.0%)" in result.output
    assert "Only other regions had capacity: 0 of 1 (0.0%)" in result.output


def test_bytes_that_are_not_text_are_a_skipped_line(env, monkeypatch):
    _config(env)
    path = _log(env, [_sample({"gpu_1x_gh200": ["us-east-3"]})])
    path.write_bytes(path.read_bytes() + b"\xff\xfe\n")
    _filesystems(monkeypatch)
    result = _report()
    assert result.exit_code == 0, result.output
    assert "Skipped: 1 unreadable lines" in result.output


def test_a_last_line_with_no_newline_still_counts(env, monkeypatch):
    _config(env)
    _log(env, [_sample({"gpu_1x_gh200": ["us-east-3"]})], end="")
    _filesystems(monkeypatch)
    output = _report().output
    assert "Samples: 1," in output
    assert "Skipped: 0 unreadable lines" in output


def test_zero_skipped_lines_are_still_shown(env, monkeypatch):
    _config(env)
    _log(env, FOUR_BUCKETS)
    _filesystems(monkeypatch)
    assert "Skipped: 0 unreadable lines" in _report().output


def test_a_region_name_that_looks_like_markup_is_printed_as_it_is(env, monkeypatch):
    _config(env)
    _log(env, [_sample({"gpu_1x_gh200": ["[/oops]"]})])
    _filesystems(monkeypatch)
    result = _report()
    assert result.exit_code == 0, result.output
    assert "[/oops]: 1 of 1 (100.0%)" in result.output


def test_sample_count_and_first_and_last_timestamps(env, monkeypatch):
    _config(env)
    _log(env, list(reversed(FOUR_BUCKETS)))  # first and last by time, not by position
    _filesystems(monkeypatch)
    assert "Samples: 5, first 2026-11-01T00:00:00Z, last 2026-11-01T00:20:00Z" in _report().output


def test_missing_log_exits_non_zero_with_no_percentages(env, monkeypatch):
    _config(env)
    _filesystems(monkeypatch)
    result = _report()
    _assert_clean_failure(result)
    assert "No capacity samples" in result.output


def test_empty_log_exits_non_zero_with_no_percentages(env, monkeypatch):
    _config(env)
    _log(env, [], end="")
    _filesystems(monkeypatch)
    result = _report()
    _assert_clean_failure(result)
    assert "No capacity samples" in result.output


def test_a_log_of_only_bad_lines_exits_non_zero_and_says_how_many(env, monkeypatch):
    _config(env)
    _log(env, ["bad", "worse"])
    _filesystems(monkeypatch)
    result = _report()
    _assert_clean_failure(result)
    assert "2 unreadable lines" in " ".join(result.output.split())  # errors wrap at the width


def test_an_unreadable_log_exits_non_zero_with_no_percentages(env, monkeypatch):
    _config(env)
    path = env / ".local" / "state" / "cloudgpu" / "capacity.jsonl"
    path.mkdir(parents=True)  # a directory where the file should be
    _filesystems(monkeypatch)
    _assert_clean_failure(_report())


def test_no_configs_exits_non_zero_with_no_percentages(env):
    _log(env, FOUR_BUCKETS)
    result = _report()
    _assert_clean_failure(result)
    assert "no configs" in result.output


def test_lambda_api_error_exits_non_zero_with_the_reason_and_no_percentages(env, monkeypatch):
    _config(env)
    _log(env, FOUR_BUCKETS)

    def fail():
        raise lambda_api.LambdaAPIError("stubbed API failure")

    monkeypatch.setattr(lambda_api, "list_filesystems", fail)
    result = _report()
    _assert_clean_failure(result)
    assert "stubbed API failure" in result.output


def test_missing_key_exits_non_zero_with_the_reason_and_no_percentages(env):
    _config(env)
    _log(env, FOUR_BUCKETS)
    result = _report()  # the real list_filesystems, with no key in the environment
    _assert_clean_failure(result)
    assert lambda_api.API_KEY_ENV in result.output


def test_a_non_json_200_exits_cleanly_with_no_percentages(env, monkeypatch):
    _config(env)
    _log(env, FOUR_BUCKETS)

    def fail():
        return json.loads("<html>")

    monkeypatch.setattr(lambda_api, "list_filesystems", fail)
    _assert_clean_failure(_report())


@pytest.mark.parametrize("response", [None, {}, {"error": "nope"}, ["not a filesystem"]])
def test_a_200_that_is_not_filesystems_is_a_failed_lookup_not_no_home_region(env, monkeypatch, response):
    _config(env)
    _log(env, FOUR_BUCKETS)
    _filesystems(monkeypatch, response)
    result = _report()
    _assert_clean_failure(result)
    assert "could not look up filesystems" in result.output


def test_a_failed_lookup_prints_no_percentages_for_a_config_with_a_region(env, monkeypatch):
    _config(env, "pinned", extra='region = "us-east-3"')
    _config(env, "unpinned")
    _log(env, FOUR_BUCKETS)

    def fail():
        raise lambda_api.LambdaAPIError("stubbed API failure")

    monkeypatch.setattr(lambda_api, "list_filesystems", fail)
    _assert_clean_failure(_report())


def test_a_config_with_no_home_region_is_named_while_another_reports(env, monkeypatch):
    _config(env, "homeless", extra='filesystem = "missing"')
    _config(env, "housed")
    _log(env, FOUR_BUCKETS)
    _filesystems(monkeypatch)
    result = _report()
    assert result.exit_code == 0, result.output
    homeless = _config_block(result.output, "homeless")
    assert "no filesystem named 'missing'" in homeless
    assert "no home region yet" in homeless
    assert "%" not in homeless
    assert "Only other regions had capacity: 2 of 5 (40.0%)" in _config_block(result.output, "housed")


def test_a_filesystem_with_no_region_gives_no_home_figures(env, monkeypatch):
    _config(env)
    _log(env, FOUR_BUCKETS)
    _filesystems(monkeypatch, [{"name": "prof"}])
    result = _report()
    assert "no home region yet" in result.output
    assert "%" not in result.output


def test_only_configs_with_no_home_region_exits_non_zero(env, monkeypatch):
    _config(env, extra='filesystem = "missing"')
    _log(env, FOUR_BUCKETS)
    _filesystems(monkeypatch, [])
    result = _report()
    _assert_clean_failure(result)
    assert "no home region yet" in result.output


def test_a_broken_config_is_named_while_another_reports(env, monkeypatch):
    (env / "configs" / "broken.toml").write_text("this is not toml = = =\n")
    _config(env, "good")
    _log(env, FOUR_BUCKETS)
    _filesystems(monkeypatch)
    result = _report()
    assert result.exit_code == 0, result.output
    assert "%" not in _config_block(result.output, "broken")
    assert "2 of 5 (40.0%)" in _config_block(result.output, "good")


def test_each_config_is_figured_from_its_own_types_and_home(env, monkeypatch):
    _config(env, "a", gpu='["gh200"]', extra='region = "us-east-3"')
    _config(env, "b", gpu='["h100"]', extra='region = "us-west-2"')
    _log(env, [_sample({"gpu_1x_gh200": ["us-west-2"], "gpu_1x_h100_pcie": ["us-west-2"]})])
    output = _report().output
    assert "Only other regions had capacity: 1 of 1" in _config_block(output, "a")
    assert "Home region had capacity:        1 of 1" in _config_block(output, "b")


def test_the_log_file_is_byte_for_byte_unchanged(env, monkeypatch):
    _config(env)
    path = _log(env, FOUR_BUCKETS + ["a bad line", ""], end="")
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns
    _filesystems(monkeypatch)
    assert _report().exit_code == 0
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == mtime
