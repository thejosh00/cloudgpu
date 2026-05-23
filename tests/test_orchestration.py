"""Tests for the launch orchestration (GPU selection + capacity polling + launch)."""

from __future__ import annotations

import pytest

from cloudgpu.local import lambda_api, orchestration
from cloudgpu.local.orchestration import OrchestrationError


class TestExpandTargets:
    def test_aliases_expand_in_order(self):
        assert orchestration.expand_targets(["gh200", "a100"]) == [
            "gpu_1x_gh200", "gpu_1x_a100", "gpu_1x_a100_sxm4",
        ]

    def test_dedupes_preserving_order(self):
        assert orchestration.expand_targets(["a100", "a100"]) == [
            "gpu_1x_a100", "gpu_1x_a100_sxm4",
        ]

    def test_full_name_passthrough(self):
        assert orchestration.expand_targets(["gpu_1x_h100_pcie"]) == ["gpu_1x_h100_pcie"]

    def test_bare_name_gets_prefixed(self):
        assert orchestration.expand_targets(["v100"]) == ["gpu_1x_v100"]


class TestPickTarget:
    def _types(self, mapping):
        # mapping: type_name -> [region names with capacity]
        return {
            name: {"regions_with_capacity_available": [{"name": r} for r in regions]}
            for name, regions in mapping.items()
        }

    def test_returns_first_preference_with_capacity_in_region(self):
        types = self._types({
            "gpu_1x_gh200": ["us-west-1"],          # capacity, wrong region
            "gpu_1x_a100": ["us-east-3", "us-x"],   # capacity in our region
        })
        targets = ["gpu_1x_gh200", "gpu_1x_a100"]
        assert orchestration.pick_target(types, targets, "us-east-3") == "gpu_1x_a100"

    def test_honors_preference_order(self):
        types = self._types({
            "gpu_1x_gh200": ["us-east-3"],
            "gpu_1x_a100": ["us-east-3"],
        })
        targets = ["gpu_1x_gh200", "gpu_1x_a100"]
        assert orchestration.pick_target(types, targets, "us-east-3") == "gpu_1x_gh200"

    def test_none_when_no_capacity_in_region(self):
        types = self._types({"gpu_1x_gh200": ["us-west-1"]})
        assert orchestration.pick_target(types, ["gpu_1x_gh200"], "us-east-3") is None


class TestPickTargetAny:
    def _types(self, mapping):
        return {
            name: {"regions_with_capacity_available": [{"name": r} for r in regions]}
            for name, regions in mapping.items()
        }

    def test_first_type_then_first_region(self):
        types = self._types({"gpu_1x_gh200": ["us-west-1", "us-east-3"]})
        assert orchestration.pick_target_any(types, ["gpu_1x_gh200"]) == ("gpu_1x_gh200", "us-west-1")

    def test_honors_preference_order_across_regions(self):
        types = self._types({"gpu_1x_a100": ["us-x"], "gpu_1x_gh200": ["us-y"]})
        targets = ["gpu_1x_gh200", "gpu_1x_a100"]
        assert orchestration.pick_target_any(types, targets) == ("gpu_1x_gh200", "us-y")

    def test_allowed_restricts_regions(self):
        types = self._types({"gpu_1x_gh200": ["us-west-1", "us-east-3"]})
        assert orchestration.pick_target_any(types, ["gpu_1x_gh200"], allowed=["us-east-3"]) == (
            "gpu_1x_gh200", "us-east-3",
        )

    def test_none_when_nothing_matches(self):
        types = self._types({"gpu_1x_gh200": ["us-west-1"]})
        assert orchestration.pick_target_any(types, ["gpu_1x_gh200"], allowed=["us-east-3"]) is None


class TestRegionForFilesystem:
    def test_finds_region(self, monkeypatch):
        monkeypatch.setattr(lambda_api, "list_filesystems", lambda: [
            {"name": "other", "region": {"name": "us-west-1"}},
            {"name": "washington", "region": {"name": "us-east-3"}},
        ])
        assert orchestration.region_for_filesystem("washington") == "us-east-3"

    def test_missing_returns_none(self, monkeypatch):
        monkeypatch.setattr(lambda_api, "list_filesystems", lambda: [])
        assert orchestration.region_for_filesystem("washington") is None


class TestAcquireInstance:
    @pytest.fixture
    def profile(self):
        return {
            "name": "wash",
            "filesystem": "washington",
            "gpu": ["gh200"],
            "region": "us-east-3",   # set explicitly to skip filesystem lookup
            "poll_seconds": 0,
            "max_hours": 1,
            "instance_name": "wash",
        }

    def test_reuses_existing_filesystem(self, profile, monkeypatch):
        # filesystem already exists -> pin to its region, do NOT create one.
        monkeypatch.setattr(orchestration, "ensure_ssh_key", lambda name: "mini")
        monkeypatch.setattr(lambda_api, "list_filesystems", lambda: [
            {"name": "washington", "id": "fs-existing", "region": {"name": "us-east-3"}},
        ])
        monkeypatch.setattr(lambda_api, "create_filesystem",
                            lambda *a, **k: pytest.fail("should not create an existing filesystem"))
        monkeypatch.setattr(lambda_api, "list_instance_types", lambda: {
            "gpu_1x_gh200": {"regions_with_capacity_available": [{"name": "us-east-3"}]},
        })
        launched = {}
        monkeypatch.setattr(lambda_api, "launch_instance",
                            lambda **kw: launched.update(kw) or {"instance_ids": ["i-1"]})
        monkeypatch.setattr(lambda_api, "get_instance", lambda i: {
            "status": "active", "ip": "1.2.3.4",
            "instance_type": {"price_cents_per_hour": 229},
        })

        rt = orchestration.acquire_instance(profile)
        assert rt["instance_id"] == "i-1"
        assert rt["host"] == "ubuntu@1.2.3.4"
        assert rt["instance_type"] == "gpu_1x_gh200"
        assert rt["region"] == "us-east-3"
        assert rt["filesystem"] == "washington"
        assert rt["filesystem_id"] == "fs-existing"
        assert rt["created_filesystem"] is False
        assert launched["file_system_names"] == ["washington"]
        assert launched["region_name"] == "us-east-3"
        assert launched["name"] == "wash"

    def test_auto_creates_filesystem_when_missing(self, monkeypatch):
        # No filesystem field -> defaults to profile name; doesn't exist -> create it
        # in the first region with capacity, then launch there.
        profile = {"name": "wash", "gpu": ["gh200"], "ssh_key": "mini",
                   "poll_seconds": 0, "max_hours": 1}
        monkeypatch.setattr(orchestration, "ensure_ssh_key", lambda name: "mini")
        monkeypatch.setattr(lambda_api, "list_filesystems", lambda: [])  # none exist
        monkeypatch.setattr(lambda_api, "list_instance_types", lambda: {
            "gpu_1x_gh200": {"regions_with_capacity_available": [{"name": "us-west-1"}]},
        })
        created = []
        monkeypatch.setattr(lambda_api, "create_filesystem",
                            lambda n, r: created.append((n, r)) or {"id": "fs-new"})
        launched = {}
        monkeypatch.setattr(lambda_api, "launch_instance",
                            lambda **kw: launched.update(kw) or {"instance_ids": ["i-9"]})
        monkeypatch.setattr(lambda_api, "get_instance", lambda i: {
            "status": "active", "ip": "5.6.7.8", "instance_type": {"price_cents_per_hour": 229},
        })

        rt = orchestration.acquire_instance(profile)
        assert created == [("wash", "us-west-1")]
        assert rt["filesystem"] == "wash"
        assert rt["filesystem_id"] == "fs-new"
        assert rt["created_filesystem"] is True
        assert rt["region"] == "us-west-1"
        assert launched["file_system_names"] == ["wash"]
        assert launched["region_name"] == "us-west-1"

    def test_timeout_when_no_capacity(self, profile, monkeypatch):
        profile["max_hours"] = 0  # deadline already passed -> no polling, immediate timeout
        monkeypatch.setattr(orchestration, "ensure_ssh_key", lambda name: "mini")
        monkeypatch.setattr(lambda_api, "list_filesystems", lambda: [
            {"name": "washington", "id": "fs-x", "region": {"name": "us-east-3"}},
        ])
        monkeypatch.setattr(lambda_api, "list_instance_types", lambda: {})
        with pytest.raises(OrchestrationError, match="Timed out"):
            orchestration.acquire_instance(profile)
