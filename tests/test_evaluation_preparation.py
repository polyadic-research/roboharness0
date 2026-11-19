from __future__ import annotations

from types import SimpleNamespace

import pytest

from rpent.evaluation.cases import EpisodeProtocol
from rpent.evaluation.preparation import PreparedEvaluation


def test_xiaomi_daemon_uses_explicit_family_and_dtype(monkeypatch, tmp_path):
    from robots.robocasa.evaluation import build_policy_daemon
    from rpent.policies.xiaomi_mibot_daemon import XiaomiMiBoTDaemon

    monkeypatch.setenv("RPENT_XIAOMI_MIBOT_SNAPSHOT", "/pinned/snapshot")
    daemon = build_policy_daemon(args=SimpleNamespace(
        vla="xiaomi-mibot", jobs=1, policy_device="cuda:1", policy_dtype="float32",
        policy_endpoint=None, seed=7), output_dir=tmp_path)
    assert isinstance(daemon, XiaomiMiBoTDaemon)
    assert daemon.device == "cuda:1"
    assert daemon.dtype == "float32"


def _protocol() -> EpisodeProtocol:
    return EpisodeProtocol(
        environment="synthetic",
        preset_name="synthetic-v1",
        horizon_ceiling=10,
        checkpoint_repo_id="local/test",
        checkpoint_revision="revision",
        binding_name="synthetic-binding-v1",
    )


def test_prepared_evaluation_requires_tasks_and_protocol() -> None:
    with pytest.raises(ValueError, match="task"):
        PreparedEvaluation(
            tasks=(),
            protocol=_protocol(),
            environment={},
            simulator={},
            policy={},
            contracts={},
            runtime={},
        )

    with pytest.raises(ValueError, match="protocol"):
        PreparedEvaluation(
            tasks=(SimpleNamespace(name="task"),),
            protocol=object(),
            environment={},
            simulator={},
            policy={},
            contracts={},
            runtime={},
        )


def test_prepared_evaluation_recursively_freezes_manifest_inputs() -> None:
    prepared = PreparedEvaluation(
        tasks=(SimpleNamespace(name="task"),),
        protocol=_protocol(),
        environment={"name": "synthetic"},
        simulator={"tasks": [{"name": "task"}]},
        policy={},
        contracts={},
        runtime={},
    )

    with pytest.raises(TypeError):
        prepared.environment["name"] = "changed"
    with pytest.raises(TypeError):
        prepared.simulator["tasks"][0]["name"] = "changed"
    assert isinstance(prepared.simulator["tasks"], tuple)


def test_xiaomi_coordinator_attaches_loaded_lane_without_loading_or_closing(monkeypatch, tmp_path):
    from robots.robocasa.evaluation import build_policy_daemon
    from rpent.policies import xiaomi_mibot_artifacts

    monkeypatch.setenv("RPENT_XIAOMI_MIBOT_SNAPSHOT", "/pinned/snapshot")
    monkeypatch.setenv("RPENT_SHARED_POLICY_LOADED", "1")
    monkeypatch.setattr(xiaomi_mibot_artifacts, "inspect_xiaomi_artifacts",
                        lambda *args: SimpleNamespace(identity_digest="a" * 64))
    daemon = build_policy_daemon(args=SimpleNamespace(
        vla="xiaomi-mibot", jobs=1, policy_device="cuda:0", policy_dtype="bfloat16",
        policy_endpoint="http://127.0.0.1:42027", seed=7), output_dir=tmp_path)
    assert daemon.attach_loaded is True
    assert daemon.expected_artifact_identity == "a" * 64
