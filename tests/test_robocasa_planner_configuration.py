"""Reviewer configuration must preserve the frozen Planner's real controls."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest


def _module():
    from tools import configure_robocasa_planner
    return configure_robocasa_planner


def test_http_timeout_is_not_an_outer_episode_timeout():
    module = _module()
    config = module.load_config(Path(module.DEFAULT_CONFIG))
    args = module.cli_arguments(config)
    assert args == ["--planner", "api", "--model", "codexgateway:gpt-6-astra",
                    "--max-tokens", "0", "--plan-generation-timeout-s", "180"]
    assert "--planner-timeout-s" not in args
    assert config["gateway"]["http_request_timeout_s"] == 330


def test_configuration_check_never_exposes_private_environment_values():
    module = _module()
    config = module.load_config(Path(module.DEFAULT_CONFIG))
    result = module.inspect_environment(config, {
        "OPENAI_BASE_URL": "https://private.example.invalid/v1",
        "OPENAI_API_KEY": "secret-fixture-key",
    })
    assert result["configured"] is True
    assert "private.example.invalid" not in str(result)
    assert "secret-fixture-key" not in str(result)
    assert result["network_requests"] == 0


@pytest.mark.parametrize("url", ["http://user:password@example.invalid/v1", "https://example.invalid/v1?key=secret", "ftp://example.invalid"])
def test_private_endpoint_configuration_rejects_credential_urls(url):
    module = _module()
    config = module.load_config(Path(module.DEFAULT_CONFIG))
    result = module.inspect_environment(config, {"OPENAI_BASE_URL": url, "OPENAI_API_KEY": "fixture"})
    assert result["configured"] is False
    assert url not in str(result)


def test_runner_cannot_silently_use_an_output_token_cap():
    module = _module()
    config = copy.deepcopy(module.load_config(Path(module.DEFAULT_CONFIG)))
    config["cli"]["max_tokens"] = 8192
    with pytest.raises(ValueError, match="max_tokens"):
        module.validate_config(config)


def test_reasoning_effort_cannot_be_declared_as_a_frozen_client_control():
    module = _module()
    config = copy.deepcopy(module.load_config(Path(module.DEFAULT_CONFIG)))
    config["gateway"]["settings_scope"] = "client_environment"
    with pytest.raises(ValueError, match="gateway"):
        module.validate_config(config)


def test_explicit_private_env_file_only_loads_required_names(tmp_path, monkeypatch):
    module = _module()
    config = module.load_config()
    private = tmp_path / "private.env"
    private.write_text("OPENAI_BASE_URL=https://fixture.invalid/v1\nOPENAI_API_KEY=fixture-key\nUNRELATED_SECRET=ignore\n")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "process-priority")
    assert module.load_private_environment(config, private) == {
        "OPENAI_BASE_URL": "https://fixture.invalid/v1", "OPENAI_API_KEY": "process-priority",
    }


def test_private_env_validation_redacts_invalid_values(tmp_path, monkeypatch):
    module = _module()
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    private = tmp_path / "private.env"
    private.write_text("OPENAI_BASE_URL=https://fixture.invalid:invalid/v1\nOPENAI_API_KEY=secret-fixture-key\n")
    with pytest.raises(ValueError) as caught:
        module.load_private_environment(module.load_config(), private)
    assert "secret-fixture-key" not in str(caught.value)
    assert "fixture.invalid" not in str(caught.value)


def test_private_env_missing_file_error_does_not_echo_path(tmp_path):
    module = _module()
    with pytest.raises(ValueError) as caught:
        module.load_private_environment(module.load_config(), tmp_path / "private-name.env")
    assert "private-name.env" not in str(caught.value)
