"""Inspect the public Planner configuration without making a model request."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "evaluations/robocasa365/2.0/planner.json"


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1 or config.get("source_commit") != "c8911c64f24393e8665bd9ecc34f819173d9c3ba":
        raise ValueError("Planner configuration must identify the unchanged 2.0 source")
    if not isinstance(config.get("provider"), str) or not config["provider"].strip():
        raise ValueError("Configured Planner provider disclosure is required")
    cli = config.get("cli", {})
    if cli.get("planner") != "api" or cli.get("model") != "codexgateway:gpt-6-astra":
        raise ValueError("Planner backend/model differ from the documented baseline")
    if type(cli.get("max_tokens")) is not int or cli["max_tokens"] != 0:
        raise ValueError("CodexGateway requires max_tokens=0")
    if cli.get("plan_generation_timeout_s") != 180 or cli.get("planner_timeout_s") is not None:
        raise ValueError("Keep nested baseline180 and outer timeout unset; historical600 overrides are separate evidence")
    gateway = config.get("gateway", {})
    if gateway.get("reasoning_effort") != "xhigh" or gateway.get("http_request_timeout_s") != 330:
        raise ValueError("gateway settings differ from the documented configuration")
    if gateway.get("settings_scope") != "gateway/server configuration; frozen client does not send reasoning effort":
        raise ValueError("Reasoning effort must be declared as gateway-side configuration")
    if config.get("private_environment") != {
        "base_url_env": "OPENAI_BASE_URL", "api_key_env": "OPENAI_API_KEY",
    }:
        raise ValueError("Only the documented private environment names are supported")


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def cli_arguments(config: dict) -> list[str]:
    validate_config(config)
    cli = config["cli"]
    return ["--planner", cli["planner"], "--model", cli["model"],
            "--max-tokens", str(cli["max_tokens"]),
            "--plan-generation-timeout-s", str(cli["plan_generation_timeout_s"])]


def inspect_environment(config: dict, environment: dict | None = None) -> dict:
    """Return presence/shape checks only; never echo a URL or credential."""
    validate_config(config)
    environment = os.environ if environment is None else environment
    names = config["private_environment"]
    base_url = environment.get(names["base_url_env"], "").strip()
    key = environment.get(names["api_key_env"], "").strip()
    valid_url = False
    try:
        parsed = urlsplit(base_url)
        parsed.port  # Reject malformed or out-of-range ports without echoing the URL.
        valid_url = bool(parsed.scheme in {"http", "https"} and parsed.hostname and
                         not (parsed.username or parsed.password or parsed.query or parsed.fragment))
    except ValueError:
        pass
    return {
        "configured": bool(valid_url and key),
        "base_url_variable": names["base_url_env"], "base_url_present": bool(base_url),
        "base_url_shape_valid": valid_url,
        "api_key_variable": names["api_key_env"], "api_key_present": bool(key),
        "private_values_disclosed": False, "network_requests": 0,
        "api_settings": "The supplied API must provide the declared model and xhigh settings.",
        "quota_checked": False, "provider_identity_audited": False,
    }


def load_private_environment(config: dict, env_file: Path | None = None) -> dict[str, str]:
    """Load only required names, with process variables overriding an explicit file."""
    validate_config(config)
    values = {}
    if env_file is not None:
        from dotenv import dotenv_values

        try:
            with env_file.open() as stream:
                values = dotenv_values(stream=stream, interpolate=False)
        except (OSError, UnicodeError):
            raise ValueError("Cannot read the explicitly requested private environment file") from None
    names = config["private_environment"].values()
    selected = {name: os.environ.get(name, values.get(name) or "").strip() for name in names}
    if not inspect_environment(config, selected)["configured"]:
        raise ValueError("Required Planner environment is missing or has an invalid endpoint shape")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--env-file", type=Path, help="Explicit private dotenv file; used only with --check-env")
    args = parser.parse_args()
    if args.env_file is not None and not args.check_env:
        parser.error("--env-file requires --check-env")
    config = load_config(args.config)
    result = {"config_valid": True, "provider": config["provider"],
              "model": config["cli"]["model"], "cli_arguments": cli_arguments(config),
              "gateway": {key: config["gateway"][key] for key in
                          ("reasoning_effort", "http_request_timeout_s", "settings_scope")},
              "network_requests": 0}
    if args.check_env:
        try:
            environment = load_private_environment(config, args.env_file)
        except ValueError as exc:
            result["private_environment"] = {"configured": False, "error": str(exc), "private_values_disclosed": False}
        else:
            result["private_environment"] = inspect_environment(config, environment)
    print(json.dumps(result, indent=2))
    return 2 if args.check_env and not result["private_environment"]["configured"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
