"""Factory for the supported episode-owning LLM runtimes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rpent.runtime.base import AgentRuntime
from rpent.utils.config import get_memory_dir, get_repo_root


def authoritative_runtime_options(runtime_type: str) -> dict[str, Any]:
    """Return backend options that preserve authoritative isolation."""
    if runtime_type == "api":
        return {}
    if runtime_type == "codex":
        return {
            "codex_sandbox": "read-only",
            "codex_approval_mode": "auto_review",
        }
    raise ValueError(
        f"planner backend {runtime_type!r} does not provide tested "
        "authoritative isolation"
    )


def build_runtime(
    runtime_type: str,
    *,
    output_dir: str | Path,
    recipe_tag: str,
    env_name: str,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 8192,
    planner_timeout_s: int | None = None,
    claude_code_max_budget_usd: float | None = None,
    dashboard: Any = None,
    recorder: Any = None,
    no_images: bool = False,
    codex_sandbox: str = "full-access",
    codex_approval_mode: str = "deny_all",
) -> AgentRuntime:
    """Build an API, Claude Code, or Codex runtime."""
    if runtime_type == "api":
        if not model:
            raise ValueError(
                "the 'api' planner requires a model id; pass --model with a "
                "provider prefix (e.g. 'anthropic:claude-opus-4-8', "
                "'openai:gpt-5.5', 'openai-chat:glm-5.2')."
            )

        from rpent.recording.llm_images import LLMImageArchive

        image_archive = LLMImageArchive(output_dir, recorder=recorder)

        if model.startswith("codexgateway:"):
            from rpent.runtime.codexgateway import CodexGatewayRuntime

            return CodexGatewayRuntime(
                model=model.removeprefix("codexgateway:"),
                base_url=base_url,
                max_tokens=max_tokens,
                timeout_s=planner_timeout_s,
                dashboard=dashboard,
                recorder=recorder,
                no_images=no_images,
                image_archive=image_archive,
            )

        import inspect

        from pydantic_ai.models import infer_model
        from pydantic_ai.providers import infer_provider, infer_provider_class

        from rpent.runtime.api import ApiAgentRuntime

        def _provider_factory(provider_name: str):
            if not base_url:
                return infer_provider(provider_name)
            provider_cls = infer_provider_class(provider_name)
            params = inspect.signature(provider_cls.__init__).parameters
            kwargs = {}
            if "base_url" in params:
                kwargs["base_url"] = base_url
            return provider_cls(**kwargs)

        api_model = infer_model(model, provider_factory=_provider_factory)
        return ApiAgentRuntime(
            model=api_model,
            max_tokens=max_tokens,
            dashboard=dashboard,
            recorder=recorder,
            no_images=no_images,
            image_archive=image_archive,
        )
    if runtime_type == "claude_code":
        from rpent.runtime.claude import ClaudeCodeAgentRuntime

        timeout_s = planner_timeout_s
        if timeout_s is None:
            timeout_s = int(os.environ.get("CELL_TIMEOUT_S", "1200"))
        budget = claude_code_max_budget_usd
        if budget is None:
            budget = float(os.environ.get("MAX_BUDGET_USD", "10"))
        return ClaudeCodeAgentRuntime(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model or "sonnet",
            timeout_s=timeout_s,
            max_budget_usd=budget,
            extra_dirs=[str(get_memory_dir(env_name))],
            output_path=Path(output_dir) / f"claude_{recipe_tag}.txt",
            dashboard=dashboard,
            recorder=recorder,
        )
    if runtime_type == "codex":
        from rpent.runtime.codex import CodexAgentRuntime

        timeout_s = planner_timeout_s
        if timeout_s is None:
            timeout_s = int(
                os.environ.get(
                    "CODEX_TIMEOUT_S",
                    os.environ.get("CELL_TIMEOUT_S", "1200"),
                )
            )
        return CodexAgentRuntime(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model,
            timeout_s=timeout_s,
            extra_dirs=[str(get_memory_dir(env_name))],
            output_path=Path(output_dir) / f"codex_{recipe_tag}.txt",
            dashboard=dashboard,
            recorder=recorder,
            sandbox=codex_sandbox,
            approval_mode=codex_approval_mode,
        )
    raise ValueError(f"unknown runtime_type: {runtime_type}")


# Compatibility with the previous CLI/API name.
build_agent_runtime = build_runtime

__all__ = [
    "authoritative_runtime_options",
    "build_agent_runtime",
    "build_runtime",
]
