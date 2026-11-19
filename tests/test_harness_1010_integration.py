"""Cross-block CLI, prompt, parser, catalog and narration contracts."""

import hashlib
import sys
from unittest.mock import patch

import pytest

from robots.robocasa import control_modes
from robots.robocasa.policy_skills import SKILL_DIRECTORY, load_policy_skill
from robots.robocasa.prompts.system import system_prompt
from rpent.cli import main as cli
from rpent.envs import get_env_spec
from rpent.planning import resolve_planning_stack
from rpent.runtime import RuntimePlanGenerator, RuntimeResult


def args_for(*flags):
    with patch.object(sys, "argv", ["rpent", "--env", "robocasa"]):
        parser = cli._build_argparser()
        get_env_spec("robocasa").add_cli_args(parser, use_dashboard=False)
        return parser.parse_args([
            "--env", "robocasa", "--task", "StoreLeftoversInBowl",
            "--vla", "xiaomi-mibot", *flags,
        ])


@pytest.mark.parametrize(
    "flag,key,section,heading,mode",
    [
        ("--no-transition-takeover", "transition_takeover_enabled", "TRANSITIONS", "Transitions", "transition"),
        ("--no-op-takeover", "op_takeover_enabled", "TAKEOVER", "Takeover", "takeover"),
    ],
)
def test_cli_ablation_reaches_both_prompts_skill_gate_and_metadata(
    flag, key, section, heading, mode, tmp_path,
):
    args = args_for(flag)
    config = get_env_spec("robocasa").parse_config(args)
    variables = config.prompt_vars
    bundle = cli._configured_prompt_bundle("robocasa", variables)
    assert section not in bundle.system()
    executor_prompt = bundle.render("system", variables={
        **variables, "harness_protocol": "protocol", "planning_startup": "startup",
        "output_dir": str(tmp_path),
    })
    assert system_prompt()[section] not in executor_prompt
    guidance = cli._planner_action_guidance("robocasa", variables)
    assert system_prompt()[section] not in guidance
    assert f"## {heading}\n" not in variables["policy_skill"]
    stack = resolve_planning_stack(
        "robocasa-hooked-gated-v1", **cli._planning_control_options(args),
    )
    generator = RuntimePlanGenerator(object(), require_outcome_program=False)
    harness = stack.build_harness(trace_path=tmp_path / "trace", plan_generator=generator)
    harness.set_physical_tools(set(control_modes.PRIVILEGED_TOOLS))
    # This is the CLI's trusted metadata bind, after stack construction.
    harness.configure_control_modes(policy_family="xiaomi-mibot", task_name=config.task)
    assert getattr(harness.state, key) is False
    assert generator._control_config[key] is False
    raw = {"id": "operation", "intent": "carry or take over", "allowed_tools": ["navigate_to"],
           "skill_row": "none", "tier": "none", "control_mode": mode}
    with pytest.raises(ValueError, match=key):
        harness.state.parse_steps([raw])
    metadata = config.task_desc["control_modes"]
    assert metadata[key] is False
    assert metadata["skill_sha256"] == hashlib.sha256(variables["policy_skill"].encode()).hexdigest()
    assert metadata["capabilities_sha256"] == hashlib.sha256((SKILL_DIRECTORY / "xiaomi-mibot.capabilities.json").read_bytes()).hexdigest()
    assert stack.metadata()["effective"][key] is False

    class Runtime:
        def solve(self, **kwargs):
            self.prompt = kwargs["system_prompt"]
            return RuntimeResult(finish_result={"strategy": "act", "steps": [{
                "id": "monitor", "intent": "continue the operation",
                "allowed_tools": ["continue_vla"], "control_mode": "monitor",
            }]})

    runtime = Runtime()
    planner = RuntimePlanGenerator(runtime, require_outcome_program=False, action_guidance=guidance)
    planner.generate(goal="continue", observations=[], constraints=[], physical_tools={"continue_vla"})
    assert system_prompt()[section] not in runtime.prompt


def test_default_prompt_is_byte_identical_and_both_switches_compose():
    config = get_env_spec("robocasa").parse_config(args_for())
    assert cli._configured_prompt_bundle("robocasa", config.prompt_vars).system() == system_prompt()
    assert config.prompt_vars["policy_skill"] == load_policy_skill("xiaomi-mibot")
    config = get_env_spec("robocasa").parse_config(args_for("--no-transition-takeover", "--no-op-takeover"))
    assert not {"TRANSITIONS", "TAKEOVER"} & cli._configured_prompt_bundle("robocasa", config.prompt_vars).system().keys()
    assert "## Transitions\n" not in config.prompt_vars["policy_skill"]
    assert "## Takeover\n" not in config.prompt_vars["policy_skill"]


def test_control_tool_sets_have_one_source():
    from robots.robocasa import capabilities, privileged_tools

    for module in (capabilities, privileged_tools):
        assert module.PRIVILEGED_PHYSICAL_TOOLS == frozenset(control_modes.PRIVILEGED_TOOLS)
        assert module.READ_ONLY_PERCEPTION_TOOLS == frozenset(control_modes.ALLOWLISTABLE_READ_ONLY_TOOLS)
    assert capabilities.TIERS == frozenset(control_modes.TIERS)
    assert capabilities.CONTROL_MODES == frozenset(control_modes.CONTROL_MODES)
    assert capabilities.VLA_TOOLS == frozenset(control_modes.POLICY_TOOLS)
