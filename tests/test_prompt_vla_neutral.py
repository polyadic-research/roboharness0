"""Harness text points at "the VLA"; facts about one policy live in its skill (1.2.0)."""

import re

import numpy as np
import pytest

import rpent.cli.main as cli
from robots.robocasa.prompts.system import system_prompt, task_rules
from robots.robocasa.toolkit import RoboCasaToolkit

VLA_NAMES = re.compile(r"Pi0\.5|π0\.5|pi0\.5|Xiaomi|MiBoT|SmolVLA|OpenVLA|GR00T")


class _Controller:
    def execute_policy_chunk(self, observation, instruction, *, max_actions=None):
        raise AssertionError("schema inspection must not run a policy")


def _toolkit(forward):
    return RoboCasaToolkit(
        _Controller(),
        initial_observation={
            "state": np.zeros(16, dtype=np.float32),
            "robot0_agentview_left": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_agentview_right": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand": np.zeros((4, 4, 3), dtype=np.uint8),
        },
        instruction="Open the drawer.",
        forward_step_guidance=forward,
    )


def test_system_prompt_sections_name_no_specific_vla():
    for name, text in system_prompt().items():
        assert not VLA_NAMES.search(text), (name, VLA_NAMES.search(text).group(0))


@pytest.mark.parametrize("family", ["pi05", "xiaomi-mibot", "smolvla"])
def test_task_rules_and_planner_guidance_without_a_skill_name_no_specific_vla(family):
    assert not VLA_NAMES.search(task_rules("OpenDrawer", policy_family=family))
    guidance = cli._planner_action_guidance("robocasa", {"task_rules": "", "policy_skill": ""})
    assert not VLA_NAMES.search(guidance), VLA_NAMES.search(guidance).group(0)


@pytest.mark.parametrize("forward", [True, False])
def test_tool_descriptions_name_no_specific_vla(forward):
    for spec in _toolkit(forward).get_tools_spec():
        text = repr(spec)
        assert not VLA_NAMES.search(text), (spec["name"], VLA_NAMES.search(text).group(0))
