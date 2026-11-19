"""Shared vocabulary for per-step control modes, policy tiers and privileged tools.

Harness 1.0.10 block A (rules and prompts) defines these names. Blocks B
(privileged tools), C (plan-step fields and ``parse_steps`` checks), D
(capability data) and F (experiment wiring) must use them unchanged so that the
prompt text, the plan schema and the capability file never drift apart. 1.0.11
adds the operation identity fields, progress windows and LLM segment quotas
(backlog A5, B1-B5). This module holds no behavior: no tool, lifecycle or
control-loop logic.
"""

from __future__ import annotations

from typing import Final

POLICY_DISPLAY_NAMES: Final[dict[str, str]] = {
    "pi05": "π0.5", "xiaomi-mibot": "Xiaomi-Robotics-1", "smolvla": "SmolVLA",
}


def policy_display_name(family: str) -> str:
    return POLICY_DISPLAY_NAMES.get(family, family)

# Plan-step fields added by block C. The prompt also asks the model to repeat
# them inside ``intent`` as ``row=<name>; tier=<tier>; control_mode=<mode>`` so
# the information survives until the schema carries the structured fields.
STEP_FIELDS: Final[tuple[str, ...]] = ("skill_row", "tier", "control_mode")

CONTROL_MODES: Final[tuple[str, ...]] = ("delegate", "monitor", "takeover", "transition")

# Policy tiers in prompt vocabulary. ``none`` means no matching capability row.
TIERS: Final[tuple[str, ...]] = ("reliable", "unstable", "weak", "none")

# Labels used by ``<policy>.capabilities.json`` (block D) mapped to prompt tiers.
TIER_LABELS: Final[dict[str, str]] = {
    "可靠": "reliable",
    "不稳定": "unstable",
    "弱": "weak",
    "无记录": "none",
}

# Default control mode for an operation of each tier. Transitions are not
# tier-driven: they are the carry between two operations at different fixtures.
TIER_CONTROL_MODE: Final[dict[str, str]] = {
    "reliable": "delegate",
    "unstable": "monitor",
    "weak": "takeover",
    "none": "takeover",
}

# Modes in which the policy executes the operation and the 1.0.9 conservative
# stall/timing rules apply.
POLICY_DRIVEN_MODES: Final[frozenset[str]] = frozenset({"delegate", "monitor"})

# Modes in which the LLM drives the robot through privileged tools and the
# policy neither runs nor receives an instruction.
LLM_DRIVEN_MODES: Final[frozenset[str]] = frozenset({"takeover", "transition"})

# Policies conditioned on language receive the environment sentence and the
# LLM's operation sentences; every other family receives its training task id.
# Harness text names neither; policy-specific facts live in the policy skill.
LANGUAGE_POLICY_FAMILIES: Final[frozenset[str]] = frozenset({"pi05", "xiaomi-mibot"})

# Physical tools added by block B. They may appear in ``allowed_tools`` only for
# LLM-driven steps; ``parse_steps`` (block C) enforces this.
PRIVILEGED_TOOLS: Final[tuple[str, ...]] = (
    "navigate_to", "move_eef_path", "adjust_torso_to", "pour_into", "place_container",
)

# 1.2.1: place_container sets held handle cookware down at a back_projected destination.
# 1.0.12: pour_into tilts a held container about its rim over a receiving point;
# the prompt reserves it for takeover steps, the parser accepts it wherever the
# other privileged tools are legal.
PRIVILEGED_TOOLS_BY_MODE: Final[dict[str, tuple[str, ...]]] = {
    "transition": ("navigate_to", "adjust_torso_to", "move_eef_path", "pour_into", "place_container"),
    "takeover": ("move_eef_path", "navigate_to", "adjust_torso_to", "pour_into", "place_container"),
}

# Read-only grounding tools. ``zoom`` is added by block B; none of these is a
# physical tool and none may be listed in ``allowed_tools``.
READ_ONLY_GROUNDING_TOOLS: Final[tuple[str, ...]] = ("observe", "back_project", "zoom")

# Read-only tools the parser may accept in an action allowlist; no motion budget.
ALLOWLISTABLE_READ_ONLY_TOOLS: Final[tuple[str, ...]] = ("zoom",)

# Policy tools. Never in transition steps; in takeover steps only for the
# contact phase of a staged takeover (Adam, 2026-10-09).
POLICY_TOOLS: Final[tuple[str, ...]] = (
    "smolvla_pick", "continue_vla", "execute_vla", "resume_vla",
)

# Plan-step identity fields (1.0.11 A5). ``operation`` is a stable name the
# planner gives one complete operation and keeps across replanning; the pick,
# carry and place steps split from one pick-and-place share it. ``subtask``
# names the part of that operation a step covers (default ``whole``; a
# transition defaults to ``carry``). ``travel_m`` is a transition's planar
# base travel, measured from public evidence, that sizes its step quota.
IDENTITY_FIELDS: Final[tuple[str, ...]] = ("operation", "subtask", "travel_m")
SUBTASK_WHOLE: Final[str] = "whole"
SUBTASK_CARRY: Final[str] = "carry"

# Real (physically executed) verified failures of one delegate/monitor subtask
# that permit bounded takeover; one destructive failure/conflict is exceptional.
FALLBACK_FAILURES: Final[int] = 2

# A delegate/monitor subtask whose exact atomic row has typical steps may
# escalate to takeover once its policy steps reach this multiple of the typical
# steps (1.0.10 rule 7) *and* a further no-progress window has passed (1.0.11
# B2). In the 2,500-episode pi05 benchmark, 6.5% of atomic episodes still
# unsolved at 1.5x typical steps succeeded later (14.5% at 1.25x, 2.9% at 2x).
ESCALATION_TYPICAL_FACTOR: Final[float] = 1.5

# Executed control steps without a progress signal (pass/progress verdict,
# maintained hold, or a shrinking measured target distance) after which a
# policy subtask counts as stalled (1.0.11 B2/B3/B7). Provisional: the backlog
# fixes 100-150 steps until the full evaluation calibrates it.
NO_PROGRESS_WINDOW_STEPS: Final[int] = 120

# A measured target distance must shrink by at least this much to count as
# progress (verify_step metrics.target_distance_m).
PROGRESS_DISTANCE_M: Final[float] = 0.02

# LLM-driven segments (takeover or transition) run on quotas instead of a
# fixed failure count (1.0.11 B4). Step quota: a transition gets
# TRANSITION_BASE_STEPS + TRANSITION_STEPS_PER_M * travel_m; a takeover gets
# TAKEOVER_TYPICAL_FACTOR x typical steps when its row has them, otherwise
# TAKEOVER_NONE_STEPS. Both are capped by the remaining horizon minus the
# policy fallback reserve and never fall below MIN_SEGMENT_STEPS. Provisional
# until the fork experiments (backlog E3) calibrate them.
TRANSITION_BASE_STEPS: Final[int] = 150
TRANSITION_STEPS_PER_M: Final[int] = 200
TRANSITION_DEFAULT_TRAVEL_M: Final[float] = 1.5
TRANSITION_MAX_TRAVEL_M: Final[float] = 6.0
TAKEOVER_TYPICAL_FACTOR: Final[float] = 1.5
# 1.0.12: a pour from the grasp pose needs about 450 steps through pour_into
# (align 180-300, tilt 150, return 100); 400 ended the D10 segment mid-return.
TAKEOVER_NONE_STEPS: Final[int] = 500
MIN_SEGMENT_STEPS: Final[int] = 50
# Horizon kept for the policy fallback when the row has no typical steps.
FALLBACK_RESERVE_STEPS: Final[int] = 200
# Physical calls (privileged or, in a staged takeover, policy) per segment:
# at least SEGMENT_CALL_QUOTA, or one per SEGMENT_STEPS_PER_CALL quota steps for
# long segments (1.0.11 canary D06 used 15 calls in 586 of 1021 quota steps).
SEGMENT_CALL_QUOTA: Final[int] = 15
SEGMENT_STEPS_PER_CALL: Final[int] = 40
# An LLM segment that goes this many executed steps without progress ends early.
SEGMENT_NO_PROGRESS_STEPS: Final[int] = 150
# LLM segments of one subtask that may end in a fallback before the subtask
# stays with the policy; bounds LLM/policy ping-pong now that a fallback is no
# longer permanent (1.0.11 B5). Segments that pass do not count: a carry split
# into several passing legs (1.0.11 canary D07) is not ping-pong.
MAX_LLM_FALLBACKS: Final[int] = 3
# Verified failure types that end an LLM segment at once (drop, spill, unsafe).
DESTRUCTIVE_FAILURE_TYPES: Final[frozenset[str]] = frozenset({"object_dropped", "unsafe_trajectory"})

# Two consecutive calls of one privileged tool whose targets differ by less
# than this are reported as a repeat (1.0.11 B4: every retry changes approach).
REPEAT_POSITION_M: Final[float] = 0.01
REPEAT_ROTATION_DEG: Final[float] = 2.0

# Prompt sections that can be removed for ablation; see ``system_prompt``.
ABLATABLE_SECTIONS: Final[tuple[str, ...]] = ("CONTROL MODES", "TRANSITIONS", "TAKEOVER")

__all__ = [
    "LANGUAGE_POLICY_FAMILIES",
    "ALLOWLISTABLE_READ_ONLY_TOOLS",
    "ABLATABLE_SECTIONS",
    "CONTROL_MODES",
    "DESTRUCTIVE_FAILURE_TYPES",
    "ESCALATION_TYPICAL_FACTOR",
    "FALLBACK_FAILURES",
    "FALLBACK_RESERVE_STEPS",
    "IDENTITY_FIELDS",
    "LLM_DRIVEN_MODES",
    "MAX_LLM_FALLBACKS",
    "MIN_SEGMENT_STEPS",
    "NO_PROGRESS_WINDOW_STEPS",
    "POLICY_DRIVEN_MODES",
    "POLICY_TOOLS",
    "PRIVILEGED_TOOLS",
    "PRIVILEGED_TOOLS_BY_MODE",
    "PROGRESS_DISTANCE_M",
    "READ_ONLY_GROUNDING_TOOLS",
    "REPEAT_POSITION_M",
    "REPEAT_ROTATION_DEG",
    "SEGMENT_CALL_QUOTA",
    "SEGMENT_NO_PROGRESS_STEPS",
    "SEGMENT_STEPS_PER_CALL",
    "STEP_FIELDS",
    "SUBTASK_CARRY",
    "SUBTASK_WHOLE",
    "TAKEOVER_NONE_STEPS",
    "TAKEOVER_TYPICAL_FACTOR",
    "TIERS",
    "TIER_CONTROL_MODE",
    "TIER_LABELS",
    "TRANSITION_BASE_STEPS",
    "TRANSITION_DEFAULT_TRAVEL_M",
    "TRANSITION_MAX_TRAVEL_M",
    "TRANSITION_STEPS_PER_M",
]
