"""Fast tests for versioned planning configuration and shared schemas."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rpent.cli.main import _build_argparser
from rpent.harness import HarnessSession
from rpent.harness.recovery import RecoveryPolicy
from rpent.harness.verification import VerificationCoordinator, VerificationPolicy
from rpent.planning import (
    DEFAULT_PLANNING_PROFILE,
    ROBOCASA_HOOKED_GATED_PLANNING_PROFILE,
    ROBOCASA_XIAOMI_MIBOT_PLANNING_PROFILE,
    ObservationSummary,
    PlanningStack,
    PlanStep,
    StepVerdict,
    TaskSpec,
    get_planning_profile,
    list_planning_profiles,
    resolve_planning_stack,
    resolve_required_planning_profile,
)


class PlanningProfileTest(unittest.TestCase):
    def test_profiles_have_stable_unique_ids(self) -> None:
        profiles = list_planning_profiles()
        ids = [profile.id for profile in profiles]
        self.assertIn(DEFAULT_PLANNING_PROFILE, ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_required_planning_profile_rejects_missing_and_canonicalizes_alias(
        self,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="--planning-profile is required for non-UI evaluation",
        ):
            resolve_required_planning_profile(None)

        self.assertEqual(
            resolve_required_planning_profile(
                "libero-authoritative-gated-v1"
            ),
            "authoritative-gated-v1",
        )

    def test_profile_defaults_and_cli_overrides_are_separate(self) -> None:
        stack = resolve_planning_stack(
            "libero-trace-v1",
            harness_mode="plan-gated",
            max_replans=7,
        )
        self.assertIsInstance(stack, PlanningStack)
        self.assertEqual(stack.profile.harness_mode, "trace")
        self.assertEqual(stack.harness_mode, "plan-gated")
        self.assertEqual(stack.max_replans, 7)
        metadata = stack.metadata()
        self.assertEqual(metadata["profile_id"], "libero-trace-v1")
        self.assertEqual(
            metadata["effective"]["harness_mode"],
            "plan-gated",
        )
        self.assertEqual(
            stack.profile.planning_algorithm_version,
            "plan-generate-register-verify-repair-v2",
        )
        self.assertIn("create_plan", stack.prompt_context())
        self.assertIn("register_plan", stack.prompt_context())
        self.assertIn("verify_step", stack.prompt_context())
        self.assertFalse(stack.uses_authoritative_session)

    def test_authoritative_profile_rejects_non_gated_mode_overrides(self) -> None:
        for mode in ("reactive", "trace"):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError, "requires harness mode 'plan-gated'"
            ):
                resolve_planning_stack(
                    "libero-authoritative-gated-v1",
                    harness_mode=mode,
                )

    def test_reactive_profile_preserves_existing_loop(self) -> None:
        stack = resolve_planning_stack(DEFAULT_PLANNING_PROFILE)
        self.assertEqual(stack.harness_mode, "reactive")
        self.assertEqual(
            stack.context_policy.id,
            "raw-tool-result-v1",
        )
        self.assertIn("normal reactive tool loop", stack.prompt_context())

    def test_fast_profile_skips_guide_reads_and_batches_localization(self) -> None:
        stack = resolve_planning_stack("libero-fast-trace-v1")
        self.assertEqual(stack.context_policy.id, "libero-compact-v1")
        self.assertIn("Do not read MEMORY.md", stack.prompt_context())
        self.assertIn("points=[[row,col]", stack.prompt_context())
        self.assertIn("Do not read", stack.startup_instruction())

    def test_authoritative_profile_enables_session_contract(self) -> None:
        stack = resolve_planning_stack("libero-authoritative-gated-v1")
        self.assertEqual(stack.harness_mode, "plan-gated")
        self.assertTrue(stack.profile.authoritative_session)
        self.assertTrue(stack.uses_authoritative_session)
        self.assertTrue(stack.profile.require_execution_id)
        self.assertFalse(stack.profile.auto_verify_tool_signals)
        self.assertFalse(stack.profile.require_environment_success)
        self.assertIn("API", stack.profile.description)
        self.assertEqual(
            stack.profile.planning_algorithm_version,
            "authoritative-outcome-session-v1",
        )
        self.assertEqual(
            stack.profile.planner_prompt_version,
            "authoritative-harness-v1",
        )
        self.assertEqual(
            stack.profile.agent_runtime_version,
            "authoritative-harness-session-v1",
        )
        prompt = stack.prompt_context()
        self.assertIn("authoritative HarnessSession", prompt)
        self.assertIn("execution_status.allowed_operations", prompt)
        self.assertIn("execution_status.required_operation", prompt)
        metadata = stack.metadata()["effective"]
        self.assertTrue(metadata["authoritative_session"])
        self.assertTrue(metadata["require_execution_id"])
        self.assertFalse(metadata["auto_verify_tool_signals"])
        self.assertFalse(metadata["require_environment_success"])
        self.assertTrue(stack.profile.metadata()["authoritative_session"])
        self.assertIn("request_exception", prompt)
        self.assertIn("apply_recovery", prompt)
        self.assertIn("configured_physical_tool is a category placeholder", prompt)
        self.assertIn("execution_status.allowed_tools", prompt)
        self.assertIn("run-output audit writing utilities", prompt)
        self.assertIn("Never reuse the previous observation result", prompt)
        self.assertIn("content-addressed refs may repeat", prompt)
        self.assertIn("new read-only observation", prompt)
        self.assertNotIn("official environment predicate", prompt)
        self.assertNotIn("tool-signal", prompt)

    def test_profiles_resolve_a_central_recovery_policy_by_default(self) -> None:
        stack = resolve_planning_stack("authoritative-gated-v1")

        self.assertEqual(stack.recovery_policy, RecoveryPolicy())
        self.assertEqual(
            stack.metadata()["effective"]["recovery_policy"],
            {
                "local_replans_per_outcome": 1,
                "stage_replans_per_stage": 1,
                "remaining_stage_repartitions_per_run": 1,
                "max_total_physical_actions_per_run": 100,
            },
        )

    def test_robocasa_authoritative_profile_owns_extended_recovery(self) -> None:
        stack = resolve_planning_stack("robocasa-authoritative-gated-v1")

        self.assertEqual(stack.max_replans, 8)
        self.assertEqual(
            stack.recovery_policy,
            RecoveryPolicy(
                local_replans_per_outcome=6,
                stage_replans_per_stage=1,
                remaining_stage_repartitions_per_run=1,
                max_total_physical_actions_per_run=9,
            ),
        )
        self.assertEqual(
            resolve_planning_stack("authoritative-gated-v1").recovery_policy,
            RecoveryPolicy(),
        )

    def test_robocasa_v2_budget_covers_eighteen_25_action_requests(self) -> None:
        stack = resolve_planning_stack("robocasa-authoritative-gated-v2")

        self.assertEqual(stack.max_replans, 17)
        self.assertEqual(
            stack.recovery_policy,
            RecoveryPolicy(
                local_replans_per_outcome=15,
                stage_replans_per_stage=1,
                remaining_stage_repartitions_per_run=1,
                max_total_physical_actions_per_run=18,
            ),
        )

    def test_profile_rejects_recovery_budget_above_replan_ceiling(self) -> None:
        profile = get_planning_profile("authoritative-gated-v1")

        with self.assertRaisesRegex(ValueError, "recovery replan budget"):
            replace(
                profile,
                max_replans=2,
                recovery_policy=RecoveryPolicy(
                    local_replans_per_outcome=2,
                    stage_replans_per_stage=1,
                    remaining_stage_repartitions_per_run=0,
                ),
            )
    def test_authoritative_profile_has_neutral_id_and_legacy_input_alias(
        self,
    ) -> None:
        canonical = resolve_planning_stack("authoritative-gated-v1")
        legacy = resolve_planning_stack("libero-authoritative-gated-v1")

        self.assertIs(legacy.profile, canonical.profile)
        self.assertEqual(canonical.profile.id, "authoritative-gated-v1")
        registered = [profile.id for profile in list_planning_profiles()]
        self.assertIn("authoritative-gated-v1", registered)
        self.assertNotIn("libero-authoritative-gated-v1", registered)

        parser = _build_argparser()
        planning_action = next(
            action
            for action in parser._actions
            if action.dest == "planning_profile"
        )
        self.assertIn("authoritative-gated-v1", planning_action.choices)
        self.assertIn("libero-authoritative-gated-v1", planning_action.choices)

    def test_authoritative_profile_rejects_unimplemented_environment_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "environment success gate"):
            resolve_planning_stack(
                "libero-authoritative-gated-v1",
                require_environment_success=True,
            )

    def test_v3_profile_preserves_legacy_identity_and_protocol(self) -> None:
        profile = next(
            profile
            for profile in list_planning_profiles()
            if profile.id == "libero-hooked-gated-v3"
        )
        stack = resolve_planning_stack(profile.id)

        self.assertFalse(profile.authoritative_session)
        self.assertFalse(stack.uses_authoritative_session)
        self.assertEqual(profile.planning_algorithm_version, "mode-hook-feedback-loop-v3")
        self.assertEqual(profile.planner_prompt_version, "libero-hooked-v3")
        self.assertEqual(profile.agent_runtime_version, "legacy-agent-runtime-v1")
        self.assertTrue(profile.auto_verify_tool_signals)
        self.assertTrue(profile.require_environment_success)
        self.assertNotIn("authoritative_session", profile.metadata())
        self.assertNotIn("authoritative_session", stack.metadata()["effective"])
        self.assertNotIn("authoritative_session", stack.component_metadata())
        self.assertIn("planning-loop-v3", stack.prompt_context())
        self.assertIn("create_plan", stack.prompt_context())
        self.assertIn("register_plan", stack.prompt_context())
        self.assertIn("one-shot full-task contract", stack.prompt_context())
        self.assertIn("no continuation or stage-advance", stack.prompt_context())
        self.assertIn("authoritative terminal evidence", stack.prompt_context())
        self.assertNotIn("start_instruction", stack.prompt_context())
        rendered = stack.prompt_context() + "\n" + stack.startup_instruction()
        self.assertIn("back_project", rendered)
        self.assertIn("segment", rendered)
        self.assertIn("Pi0-pick", rendered)

    def test_episode_hooked_profile_copies_v3_control_contract(self) -> None:
        generic = resolve_planning_stack("episode-hooked-gated-v1")
        legacy = resolve_planning_stack("libero-hooked-gated-v3")

        self.assertEqual(
            generic.profile.planning_algorithm_version,
            "mode-hook-feedback-loop-v3",
        )
        self.assertEqual(generic.profile.planner_prompt_version, "episode-hooked-v1")
        self.assertEqual(
            generic.profile.verifier_prompt_version,
            "structured-feedback-verifier-v2",
        )
        self.assertEqual(
            generic.profile.observation_version,
            "episode-observation-summary-v1",
        )
        self.assertEqual(
            generic.profile.context_policy_version,
            "episode-compact-v1",
        )
        self.assertEqual(generic.context_policy.id, "episode-compact-v1")
        self.assertEqual(
            (
                generic.harness_mode,
                generic.max_replans,
                generic.profile.require_execution_id,
                generic.profile.auto_verify_tool_signals,
                generic.require_environment_success,
            ),
            (
                legacy.harness_mode,
                legacy.max_replans,
                legacy.profile.require_execution_id,
                legacy.profile.auto_verify_tool_signals,
                legacy.require_environment_success,
            ),
        )

    def test_episode_hooked_protocol_is_simulator_neutral(self) -> None:
        stack = resolve_planning_stack("episode-hooked-gated-v1")
        rendered = stack.prompt_context() + "\n" + stack.startup_instruction()

        self.assertIn("planning-loop-v3", rendered)
        self.assertIn("create_plan", rendered)
        self.assertIn("register_plan", rendered)
        self.assertIn("allowed_tools", rendered)
        self.assertIn("environment predicate", rendered)
        for forbidden in (
            "view_driver_state",
            "back_project",
            "segment",
            "pi0_pick",
            "set_gripper",
            "execute_vla",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_robocasa_hooked_profile_uses_legacy_episode_contract(self) -> None:
        stack = resolve_planning_stack(ROBOCASA_HOOKED_GATED_PLANNING_PROFILE)

        self.assertEqual(stack.profile.id, "robocasa-hooked-gated-v1")
        self.assertFalse(stack.uses_authoritative_session)
        self.assertEqual(
            stack.profile.planning_algorithm_version,
            "mode-hook-feedback-loop-v3",
        )
        self.assertEqual(
            stack.profile.planner_prompt_version,
            "episode-hooked-v1",
        )
        self.assertEqual(
            stack.profile.observation_version,
            "episode-observation-summary-v1",
        )
        self.assertEqual(stack.context_policy.id, "episode-compact-v1")
        self.assertEqual(stack.harness_mode, "plan-gated")
        self.assertEqual(stack.max_replans, 8)
        self.assertTrue(stack.profile.require_execution_id)
        self.assertTrue(stack.profile.auto_verify_tool_signals)
        self.assertTrue(stack.require_environment_success)
        self.assertNotIn(
            "authoritative_session",
            stack.metadata()["effective"],
        )

    def test_xiaomi_profile_clones_harness_behavior_and_adds_policy_identity(self) -> None:
        existing = get_planning_profile(ROBOCASA_HOOKED_GATED_PLANNING_PROFILE)
        xiaomi = get_planning_profile(ROBOCASA_XIAOMI_MIBOT_PLANNING_PROFILE)

        ignored = {
            "id",
            "description",
            "policy_family",
            "checkpoint_repo_id",
            "checkpoint_revision",
            "binding_name",
            "observation_history",
            "observation_interval",
            "crop_ratio",
            "generated_chunk_size",
            "native_action_shape",
            "canonical_task_input",
            "supports_articulated_control",
        }
        assert {
            key: value
            for key, value in vars(existing).items()
            if key not in ignored
        } == {
            key: value for key, value in vars(xiaomi).items() if key not in ignored
        }
        assert existing.metadata().get("policy_family") is None
        assert xiaomi.policy_family == "xiaomi-mibot"
        assert xiaomi.checkpoint_revision == (
            "0d1aa76d0d82debc9b611e4d1e231096434d5be4"
        )
        assert xiaomi.observation_history == 4
        assert xiaomi.observation_interval == 2
        assert xiaomi.generated_chunk_size == 16
        assert xiaomi.native_action_shape == (12,)
        assert xiaomi.require_outcome_program is False

    def test_authoritative_session_prompt_excludes_legacy_control_tools(self) -> None:
        stack = resolve_planning_stack("libero-authoritative-gated-v1")
        chinese_stack = resolve_planning_stack(
            "libero-authoritative-gated-v1", use_chinese=True
        )
        for current in (stack, chinese_stack):
            rendered = current.prompt_context() + "\n" + current.startup_instruction()
            for forbidden in (
                "create_plan",
                "register_plan",
                "replan",
                "request_replan",
                "select_outcome",
                "advance_stage",
                "view_plan",
            ):
                self.assertNotIn(forbidden, rendered)
        self.assertIn("start_instruction", stack.startup_instruction())
        self.assertLess(
            stack.startup_instruction().index("view_driver_state"),
            stack.startup_instruction().index("start_instruction"),
        )
        self.assertIn("state='complete'", stack.prompt_context())
        self.assertIn("contract_success=true", stack.prompt_context())

    def test_only_authoritative_profiles_use_authoritative_session(self) -> None:
        profiles = list_planning_profiles()
        authoritative = [profile.id for profile in profiles if profile.authoritative_session]

        self.assertEqual(
            authoritative,
            [
                "authoritative-gated-v1",
                "robocasa-authoritative-gated-v1",
                "robocasa-authoritative-gated-v2",
            ],
        )
        for profile in profiles:
            stack = resolve_planning_stack(profile.id)
            self.assertEqual(stack.uses_authoritative_session, profile.authoritative_session)

    def test_authoritative_session_factory_wires_verifiers_and_profile_policy(
        self,
    ) -> None:
        class NeverCalledVerifier:
            name = "test-verifier"
            version = "v1"

            def verify(self, request):  # pragma: no cover - construction only
                raise AssertionError(request)

        verifier = NeverCalledVerifier()
        stack = resolve_planning_stack("libero-authoritative-gated-v1")
        with TemporaryDirectory() as tmp:
            session = stack.build_harness_session(
                trace_path=Path(tmp) / "trace.jsonl",
                predicate_verifiers=(verifier,),
            )

        self.assertIsInstance(session, HarnessSession)
        coordinator = session.controller.verification_coordinator
        self.assertIsInstance(coordinator, VerificationCoordinator)
        self.assertEqual(coordinator._verifiers, (verifier,))
        self.assertIsInstance(coordinator._policy, VerificationPolicy)
        self.assertEqual(
            coordinator._policy.max_observation_age_s,
            stack.profile.verification_max_observation_age_s,
        )
        self.assertEqual(
            coordinator._policy.min_confidence,
            stack.profile.verification_min_confidence,
        )
        self.assertEqual(
            session.controller.recovery_tracker.policy,
            stack.recovery_policy,
        )

    def test_legacy_profile_rejects_authoritative_session_factory(self) -> None:
        stack = resolve_planning_stack("libero-plan-gated-v1")
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "authoritative HarnessSession"
        ):
            stack.build_harness_session(
                trace_path=Path(tmp) / "trace.jsonl",
                predicate_verifiers=(),
            )

    def test_authoritative_session_factory_requires_a_predicate_verifier(self) -> None:
        stack = resolve_planning_stack("libero-authoritative-gated-v1")
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "predicate verifier"
        ):
            stack.build_harness_session(
                trace_path=Path(tmp) / "trace.jsonl",
                predicate_verifiers=(),
            )


    def test_v3_can_treat_environment_success_as_audit_only(self) -> None:
        stack = resolve_planning_stack(
            "libero-hooked-gated-v3",
            require_environment_success=False,
        )
        self.assertFalse(stack.require_environment_success)
        self.assertFalse(
            stack.metadata()["effective"]["require_environment_success"]
        )
        prompt = stack.prompt_context()
        self.assertIn("audit-only", prompt)
        self.assertIn("finish(status='success'", prompt)
        self.assertIn("finish(status='failure'", prompt)
        self.assertIn("base_task_success=true", prompt)
        self.assertIn("Never use status='stuck'", prompt)

    def test_chinese_planning_override_is_recorded_and_prompted(self) -> None:
        stack = resolve_planning_stack(
            "libero-plan-gated-v1",
            use_chinese=True,
        )
        self.assertTrue(stack.use_chinese)
        self.assertTrue(stack.metadata()["effective"]["use_chinese"])
        prompt = stack.prompt_context()
        self.assertIn("Simplified Chinese", prompt)
        self.assertIn("verify_step evidence/summary", prompt)

    def test_cli_accepts_requested_and_canonical_chinese_flags(self) -> None:
        parser = _build_argparser()
        requested = parser.parse_args(["--env", "robocasa", "--use_chines"])
        canonical = parser.parse_args(["--env", "robocasa", "--use-chinese"])
        default = parser.parse_args(["--env", "robocasa"])
        self.assertTrue(requested.use_chines)
        self.assertTrue(canonical.use_chines)
        self.assertFalse(default.use_chines)

    def test_cli_has_separate_outer_and_nested_planner_timeouts(self) -> None:
        parser = _build_argparser()
        defaults = parser.parse_args(["--env", "robocasa"])
        configured = parser.parse_args(
            [
                "--env",
                "robocasa",
                "--planner-timeout-s",
                "3600",
                "--plan-generation-timeout-s",
                "180",
            ]
        )
        self.assertIsNone(defaults.planner_timeout_s)
        self.assertEqual(defaults.plan_generation_timeout_s, 180)
        self.assertEqual(configured.planner_timeout_s, 3600)
        self.assertEqual(configured.plan_generation_timeout_s, 180)



class PlanningSchemaTest(unittest.TestCase):
    def test_shared_contracts_accept_a_grounded_step(self) -> None:
        task = TaskSpec(
            raw_instruction="Put the black bowl on the plate",
            verb="pick_and_place",
            target_object="black bowl",
            destination="plate",
            success_description="black bowl rests on plate",
        )
        observation = ObservationSummary(
            step=0,
            image_paths=("images_cam_hi/image_cam_hi_00.png",),
            eef_position=(0.0, 0.0, 0.4),
        )
        step = PlanStep(
            id="grasp",
            intent="grasp the visually grounded black bowl",
            allowed_tools=("pi0_pick",),
            expected_evidence=("bowl appears in wrist camera",),
        )
        verdict = StepVerdict(
            verdict="pass",
            evidence=("wrist camera contains the black bowl",),
            summary="target grasp confirmed",
        )
        self.assertEqual(task.destination, "plate")
        self.assertEqual(observation.step, 0)
        self.assertEqual(step.allowed_tools, ("pi0_pick",))
        self.assertEqual(verdict.verdict, "pass")

    def test_verdict_requires_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "evidence"):
            StepVerdict(
                verdict="pass",
                evidence=(),
                summary="unsupported success claim",
            )


if __name__ == "__main__":
    unittest.main()
