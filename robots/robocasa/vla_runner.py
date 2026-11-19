"""Direct RoboCasa policy rollouts through the canonical Episode worker."""

from __future__ import annotations

import os
import time
from argparse import Namespace
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from robots.robocasa.config import parse_object_registries
from rpent.envs.base import get_env_spec
from rpent.episode.runtime import open_episode
from rpent.evaluation.worker import HarnessRunReport, PendingCaseRecorder
from rpent.recording.notifications import notification_sink
from rpent.utils.logging import init_output_dir


def run_vla_only_case(case, manifest, artifact_dir: Path) -> HarnessRunReport:
    """Retain real policy/transition events without constructing an LLM runtime."""
    if manifest.launch.get('execution_mode') != 'vla-only':
        raise ValueError('VLA-only runner requires a vla-only manifest')
    effective = manifest.launch.get('effective', {})
    if not isinstance(effective, Mapping):
        raise ValueError('manifest launch effective config must be a mapping')
    artifact_dir = init_output_dir(Path(artifact_dir), verbose=False)
    with _capture_physics(artifact_dir, case.task_name):
        return _run_rollout(case, manifest, artifact_dir, effective)


@contextmanager
def _capture_physics(artifact_dir, task):
    values = {'RPENT_NARRATION_STATES_DIR': str(artifact_dir / 'physics-state'),
              'RPENT_FROZEN_TASK': task}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_rollout(case, manifest, artifact_dir, effective):
    args = _runtime_args(case, manifest, effective)
    bootstrap = get_env_spec('robocasa').build_episode_bootstrap(args, artifact_dir, None)
    bootstrap.runtime_metadata['execution_mode'] = 'vla-only'
    pending = PendingCaseRecorder(manifest, case, artifact_dir.name, artifact_dir,
        notify=notification_sink(manifest.runtime.get('notification_endpoint'),
                                 os.environ.get('RPENT_NOTIFICATION_TOKEN')))
    recorder = None
    try:
        with open_episode(bootstrap) as runtime:
            recorder = pending.bind(bootstrap, runtime)
            instruction = runtime.simulator.instruction
            pending.record('vla_rollout_started', instruction=instruction,
                           horizon=case.horizon, actions_per_request=args.actions_per_policy_request)
            observation = runtime.observation
            initial_state = np.asarray(observation['state'], dtype=np.float64)
            max_state_delta = 0.0
            count, chunks, final = 0, 0, None
            while count < case.horizon:
                started = time.monotonic()
                result = runtime.controller.execute_policy_chunk(
                    observation, instruction,
                    max_actions=min(args.actions_per_policy_request, case.horizon-count))
                if not result.transitions:
                    raise RuntimeError('policy rollout made no confirmed transitions')
                count += result.executed_actions
                for transition in result.transitions:
                    delta = np.asarray(transition.observation['state'], dtype=np.float64)-initial_state
                    max_state_delta = max(max_state_delta, float(np.max(np.abs(delta))))
                chunks += 1
                final = result.final
                observation = final.observation
                pending.record('vla_chunk_completed', chunk_index=chunks,
                    requested_actions=result.requested_actions, executed_actions=result.executed_actions,
                    step_id=final.step_id, success=final.success,
                    chunk_elapsed_s=time.monotonic()-started, stop_reason=result.stop_reason)
                if final.terminal or result.stop_reason is not None:
                    break
            success = final is not None and final.success
            pending.record('vla_rollout_completed', success=success, chunks_used=chunks,
                           executed_actions=count, truncated=bool(final and final.truncated),
                           max_state_delta=max_state_delta,
                           robot_motion_observed=max_state_delta > 1e-4)
        return HarnessRunReport(exit_code=0, terminal_classification='SUCCESS' if success else 'FAILED',
            harness_state='complete' if success else 'terminal_failure',
            simulator_success=bool(success), contract_success=bool(success),
            terminated=bool(final and (final.terminated or final.success)),
            truncated=bool(final and final.truncated),
            step_count=int(recorder.snapshot()['transition_count']), action_count=count,
            error_type=None, error_message=None, recorder=recorder)
    except BaseException as error:
        if recorder is not None:
            error.rpent_recorder = recorder
        raise


def _runtime_args(case, manifest, effective):
    return Namespace(
        task=case.task_name, seed=case.seed, max_episode_steps=case.horizon,
        vla=manifest.policy.get('policy_family', effective.get('vla', 'smolvla')),
        planning_profile=effective.get('planning_profile', 'robocasa-horizon-gated-v1'),
        object_registries=parse_object_registries(effective.get('object_registries', 'lightwheel,objaverse')),
        env_endpoint=manifest.runtime.get('simulator_endpoint'),
        policy_endpoint=os.environ.get('RPENT_SHARED_POLICY_ENDPOINT') or manifest.runtime.get('policy_endpoint'),
        policy_device=str(effective.get('policy_device', 'cuda:0')),
        policy_dtype=effective.get('policy_dtype'),
        actions_per_policy_request=int(effective.get('actions_per_policy_request') or
                                      manifest.policy.get('executed_actions_per_request', 16)),
        no_forward_step_guidance=bool(effective.get('no_forward_step_guidance', False)),
    )
