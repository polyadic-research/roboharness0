import os
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from rpent.envs.base import get_env_spec
from rpent.episode.contracts import ChunkExecutionResult, StepResult
from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.manifest import EvaluationManifest
from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT as checkpoint


def manifest(tmp_path):
    case = EpisodeCase(ordinal=0, environment='robocasa',
        preset_name='robocasa365-pandaomron-xiaomi-mibot-v1',
        task_name='SlideDishwasherRack', task_index=0, instruction='Packaged instruction',
        seed=7, episode_index=0, horizon=34, checkpoint_repo_id=checkpoint.repo_id,
        checkpoint_revision=checkpoint.revision, binding_name=checkpoint.binding_name)
    return EvaluationManifest(rpent_commit='b'*40, rpent_dirty=False,
        cli_arguments={'argv': []}, environment={'name': 'robocasa'}, simulator={},
        policy={'policy_family': 'xiaomi-mibot'}, task_selector=case.task_name,
        cases=(case,), schedule={}, contracts={}, runtime={'evaluation_root': str(tmp_path)},
        launch={'execution_mode': 'vla-only', 'preset_version': 1,
                'runner': 'robocasa-vla-only-v1', 'submitted': {},
                'effective': {'vla': 'xiaomi-mibot', 'actions_per_policy_request': 16}})


def test_robocasa_registers_canonical_pure_worker_runner():
    assert callable(get_env_spec('robocasa').evaluation_runners.get('vla-only'))


def fake_runtime(monkeypatch, tmp_path, *, zero=False, fail=False):
    import robots.robocasa.vla_runner as module
    calls, events = [], []
    confirmed = [0]
    recorder = SimpleNamespace(snapshot=lambda: {'transition_count': confirmed[0]})
    class Pending:
        def __init__(self, *args, **kwargs): pass
        def bind(self, bootstrap, runtime):
            assert bootstrap.runtime_metadata['execution_mode'] == 'vla-only'
            return recorder
        def record(self, kind, **payload): events.append((kind, payload))
    class Controller:
        def execute_policy_chunk(self, observation, instruction, *, max_actions):
            calls.append((instruction, max_actions))
            if fail and len(calls) == 2:
                raise RuntimeError('prediction failed')
            if zero:
                return ChunkExecutionResult((), 16)
            count = 16 if len(calls) == 1 else 2
            rows = []
            for index in range(count):
                confirmed[0] += 1
                success = len(calls) == 2 and index == 1
                rows.append(StepResult('episode-1', confirmed[0],
                    {'state': np.ones(16, dtype=np.float32)*confirmed[0]},
                    float(success), success, success, False))
            return ChunkExecutionResult(tuple(rows), 16)
    runtime = SimpleNamespace(episode_id='episode-1',
        simulator=SimpleNamespace(instruction='Actual environment sentence'),
        observation={'state': np.zeros(16, dtype=np.float32)}, controller=Controller())
    @contextmanager
    def opened(bootstrap): yield runtime
    def bootstrap(args, *unused):
        assert args.task == 'SlideDishwasherRack' and args.seed == 7
        assert args.actions_per_policy_request == 16 and args.vla == 'xiaomi-mibot'
        assert os.environ['RPENT_NARRATION_STATES_DIR'].endswith('/physics-state')
        assert os.environ['RPENT_FROZEN_TASK'] == args.task
        return SimpleNamespace(runtime_metadata={})
    monkeypatch.setattr(module, 'get_env_spec', lambda name: SimpleNamespace(build_episode_bootstrap=bootstrap))
    monkeypatch.setattr(module, 'open_episode', opened)
    monkeypatch.setattr(module, 'PendingCaseRecorder', Pending)
    return module, calls, events, recorder


def test_pure_runner_uses_environment_language_and_cancels_terminal_suffix(monkeypatch, tmp_path):
    module, calls, events, recorder = fake_runtime(monkeypatch, tmp_path)
    m = manifest(tmp_path)
    report = module.run_vla_only_case(m.cases[0], m, tmp_path/'attempt')
    assert calls == [('Actual environment sentence', 16)]*2
    assert report.simulator_success and report.contract_success
    assert report.step_count == report.action_count == 18
    assert report.recorder is recorder
    chunks = [payload for kind, payload in events if kind == 'vla_chunk_completed']
    assert [c['executed_actions'] for c in chunks] == [16, 2]
    assert events[-1][0] == 'vla_rollout_completed'
    assert events[-1][1]['robot_motion_observed'] is True
    assert events[-1][1]['max_state_delta'] == 18


def test_pure_runner_fails_closed_when_no_transition_is_confirmed(monkeypatch, tmp_path):
    module, _, _, recorder = fake_runtime(monkeypatch, tmp_path, zero=True)
    m = manifest(tmp_path)
    with pytest.raises(RuntimeError, match='no confirmed transitions') as error:
        module.run_vla_only_case(m.cases[0], m, tmp_path/'attempt')
    assert error.value.rpent_recorder is recorder


def test_pure_runner_preserves_recorder_after_prediction_error(monkeypatch, tmp_path):
    module, _, _, recorder = fake_runtime(monkeypatch, tmp_path, fail=True)
    m = manifest(tmp_path)
    with pytest.raises(RuntimeError, match='prediction failed') as error:
        module.run_vla_only_case(m.cases[0], m, tmp_path/'attempt')
    assert error.value.rpent_recorder is recorder
    assert recorder.snapshot()['transition_count'] == 16


def test_pure_runner_attaches_exclusive_coordinator_lane(monkeypatch, tmp_path):
    from robots.robocasa.vla_runner import _runtime_args

    monkeypatch.setenv('RPENT_SHARED_POLICY_ENDPOINT', 'http://127.0.0.1:20001')
    m = manifest(tmp_path)
    args = _runtime_args(m.cases[0], m, m.launch['effective'])
    assert args.policy_endpoint == 'http://127.0.0.1:20001'
