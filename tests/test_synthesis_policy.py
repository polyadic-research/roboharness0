"""Evidence-gated semantic reviews are independent of bounded motor recovery."""
from copy import deepcopy

import pytest

from robots.robocasa.capabilities import load_capabilities
from rpent.harness.state import PlanStateMachine
from rpent.planning.schemas import ActionFeedback

SENTENCE = 'Move both objects onto the board, then close the cabinet.'
TOOLS = {'continue_vla', 'execute_vla', 'move_eef_path'}


def machine(task='BreadSelection'):
    events = []
    state = PlanStateMachine(
        max_replans=None, environment_horizon_only=True, physical_tools=TOOLS,
        policy_family='xiaomi-mibot', task_name=task,
        record=lambda name, **payload: events.append({'event': name, **payload}),
    )
    state.set_policy_instruction(SENTENCE)
    row = load_capabilities('xiaomi-mibot')['rows'].get(task)
    plan = {'id': 'whole', 'operation': 'whole_task', 'subtask': 'whole',
            'intent': SENTENCE, 'allowed_tools': ['continue_vla', 'execute_vla'],
            'max_attempts': 64, 'skill_row': task if row else 'none',
            'tier': row['tier'] if row else 'none', 'control_mode': 'monitor'}
    assert state.register_plan(SENTENCE, [plan])['ok']
    return state, events, plan


def act(state, step, *, failure=None, verdict='retry', progress=None, executed=16):
    eid = f'execution-{step}'
    state.record_action_feedback(ActionFeedback(
        execution_id=eid, tool_name='continue_vla', physical=True,
        execution_status='completed', raw_result={
            'executed_actions': executed, 'environment': {'step_id': step},
            'grasp_status': {'retention_monitor_active': True},
        }), input_dict={})
    result = state.verify_step(verdict, ['Fresh public image and measured state'], 'Observed',
                               execution_id=eid, failure_type=failure, progress_evidence=progress)
    assert result.get('ok'), result
    return result


def phase(step, *, target='red cup', complete=False, distance=None, description='Moved toward current goal'):
    value = {'target_id': target, 'phase': 'transport', 'observation_step': step,
             'observed_change': description, 'phase_complete': complete}
    if distance is not None:
        value['target_distance_m'] = distance
    return value


def test_atomic_two_explicit_failures_offer_but_never_force_takeover():
    state, _, plan = machine('PickPlaceCounterToSink')
    act(state, 16, failure='grasp_missed')
    assert not state.escalated_operations() and not state.needs_replan
    act(state, 32, failure='grasp_missed')
    assert state.escalated_operations()
    assert not state.needs_replan
    replacement = {**plan, 'control_mode': 'takeover', 'allowed_tools': ['move_eef_path', 'continue_vla']}
    assert state.replan('Choose a grounded correction', [replacement])['ok']


def test_unknown_first_miss_opens_semantic_only_replan_and_preserves_identity():
    state, _, plan = machine()
    act(state, 16, failure='grasp_missed')
    assert state.needs_replan and state.semantic_policy_replanning_allowed
    assert not state.escalated_operations()
    assert 'error' in state.replan('No motor permission yet', [{
        **plan, 'control_mode': 'takeover', 'allowed_tools': ['move_eef_path'],
    }])
    assert 'error' in state.replan('Cannot reset counters by renaming', [{**plan, 'operation': 'new'}])
    assert state.replan('Review current goal, keep native instruction', [plan])['ok']
    act(state, 32, failure='grasp_missed')
    assert state.escalated_operations()
    assert state.replan('Second real miss permits bounded motor recovery', [{
        **plan, 'control_mode': 'takeover', 'allowed_tools': ['move_eef_path'],
    }])['ok']


@pytest.mark.parametrize('failure', ['object_dropped', 'unsafe_trajectory'])
def test_one_destructive_failure_permits_recovery_without_two_failure_deadlock(failure):
    state, _, plan = machine()
    act(state, 16, failure=failure)
    assert state.escalated_operations()
    assert state.replan('Bounded safe recovery', [{
        **plan, 'control_mode': 'takeover', 'allowed_tools': ['move_eef_path'],
    }])['ok']


def test_explicit_target_conflict_can_open_bounded_recovery_in_native_scope():
    state, _, _ = machine()
    state.report_task_conflict({'phase': 'grasp', 'conflict_type': 'wrong_target',
                               'expected_target': 'cup', 'observed_target': 'bottle',
                               'evidence': 'Fresh public wrist image'})
    assert state.escalated_operations() and not state.whole_task_policy_first_active


@pytest.mark.parametrize('task', ['BreadSelection', 'CategorizeCondiments'])
def test_composite_old_hold_without_current_goal_progress_opens_only_semantic_review(task):
    state, _, plan = machine(task)
    act(state, 60, failure='timeout', verdict='progress', executed=60)
    act(state, 120, failure='timeout', verdict='progress', executed=60)
    assert state.needs_replan and state.semantic_policy_replanning_allowed
    assert not state.escalated_operations()
    assert state.replan('Inspect remaining goal; same VLA instruction', [plan])['ok']


def test_target_phase_marker_ignores_paraphrased_completed_facts_and_stale_steps():
    state, events, plan = machine()
    act(state, 16, verdict='progress', progress=phase(16, complete=True))
    assert state.replan('Current phase passed, inspect remaining original goal', [plan])['ok']
    act(state, 32, verdict='progress', progress=phase(
        32, target='RED-CUP', complete=True, description='Same fact with different wording'))
    assert events[-1]['progress_signal'] is False
    act(state, 48, verdict='progress', progress=phase(32, distance=.1))
    assert events[-1]['progress_signal'] is False


def test_new_target_distance_baseline_never_resets_without_measured_decrease():
    state, events, _ = machine()
    act(state, 16, verdict='progress', progress=phase(16, distance=.3))
    assert events[-1]['progress_signal'] is False
    act(state, 32, verdict='progress', progress=phase(32, target='RED-CUP', distance=.25))
    assert events[-1]['progress_signal'] is True


def test_structured_progress_is_optional_and_cannot_change_success_authority():
    state, _, _ = machine()
    act(state, 16, verdict='progress')
    assert not state.environment_success
    malformed = deepcopy(phase(32))
    malformed['phase'] = 'I changed the wording'
    with pytest.raises(ValueError, match='phase'):
        from rpent.planning.schemas import StepVerdict
        StepVerdict(verdict='progress', evidence=('image',), summary='observed', progress_evidence=malformed)
