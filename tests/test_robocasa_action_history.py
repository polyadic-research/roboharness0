"""Executed VLA commands must reach both executor and planning model inputs."""
import json

import numpy as np
import pytest

from robots.robocasa.binding import make_robocasa_binding
from robots.robocasa.toolkit import RoboCasaToolkit
from rpent.context.projectors import EpisodeCompactContextV1, RawToolContextV1
from rpent.episode.contracts import StepResult
from rpent.episode.controller import EpisodeController
from rpent.runtime import RuntimePlanGenerator, RuntimeResult


def observation():
    state = np.zeros(16, dtype=np.float32)
    state[6] = state[13] = 1
    return {'state': state, **{
        camera: np.zeros((256, 256, 3), dtype=np.uint8)
        for camera in ('robot0_agentview_center', 'robot0_agentview_left',
                       'robot0_agentview_right', 'robot0_eye_in_hand')
    }}


class Simulator:
    def __init__(self, terminal_at=None, fail_at=None):
        self.step_id = 0
        self.terminal_at = terminal_at
        self.fail_at = fail_at
        self.actions = []

    def step(self, action):
        if self.step_id + 1 == self.fail_at:
            raise RuntimeError('injected simulator failure')
        self.actions.append(action.copy())
        self.step_id += 1
        return StepResult('scene-one', self.step_id, observation(), 0,
                          self.step_id == self.terminal_at, False, False)


class Policy:
    def __init__(self, actions):
        self.actions = actions

    def predict(self, obs, instruction, *, max_actions=None):
        return self.actions[:max_actions].copy()


def toolkit(actions, *, terminal_at=None, fail_at=None):
    sim = Simulator(terminal_at, fail_at)
    controller = EpisodeController(sim, Policy(actions), make_robocasa_binding())
    tools = RoboCasaToolkit(
        controller, initial_observation=observation(), instruction='Pull the rack out.',
        forward_step_guidance=True, actions_per_policy_request=len(actions),
        episode_horizon=450,
    )
    tools.set_context_policy(EpisodeCompactContextV1())
    return tools, sim


def vectors(count=3):
    return np.linspace(-1, 1, count * 12, dtype=np.float32).reshape(count, 12)


def test_repeated_policy_chunks_reach_llm_without_duplicates():
    commands = vectors()
    tools, sim = toolkit(commands)
    result = tools.execute_tool('continue_vla', {
        'intervene': True, 'instruction': 'Pull the rack.', 'max_chunks': 2,
    })
    projected = json.loads(result.projection.text)
    history = projected['action_history']
    assert history['scope'] == 'since_previous_tool_result'
    assert len(history['records']) == 2
    assert [r['step_ids'] for r in history['records']] == [[1, 2, 3], [4, 5, 6]]
    assert all(r['control_source'] == 'vla' for r in history['records'])
    for record in history['records']:
        np.testing.assert_array_equal(np.asarray(record['actions'], dtype=np.float32), commands)
        assert record['instruction'] == 'Pull the rack.'
    np.testing.assert_array_equal(np.concatenate([r['actions'] for r in history['records']]), np.asarray(sim.actions))
    fresh = tools.execute_tool('observe', {})
    assert 'action_history' not in json.loads(fresh.projection.text)
    result2 = tools.execute_tool('execute_vla', {
        'intervene': False, 'instruction': 'Continue pulling.',
    })
    assert result2.result['instruction'] == 'Pull the rack.'
    assert json.loads(result2.projection.text)['instruction'] == 'Pull the rack.'
    assert json.loads(result2.projection.text)['action_history']['records'][0]['step_ids'] == [7, 8, 9]
    assert tools.action_history_snapshot()['total_executed_actions'] == 9


def test_terminal_suffix_is_distinguished_from_executed_actions():
    commands = vectors()
    tools, sim = toolkit(commands, terminal_at=2)
    result = tools.execute_tool('execute_vla', {'instruction': 'Pull the rack.'})
    record = json.loads(result.projection.text)['action_history']['records'][0]
    assert record['step_ids'] == [1, 2]
    assert record['executed_actions'] == 2
    assert record['cancelled_actions'] == 1
    np.testing.assert_array_equal(np.asarray(record['actions'], dtype=np.float32), commands)
    assert len(sim.actions) == 2


def test_partial_fault_keeps_confirmed_actions_and_failed_attempt_distinct():
    commands = vectors()
    tools, sim = toolkit(commands, fail_at=3)
    result = tools.execute_tool('execute_vla', {'instruction': 'Pull the rack.'})
    record = json.loads(result.projection.text)['action_history']['records'][0]
    assert 'error' in result.result
    assert record['step_ids'] == [1, 2]
    assert record['status'] == 'faulted'
    assert record['attempted_action_index'] == 2
    np.testing.assert_array_equal(np.asarray(record['actions'], dtype=np.float32), commands)
    assert len(sim.actions) == 2
    assert tools.action_history_snapshot()['total_executed_actions'] == 2
    assert tools.observe()['step'] == 2


@pytest.mark.parametrize('projector', [RawToolContextV1(), EpisodeCompactContextV1()])
def test_action_history_survives_projection_character_limit(projector):
    history = {'schema': 'robocasa-action-history/v1', 'records': [{'actions': vectors(250).tolist()}]}
    projected = projector.project_tool_result(
        tool_name='observe', result={'step': 250, 'state': {'proprioception': [0.] * 16},
                                     'action_history': history}, max_text_chars=100,
    )
    assert json.loads(projected.text)['action_history'] == history


class PlanningRuntime:
    def __init__(self):
        self.requests = []
        self.page = None

    def solve(self, *, system_prompt, user_message, toolkit, **kwargs):
        self.requests.append(json.loads(user_message))
        assert 'read_action_history' in [t['name'] for t in toolkit.get_tools_spec()]
        self.page = toolkit.execute_tool('read_action_history', {'start_chunk': 0}).result
        return RuntimeResult(finish_result={'strategy': 'Keep pulling.', 'steps': [{
            'id': 'pull', 'intent': 'Pull rack out', 'allowed_tools': ['continue_vla'],
            'expected_evidence': ['The rack extends.'],
        }]})


def test_independent_planner_receives_history_and_can_page_older_commands():
    tools, _ = toolkit(vectors())
    tools.continue_vla('Pull the rack.', max_chunks=2)
    runtime = PlanningRuntime()
    generator = RuntimePlanGenerator(runtime, require_outcome_program=False)
    generator.set_action_history_source(tools.action_history_snapshot)
    generator.generate(goal='Pull the rack out', observations=['Rack is retracted'],
                       constraints=[], physical_tools={'continue_vla'})
    assert runtime.requests[-1]['action_history']['total_executed_actions'] == 6
    assert runtime.page['action_history']['records'][0]['step_ids'] == [1, 2, 3]
    generator.generate_replan(goal='Pull the rack out', reason='No extension',
                             observations=['Rack remains inside'], constraints=[],
                             current_plan={}, physical_tools={'continue_vla'})
    assert runtime.requests[-1]['action_history']['total_executed_actions'] == 6


def test_history_does_not_cross_scene_instances():
    a, _ = toolkit(vectors())
    b, _ = toolkit(vectors())
    a.execute_vla(instruction='Pull rack.')
    assert b.action_history_snapshot()['records'] == []


def test_large_history_remains_exact_and_readable_in_pages():
    from rpent.runtime.plan_generation import (
        _action_history_page,
        _recent_action_history,
    )

    tools, _ = toolkit(vectors(5))
    for _ in range(60):
        tools.execute_vla(instruction='Pull the rack.')
    history = tools.action_history_snapshot()
    recent = _recent_action_history(history)
    assert recent['total_executed_actions'] == 300
    # 1.0.11: the window is also bounded by characters (gateway prompt limit),
    # keeping whole most-recent records.
    import json

    from rpent.runtime.plan_generation import _HISTORY_WINDOW_CHARS
    assert 0 < sum(len(r['actions']) for r in recent['records']) <= 250
    assert len(json.dumps(recent['records'], separators=(',', ':'))) <= _HISTORY_WINDOW_CHARS
    assert recent['records'] == history['records'][-len(recent['records']):]
    seen = []
    cursor = 0
    while cursor is not None:
        page = _action_history_page(history, cursor)['action_history']
        seen.extend(page['records'])
        cursor = page['next_chunk']
    assert seen == history['records']
    seen[0]['actions'][0][0] = 999
    assert tools.action_history_snapshot()['records'][0]['actions'][0][0] == -1


def test_gateway_wire_contains_the_actual_native_command_rows():
    import asyncio

    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai import Agent
    from pydantic_ai.providers.openai import OpenAIProvider

    from rpent.runtime.api import _build_tools
    from rpent.runtime.codexgateway import _CodexGatewayChatModel

    tools, simulator = toolkit(vectors())
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'native-command-call', 'type': 'function', 'function': {
                    'name': 'continue_vla',
                    'arguments': json.dumps({'instruction': 'Pull rack.', 'max_chunks': 2}),
                },
            }]}
        else:
            message = {'role': 'assistant', 'content': 'done'}
        return httpx.Response(200, json={
            'id': 'chat-test', 'object': 'chat.completion', 'created': 1,
            'model': 'gpt-6.1-sol', 'choices': [{
                'index': 0, 'message': message,
                'finish_reason': 'tool_calls' if len(requests) == 1 else 'stop',
            }],
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            client = AsyncOpenAI(api_key='test-only', base_url='http://gateway.invalid/v1', http_client=http)
            model = _CodexGatewayChatModel('gpt-6.1-sol', provider=OpenAIProvider(openai_client=client))
            agent = Agent(model, instructions='Use the VLA.', tools=_build_tools(tools))
            await agent.run('Pull rack out.')

    asyncio.run(run())
    returned = next(m for m in requests[1]['messages'] if m['role'] == 'tool')
    history = json.loads(returned['content'])['action_history']
    np.testing.assert_array_equal(
        np.concatenate([r['actions'] for r in history['records']]), simulator.actions,
    )
