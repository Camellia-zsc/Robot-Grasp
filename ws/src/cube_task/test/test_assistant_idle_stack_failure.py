"""Idle physical motion invalidates completed stacks without commanding recovery."""
import copy
import socket

import pytest

from cube_task.assistant.server import DashboardController
from test_assistant_skill_orchestrator import Session, Settings
from test_assistant_skill_protocol import context


GOALS = [dict(label='apple', zone_name=None,
              destination=dict(type='on_object', label='red'))]


class NoModelCalls:
    def plan_skills(self, *args, **kwargs):
        raise AssertionError('idle physics must not call a model')


@pytest.fixture
def completed_stack(context, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('idle invalidation tests must not use the network')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    session = Session(context['scene'])
    ctl = DashboardController(session_factory=lambda **kwargs: session,
                              client=NoModelCalls(), settings=Settings())
    assert ctl.wait_idle(2)
    with ctl._lock:
        ctl._skill_memory = dict(instruction='把苹果放在红色方块上',
            goals=copy.deepcopy(GOALS), protected_goals=[], originals=[],
            obligations={}, restorations=[])
        ctl._last_result = dict(ok=True, camera_verified=True, verified_revision=0,
            truth={'success': True}, goals=copy.deepcopy(GOALS), skill_task=True,
            items=[dict(goal=copy.deepcopy(GOALS[0]), ok=True,
                        truth={'success': True}, verified_revision=0)])
        ctl._skill_task.update(status='complete', goals=copy.deepcopy(GOALS))
        ctl._status = 'complete'
    yield ctl, session
    ctl._busy = False
    ctl.close()


@pytest.mark.parametrize('label', ['apple', 'red'])
def test_idle_upper_or_support_motion_invalidates_success_and_allows_manual_resume(
        completed_stack, monkeypatch, label):
    ctl, session = completed_stack
    memory = copy.deepcopy(ctl._skill_memory)
    session._environment_revision += 1
    ctl._on_sim_event('physics_scene_changed', dict(labels=[label], message='物品发生物理位移'))
    state = ctl.snapshot()
    assert state['status'] == 'error'
    assert state['skill_task']['status'] == 'failed'
    assert state['last_result']['ok'] is False
    assert state['last_result']['camera_verified'] is False
    assert state['last_result']['truth']['success'] is False
    assert state['last_result']['items'][0]['ok'] is False
    assert state['last_result']['error'] == 'stack_scene_changed'
    assert state['can_resume_recovery']
    assert ctl._skill_memory == memory
    assert session.calls == []

    resumed = []
    def run(_self, instruction, **kwargs):
        resumed.append((instruction, kwargs))
    monkeypatch.setattr('cube_task.assistant.skill_orchestrator.SkillOrchestrator.run', run)
    ctl.resume_recovery()
    assert ctl.wait_idle(2)
    assert len(resumed) == 1
    assert resumed[0][1]['goals'] == GOALS
    assert resumed[0][1]['resume'] is True
    assert resumed[0][1]['failure']['error'] == 'stack_scene_changed'
    assert not session.calls


@pytest.mark.parametrize('case', ['unrelated', 'busy', 'table_goal', 'already_failed'])
def test_idle_invalidation_does_not_replace_other_task_feedback(completed_stack, case):
    ctl, session = completed_stack
    labels = ['green'] if case == 'unrelated' else ['red']
    if case == 'busy':
        ctl._busy = True
    elif case == 'table_goal':
        ctl._skill_memory['goals'] = [dict(label='apple', zone_name='zone_red')]
    elif case == 'already_failed':
        ctl._last_result['ok'] = False
        ctl._last_result['error'] = 'existing_failure'
        ctl._skill_task['status'] = 'failed'
    before = copy.deepcopy(ctl._last_result)
    ctl._on_sim_event('physics_scene_changed', dict(labels=labels, message='物品发生物理位移'))
    assert ctl._last_result == before
    assert not session.calls


def test_editor_cleared_result_does_not_hide_a_subsequent_physical_fall(completed_stack):
    ctl, session = completed_stack
    ctl._last_result = None
    ctl._status = 'ready'
    ctl._on_sim_event('physics_scene_changed', dict(labels=['apple']))
    state = ctl.snapshot()
    assert state['skill_task']['status'] == 'failed'
    assert state['last_result']['ok'] is False
    assert state['last_result']['goals'] == GOALS
    assert state['can_resume_recovery']
    assert session.calls == []
