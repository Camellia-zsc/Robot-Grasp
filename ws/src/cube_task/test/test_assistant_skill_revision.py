"""An approved semantic plan expires even if a new image can track its target."""
import threading
from types import SimpleNamespace

import pytest

from cube_task.assistant.skill_execution import SkillExecutor
from test_task_fakes import make_det


def session():
    return SimpleNamespace(
        _operation_lock=threading.RLock(), _check_open=lambda: None,
        cancel_event=SimpleNamespace(is_set=lambda: False, external=None),
        _environment_revision=0, _interventions=[], snapshot=lambda: {},
        runner=SimpleNamespace(has_held_recovery=False, class_to_zone={'red': 'zone_red'},
                               skill=SimpleNamespace(execution_guard=lambda stage: None)))


@pytest.mark.parametrize('pending', [False, True])
def test_stale_or_queued_scene_is_refused_before_any_skill_dispatch(pending):
    s = session()
    if pending:
        s._interventions.append(('move', {}))
    else:
        s._environment_revision = 1
    executor = SkillExecutor(s)
    def forbidden(*args):
        pytest.fail('expired skill was dispatched')
    executor._pick = forbidden
    result = executor.execute({'id': 'p', 'skill': 'pick', 'label': 'red'}, [], expected_revision=0)
    assert not result['ok'] and result['error'] == 'scene_changed'


@pytest.mark.parametrize('skill', ['pick', 'move'])
def test_scene_change_during_new_camera_measurement_cancels_grasp(skill):
    s = session()
    executor = SkillExecutor(s)
    if skill == 'move':
        executor._resolve = lambda value, goals: ({'type': 'named_region', 'name': 'zone_red'}, None)
    def observe(*args):
        s._environment_revision += 1
        return make_det('red', .3, 0.), {'object_id': 'red'}
    executor._fresh_target = observe
    result = executor.execute({'id': 'p', 'skill': skill, 'label': 'red',
                               'destination': {'type': 'zone', 'name': 'zone_red'}}, [], expected_revision=0)
    assert not result['ok'] and result['error'] == 'scene_changed'


def test_scene_guard_stays_bound_to_approved_revision_and_is_restored():
    s = session()
    original = s.runner.skill.execution_guard
    executor = SkillExecutor(s)
    def during_move(*args):
        s._environment_revision += 1
        error = s.runner.skill.execution_guard('close')
        return {'ok': False, 'error': error}
    executor._move = during_move
    result = executor.execute({'id': 'm', 'skill': 'move', 'label': 'red'}, [], expected_revision=0)
    assert not result['ok'] and result['error'] == 'scene_changed'
    assert s.runner.skill.execution_guard is original
