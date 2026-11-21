"""Unmeasured final moves are pending work, never permission to grasp blind."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cube_task.assistant.skill_execution import SkillExecutor, _PlanChanged, _TargetUnavailable
from cube_task.core.types import Detection
import numpy as np


def executor(*, valid=False, held=False, healthy=True):
    scene = dict(scene_version=2, objects=[dict(label='pear', valid=valid,
        center_base=[.3, -.1, .15], top_z=.15, height=.1, length=.08, width=.08, yaw=0.,
        reject_reason='' if valid else 'touches_image_border')], zones=[], class_to_zone={'pear': 'zone_green'})
    session = SimpleNamespace(scene=scene, cancel_event=threading.Event(),
        _operation_lock=threading.RLock(), _check_open=lambda: None,
        _environment_revision=0, _interventions=[],
        runner=SimpleNamespace(has_held_recovery=held),
        arm=SimpleNamespace(health=lambda: SimpleNamespace(ok=healthy)),
        snapshot=lambda: dict(held_object='pear' if held else None, editor={}))
    ex = SkillExecutor(session)
    ex._move = Mock(return_value=dict(ok=True))
    goals = [dict(label='pear', zone_name='zone_green')]
    step = dict(id='pear-final', skill='move', label='pear', destination=dict(type='goal', index=0))
    return ex, goals, step


def test_unreliable_final_move_is_deferred_without_scanning_or_manipulation():
    ex, goals, step = executor()
    ex.session.observe = Mock(side_effect=AssertionError('deferral must not scan'))
    result = ex.execute(step, goals)
    assert result['ok'] is False and result['deferred'] is True
    assert result['manipulation_started'] is False and result['changed'] is False
    assert not result['holding'] and result['error'] == 'target_not_observed'
    ex._move.assert_not_called()
    ex.session.observe.assert_not_called()
    assert goals == [dict(label='pear', zone_name='zone_green')]


@pytest.mark.parametrize('destination', [dict(type='auto_free'), dict(type='original', label='pear'),
                                       dict(type='zone', name='zone_red')])
def test_temporary_or_dependent_rearrangement_never_soft_skips(destination):
    ex, goals, step = executor()
    step['destination'] = destination
    assert ex.defer_unobserved(step, goals) is None


@pytest.mark.parametrize('skill', ['pick', 'place'])
def test_pick_place_dependencies_are_never_soft_skipped(skill):
    ex, goals, step = executor()
    step['skill'] = skill
    assert ex.defer_unobserved(step, goals) is None


@pytest.mark.parametrize('held,healthy', [(True, True), (False, False)])
def test_hold_and_controller_fault_do_not_become_benign_deferral(held, healthy):
    ex, goals, step = executor(held=held, healthy=healthy)
    assert ex.defer_unobserved(step, goals) is None


def test_cancelled_or_changed_scene_keeps_existing_stop_semantics():
    ex, goals, step = executor()
    ex.session.cancel_event.set()
    result = ex.execute(step, goals)
    assert result['error'] == 'aborted' and not result.get('deferred')
    ex.session.cancel_event.clear()
    result = ex.execute(step, goals, expected_revision=1)
    assert result['error'] == 'scene_changed' and not result.get('deferred')


def test_valid_target_still_uses_fresh_execution_checks():
    ex, goals, step = executor(valid=True)
    assert ex.execute(step, goals)['ok']
    ex._move.assert_called_once()


def test_fresh_target_disappearing_before_grasp_becomes_pending():
    ex, goals, step = executor(valid=True)
    ex._move.side_effect = _TargetUnavailable('fresh target missing')
    result = ex.execute(step, goals)
    assert result['deferred'] and not result['manipulation_started']


@pytest.mark.parametrize('exception,error', [(_PlanChanged('changed'), 'scene_changed'),
                                           (ValueError('real planning refusal'), 'invalid_step')])
def test_other_execution_failures_remain_failures(exception, error):
    ex, goals, step = executor(valid=True)
    ex._move.side_effect = exception
    result = ex.execute(step, goals)
    assert result['error'] == error and not result.get('deferred')


def test_partial_final_check_keeps_each_camera_verdict_before_truth_combination():
    ex, goals, _step = executor(valid=True)
    goals.append(dict(label='banana', zone_name='zone_yellow'))
    ex.session._observed_detections = [Detection('pear', 1., center_base=np.array([.3, .1, .1]))]
    ex.session._evaluation_object = lambda measured: measured
    ex.session._verify_zone = lambda obj, zone: True
    ex.session.verify_goals = lambda selected: dict(ok=False, items=[
        dict(goal=goals[0], ok=True), dict(goal=goals[1], ok=False)], verified_revision=0)
    result = ex.execute(dict(id='final', skill='verify'), goals)
    assert not result['ok'] and not result['camera_verified']
    assert result['items'][0]['ok'] and result['items'][0]['camera_verified'] is True
    assert result['items'][1]['camera_verified'] is False
