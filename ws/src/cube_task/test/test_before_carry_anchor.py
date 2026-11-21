"""The first carry must honor the measured held-reference confirmation hook."""
import numpy as np
import pytest

from cube_task.task.skills import _StepFailed
from test_task_skills import _setup


def test_before_carry_refusal_holds_after_lift_before_first_transit():
    _cfg, _clock, _world, arm, _log, skill, grasp, place = _setup()
    calls = []
    def unconfirmed(g, p, result):
        calls.append((len(arm.motions()), result.grasp_width))
        return False, 'held anchor has no current depth evidence', None
    skill.before_carry = unconfirmed
    result = skill.execute(grasp, place)
    assert calls == [(3, .03)]
    assert not result.ok and result.detail['holding'] and not result.detail['fault']
    assert result.error == 'placement_changed' and result.stage == 'held_transit'
    assert len(arm.motions()) == 3 and len(arm.gripper_cmds()) == 2


@pytest.mark.parametrize('failure', ['stop', 'scene', 'fault'])
def test_before_carry_preserves_stop_scene_and_driver_fault(failure):
    _cfg, _clock, _world, arm, _log, skill, grasp, place = _setup()
    changed = []
    skill.execution_guard = lambda stage: 'scene_changed' if changed else None
    def interrupted(g, p, result):
        if failure == 'stop':
            arm.abort_event.set()
        elif failure == 'scene':
            changed.append(True)
        else:
            raise _StepFailed('held_transit', 'comm', 'lost feedback', fault=True)
        return True, '', None
    skill.before_carry = interrupted
    result = skill.execute(grasp, place)
    assert not result.ok and result.detail['holding']
    assert len(arm.motions()) == 3 and len(arm.gripper_cmds()) == 2
    assert result.error == {'stop': 'aborted', 'scene': 'scene_changed', 'fault': 'comm'}[failure]
    assert result.detail['fault'] is (failure != 'scene')


def test_confirmed_before_carry_is_optional_and_runs_once_before_transit():
    _cfg, _clock, _world, arm, _log, skill, grasp, place = _setup()
    calls = []
    skill.before_carry = lambda g, p, r: (calls.append(len(arm.motions())) or True, '', None)
    result = skill.execute(grasp, place)
    assert result.ok and calls == [3]


def test_confirmed_first_carry_executes_checked_route_once_and_never_replans_it():
    _cfg, _clock, _world, arm, _log, skill, grasp, place = _setup()
    middle = place.T_pre.copy()
    middle[1, 3] += .01
    route = [middle, place.T_pre.copy()]
    skill.before_carry = lambda g, p, r: (True, '', route)
    skill.before_held_transit = lambda *args: pytest.fail('a supplied route must not be replanned')
    result = skill.execute(grasp, place)
    assert result.ok
    motions = arm.motions()
    assert len(motions) == 7
    np.testing.assert_array_equal(motions[3][1], middle)
    np.testing.assert_array_equal(motions[4][1], place.T_pre)
    assert motions[3][2] is True and motions[4][2] is True
    # The following command is the release, with no repeated P2P to T_pre.
    np.testing.assert_array_equal(motions[5][1], place.T_release)


@pytest.mark.parametrize('failure', ['width', 'endpoint', 'empty', 'uncertainty'])
def test_preplanned_first_carry_keeps_the_existing_route_feedback_checks(failure):
    _cfg, _clock, _world, arm, _log, skill, grasp, place = _setup()
    def checked(g, p, result):
        route = [place.T_pre.copy()]
        if failure == 'width':
            arm.width += .005
        elif failure == 'endpoint':
            route[-1][0, 3] += .01
        elif failure == 'empty':
            route = []
        else:
            result.detail['held_transit_opening_uncertainty'] = .002
        return True, '', route
    skill.before_carry = checked
    result = skill.execute(grasp, place)
    assert not result.ok and result.detail['holding'] and not result.detail['fault']
    assert result.error == 'placement_changed'
    assert len(arm.motions()) == 3 and len(arm.gripper_cmds()) == 2
