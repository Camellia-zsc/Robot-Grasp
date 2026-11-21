"""A wide-opening proof also requires its actual-start geometry guard."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.core.transforms import rot_x
from cube_task.core.types import MotionResult
from cube_task.task.skills import _StepFailed
from test_runner_stack_opening import replan, stacked_scene
from test_stack_release_clearance import checked_wide_release


@pytest.mark.parametrize('checker_kind', ['missing', 'refused'])
def test_missing_or_refused_actual_exit_check_never_falls_back_to_opening(checker_kind):
    arm, log, skill, grasp, place, preflights = checked_wide_release()
    calls = []
    if checker_kind == 'missing':
        del skill.current_release_exit_check
    else:
        def refused(g, p, actual, width):
            calls.append((g, p, actual.copy(), width))
            return False
        skill.current_release_exit_check = refused
    result = skill.place_held(grasp, place, arm.width)
    assert not result.ok and result.detail['holding'] and not result.detail['fault']
    assert result.error == 'release_clearance_unconfirmed'
    assert not arm.gripper_cmds()
    assert len(arm.motions()) == 3
    assert len(preflights) == 1  # Never switch to the vertical fallback.
    assert not any(step['stage'] in ('release_clearance', 'retreat')
                   for step in log.kinds('skill_step'))
    if checker_kind == 'refused':
        assert len(calls) == 1 and calls[0][0] is grasp and calls[0][1] is place
        np.testing.assert_array_equal(calls[0][2], place.T_release)
        assert calls[0][3] == place.open_width


@pytest.mark.parametrize('failure', ['stop', 'scene', 'driver_fault', 'unhealthy'])
def test_actual_exit_checker_preserves_stop_scene_and_driver_fault(failure):
    arm, _log, skill, grasp, place, _preflights = checked_wide_release()
    changed = []
    skill.execution_guard = lambda stage: 'scene_changed' if changed else None

    def interrupted(*args):
        if failure == 'stop':
            arm.abort_event.set()
        elif failure == 'scene':
            changed.append(True)
        elif failure == 'unhealthy':
            arm.health = lambda: SimpleNamespace(ok=False)
        else:
            raise _StepFailed('release_clearance', 'comm', 'lost feedback', fault=True)
        return True

    skill.current_release_exit_check = interrupted
    result = skill.place_held(grasp, place, arm.width)
    assert not result.ok and result.detail['holding']
    assert result.error == dict(stop='aborted', scene='scene_changed',
                               driver_fault='comm', unhealthy='arm_error')[failure]
    assert result.detail['fault'] is (failure != 'scene')
    assert not arm.gripper_cmds() and len(arm.motions()) == 3


def test_real_runner_checker_accepts_clear_actual_rotated_start_and_completes_release():
    parts = stacked_scene()
    runner, _target, grasp, _previous, _support, _zone, _polygon = parts
    candidate, why = replan(parts, observation_filter=lambda *args, **kwargs: True)
    assert candidate is not None, why
    assert hasattr(candidate, 'release_opening_checked')
    actual = candidate.T_release.copy()
    actual[:3, :3] = rot_x(-math.radians(2.9)) @ actual[:3, :3]
    arm, _log, skill, _old_grasp, _old_place, _preflights = checked_wide_release()
    arm.width = .048
    arm.tcp = candidate.T_pre.copy()
    calls = []

    def actual_checker(g, p, pose, width):
        calls.append(pose.copy())
        return runner.current_release_exit_check(g, p, pose, width)

    def reached_with_admitted_error(_kind, pose, _linear, _index):
        if np.array_equal(pose, candidate.T_release):
            arm.tcp = actual.copy()
            return MotionResult(ok=True, pos_err=0.)
        return None

    skill.current_release_exit_check = actual_checker
    arm.fail_fn = reached_with_admitted_error
    result = skill.place_held(grasp, candidate, arm.width)
    assert result.ok, result.error
    assert not result.detail['holding'] and not result.detail['fault']
    assert result.detail['release_exit_mode'] == 'checked_wide_opening'
    assert arm.gripper_cmds() == [candidate.open_width]
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], actual)
    np.testing.assert_array_equal(arm.motions()[-1][1], candidate.T_retreat)
