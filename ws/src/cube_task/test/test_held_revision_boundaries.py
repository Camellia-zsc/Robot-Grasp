"""A changed held scene stops before more transport or camera route planning."""
import numpy as np
import pytest

from test_task_skills import _setup


@pytest.mark.parametrize('changed_during,expected_motions', [('lift', 3), ('transit', 4)])
def test_changed_held_scene_stops_before_next_motion_or_release_observation(
        changed_during, expected_motions):
    _cfg, _clock, world, arm, _log, skill, grasp, place = _setup()
    changed = []
    inspected = []
    skill.on_stage = lambda stage: changed.append(stage) if stage == changed_during else None
    skill.execution_guard = lambda _stage: 'scene_changed' if changed else None
    skill.before_release = lambda *args: inspected.append(args) or place

    result = skill.execute(grasp, place)

    assert not result.ok and result.error == 'scene_changed'
    assert result.stage == 'held_transit'
    assert result.detail['holding'] and not result.detail['fault']
    assert world.held is not None and not world.dropped and not world.placed
    assert not inspected
    assert len(arm.motions()) == expected_motions
    assert arm.gripper_cmds() == [grasp.open_width, grasp.close_width]


@pytest.mark.parametrize('changed_before_planning', [True, False])
def test_changed_held_continuation_checks_scene_before_and_after_route_planning(
        changed_before_planning):
    _cfg, _clock, world, arm, _log, skill, grasp, place = _setup()
    skill.execution_guard = lambda stage: 'scene_changed' if stage == 'release' else None
    initial = skill.execute(grasp, place)
    assert initial.detail['holding'] and initial.error == 'scene_changed'
    previous_motions = len(arm.motions())
    previous_gripper = arm.gripper_cmds()
    changed = [changed_before_planning]
    planned = []

    def plan(*_args):
        planned.append(True)
        changed[0] = True
        return [np.array(place.T_pre, copy=True)]

    skill.execution_guard = lambda _stage: 'scene_changed' if changed[0] else None
    skill.before_held_transit = plan
    result = skill.place_held(grasp, place, initial.grasp_width)

    assert result.stage == 'held_transit' and result.error == 'scene_changed'
    assert result.detail['holding'] and not result.detail['fault']
    assert bool(planned) is not changed_before_planning
    assert len(arm.motions()) == previous_motions
    assert arm.gripper_cmds() == previous_gripper
    assert world.held is not None and not world.placed and not world.dropped
