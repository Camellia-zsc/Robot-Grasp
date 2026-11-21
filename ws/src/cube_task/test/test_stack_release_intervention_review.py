"""A checked aperture cannot authorize retreat after the observed scene changes."""
from types import SimpleNamespace

import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.core.types import MotionResult

from test_stack_release_clearance import checked_wide_release


@pytest.mark.parametrize('wide', [True, False])
def test_intervention_during_opening_stops_before_exit_and_retains_released_state(wide):
    arm, log, skill, grasp, place, checks = checked_wide_release()
    if not wide:
        del place.release_opening_checked
    changed = [False]
    guarded = []
    original = arm.gripper
    session = SimpleNamespace(_environment_revision=0, _active_revision=0,
                               _apply_interventions=lambda: None)

    def guard(stage):
        guarded.append((stage, changed[0]))
        return SimulationSession._execution_guard(session, stage)

    def open_then_intervene(*args, **kwargs):
        feedback = original(*args, **kwargs)
        changed[0] = True
        session._environment_revision += 1
        return feedback

    skill.execution_guard = guard
    arm.gripper = open_then_intervene
    result = skill.place_held(grasp, place, arm.width)

    assert changed[0] and arm.gripper_cmds() == [place.open_width]
    # The completed opening released the object: an invalidated exit must not
    # resurrect a held-object recovery or close the now empty gripper.
    assert result.detail['holding'] is False
    assert result.ok is False
    assert result.error == 'retreat_scene_changed'
    assert any(after_open for _stage, after_open in guarded)
    assert len(arm.motions()) == 3  # transit, inspection-hover, release only.
    assert not any(step['stage'] in ('release_clearance', 'retreat')
                   for step in log.kinds('skill_step'))


@pytest.mark.parametrize('wide', [True, False])
def test_stop_after_successful_opening_is_not_reported_as_task_success(wide):
    arm, _log, skill, grasp, place, _checks = checked_wide_release()
    if not wide:
        del place.release_opening_checked
    original = arm.gripper
    def open_then_stop(*args, **kwargs):
        feedback = original(*args, **kwargs)
        arm.abort_event.set()
        return feedback
    arm.gripper = open_then_stop
    result = skill.place_held(grasp, place, arm.width)
    assert result.ok is False
    assert result.error == 'retreat_aborted'
    assert result.detail['fault'] is True
    assert result.detail['holding'] is False
    assert len(arm.motions()) == 3


@pytest.mark.parametrize('wide', [True, False])
def test_exit_driver_fault_preserves_fault_and_released_state(wide):
    arm, _log, skill, grasp, place, _checks = checked_wide_release()
    if not wide:
        del place.release_opening_checked
    arm.fail_fn = lambda _kind, _target, _linear, index: (
        MotionResult(ok=False, error='comm') if index == 3 else None)
    result = skill.place_held(grasp, place, arm.width)
    assert result.ok is False
    assert result.error == 'retreat_comm'
    assert result.detail['fault'] is True
    assert result.detail['holding'] is False
    assert len(arm.motions()) == 4
