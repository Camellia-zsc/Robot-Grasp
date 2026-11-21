"""Stacked rigid bodies receive a guarded stationary interval after opening."""
from types import SimpleNamespace

import pytest

from cube_task.assistant.simulation import SimulationSession
from test_stack_release_clearance import checked_wide_release


def timed_release(*, wide=True, stacked=True):
    arm, log, skill, grasp, place, checks = checked_wide_release()
    if not wide:
        del place.release_opening_checked
    if not stacked:
        grasp.target.extra.pop('_stack_support')
    opened, exits, held_during_wait = [], [], []
    results = []
    skill.before_open = lambda g, p, result: (results.append(result) or True, '')
    original_open = arm.gripper
    def open_and_timestamp(*args, **kwargs):
        feedback = original_open(*args, **kwargs)
        opened.append(arm.clock.now())
        return feedback
    arm.gripper = open_and_timestamp
    original_move = skill._move
    def move(result, stage, pose, linear):
        if stage in ('release_clearance', 'retreat'):
            exits.append((stage, arm.clock.now()))
        return original_move(result, stage, pose, linear)
    skill._move = move
    original_sleep = arm.clock.sleep
    def sleep(seconds):
        if abs(seconds-1.2) < 1e-9:
            held_during_wait.append(results[-1].detail['holding'])
        original_sleep(seconds)
    arm.clock.sleep = sleep
    return arm, skill, grasp, place, opened, exits, held_during_wait


@pytest.mark.parametrize('wide', [True, False])
def test_stack_is_released_then_waits_before_any_exit_motion(wide):
    arm, skill, grasp, place, opened, exits, held = timed_release(wide=wide)
    result = skill.place_held(grasp, place, arm.width)
    assert result.ok and not result.detail['holding']
    assert held == [False]
    assert exits[0][1]-opened[-1] >= 1.2-1e-9
    assert result.detail['release_settle_s'] == pytest.approx(1.2)


def test_nonstack_release_keeps_its_existing_exit_timing():
    arm, skill, grasp, place, opened, exits, held = timed_release(stacked=False)
    result = skill.place_held(grasp, place, arm.width)
    assert result.ok and not held
    assert exits[0][1]-opened[-1] == pytest.approx(0.)


@pytest.mark.parametrize('intervention', ['stop', 'scene'])
def test_change_during_opened_stack_wait_never_moves_the_exit(intervention):
    arm, skill, grasp, place, opened, exits, held = timed_release()
    session = SimpleNamespace(_environment_revision=0, _active_revision=0,
                               _apply_interventions=lambda: None)
    skill.execution_guard = lambda stage: SimulationSession._execution_guard(session, stage)
    original_sleep = arm.clock.sleep
    def sleep(seconds):
        original_sleep(seconds)
        if abs(seconds-1.2) < 1e-9:
            if intervention == 'stop':
                arm.abort_event.set()
            else:
                session._environment_revision += 1
    arm.clock.sleep = sleep
    result = skill.place_held(grasp, place, arm.width)
    assert opened and held == [False]
    assert not result.ok and not result.detail['holding'] and not exits
    assert result.error == ('retreat_aborted' if intervention == 'stop' else 'retreat_scene_changed')
    assert result.detail['fault'] is (intervention == 'stop')
