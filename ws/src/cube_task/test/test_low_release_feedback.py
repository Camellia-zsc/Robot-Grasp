"""Opening must follow actual near-surface, stationary feedback, including failures."""
import numpy as np
import pytest

from cube_task.core.transforms import rot_y
from cube_task.core.types import MotionResult, Stamped
from test_task_skills import _setup


@pytest.mark.parametrize('height', [.020, .079])
def test_stalled_descent_never_opens_high_above_the_destination(height):
    _cfg, _clock, world, arm, _log, skill, grasp, place = _setup()

    def stopped(kind, target, linear, index):
        if index == 4:
            arm.tcp = place.T_release.copy()
            arm.tcp[2, 3] += height
            return MotionResult(ok=False, error='stalled')
        return None

    arm.fail_fn = stopped
    result = skill.execute(grasp, place)
    assert not result.ok
    assert not world.placed


@pytest.mark.parametrize('feedback', ['high', 'lateral', 'rotated', 'stale', 'invalid', 'moving'])
def test_successful_motion_cannot_open_without_current_stationary_release_pose(feedback):
    _cfg, clock, world, arm, _log, skill, grasp, place = _setup()
    real_pose = arm.get_tcp_pose
    reads = [0]

    def pose():
        sample = real_pose()
        if arm.n_motion != 5:
            return sample
        reads[0] += 1
        value = sample.value.copy()
        if feedback == 'high':
            value[2, 3] += .02
        elif feedback == 'lateral':
            value[0, 3] += .015
        elif feedback == 'rotated':
            value[:3, :3] = rot_y(np.deg2rad(15)) @ value[:3, :3]
        elif feedback == 'stale':
            return Stamped(value, clock.now() - 10.)
        elif feedback == 'invalid':
            value[2, 3] = np.nan
        elif feedback == 'moving':
            value[2, 3] += .001 if reads[0] % 2 else -.001
        return Stamped(value, sample.stamp)

    arm.get_tcp_pose = pose
    result = skill.execute(grasp, place)
    assert not result.ok and result.detail['holding'], result
    assert len(arm.gripper_cmds()) == 2
    assert not world.placed


def test_near_surface_stalled_descent_can_release_after_stationary_feedback():
    _cfg, _clock, world, arm, _log, skill, grasp, place = _setup()

    def stopped(kind, target, linear, index):
        if index == 4:
            arm.tcp = place.T_release.copy()
            arm.tcp[2, 3] += .001
            return MotionResult(ok=False, error='stalled')
        return None

    arm.fail_fn = stopped
    result = skill.execute(grasp, place)
    assert result.ok and world.placed == {'zone_red': ['red']}, result


def test_stop_during_release_settle_does_not_open():
    _cfg, clock, world, arm, _log, skill, grasp, place = _setup()
    real_sleep = clock.sleep

    def sleep(duration):
        if arm.n_motion == 5 and duration == pytest.approx(.12):
            arm.abort_event.set()
        real_sleep(duration)

    clock.sleep = sleep
    result = skill.execute(grasp, place)
    assert not result.ok and result.detail['holding'], result
    assert len(arm.gripper_cmds()) == 2
    assert not world.placed


@pytest.mark.parametrize('label', ['banana', 'pear', 'apple', 'red'])
@pytest.mark.parametrize('stack', [False, True])
def test_interactive_clearance_leaves_depth_error_margin_for_every_label_and_support(label, stack):
    from cube_task.assistant.catalog_perception import configure_sandbox
    from cube_task.assistant.stacking import planning_options
    from cube_task.perception.grasp import PlacePlanner
    from cube_task.core.types import Zone
    from test_assistant_stacking_geometry import measured_session, DESTINATION

    cfg, *_ = _setup()
    configure_sandbox(cfg)
    zone = (planning_options(measured_session(), DESTINATION)[0]['placement_zone'] if stack
            else Zone('table', np.array([.3, 0., 0.]), (.2, .2)))
    offset = .03
    plan = PlacePlanner(cfg).plan(zone, offset, .2,
        lower=(cfg.get('place.release_lower', {}) or {}).get(label, 0.))
    assert plan.feasible
    assert plan.T_release[2, 3] - offset - zone.center[2] == pytest.approx(.004)
