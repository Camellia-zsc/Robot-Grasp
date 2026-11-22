"""A held continuation follows its checked entry route after camera inspection."""
import numpy as np
import pytest

from cube_task.core.transforms import rot_y
from test_task_skills import _setup


def held_scene():
    cfg, clock, world, arm, log, skill, grasp, place = _setup()
    picked = skill.pick_only(grasp)
    assert picked.ok and picked.detail['holding']
    # A reachable camera inspection can end below transit with another tilt;
    # blindly raising this pose vertically need not be reachable.
    arm.tcp = grasp.T_lift.copy()
    arm.tcp[2, 3] = place.T_pre[2, 3] - .04
    arm.tcp[:3, :3] = rot_y(.3) @ arm.tcp[:3, :3]
    arm.commands.clear()
    arm.n_motion = 0
    waypoint = arm.tcp.copy()
    waypoint[0, 3] -= .06
    return clock, world, arm, skill, grasp, place, waypoint, picked.grasp_width


def test_checked_entry_route_replaces_unplanned_vertical_lift_and_p2p():
    _clock, world, arm, skill, grasp, place, waypoint, width = held_scene()
    skill.before_held_transit = lambda *args: [waypoint, place.T_pre]
    result = skill.place_held(grasp, place, width)
    assert result.ok, result
    moves = arm.motions()
    np.testing.assert_allclose(moves[0][1], waypoint)
    np.testing.assert_allclose(moves[1][1], place.T_pre)
    assert moves[0][2] is True and moves[1][2] is True
    assert world.placed == {'zone_red': ['red']}


@pytest.mark.parametrize('route', [None, [], 'invalid', [np.eye(3)], [np.eye(4)]])
def test_unavailable_or_incomplete_route_keeps_grip_without_motion(route):
    _clock, world, arm, skill, grasp, place, _waypoint, width = held_scene()
    def check(*args):
        args[-1].detail['held_transit_planning_reason'] = 'no checked path from inspected pose'
        return route
    skill.before_held_transit = check
    result = skill.place_held(grasp, place, width)
    assert not result.ok and result.error == 'placement_changed'
    assert result.detail['holding'] and not result.detail['fault']
    assert not arm.commands and not world.placed


@pytest.mark.parametrize('drift,allowed', [(.0002, True), (.002, False)])
def test_each_route_segment_requires_opening_inside_checked_finger_sweep(drift, allowed):
    _clock, world, arm, skill, grasp, place, waypoint, width = held_scene()
    skill.before_held_transit = lambda *args: [waypoint, place.T_pre]
    arm.width_fn = lambda current: width + drift if current.n_motion >= 1 else width
    result = skill.place_held(grasp, place, width)
    if allowed:
        assert result.ok, result
        assert world.placed
    else:
        assert not result.ok and result.error == 'placement_changed'
        assert result.detail['holding'] and not result.detail['fault']
        assert len(arm.motions()) == 1 and not arm.gripper_cmds()


def test_scene_change_between_checked_segments_cannot_continue_or_open():
    _clock, _world, arm, skill, grasp, place, waypoint, width = held_scene()
    skill.before_held_transit = lambda *args: [waypoint, place.T_pre]
    skill.execution_guard = lambda stage: 'scene_changed' if arm.n_motion >= 1 else None
    result = skill.place_held(grasp, place, width)
    assert not result.ok and result.error == 'scene_changed'
    assert result.detail['holding'] and not result.detail['fault']
    assert len(arm.motions()) == 1 and not arm.gripper_cmds()


def test_stop_during_entry_planning_cannot_dispatch_a_route():
    _clock, _world, arm, skill, grasp, place, waypoint, width = held_scene()
    def check(*args):
        arm.abort_event.set()
        return [waypoint, place.T_pre]
    skill.before_held_transit = check
    result = skill.place_held(grasp, place, width)
    assert not result.ok and result.error == 'aborted'
    assert result.detail['holding'] and not arm.commands


@pytest.mark.parametrize('width_after', [0., None])
def test_final_route_segment_requires_feedback_even_with_legacy_verification_disabled(width_after):
    _clock, world, arm, skill, grasp, place, waypoint, width = held_scene()
    skill.verify = False
    skill.before_held_transit = lambda *args: [waypoint, place.T_pre]
    arm.width_fn = lambda current: width_after if current.n_motion >= 2 else width
    result = skill.place_held(grasp, place, width)
    assert not result.ok
    assert len(arm.motions()) == 2 and not arm.gripper_cmds() and not world.placed
    assert result.error == ('no_gripper_feedback' if width_after is None else 'dropped_in_transit')


@pytest.mark.parametrize('width_after', [0., None])
def test_real_lost_or_missing_feedback_stops_route_before_any_release(width_after):
    _clock, _world, arm, skill, grasp, place, waypoint, width = held_scene()
    skill.before_held_transit = lambda *args: [waypoint, place.T_pre]
    arm.width_fn = lambda current: width_after if current.n_motion >= 1 else width
    result = skill.place_held(grasp, place, width)
    assert not result.ok and len(arm.motions()) == 1 and not arm.gripper_cmds()
    if width_after is None:
        assert result.detail['holding'] and result.detail['fault']
        assert result.error == 'no_gripper_feedback'
    else:
        assert not result.detail['holding'] and not result.detail['fault']
        assert result.error == 'dropped_in_transit'


@pytest.mark.parametrize('width_after', [0., None])
def test_rejected_route_still_reports_loss_or_missing_feedback_during_planning(width_after):
    _clock, _world, arm, skill, grasp, place, _waypoint, width = held_scene()
    def check(*args):
        arm.width_fn = lambda current: width_after
        return None
    skill.before_held_transit = check
    result = skill.place_held(grasp, place, width)
    assert not result.ok and not arm.commands
    if width_after is None:
        assert result.detail['holding'] and result.detail['fault']
        assert result.error == 'no_gripper_feedback'
    else:
        assert not result.detail['holding'] and not result.detail['fault']
        assert result.error == 'dropped_in_transit'


@pytest.mark.parametrize('stale', [False, True])
def test_checked_joint_transfer_executes_exact_solution_and_refuses_changed_start(stale):
    from cube_task.assistant.placement_observation import JointTransitRoute
    _clock, world, arm, skill, grasp, place, _waypoint, width = held_scene()
    start, end = np.zeros(6), np.ones(6)*.1
    route = JointTransitRoute(place.T_pre, start+(.01 if stale else 0.), end)
    commands = []
    def move_joints(q):
        commands.append(np.array(q))
        return arm.move_tcp(place.T_pre, linear=False)
    arm.move_joints = move_joints
    skill.before_held_transit = lambda *args: route
    result = skill.place_held(grasp, place, width)
    if stale:
        assert not result.ok and result.detail['holding'] and not commands
        assert not arm.motions()
    else:
        assert result.ok and len(commands) == 1
        np.testing.assert_array_equal(commands[0], end)
        assert arm.motions()[0][2] is False


@pytest.mark.parametrize('available', [True, False])
def test_return_from_camera_inspection_cannot_bypass_checked_held_transit(available):
    from cube_task.core.types import PickResult
    from cube_task.task.skills import _StepFailed
    _clock, _world, arm, skill, grasp, place, waypoint, width = held_scene()
    skill.before_release = lambda *args: place
    checked = []
    def entry(*args):
        checked.append(True)
        return [waypoint, place.T_pre] if available else None
    skill.before_held_transit = entry
    result = PickResult(False, grasp_width=width, detail={'holding': True, 'fault': False, 'steps': []})
    if available:
        skill._inspect_release(result, grasp, place)
        np.testing.assert_allclose(arm.motions()[0][1], waypoint)
        assert arm.motions()[0][2] is True
    else:
        with pytest.raises(_StepFailed, match='placement_changed'):
            skill._inspect_release(result, grasp, place)
        assert not arm.motions()
    assert checked == [True]


@pytest.mark.parametrize('settles', [True, False])
def test_measured_anchor_shift_gets_one_checked_local_replan_before_opening(settles):
    from cube_task.core.types import PickResult
    _clock, _world, arm, skill, grasp, place, _waypoint, width = held_scene()
    arm.tcp = place.T_release.copy()
    checks, replans = [], []
    def before_open(g, p, result):
        checks.append(True)
        if len(checks) == 1 or not settles:
            result.detail['held_anchor_replan_required'] = True
            return False, 'newly measured grip translation'
        return True, ''
    def replan(*args):
        replans.append(True)
        return place
    skill.before_open, skill.before_release = before_open, replan
    skill.before_held_transit = lambda *args: [place.T_pre]
    result = PickResult(False, grasp_width=width,
                        detail={'holding': True, 'fault': False, 'steps': []})
    skill._release_and_retreat(result, grasp, place)
    assert len(replans) == 1 and len(checks) == 2
    np.testing.assert_allclose(arm.motions()[0][1], place.T_pre)
    np.testing.assert_allclose(arm.motions()[1][1], place.T_release)
    if settles:
        assert arm.gripper_cmds() and not result.detail['holding']
    else:
        assert result.error == 'held_anchor_unconfirmed' and result.detail['holding']
        assert not arm.gripper_cmds()
