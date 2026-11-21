"""The pick-only runner seam checks safety without requiring a release plan."""
import copy

import numpy as np
import pytest

from cube_task.core.types import MotionResult
from test_task_fakes import Rig, make_det


def test_pick_only_does_not_need_capacity_and_keeps_camera_measured_anchor():
    rig = Rig('L1')
    target = rig.world.add(make_det('red', .3, .05))
    before = copy.deepcopy(target)
    runner = rig.runner()
    runner.placer.plan = lambda *args, **kwargs: pytest.fail('pick-only must not plan a release')
    result, why = runner.pick_only_detection(target)
    assert result is not None and result.ok and result.detail['holding'], why
    context = runner._held_recovery_context
    assert context['zone'] == 'held' and context['width'] == pytest.approx(.03)
    assert context['grasp'].target.extra['_closed_grip_width'] == pytest.approx(.03)
    np.testing.assert_array_equal(context['det'].center_base, before.center_base)
    assert len(rig.arm.motions()) == 3 and not runner._placed and not runner._placed_labels
    # Attempting another pick cannot open the held grasp.
    commands = len(rig.arm.commands)
    result, why = runner.pick_only_detection(make_det('green', .4, -.1))
    assert result is None and 'already held' in why
    assert len(rig.arm.commands) == commands


def test_pick_only_retains_all_neighbors_for_grasp_rejection():
    from cube_task.perception.grasp import GraspPlanner
    rig = Rig('L1')
    target = rig.world.add(make_det('red', .3, 0.))
    blocker = make_det('unknown', .33, .03)
    blocker.extra['_points_xy'] = np.array([[.33, 0.], [.3, .03]])
    runner = rig.runner()
    runner.planner = GraspPlanner(rig.cfg, 'Sandbox')
    result, why = runner.pick_only_detection(target, neighbors=[target, blocker])
    assert result is None and 'grasp infeasible' in why
    assert not rig.arm.commands


def test_unreachable_pick_only_pose_sends_no_motion_or_gripper_commands():
    rig = Rig('L1')
    runner = rig.runner()
    rig.arm.check_reachable = lambda *args, **kwargs: (False, 'IK refused')
    result, why = runner.pick_only_detection(make_det('red', .3, .05))
    assert result is None and 'IK refused' in why
    assert not rig.arm.commands and not runner.has_held_recovery


def test_failed_lift_saves_same_grip_for_later_explicit_place():
    rig = Rig('L1')
    target = rig.world.add(make_det('red', .3, .05))
    runner = rig.runner()
    rig.arm.fail_fn = lambda kind, pose, linear, index: (
        MotionResult(False, 'stalled') if index == 2 else None)
    result, why = runner.pick_only_detection(target)
    assert result is not None and not result.ok and result.detail['holding'], why
    assert runner.has_held_recovery
    assert len(rig.arm.gripper_cmds()) == 2 and len(rig.arm.motions()) == 3
