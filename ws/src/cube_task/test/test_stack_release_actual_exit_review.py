"""The certified tool sweep must cover release feedback's admitted pose error."""
import math

import numpy as np

from cube_task.core.transforms import rot_x
from cube_task.core.types import Detection, MotionResult
from test_runner_stack_opening import replan, stacked_scene
from test_stack_release_clearance import checked_wide_release


def test_admitted_actual_release_rotation_cannot_open_into_measured_neighbor():
    parts = stacked_scene()
    runner, target, grasp, _previous, support, _zone, polygon = parts
    neighbor = Detection('unknown', 1., center_base=np.array([.444, .0295, .35]),
        height=.35, top_z=.35, length=.002, width=.002, yaw=0.)
    candidate, why = replan(parts, [neighbor],
                            observation_filter=lambda *args, **kwargs: True)
    # A planner that reserves the complete feedback envelope may refuse this
    # narrow gap before motion. Otherwise the actual-pose gate must refuse it.
    if candidate is None:
        assert why
        return
    actual = candidate.T_release.copy()
    actual[:3, :3] = rot_x(-math.radians(2.9)) @ actual[:3, :3]
    allowed = runner._strict_place_filter(target, grasp, [target, support, neighbor],
        polygon, release_opening=candidate.open_width, gripper_only=True)
    assert allowed(0., 0., 0., gripper_pose=candidate.T_release,
                   gripper_open_width=candidate.open_width)
    assert not allowed(0., 0., 0., gripper_pose=actual,
                       gripper_open_width=candidate.open_width)

    arm, _log, skill, _old_grasp, _old_place, _checks = checked_wide_release()
    skill.current_release_exit_check = runner.current_release_exit_check
    arm.width = .048
    arm.tcp = actual.copy()
    feedback, reason = skill._release_feedback(candidate)
    assert feedback is not None and not reason  # 2.9 degrees is admitted.
    arm.tcp = candidate.T_pre.copy()

    def reached_with_admitted_error(_kind, pose, _linear, _index):
        if np.array_equal(pose, candidate.T_release):
            arm.tcp = actual.copy()
            return MotionResult(ok=True, pos_err=0.)
        return None

    arm.fail_fn = reached_with_admitted_error
    result = skill.place_held(grasp, candidate, arm.width)
    assert not arm.gripper_cmds(), 'the actual palm intersects a measured neighbor'
    assert not result.ok and result.detail['holding']


def test_feedback_rotation_tolerance_still_allows_a_clear_real_plan():
    parts = stacked_scene()
    candidate, why = replan(parts, observation_filter=lambda *args, **kwargs: True)
    assert candidate is not None, why
    arm, _log, skill, _old_grasp, _old_place, _checks = checked_wide_release()
    actual = candidate.T_release.copy()
    actual[:3, :3] = rot_x(-math.radians(2.9)) @ actual[:3, :3]
    arm.tcp = actual
    feedback, reason = skill._release_feedback(candidate)
    assert feedback is not None and not reason
    assert hasattr(candidate, 'release_opening_checked')
