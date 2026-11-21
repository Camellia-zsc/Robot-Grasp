"""A safe release endpoint does not establish a clear rotating exit sweep."""
import math

import numpy as np

from cube_task.core.transforms import rot_y, top_down_T
from cube_task.core.types import Detection
from cube_task.robot.safety import interpolate_pose
from test_runner_stack_opening import replan, stacked_scene


def rotating_exit_scene():
    parts = stacked_scene()
    runner = parts[0]
    down = top_down_T(0., 0., 0., 0.)[:3, :3]

    def find_tilt(pose, q_seed=None, max_tilt_deg=None):
        adjusted = pose.copy()
        angle = math.radians(0. if pose[2, 3] < .2 else 20.)
        adjusted[:3, :3] = rot_y(angle) @ down
        return adjusted, angle, ''

    def reachable(pose, **kwargs):
        tilt = math.degrees(math.acos(-float(pose[2, 2])))
        return bool(pose[2, 3] < .2 or tilt >= 19.99), ''

    # Only the arm's reachable wrist attitudes are controlled. Occupancy,
    # release search, body/finger geometry and certificate production are real.
    runner.arm.find_reachable_tilt = find_tilt
    runner.arm.check_reachable = reachable
    return parts


def test_unobstructed_rotating_stack_exit_keeps_full_checked_opening():
    candidate, why = replan(rotating_exit_scene(),
                            observation_filter=lambda *args, **kwargs: True)
    assert candidate is not None, why
    assert hasattr(candidate, 'release_opening_checked')
    assert candidate.open_width >= .095


def test_vertical_exit_beside_same_tall_neighbor_is_still_certified():
    parts = stacked_scene()
    neighbor = Detection('unknown', 1., center_base=np.array([.482, 0., .35]),
        height=.35, top_z=.35, length=.008, width=.008, yaw=0.)
    candidate, why = replan(parts, [neighbor],
                            observation_filter=lambda *args, **kwargs: True)
    assert candidate is not None, why
    assert hasattr(candidate, 'release_opening_checked')
    np.testing.assert_array_equal(candidate.T_retreat[:3, :3], candidate.T_release[:3, :3])


def test_exit_checks_empty_gripper_instead_of_moving_a_phantom_released_body():
    parts = rotating_exit_scene()
    neighbor = Detection('unknown', 1., center_base=np.array([.330, 0., .18]),
        height=.18, top_z=.18, length=.008, width=.008, yaw=0.)
    # The upper body stays at its support after opening. Carrying its original
    # source prism along the now-empty TCP would sweep that phantom body's
    # bottom left edge through x=.3285,z=.174, into this clear-side neighbor.
    # The real opened fingers/palm stay to the right of x=.340 throughout.
    candidate, why = replan(parts, [neighbor],
                            observation_filter=lambda *args, **kwargs: True)
    assert candidate is not None, why
    assert hasattr(candidate, 'release_opening_checked')


def test_release_certificate_cannot_authorize_a_known_exit_box_collision():
    parts = rotating_exit_scene()
    runner, target, grasp, _previous, support, _zone, polygon = parts
    clear, why = replan(parts, observation_filter=lambda *args, **kwargs: True)
    assert clear is not None, why
    neighbor = Detection('unknown', 1., center_base=np.array([.482, 0., .35]),
        height=.35, top_z=.35, length=.008, width=.008, yaw=0.)
    allowed = runner._strict_place_filter(target, grasp, [target, support, neighbor],
                                          polygon, release_opening=clear.open_width)

    def boxes_clear(fraction):
        pose = interpolate_pose(clear.T_release, clear.T_retreat, fraction)
        closing = math.atan2(float(pose[1, 1]), float(pose[0, 1]))
        return allowed(.4, 0., closing, tcp_xy=pose[:2, 3], gripper_pose=pose,
                       gripper_open_width=clear.open_width)

    # This obstacle clears the actual open gripper at release, then intersects
    # the palm as the retreat tilts. Checking release plus IK misses the bug.
    assert boxes_clear(0.)
    assert not boxes_clear(.75)
    assert not boxes_clear(1.)
    candidate, _why = replan(parts, [neighbor],
                             observation_filter=lambda *args, **kwargs: True)
    assert candidate is None or not hasattr(candidate, 'release_opening_checked'), (
        'a measured obstacle intersects the checked wide-opening retreat')


def test_open_tool_sweep_checks_an_obstacle_between_two_clear_endpoints():
    parts = rotating_exit_scene()
    runner, target, grasp, _previous, support, _zone, polygon = parts
    clear, why = replan(parts, observation_filter=lambda *args, **kwargs: True)
    assert clear is not None, why
    # A small measured obstruction above the release fingers and below their
    # final pose intersects the finger band halfway through the upward exit.
    neighbor = Detection('unknown', 1., center_base=np.array([.457, 0., .213]),
        height=.003, top_z=.213, length=.003, width=.003, yaw=0.)
    allowed = runner._strict_place_filter(target, grasp, [target, support, neighbor],
        polygon, release_opening=clear.open_width, gripper_only=True)
    def tool_clear(fraction):
        pose = interpolate_pose(clear.T_release, clear.T_retreat, fraction)
        return allowed(0., 0., 0., gripper_pose=pose,
                       gripper_open_width=clear.open_width)
    assert tool_clear(0.) and tool_clear(1.)
    assert not tool_clear(.5)
    candidate, _why = replan(parts, [neighbor],
                             observation_filter=lambda *args, **kwargs: True)
    assert candidate is None
