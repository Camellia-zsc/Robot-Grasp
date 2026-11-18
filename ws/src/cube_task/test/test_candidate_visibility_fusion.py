"""Every release sample needs real table evidence in at least one fresh view."""
import copy
import math

import numpy as np
import pytest

from cube_task.assistant.placement_observation import candidate_visibility_filter
from cube_task.core.transforms import make_T, rot_y, top_down_T
from cube_task.core.types import CameraIntrinsics, Detection, Frame, GraspPlan


def scene():
    target = Detection('green', 1., center_base=np.array([0., 0., .05]), top_z=.05,
                       height=.05, length=.05, width=.05, yaw=0.)
    pose = top_down_T(0., 0., .02, 0.)
    grasp = GraspPlan(target, pose.copy(), pose, pose.copy(), .1, 0., .05, 0.)
    intr = CameraIntrinsics(400., 400., 200., 200., 400, 400)
    frame = Frame(np.zeros((400, 400, 3), np.uint8), np.ones((400, 400), np.float32), intr, 1.)
    camera = make_T(np.diag([1., -1., -1.]), [0., 0., 1.])
    return target, grasp, frame, camera


def complementary_frames():
    target, grasp, first, camera = scene()
    second = copy.deepcopy(first)
    first.depth[:, :200] = .6
    second.depth[:, 200:] = .6
    return target, grasp, first, second, camera


def test_two_complementary_occluded_views_prove_the_complete_release():
    target, grasp, first, second, camera = complementary_frames()
    assert not candidate_visibility_filter(target, grasp, first, camera)(0., 0., 0.)
    assert not candidate_visibility_filter(target, grasp, second, camera)(0., 0., 0.)
    fused = candidate_visibility_filter(target, grasp, first, camera,
                                        observations=[(second, camera)])
    assert fused(0., 0., 0.)


def test_an_area_hidden_in_every_view_remains_unknown():
    target, grasp, first, camera = scene()
    first.depth[:, :200] = .6
    second = copy.deepcopy(first)
    assert not candidate_visibility_filter(target, grasp, first, camera,
        observations=[(second, camera)])(0., 0., 0.)


@pytest.mark.parametrize('hole', [0., np.nan, np.inf])
def test_fusion_cannot_fill_a_depth_hole_that_no_view_measured(hole):
    target, grasp, first, second, camera = complementary_frames()
    first.depth[195:206, 195:206] = hole
    second.depth[195:206, 195:206] = hole
    assert not candidate_visibility_filter(target, grasp, first, camera,
        observations=[(second, camera)])(0., 0., 0.)


def test_each_view_uses_its_own_measured_camera_pose():
    target, grasp, first, camera = scene()
    second = copy.deepcopy(first)
    left_camera, right_camera = camera.copy(), camera.copy()
    left_camera[0, 3], right_camera[0, 3] = -.05, .05
    first.depth[:, :220] = .6
    second.depth[:, 180:] = .6
    assert not candidate_visibility_filter(target, grasp, first, left_camera)(0., 0., 0.)
    assert not candidate_visibility_filter(target, grasp, second, right_camera)(0., 0., 0.)
    assert candidate_visibility_filter(target, grasp, first, left_camera,
        observations=[(second, right_camera)])(0., 0., 0.)


def test_final_leaned_open_fingers_need_union_evidence_for_their_full_projection():
    target, grasp, first, second, camera = complementary_frames()
    release = top_down_T(0., 0., .04, 0.)
    release[:3, :3] = rot_y(math.radians(35.)) @ release[:3, :3]
    kwargs = dict(gripper_pose=release, opening=.1)
    assert not candidate_visibility_filter(target, grasp, first, camera)(0., 0., 0., **kwargs)
    assert not candidate_visibility_filter(target, grasp, second, camera)(0., 0., 0., **kwargs)
    assert candidate_visibility_filter(target, grasp, first, camera,
        observations=[(second, camera)])(0., 0., 0., **kwargs)
    # Beyond x=45 mm the leaned finger extends outside the carried box.
    # Seeing its support in neither image must still refuse this final pose.
    first.depth[190:211, 218:241] = .6
    second.depth[190:211, 218:241] = .6
    assert not candidate_visibility_filter(target, grasp, first, camera,
        observations=[(second, camera)])(0., 0., 0., **kwargs)


def test_missing_extra_depth_cannot_replace_or_invalidate_a_complete_current_view():
    target, grasp, current, camera = scene()
    unavailable = copy.deepcopy(current)
    unavailable.depth = None
    assert candidate_visibility_filter(target, grasp, current, camera,
        observations=[(unavailable, camera)])(0., 0., 0.)


def test_pixels_cropped_in_one_view_need_real_coverage_in_another():
    target, grasp, current, camera = scene()
    other = copy.deepcopy(current)
    left_camera, right_camera = camera.copy(), camera.copy()
    left_camera[0, 3], right_camera[0, 3] = -.46, .46
    assert not candidate_visibility_filter(target, grasp, current, left_camera)(0., 0., 0.)
    assert not candidate_visibility_filter(target, grasp, other, right_camera)(0., 0., 0.)
    assert candidate_visibility_filter(target, grasp, current, left_camera,
        observations=[(other, right_camera)])(0., 0., 0.)


def test_a_support_behind_the_camera_cannot_supply_positive_depth_evidence():
    target, grasp, current, camera = scene()
    current.depth[:] = 0.
    other = copy.deepcopy(current)
    other.depth[:] = 1.
    looking_away = make_T(np.eye(3), [0., 0., 1.])
    assert not candidate_visibility_filter(target, grasp, current, camera,
        observations=[(other, looking_away)])(0., 0., 0.)
