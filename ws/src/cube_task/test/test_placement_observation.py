"""Release permission needs fresh table visibility, including the finger sweep."""
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.placement_observation import (
    candidate_visibility_filter, safe_inspection_poses,
)
from cube_task.core.transforms import make_T, top_down_T
from cube_task.core.types import CameraIntrinsics, Detection, Frame, GraspPlan


def inputs():
    target = Detection('red', 1., center_base=np.array([0., 0., .05]), top_z=.05,
                       height=.05, length=.05, width=.05, yaw=0.)
    grasp = GraspPlan(target, top_down_T(0., 0., .15, 0.), top_down_T(0., 0., .02, 0.),
                      top_down_T(0., 0., .2, 0.), .06, 0., .05, 0.)
    intr = CameraIntrinsics(400., 400., 200., 200., 400, 400)
    frame = Frame(np.zeros((400, 400, 3), np.uint8), np.ones((400, 400), np.float32), intr, 1.)
    camera = make_T(np.diag([1., -1., -1.]), [0., 0., 1.])
    return target, grasp, frame, camera


def test_clear_table_and_inside_frame_allows_local_release():
    target, grasp, frame, camera = inputs()
    allowed = candidate_visibility_filter(target, grasp, frame, camera)
    assert allowed(0., 0., 0.)
    assert allowed(0., 0., 0., gripper_pose=top_down_T(0., 0., .04, 0.), opening=.07)


@pytest.mark.parametrize('case', ['carried_occlusion', 'hole', 'zero', 'crop'])
def test_a_cropped_or_occluded_landing_region_is_not_free_space(case):
    target, grasp, frame, camera = inputs()
    x = 0.
    if case == 'carried_occlusion':
        frame.depth[194:207, 194:207] = .7
    elif case == 'hole':
        frame.depth[194:207, 194:207] = np.nan
    elif case == 'zero':
        frame.depth[194:207, 194:207] = 0.
    elif case == 'crop':
        x = .46
    assert not candidate_visibility_filter(target, grasp, frame, camera)(x, 0., 0.)


def test_finger_area_must_be_observed_even_when_the_object_footprint_is_clear():
    target, grasp, frame, camera = inputs()
    # +4 cm along the closing axis is beyond the object's +2.5 cm edge.
    frame.depth[197:204, 215:220] = .8
    assert not candidate_visibility_filter(target, grasp, frame, camera)(0., 0., 0.)


def test_free_slot_outside_carried_objects_image_occlusion_remains_available():
    target, grasp, frame, camera = inputs()
    frame.depth[192:209, 192:209] = .7
    allowed = candidate_visibility_filter(target, grasp, frame, camera)
    assert not allowed(0., 0., 0.)
    assert allowed(.12, 0., 0.)


def test_missing_depth_never_confirms_landing_space():
    target, grasp, frame, camera = inputs()
    frame.depth = None
    assert not candidate_visibility_filter(target, grasp, frame, camera)(0., 0., 0.)


def test_inspection_pose_proposals_are_guarded_and_clear_all_intermediate_rotations():
    target, grasp, _frame, _camera = inputs()
    checked = []

    def check(T, max_tilt_deg=None):
        checked.append((T.copy(), max_tilt_deg))
        return True, ''

    arm = SimpleNamespace(check_reachable=check)
    # Example fixed hand-eye calibration; no physical world is needed.
    T_fc = make_T(np.diag([1., -1., -1.]), [0., 0., .05])
    poses, minimum = safe_inspection_poses(arm, grasp, T_fc, np.eye(4), [.3, 0.], .11)
    assert poses and len(checked) == len(poses)
    radius = np.linalg.norm([.025, .025, .03])
    assert minimum == pytest.approx(.11 + .04 + radius)
    assert all(pose[2, 3] >= minimum for pose in poses)
    assert all(tilt == 40. for _pose, tilt in checked)


def test_inspection_without_reachability_check_refuses_to_offer_unchecked_motion():
    _target, grasp, _frame, _camera = inputs()
    poses, _minimum = safe_inspection_poses(SimpleNamespace(), grasp, np.eye(4), np.eye(4),
                                           [.3, 0.], .1)
    assert poses == []
