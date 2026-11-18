"""Release permission needs fresh table visibility, including the finger sweep."""
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.placement_observation import (
    candidate_visibility_filter, safe_inspection_poses, safe_inspection_route,
    iter_safe_inspection_poses, inspection_history, remember_inspection, forget_inspection,
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


def test_stack_landing_sees_support_top_instead_of_demanding_table_through_base():
    target, grasp, frame, camera = inputs()
    target.extra['_stack_support'] = dict(top_z=.12)
    frame.depth[:] = .88
    assert candidate_visibility_filter(target, grasp, frame, camera)(0., 0., 0.)
    frame.depth[:] = .82
    assert not candidate_visibility_filter(target, grasp, frame, camera)(0., 0., 0.)


def test_stack_landing_height_does_not_shift_source_geometry_during_final_tilt(monkeypatch):
    from cube_task.assistant import placement_observation
    target, grasp, frame, camera = inputs()
    pose = top_down_T(0., 0., .2, 0.)
    angle = np.deg2rad(20.)
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)],
                         [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    pose[:3, :3] = rotation @ pose[:3, :3]
    seen = []
    original = placement_observation._samples
    monkeypatch.setattr(placement_observation, '_samples',
        lambda polygon, step: seen.append(polygon.copy()) or original(polygon, step))
    candidate_visibility_filter(target, grasp, frame, camera)(0., 0., 0., gripper_pose=pose)
    source_projection = seen[0]
    seen.clear()
    target.extra['_stack_support'] = dict(top_z=.12)
    candidate_visibility_filter(target, grasp, frame, camera)(0., 0., 0., gripper_pose=pose)
    np.testing.assert_array_equal(seen[0], source_projection)


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


def test_inspection_checks_native_path_and_can_retract_without_an_unreachable_lift():
    _target, grasp, _frame, _camera = inputs()
    current = top_down_T(.4, 0., .15, 0.)
    destination = top_down_T(.2, .08, .18, 0.)
    checked = []

    def path(a, b, _seed, _tilt):
        checked.append((a.copy(), b.copy()))
        # A direct orientation/translation path is unavailable from extension;
        # the fixed-orientation inward waypoint remains reachable.
        direct = np.allclose(a, current) and np.allclose(b, destination)
        return not direct, 'direct path unavailable' if direct else ''

    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''), _check_linear_path=path)
    route, why = safe_inspection_route(arm, grasp, current, destination, [], .05)
    assert len(route) == 2 and not why
    np.testing.assert_allclose(route[0][:3, :3], current[:3, :3])
    assert route[0][2, 3] == current[2, 3]
    assert route[0][0, 3] < current[0, 3]
    assert len(checked) == 3  # refused direct, accepted inward, accepted final


def test_inspection_swept_carried_box_cannot_pass_through_a_measured_neighbor():
    _target, grasp, _frame, _camera = inputs()
    obstacle = Detection('green', 1., center_base=np.array([.2, 0., .19]),
                         top_z=.19, height=.19, length=.06, width=.06, yaw=0.)
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''))
    route, why = safe_inspection_route(arm, grasp, top_down_T(0., 0., .15, 0.),
                                      top_down_T(.4, 0., .15, 0.), [obstacle], .05)
    assert route == [] and 'overlaps' in why


def test_far_tall_neighbor_does_not_raise_the_local_inspection_path():
    _target, grasp, _frame, _camera = inputs()
    far = Detection('green', 1., center_base=np.array([.2, .3, .6]),
                    top_z=.6, height=.6, length=.06, width=.06, yaw=0.)
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''))
    route, why = safe_inspection_route(arm, grasp, top_down_T(0., 0., .15, 0.),
                                      top_down_T(.25, 0., .15, 0.), [far], .05)
    assert len(route) == 1 and not why
    assert route[0][2, 3] == .15


def test_no_inspection_motion_is_offered_when_every_native_path_is_rejected():
    _target, grasp, _frame, _camera = inputs()
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''),
                          _check_linear_path=lambda *a: (False, 'joint path unavailable'))
    route, why = safe_inspection_route(arm, grasp, top_down_T(.4, 0., .15, 0.),
                                      top_down_T(.2, 0., .18, 0.), [], .05)
    assert not route and why == 'joint path unavailable'


def test_already_measured_view_is_skipped_before_ik_for_same_held_scene():
    _target, grasp, _frame, _camera = inputs()
    checks = []
    arm = SimpleNamespace(check_reachable=lambda pose, **kw: checks.append(pose.copy()) or (True, ''))
    T_fc = make_T(np.diag([1., -1., -1.]), [0., 0., .05])
    first = next(iter_safe_inspection_poses(arm, grasp, T_fc, np.eye(4), [.3, 0.], .11,
                                            max_poses=1))
    checks.clear()
    second = next(iter_safe_inspection_poses(arm, grasp, T_fc, np.eye(4), [.3, 0.], .11,
                                             exclude_poses=[first], max_poses=1))
    trace = np.sum(first[:3, :3] * second[:3, :3])
    assert np.linalg.norm(first[:3, 3] - second[:3, 3]) >= .005 or trace <= 1 + 2*np.cos(np.deg2rad(3))
    assert len(checks) == 1


def test_inspection_memory_expires_for_user_edit_or_new_grasp():
    _target, grasp, _frame, _camera = inputs()
    session = SimpleNamespace(_environment_revision=3)
    pose = top_down_T(.3, 0., .2, 0.)
    remember_inspection(session, grasp, pose)
    pose[0, 3] = 9.
    assert inspection_history(session, grasp)[0][0, 3] == .3
    session._environment_revision += 1
    assert inspection_history(session, grasp) == []
    remember_inspection(session, grasp, pose)
    grasp.T_grasp[0, 3] += .01
    assert inspection_history(session, grasp) == []


def test_releasing_and_regrasping_same_anchor_allows_old_views_again():
    _target, grasp, _frame, _camera = inputs()
    session = SimpleNamespace(_environment_revision=3,
        _skill_coverage_revision=3, _skill_observation_coverage={'focus': np.array([True])})
    remember_inspection(session, grasp, top_down_T(.3, 0., .2, 0.))
    assert inspection_history(session, grasp)
    forget_inspection(session)
    assert inspection_history(session, grasp) == []
    assert session._skill_coverage_revision is None and not session._skill_observation_coverage
