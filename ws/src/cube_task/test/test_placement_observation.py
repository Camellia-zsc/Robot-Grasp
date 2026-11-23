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


@pytest.mark.parametrize('resting_pose,allowed', [('upright', True), ('uncertain', False)])
def test_round_body_visibility_does_not_demand_clearance_for_empty_box_corners(resting_pose, allowed):
    target, grasp, frame, camera = inputs()
    target.label = 'pear'
    target.length = target.width = .085
    target.center_base[2] = target.top_z = target.height = .112
    grasp.T_grasp[2, 3] = .077
    target.extra['resting_pose'] = resting_pose
    # Foreground lies outside the full round body and both finger sweeps,
    # but inside the old 85 x 85 mm enclosing square's empty corner.
    frame.depth[181:187, 213:219] = .7
    check = candidate_visibility_filter(target, grasp, frame, camera)
    assert check(0., 0., 0.) is allowed
    assert check(0., 0., 0., gripper_pose=top_down_T(0., 0., .077, 0.)) is allowed


@pytest.mark.parametrize('height,allowed', [(.05, True), (.09, False)])
def test_visible_low_neighbor_below_fingers_does_not_block_the_free_landing_surface(height, allowed):
    target, grasp, frame, camera = inputs()
    target.center_base[2] = target.top_z = target.height = .10
    grasp.T_grasp[2, 3] = .08
    # The 50 mm neighbor is outside the object's footprint and below the
    # 80 mm fingertips. A 90 mm neighbor would obstruct the actual fingers.
    frame.depth[196:205, 213:222] = 1. - height
    check = candidate_visibility_filter(target, grasp, frame, camera)
    assert check(0., 0., 0.) is allowed
    assert check(0., 0., 0., gripper_pose=top_down_T(0., 0., .08, 0.)) is allowed


def test_low_foreground_under_the_object_still_blocks_the_landing_plane():
    target, grasp, frame, camera = inputs()
    target.center_base[2] = target.top_z = target.height = .10
    grasp.T_grasp[2, 3] = .08
    frame.depth[190:211, 190:211] = .95
    check = candidate_visibility_filter(target, grasp, frame, camera)
    assert not check(0., 0., 0.)
    assert not check(0., 0., 0., gripper_pose=top_down_T(0., 0., .08, 0.))


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


def test_pose_aware_inspection_can_see_above_unknown_body_without_a_spherical_lift():
    target, grasp, _frame, _camera = inputs()
    target.length, target.width, target.height, target.top_z = .16, .04, .04, .04
    target.center_base[2] = .04
    grasp.T_grasp = top_down_T(0., 0., .008, 0.)
    grasp.expected_width = .04
    current = top_down_T(.3, 0., .23, 0.)
    arm = SimpleNamespace(get_tcp_pose=lambda: SimpleNamespace(value=current),
        check_reachable=lambda T, **kw: (.20 <= T[2, 3] <= .25, 'height limit'),
        _check_linear_path=lambda *a, **kw: (True, ''))
    T_fc = make_T(np.diag([1., -1., -1.]), [0., 0., .05])
    poses = list(iter_safe_inspection_poses(arm, grasp, T_fc, np.eye(4), [.3, 0.], .14,
        orientation_clearance=True, max_poses=2))
    assert poses
    obstacle = Detection('unobserved_pear', 0., valid=False,
        center_base=np.array([.3, 0., .14]), top_z=.14, height=.14,
        length=1., width=1., yaw=0.)
    # Candidate generation grants no motion permission: the unchanged full
    # swept-body/finger check must still pass above the uncertainty volume.
    route, why = safe_inspection_route(arm, grasp, current, poses[0], [obstacle], .04)
    assert route, why


def test_inspection_without_reachability_check_refuses_to_offer_unchecked_motion():
    _target, grasp, _frame, _camera = inputs()
    poses, _minimum = safe_inspection_poses(SimpleNamespace(), grasp, np.eye(4), np.eye(4),
                                           [.3, 0.], .1)
    assert poses == []


def test_tapered_contact_inspects_by_translation_and_rejects_large_bending_routes():
    target, grasp, _frame, _camera = inputs()
    target.extra['grasp'] = {'neck': True}
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''),
                         _check_linear_path=lambda *a: (True, ''))
    T_fc = make_T(np.diag([1., -1., -1.]), [0., 0., .05])
    poses, _ = safe_inspection_poses(arm, grasp, T_fc, np.eye(4), [.3, 0.], .11)
    assert poses
    assert all(np.arccos(np.clip(grasp.T_grasp[:3, 2] @ pose[:3, 2], -1, 1))
               <= np.deg2rad(15.) for pose in poses)
    np.testing.assert_allclose(poses[0][:3, :3], grasp.T_grasp[:3, :3])
    from cube_task.robot.kinematics import tilt_outward
    start = top_down_T(.3, 0., .3, 0.)
    end = tilt_outward(start, np.deg2rad(30.))
    route, why = safe_inspection_route(arm, grasp, start, end, [], .05, bend_reference=start)
    assert not route and 'tapered contact' in why


def test_tapered_contact_can_turn_around_world_vertical_without_tipping_the_body():
    from cube_task.core.transforms import rot_z
    from cube_task.robot.kinematics import tilt_outward
    target, grasp, _frame, _camera = inputs()
    target.extra['grasp'] = {'neck': True}
    start = tilt_outward(top_down_T(.3, 0., .3, 0.), np.deg2rad(35.))
    grasp.T_grasp[:3, :3] = start[:3, :3]
    end = start.copy()
    end[:3, :3] = rot_z(np.deg2rad(90.)) @ start[:3, :3]
    end[:2, 3] = [0., .3]
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''),
                         _check_linear_path=lambda *a: (True, ''))
    route, why = safe_inspection_route(arm, grasp, start, end, [], .05, bend_reference=start)
    assert route, why


def test_tapered_contact_rejects_tool_axis_spin_that_tips_a_leaned_carried_body():
    from cube_task.core.transforms import rot_z
    from cube_task.robot.kinematics import tilt_outward
    target, grasp, _frame, _camera = inputs()
    target.extra['grasp'] = {'neck': True}
    start = tilt_outward(top_down_T(.3, 0., .3, 0.), np.deg2rad(35.))
    grasp.T_grasp[:3, :3] = start[:3, :3]
    end = start.copy()
    end[:3, :3] = start[:3, :3] @ rot_z(np.deg2rad(90.))
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''),
                         _check_linear_path=lambda *a: (True, ''))
    route, why = safe_inspection_route(arm, grasp, start, end, [], .05, bend_reference=start)
    assert not route and 'tapered contact' in why


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


def test_observation_stop_reserves_clearance_for_the_next_route_padding():
    _target, grasp, _frame, _camera = inputs()
    obstacle = Detection('green', 1., center_base=np.array([.2, .070, .20]),
                         top_z=.20, height=.20, length=.06, width=.06, yaw=0.)
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''))
    start, end = top_down_T(0., 0., .15, 0.), top_down_T(.2, 0., .15, 0.)
    route, _why = safe_inspection_route(arm, grasp, start, end, [obstacle], .05)
    assert route  # The incident's first, finely sampled approach could pass.
    route, why = safe_inspection_route(arm, grasp, start, end, [obstacle], .05,
                                      endpoint_clearance=.02)
    assert not route and 'overlaps' in why


def test_no_inspection_motion_is_offered_when_every_native_path_is_rejected():
    _target, grasp, _frame, _camera = inputs()
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''),
                          _check_linear_path=lambda *a: (False, 'joint path unavailable'))
    route, why = safe_inspection_route(arm, grasp, top_down_T(.4, 0., .15, 0.),
                                      top_down_T(.2, 0., .18, 0.), [], .05)
    assert not route and why == 'joint path unavailable'


def test_inspection_opening_feedback_envelope_keeps_nearby_fingers_out_of_obstacles():
    _target, grasp, _frame, _camera = inputs()
    obstacle = Detection('green', 1., center_base=np.array([.0522, 0., .20]),
                         top_z=.20, height=.20, length=.006, width=.006, yaw=0.)
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''),
                         _check_linear_path=lambda *a: (True, ''))
    start, end = top_down_T(0., 0., .15, 0.), top_down_T(0., .4, .15, 0.)
    nominal, why = safe_inspection_route(arm, grasp, start, end, [obstacle], .05)
    assert nominal and not why
    guarded, why = safe_inspection_route(arm, grasp, start, end, [obstacle], .05,
                                         opening_uncertainty=.001)
    assert not guarded and 'overlaps' in why


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


@pytest.mark.parametrize('obstacle', [False, True])
def test_joint_fallback_checks_actual_fk_sweep_after_cartesian_path_refusal(obstacle):
    target, grasp, _frame, _camera = inputs()
    start, end = top_down_T(.3, 0., .2, 0.), top_down_T(.4, 0., .2, 0.)
    q0, q1 = np.array([.3, 0., 0., 0., 0., 0.]), np.array([.4, 0., 0., 0., 0., 0.])
    def fk(q):
        return top_down_T(q[0], .1*np.sin(np.pi*(q[0]-.3)/.1), .2, 0.)
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''),
        _check_linear_path=lambda *a: (False, 'Cartesian wrist singularity'),
        get_joints=lambda: SimpleNamespace(value=q0),
        _ik_tcp=lambda *a: (q1, 0., 0.), _kin=SimpleNamespace(fk=fk),
        _T_ft=np.eye(4), _guard=SimpleNamespace(check_tcp=lambda *a, **k: (True, '')))
    neighbors = [Detection('green', 1., center_base=np.array([.35, .1, .3]),
        top_z=.3, height=.3, length=.03, width=.03, yaw=0.)] if obstacle else []
    route, why = safe_inspection_route(arm, grasp, start, end, neighbors, .05,
                                     allow_joint_route=True)
    if obstacle:
        assert not route and 'joint sweep' in why
    else:
        assert route and not why
        np.testing.assert_array_equal(route.start_joints, q0)
        np.testing.assert_array_equal(route.end_joints, q1)
        np.testing.assert_array_equal(route[-1], end)


def test_remeasured_low_release_can_only_depart_support_by_a_straight_raise():
    _target, grasp, _frame, _camera = inputs()
    start, end = top_down_T(.3, 0., .02, 0.), top_down_T(.4, 0., .2, 0.)
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''),
                          _check_linear_path=lambda *a: (True, ''))
    route, why = safe_inspection_route(arm, grasp, start, end, [], .05,
                                     allow_table_departure=True)
    assert route and not why and len(route) == 2
    np.testing.assert_allclose(route[0][:2, 3], start[:2, 3])
    np.testing.assert_allclose(route[0][:3, :3], start[:3, :3])
    assert route[0][2, 3] >= start[2, 3] + .015


def test_low_departure_does_not_authorize_an_object_already_below_the_support():
    _target, grasp, _frame, _camera = inputs()
    start, end = top_down_T(.3, 0., .015, 0.), top_down_T(.4, 0., .2, 0.)
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''),
                          _check_linear_path=lambda *a: (True, ''))
    route, why = safe_inspection_route(arm, grasp, start, end, [], .05,
                                     allow_table_departure=True)
    assert not route and 'table' in why


@pytest.mark.parametrize('case,allowed', [('clear', True), ('penetrating', False),
    ('unconfirmed', False), ('other_support', False), ('end_too_low', False)])
def test_lift_away_from_observed_source_support_restores_full_route_clearance(case, allowed):
    target, grasp, _frame, _camera = inputs()
    target.center_base[2] = target.top_z = .1
    target.extra.update(table_z=.05, support_z=.05, support_label='yellow', support_observed=True)
    grasp.T_grasp[2, 3] = .07
    start = top_down_T(.3, 0., .072, 0.)  # Body is 2 mm above its old support.
    end = top_down_T(.4, 0., .2, 0.)
    if case == 'penetrating': start[2, 3] = .069
    if case == 'unconfirmed': target.extra['support_observed'] = False
    if case == 'other_support': target.extra['support_label'] = 'green'
    if case == 'end_too_low': end = top_down_T(.3, 0., .076, 0.)
    base = Detection('yellow', 1., center_base=np.array([.3, 0., .05]),
        top_z=.05, height=.05, length=.05, width=.05, yaw=0.)
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''),
                          _check_linear_path=lambda *a: (True, ''))
    route, why = safe_inspection_route(arm, grasp, start, end, [base], .05)
    assert bool(route) is allowed, why
    if allowed:
        assert len(route) == 2
        np.testing.assert_allclose(route[0][:2, 3], start[:2, 3])
        np.testing.assert_allclose(route[0][:3, :3], start[:3, :3])
        assert route[0][2, 3] >= start[2, 3] + .015


@pytest.mark.parametrize('resting_pose,clear', [('upright', True), ('uncertain', False)])
def test_round_carried_body_avoids_box_corners_without_shrinking_unknown_shapes(resting_pose, clear):
    target, grasp, _frame, _camera = inputs()
    target.label = 'pear'
    target.extra['resting_pose'] = resting_pose
    start, end = top_down_T(.3, 0., .15, 0.), top_down_T(.25, 0., .15, 0.)
    obstacle = Detection('green', 1., center_base=np.array([.333, .033, .19]),
        top_z=.19, height=.1, length=.004, width=.004, yaw=0.)
    arm = SimpleNamespace(check_reachable=lambda *a, **k: (True, ''),
                          _check_linear_path=lambda *a: (True, ''))
    route, why = safe_inspection_route(arm, grasp, start, end, [obstacle], .05)
    assert bool(route) is clear
    assert not why if clear else 'overlaps' in why
