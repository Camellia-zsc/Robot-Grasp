"""Measured stacking postconditions and designated contact exceptions."""
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.stacking import (
    annotate, camera_confirmed, center_supported, measured_support, planning_options,
)
from cube_task.core.transforms import top_down_T
from test_runner_local_release import scene as release_scene
from test_task_fakes import make_det


DESTINATION = dict(type='on_object', label='red')


@pytest.fixture(autouse=True)
def no_online_api(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('stack geometry tests must never call an online API')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


class NoTruthWorld:
    held = None

    @property
    def objects(self):
        raise AssertionError('camera geometry must not read simulator object poses')

    def zone_of(self, *args):
        raise AssertionError('camera geometry must not use simulator placement truth')


def measured_session():
    lower = make_det('red', .35, 0., size=.05, height=.05)
    upper = make_det('apple', .35, 0., size=.062, height=.062, table_z=.05)
    return SimpleNamespace(_physics=object(), _observed_detections=[lower, upper],
        _last_fresh=[lower, upper], _observation_revision=7, _environment_revision=7,
        _interventions=[], world=NoTruthWorld(),
        _placement={'table_bounds': {'x': [.08, .6], 'y': [-.4, .4]}})


def test_planning_uses_only_measured_base_top_and_support_center():
    session = measured_session()
    options, support = planning_options(session, DESTINATION)
    zone = options['placement_zone']
    assert zone.name == 'on_object_red'
    np.testing.assert_allclose(zone.center, [.35, 0., .05])
    assert zone.release_height > 0.
    assert options['exact_xy'] is False
    assert support['label'] == 'red' and support['top_z'] == .05
    assert support['xy'] == [.35, 0.]
    # This larger polygon bounds the real table. It must not force a 62 mm
    # apple footprint into a 50 mm box's top-face rectangle.
    assert np.ptp(options['placement_polygon'], axis=0).min() > .062


@pytest.mark.parametrize('label', ['apple', 'pear', 'banana'])
def test_curved_base_is_not_claimed_to_be_a_confirmed_flat_support(label):
    det = make_det(label, .35, 0., size=.06, height=.07)
    with pytest.raises(ValueError, match='平顶|曲面'):
        measured_support([det], label)


def test_missing_or_ambiguous_camera_base_refuses_planning():
    session = measured_session()
    session._observed_detections = []
    with pytest.raises(ValueError, match='相机'):
        planning_options(session, DESTINATION)
    session._observed_detections = [make_det('red', .3, 0.), make_det('red', .4, 0.)]
    with pytest.raises(ValueError, match='相机'):
        planning_options(session, DESTINATION)


def test_no_physics_cannot_silently_execute_stacking_as_a_table_placement():
    session = measured_session()
    session._physics = None
    with pytest.raises(ValueError, match='MuJoCo'):
        planning_options(session, DESTINATION)


def test_upper_center_can_be_supported_while_its_footprint_overhangs_base():
    support = measured_support(measured_session()._last_fresh, 'red')
    assert center_supported([.35, .0], support)
    assert center_supported([.36, .0], support)
    assert not center_supported([.38, .0], support)
    rotated = dict(support, yaw=np.pi / 2, length=.08, width=.03)
    assert center_supported([.35, .03], rotated)
    assert not center_supported([.37, .0], rotated)


def test_annotation_copies_measured_support_without_mutating_shared_measurements():
    session = measured_session()
    target = session._last_fresh[1]
    support = measured_support(session._last_fresh, 'red')
    annotate(target, support)
    support['xy'][0] = 100.
    assert target.extra['_stack_support']['xy'] == [.35, 0.]


def test_camera_confirmation_uses_current_upper_and_measured_base_height():
    assert camera_confirmed(measured_session(), 'apple', DESTINATION)


@pytest.mark.parametrize('change', ['revision', 'pending', 'held', 'missing_upper',
                                   'wrong_base', 'wrong_height', 'outside_center'])
def test_camera_confirmation_rejects_stale_wrong_or_unsupported_scene(change):
    session = measured_session()
    if change == 'revision':
        session._environment_revision += 1
    elif change == 'pending':
        session._interventions.append(dict(action='move'))
    elif change == 'held':
        session.world.held = session._last_fresh[1]
    elif change == 'missing_upper':
        session._last_fresh = session._last_fresh[:1]
    elif change == 'wrong_base':
        session._last_fresh[0].label = 'green'
    elif change == 'wrong_height':
        session._last_fresh[1].extra['table_z'] = 0.
    else:
        session._last_fresh[1].center_base[0] += .06
    assert not camera_confirmed(session, 'apple', DESTINATION)


def test_completely_occluded_base_cannot_be_replaced_by_old_observed_coordinates():
    session = measured_session()
    session._last_fresh = session._last_fresh[1:]
    assert not camera_confirmed(session, 'apple', DESTINATION)


def test_partial_visible_support_evidence_confirms_label_and_measured_height():
    session = measured_session()
    upper = session._last_fresh[1]
    session._last_fresh = [upper]
    # A real apple hides the center of the box's top face. The currently seen
    # red front strip is evidence of the support plane, not its full outline.
    upper.extra.update(support_observed=True, support_label='red', support_z=.05,
        support_pixel_count=120,
        support_polygon=[[.328, -.024], [.372, -.024], [.372, -.013], [.328, -.013]])
    assert camera_confirmed(session, 'apple', DESTINATION)
    upper.extra['support_label'] = 'green'
    assert not camera_confirmed(session, 'apple', DESTINATION)


def test_remote_same_color_patch_does_not_confirm_support():
    session = measured_session()
    upper = session._last_fresh[1]
    session._last_fresh = [upper]
    upper.extra.update(support_observed=True, support_label='red', support_z=.05,
        support_pixel_count=120,
        support_polygon=[[.428, -.024], [.472, -.024], [.472, -.013], [.428, -.013]])
    assert not camera_confirmed(session, 'apple', DESTINATION)


def stack_release_filter(extra=(), annotated=True):
    runner, _old, grasp, _place = release_scene()
    target = make_det('apple', .2, .15, size=.062, height=.062)
    base = make_det('red', .35, 0., size=.05, height=.05)
    grasp.target = target
    grasp.expected_width = .062
    runner.nominal['apple'] = (.062, .062)
    if annotated:
        annotate(target, measured_support([base], 'red'))
    polygon = np.array([[.08, -.35], [.6, -.35], [.6, .35], [.08, .35]])
    return runner._strict_place_filter(target, grasp, [target, base, *extra], polygon)


def test_designated_base_allows_upper_footprint_contact_and_overhang():
    allowed = stack_release_filter()
    assert allowed(.35, 0., 0., gripper_pose=top_down_T(.35, 0., .09, 0.))


def test_same_xy_overlap_remains_blocked_for_an_ordinary_table_placement():
    assert not stack_release_filter(annotated=False)(.35, 0., 0.)


def test_other_object_remains_an_obstacle_beside_the_designated_base():
    obstruction = make_det('green', .37, 0., size=.05, height=.08)
    assert not stack_release_filter([obstruction])(.35, 0., 0.)


def test_support_contact_exception_does_not_allow_gripper_to_penetrate_base():
    allowed = stack_release_filter()
    assert not allowed(.35, 0., 0., gripper_pose=top_down_T(.35, 0., .025, 0.),
                       gripper_open_width=.02)


def test_carried_bottom_must_not_penetrate_designated_support_even_with_high_fingers():
    assert not stack_release_filter()(.35, 0., 0.,
        gripper_pose=top_down_T(.35, 0., .065, 0.))


def test_release_center_cannot_move_outside_designated_support():
    assert not stack_release_filter()(.38, 0., 0.)


def test_release_capacity_excludes_only_designated_base_from_2d_occupancy():
    runner, target, grasp, place = release_scene()
    base = make_det('green', .4, 0., size=.05, height=.05)
    neighbor = make_det('yellow', .47, 0., size=.05, height=.05)
    annotate(target, measured_support([base], 'green'))
    search, why = runner._release_search_geometry(target, grasp, place, .05,
                                                  [target, base, neighbor])
    assert search is not None, why
    assert search['neighbors'] == [target, base, neighbor]
    assert len(search['occupied']) == 1
    assert search['occupied'][0][:2] == (.47, 0.)
    base.center_base[0] += .03
    search, why = runner._release_search_geometry(target, grasp, place, .05,
                                                  [target, base, neighbor])
    assert search is not None, why
    assert len(search['occupied']) == 2


def test_missing_current_support_measurement_cannot_be_an_intentional_contact():
    runner, target, grasp, _place = release_scene()
    annotate(target, measured_support([make_det('green', .4, 0., size=.05, height=.05)], 'green'))
    polygon = np.array([[.08, -.35], [.6, -.35], [.6, .35], [.08, .35]])
    allowed = runner._strict_place_filter(target, grasp, [target], polygon)
    assert not allowed(.4, 0., 0., gripper_pose=top_down_T(.4, 0., .09, 0.))


def test_round_upright_pear_release_does_not_invent_diagonal_square_corners():
    runner, _old, grasp, place = release_scene()
    runner.arm.guard = SimpleNamespace(x=(.08, .6), y=(-.4, .4), r_max=.62)
    target = make_det('pear', .19, .324, size=.075, height=.113)
    target.yaw = np.pi / 4
    target.extra['resting_pose'] = 'upright'
    runner.nominal['pear'] = (.085, .085)
    grasp.target = target
    grasp.closing_angle = 0.
    grasp.T_grasp = top_down_T(.19, .324, .09, 0.)
    grasp.expected_width = .05
    base = make_det('green', .121, .36, size=.05, height=.05)
    annotate(target, measured_support([base], 'green'))
    polygon = np.array([[.08, -.4], [.6, -.4], [.6, .4], [.08, .4]])
    allowed = runner._strict_place_filter(target, grasp, [target, base], polygon)
    assert allowed(.132, .349, 0.)
    # The unchanged physical table/workspace edges still reject a true overhang.
    assert not allowed(.121, .36, 0.)
    place.T_release = top_down_T(.132, .349, .144, 0.)
    place.T_pre = top_down_T(.132, .349, .2, 0.)
    place.T_retreat = top_down_T(.132, .349, .24, 0.)
    ok, why = runner._adjust_strict_release(target, grasp, place, [target, base],
        polygon, [.132, .349], .09)
    assert ok, why


def test_stack_planner_can_offset_inside_measured_support_at_workspace_edge():
    from cube_task.task.runner import held_shape
    runner, _old, grasp, _place = release_scene()
    runner.arm.guard = SimpleNamespace(x=(.08, .6), y=(-.4, .4), r_max=.62)
    target = make_det('pear', .19, .324, size=.075, height=.113)
    target.extra['resting_pose'] = 'upright'
    target.yaw = np.pi / 4
    runner.nominal['pear'] = (.085, .085)
    grasp.target = target
    grasp.T_grasp = top_down_T(.19, .324, .09, 0.)
    grasp.expected_width = .05
    base = make_det('green', .121, .36, size=.05, height=.05)
    session = measured_session()
    session._observed_detections = [target, base]
    options, support = planning_options(session, dict(type='on_object', label='green'))
    annotate(target, support)
    allowed = runner._strict_place_filter(target, grasp, [target, base], options['placement_polygon'])
    shape = held_shape(target, grasp, runner.nominal['pear'])
    plan = runner.placer.plan(options['placement_zone'], .09, .2,
        held=shape, strict=True, fixed_xy=options['exact_xy'], pose_filter=allowed)
    assert plan.feasible, plan.reason
    xy = runner.placer.last_info['xy']
    assert center_supported(xy, support)
    assert np.linalg.norm(np.asarray(xy) - support['xy']) > .004


def test_small_support_retries_finger_angle_without_banning_neighboring_centers():
    runner, target, grasp, previous = release_scene()
    base = make_det('green', .4, 0., size=.05, height=.05)
    session = measured_session()
    session._observed_detections = [target, base]
    options, support = planning_options(session, dict(type='on_object', label='green'))
    annotate(target, support)
    visited = []
    previous.zone = options['placement_zone'].name
    def check_final(det, gp, candidate, neighbors, polygon, desired, held, **kwargs):
        visited.append(tuple(desired))
        return (False, 'final finger blocked') if len(visited) < 3 else (True, '')
    runner._adjust_strict_release = check_final
    updated, why = runner.replan_release(target, grasp, previous, .05,
        [target, base], **options)
    assert updated is not None, why
    assert len(visited) == 3
    assert visited[0] == visited[1] == visited[2]
