"""Checked stack releases plan and validate the aperture actually commanded."""
import numpy as np
import pytest

from cube_task.core.transforms import top_down_T
from cube_task.core.types import Detection, PickResult, Zone
from cube_task.task.skills import RELEASE_OPEN_QUANTIZATION_TOL
from test_runner_local_release import scene


def stacked_scene():
    runner, target, grasp, previous = scene()
    target.label = 'pear'
    target.center_base = np.array([.2, .15, .1123])
    target.height = target.top_z = .1123
    target.length, target.width = .075, .050
    target.extra.update(resting_pose='upright', _restore_yaw=0.)
    target.extra['_stack_support'] = dict(label='green', xy=[.4, 0.],
        top_z=.05, length=.05, width=.05, yaw=0.)
    grasp.T_grasp = top_down_T(.2, .15, .0923, 0.)
    grasp.expected_width = .048
    runner.nominal['pear'] = (.085, .085)
    zone = Zone('on_object_green', np.array([.4, 0., .05]), (.30, .30),
                release_height=.004)
    previous.zone = zone.name
    support = Detection('green', 1., center_base=np.array([.4, 0., .05]),
        height=.05, top_z=.05, length=.05, width=.05, yaw=0.)
    polygon = np.array([[.1, -.3], [.6, -.3], [.6, .3], [.1, .3]])
    return runner, target, grasp, previous, support, zone, polygon


def replan(parts, extra=(), **kwargs):
    runner, target, grasp, previous, support, zone, polygon = parts
    return runner.replan_release(target, grasp, previous, .048,
        [target, support, *extra], placement_zone=zone,
        placement_polygon=polygon, exact_xy=True, **kwargs)


def test_checked_stack_opening_covers_nominal_body_and_binds_owned_matrices():
    parts = stacked_scene()
    checks = []
    def observed(*args, **kwargs):
        checks.append(kwargs)
        return True
    candidate, why = replan(parts, observation_filter=observed)
    assert candidate is not None, why
    required = .085 + .010 + RELEASE_OPEN_QUANTIZATION_TOL
    assert candidate.open_width == pytest.approx(required)
    assert checks[-1]['opening'] == pytest.approx(candidate.open_width)
    proof = candidate.release_opening_checked
    assert proof['width'] == pytest.approx(candidate.open_width)
    for name, attr in [('release_T', 'T_release'), ('retreat_T', 'T_retreat')]:
        np.testing.assert_array_equal(proof[name], getattr(candidate, attr))
        assert not np.shares_memory(proof[name], getattr(candidate, attr))
        before = proof[name].copy()
        getattr(candidate, attr)[0, 3] += .001
        np.testing.assert_array_equal(proof[name], before)


def test_wider_release_rejects_neighbor_that_old_grip_based_opening_cleared():
    parts = stacked_scene()
    neighbor = Detection('unknown', 1., center_base=np.array([.457, 0., .2]),
        height=.2, top_z=.2, length=.008, width=.008, yaw=0.)
    candidate, why = replan(parts, [neighbor], observation_filter=lambda *a, **k: True)
    assert candidate is None, 'the 95 mm aperture must not open into the measured neighbor'
    assert why == 'no_clear_supported_release'


@pytest.mark.parametrize('final_pose', [False, True])
def test_strict_body_filter_checks_supplied_actual_aperture_at_both_stages(final_pose):
    runner, target, grasp, previous = scene()
    neighbor = Detection('unknown', 1., center_base=np.array([.457, 0., .2]),
        height=.2, top_z=.2, length=.008, width=.008, yaw=0.)
    allowed = runner._strict_place_filter(target, grasp, [target, neighbor],
                                          runner.zones['zone_red'].corners_xy())
    kwargs = {'gripper_open_width': .095002}
    if final_pose:
        kwargs['gripper_pose'] = top_down_T(.4, 0., .05, 0.)
    assert allowed(.4, 0., 0., gripper_open_width=.068)
    assert not allowed(.4, 0., 0., **kwargs)


def test_stack_without_final_fresh_visibility_has_no_opening_certificate():
    candidate, why = replan(stacked_scene())
    assert candidate is not None, why
    assert not hasattr(candidate, 'release_opening_checked')


def test_stack_too_wide_for_stroke_retains_vertical_clearance_fallback():
    parts = stacked_scene()
    parts[0].placer.open_width = .09
    candidate, why = replan(parts, observation_filter=lambda *a, **k: True)
    assert candidate is not None, why
    assert candidate.open_width == pytest.approx(.068)
    assert not hasattr(candidate, 'release_opening_checked')


def test_nonstack_release_keeps_existing_grip_based_aperture():
    runner, target, grasp, previous = scene()
    candidate, why = runner.replan_release(target, grasp, previous, .05, [target],
                                          observation_filter=lambda *a, **k: True)
    assert candidate is not None, why
    assert candidate.open_width == pytest.approx(.07)
    assert not hasattr(candidate, 'release_opening_checked')


@pytest.mark.parametrize('error', ['release_clearance_unconfirmed', 'held_anchor_unconfirmed'])
def test_non_driver_release_refusal_preserves_held_recovery_context(error):
    runner, target, grasp, previous = scene()
    result = PickResult(False, stage='release', error=error, grasp_width=.05,
        detail={'holding': True, 'fault': False, 'steps': []})
    runner._save_held_recovery(target, grasp, previous, result)
    assert runner.has_held_recovery
    assert runner._held_recovery_context['place'].zone == previous.zone
    np.testing.assert_array_equal(runner._held_recovery_context['grasp'].T_grasp,
                                  grasp.T_grasp)


def test_explicit_stack_opening_is_respected_by_real_place_planner():
    parts = stacked_scene()
    runner, target, grasp, previous, support, zone, polygon = parts
    search, why = runner._release_search_geometry(target, grasp, previous, .048,
        [target, support], zone, polygon)
    assert search is not None, why
    search['shape']['requested_release_opening'] = .097
    candidate = runner.placer.plan(zone, search['held'], search['transit'],
        held=search['shape'], strict=True, fixed_xy=True)
    assert candidate.feasible, candidate.reason
    assert candidate.open_width == pytest.approx(.097)


def test_final_leaned_body_check_receives_the_actual_wider_aperture():
    parts = stacked_scene()
    runner = parts[0]
    original = runner._strict_place_filter
    checked = []
    def factory(*args, **kwargs):
        allowed = original(*args, **kwargs)
        def check(*values, **options):
            if options.get('gripper_pose') is not None:
                checked.append(options['gripper_open_width'])
            return allowed(*values, **options)
        return check
    runner._strict_place_filter = factory
    candidate, why = replan(parts, observation_filter=lambda *a, **k: True)
    assert candidate is not None, why
    assert checked and all(value == pytest.approx(candidate.open_width) for value in checked)


@pytest.mark.parametrize('failure', ['reach', 'geometry', 'visibility', 'cancelled'])
def test_no_certificate_is_published_before_all_final_checks_pass(monkeypatch, failure):
    parts = stacked_scene()
    runner = parts[0]
    planned = []
    original = runner.placer.plan
    def record(*args, **kwargs):
        candidate = original(*args, **kwargs)
        planned.append(candidate)
        return candidate
    runner.placer.plan = record
    if failure == 'reach':
        monkeypatch.setattr('cube_task.task.reach.adjust_plan_poses',
                            lambda *a, **k: (False, 'unreachable', {}))
    if failure == 'geometry':
        runner._adjust_strict_release = lambda *a, **k: (False, 'no finger clearance')
    stopped = []
    def observed(*args, **kwargs):
        if kwargs.get('gripper_pose') is not None:
            if failure == 'cancelled':
                stopped.append(True)
            return failure != 'visibility'
        return True
    candidate, why = replan(parts, observation_filter=observed, max_attempts=1,
                            cancelled=lambda: bool(stopped))
    assert candidate is None, why
    assert planned
    assert all(not hasattr(value, 'release_opening_checked') for value in planned)


@pytest.mark.parametrize('aperture', [.101, -.01, float('nan')])
def test_invalid_explicit_opening_does_not_silently_clamp_or_get_a_plan(aperture):
    parts = stacked_scene()
    runner, target, grasp, previous, support, zone, polygon = parts
    search, why = runner._release_search_geometry(target, grasp, previous, .048,
        [target, support], zone, polygon)
    assert search is not None, why
    search['shape']['requested_release_opening'] = aperture
    candidate = runner.placer.plan(zone, search['held'], search['transit'],
        held=search['shape'], strict=True, fixed_xy=True)
    assert not candidate.feasible
