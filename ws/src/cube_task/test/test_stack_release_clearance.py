"""A tilted stack retreat must clear the released object before rotating."""
import math
import copy

import numpy as np
import pytest

from cube_task.core.transforms import rot_y, top_down_T
from cube_task.core.types import MotionResult, PickResult, Stamped
from test_task_skills import _setup


def stack_release():
    _cfg, _clock, _world, arm, log, skill, grasp, place = _setup()
    arm.world = None
    arm.width = .0385
    grasp.target.label = 'pear'
    grasp.target.height = .1123
    grasp.target.extra['_stack_support'] = {'top_z': .048}
    place.zone = 'on_object_yellow'
    place.T_release = top_down_T(.337, -.164, .1445, 0.)
    place.T_release[:3, :3] = rot_y(math.radians(10)) @ place.T_release[:3, :3]
    place.T_pre = place.T_release.copy()
    place.T_pre[2, 3] = .203
    place.T_retreat = place.T_release.copy()
    place.T_retreat[2, 3] = .245
    place.T_retreat[:3, :3] = rot_y(math.radians(15)) @ place.T_retreat[:3, :3]
    place.open_width = .0676
    arm.tcp = place.T_release.copy()
    checks = []
    arm.check_reachable = lambda target, **kwargs: (True, '')
    def linear(start, end, seed, limit):
        checks.append((start.copy(), end.copy(), seed.copy(), limit, len(arm.gripper_cmds())))
        return True, ''
    arm._check_linear_path = linear
    return arm, log, skill, grasp, place, checks


def test_stack_rotation_prechecks_actual_vertical_exit_before_opening():
    arm, log, skill, grasp, place, checks = stack_release()
    result = skill.place_held(grasp, place, arm.width)
    assert result.ok and not result.detail['holding']
    assert len(checks) == 1 and checks[0][-1] == 0
    start, clear, seed, _limit, _open_count = checks[0]
    np.testing.assert_array_equal(start, place.T_release)
    np.testing.assert_array_equal(clear[:3, :3], start[:3, :3])
    np.testing.assert_array_equal(clear[:2, 3], start[:2, 3])
    np.testing.assert_array_equal(seed, np.zeros(6))
    # A 125 mm library pear is taller than the 112.3 mm visible surface;
    # include the low fingertip caused by the ten-degree release lean.
    expected_z = .048 + .125 + .005 + (.0676 / 2 + .012) * math.sin(math.radians(10))
    assert clear[2, 3] == pytest.approx(expected_z)
    steps = log.kinds('skill_step')
    assert [s['stage'] for s in steps][-3:] == ['open_release', 'release_clearance', 'retreat']
    assert steps[-2]['linear'] is True
    np.testing.assert_array_equal(arm.motions()[-2][1], clear)
    np.testing.assert_array_equal(arm.motions()[-1][1], place.T_retreat)


@pytest.mark.parametrize('kind', ['plain', 'same_orientation'])
def test_ordinary_or_nonrotating_release_keeps_original_command_sequence(kind):
    arm, log, skill, grasp, place, checks = stack_release()
    if kind == 'plain':
        grasp.target.extra.clear()
    else:
        place.T_retreat[:3, :3] = place.T_release[:3, :3]
    result = skill.place_held(grasp, place, arm.width)
    assert result.ok and not checks
    assert not any(s['stage'] == 'release_clearance' for s in log.kinds('skill_step'))


@pytest.mark.parametrize('missing', ['height', 'height_nan', 'support', 'support_nan', 'native_path'])
def test_unknown_clearance_geometry_or_preflight_never_opens(missing):
    arm, _log, skill, grasp, place, _checks = stack_release()
    if missing == 'height':
        grasp.target.height = None
    elif missing == 'height_nan':
        grasp.target.height = math.nan
    elif missing == 'support':
        grasp.target.extra['_stack_support'] = {}
    elif missing == 'support_nan':
        grasp.target.extra['_stack_support']['top_z'] = math.nan
    else:
        del arm._check_linear_path
    result = skill.place_held(grasp, place, arm.width)
    assert not result.ok and result.detail['holding'] and not result.detail['fault']
    assert result.stage == 'release_clearance'
    assert result.error == 'release_clearance_unconfirmed'
    assert not arm.gripper_cmds()


@pytest.mark.parametrize('endpoint', [False, True])
def test_unreachable_clearance_is_a_closed_geometric_refusal(endpoint):
    arm, _log, skill, grasp, place, _checks = stack_release()
    if endpoint:
        arm.check_reachable = lambda *a, **k: (False, 'no endpoint IK')
    else:
        arm._check_linear_path = lambda *a, **k: (False, 'middle of vertical exit has no IK')
    result = skill.place_held(grasp, place, arm.width)
    assert not result.ok and result.detail['holding'] and not result.detail['fault']
    assert result.stage == 'release_clearance' and not arm.gripper_cmds()


def test_stop_during_preflight_keeps_closed_grip_and_stops_dispatch():
    arm, _log, skill, grasp, place, _checks = stack_release()
    def stop(*args):
        arm.abort_event.set()
        return False, 'aborted'
    arm._check_linear_path = stop
    result = skill.place_held(grasp, place, arm.width)
    assert not result.ok and result.detail['holding'] and result.detail['fault']
    assert result.error == 'aborted' and not arm.gripper_cmds()


def test_preflight_exception_never_opens_and_retains_fault():
    arm, _log, skill, grasp, place, _checks = stack_release()
    def broken(*args):
        raise RuntimeError('preflight feedback failed')
    arm._check_linear_path = broken
    result = skill.place_held(grasp, place, arm.width)
    assert not result.ok and result.detail['holding'] and result.detail['fault']
    assert not arm.gripper_cmds()


def test_clearance_checks_actual_release_start_and_stays_linear_in_p2p_mode():
    arm, _log, skill, grasp, place, checks = stack_release()
    skill.linear_moves = False
    def actual_release(_kind, target, _linear, _index):
        if np.array_equal(target, place.T_release):
            arm.tcp = target.copy()
            arm.tcp[0, 3] += .0004
            return MotionResult(ok=True, pos_err=.0004)
        return None
    arm.fail_fn = actual_release
    result = skill.place_held(grasp, place, arm.width)
    assert result.ok
    assert checks[0][0][0, 3] == pytest.approx(place.T_release[0, 3] + .0004)
    np.testing.assert_array_equal(checks[0][1][:2, 3], checks[0][0][:2, 3])
    assert arm.motions()[-2][2] is True
    assert arm.motions()[-1][2] is False


def test_stale_joints_do_not_authorize_opening():
    arm, _log, skill, grasp, place, _checks = stack_release()
    arm.get_joints = lambda: Stamped(np.zeros(6), arm.clock.now() - 10.)
    result = skill.place_held(grasp, place, arm.width)
    assert not result.ok and result.detail['holding'] and not result.detail['fault']
    assert not arm.gripper_cmds()


def test_pose_changed_during_preflight_keeps_object_held():
    arm, _log, skill, grasp, place, _checks = stack_release()
    def moved(*args):
        arm.tcp[0, 3] += .002
        return True, ''
    arm._check_linear_path = moved
    result = skill.place_held(grasp, place, arm.width)
    assert not result.ok and result.detail['holding'] and not result.detail['fault']
    assert not arm.gripper_cmds()


def test_failed_clearance_move_stops_before_the_rotating_retreat():
    arm, _log, skill, grasp, place, _checks = stack_release()
    arm.fail_fn = lambda _kind, _target, _linear, index: (
        MotionResult(ok=False, error='comm') if index == 3 else None)
    result = skill.place_held(grasp, place, arm.width)
    # Physical release already happened, but its controller halt must survive.
    assert result.detail['fault'] and not result.detail['holding']
    assert result.error == 'retreat_comm'
    assert len(arm.motions()) == 4 and len(arm.gripper_cmds()) == 1
    assert not np.array_equal(arm.motions()[-1][1], place.T_retreat)


def checked_wide_release():
    arm, log, skill, grasp, place, checks = stack_release()
    arm.max_width = skill.max_width = .1
    place.open_width = .095
    place.release_opening_checked = dict(width=.095,
        release_T=place.T_release.copy(), retreat_T=place.T_retreat.copy())
    return arm, log, skill, grasp, place, checks


def test_checked_complete_opening_can_use_preflighted_original_retreat():
    arm, log, skill, grasp, place, checks = checked_wide_release()
    skill.linear_moves = False
    result = skill.place_held(grasp, place, arm.width)
    assert result.ok and result.detail['release_exit_mode'] == 'checked_wide_opening'
    assert len(checks) == 1 and checks[0][-1] == 0
    np.testing.assert_array_equal(checks[0][0], place.T_release)
    np.testing.assert_array_equal(checks[0][1], place.T_retreat)
    assert arm.gripper_cmds() == [.095]
    assert arm.motions()[-1][2] is True
    assert not any(s['stage'] == 'release_clearance' for s in log.kinds('skill_step'))


@pytest.mark.parametrize('change', ['width', 'release', 'retreat', 'insufficient_width'])
def test_stale_or_insufficient_opening_evidence_uses_vertical_clearance(change):
    arm, log, skill, grasp, place, checks = checked_wide_release()
    if change == 'width':
        place.release_opening_checked['width'] = .09
    elif change == 'release':
        place.T_release[0, 3] += .001
    elif change == 'retreat':
        place.T_retreat[:3, :3] = rot_y(.01) @ place.T_retreat[:3, :3]
    else:
        grasp.target.length = .102
    result = skill.place_held(grasp, place, arm.width)
    assert result.ok
    assert result.detail.get('release_exit_mode') != 'checked_wide_opening'
    assert any(s['stage'] == 'release_clearance' for s in log.kinds('skill_step'))
    np.testing.assert_array_equal(checks[0][1][:3, :3], place.T_release[:3, :3])


@pytest.mark.parametrize('verify', [False, True])
def test_wide_mode_requires_full_checked_feedback_not_just_past_held_width(verify):
    arm, _log, skill, grasp, place, _checks = checked_wide_release()
    skill.verify = verify
    original = arm.gripper
    def short(width, **kwargs):
        feedback = original(width, **kwargs)
        arm.width = feedback.width = .060
        return feedback
    arm.gripper = short
    result = skill.place_held(grasp, place, arm.width)
    assert not result.ok and result.detail['fault'] and result.detail['holding']
    assert len(arm.motions()) == 3  # No exit motion after insufficient opening.


def test_wide_mode_rereads_actual_opening_before_moving():
    arm, _log, skill, grasp, place, _checks = checked_wide_release()
    arm.width_fn = lambda current: (.060 if current.gripper_cmds() else current.width)
    result = skill.place_held(grasp, place, arm.width)
    assert not result.ok and result.detail['fault'] and result.detail['holding']
    assert len(arm.motions()) == 3


def test_fresh_refinement_without_opening_proof_cannot_reuse_previous_marker():
    arm, _log, skill, grasp, place, _checks = checked_wide_release()
    updated = copy.deepcopy(place)
    del updated.release_opening_checked
    skill.before_release = lambda *args: updated
    arm.tcp = place.T_pre.copy()
    result = PickResult(False, grasp_width=.0385, detail={'holding': True, 'steps': []})
    skill._inspect_release(result, grasp, place)
    assert not hasattr(place, 'release_opening_checked')


@pytest.mark.parametrize('shortfall,accepted', [(1e-6, True), (.0005, False)])
def test_checked_opening_allows_sensor_quantization_but_not_real_shortfall(shortfall, accepted):
    arm, _log, skill, grasp, place, _checks = checked_wide_release()
    original = arm.gripper
    def quantized(width, **kwargs):
        feedback = original(width, **kwargs)
        arm.width = feedback.width = width - shortfall
        return feedback
    arm.gripper = quantized
    result = skill.place_held(grasp, place, arm.width)
    assert result.ok is accepted
    assert len(arm.motions()) == (4 if accepted else 3)
    if not accepted:
        assert result.detail['holding'] and result.detail['fault']
