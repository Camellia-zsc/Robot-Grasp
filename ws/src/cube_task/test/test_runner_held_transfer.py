"""A temporary held release preserves the measured grip and the outer goal."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cube_task.core.types import Detection, PickResult, Zone
from test_runner_local_release import scene


def transfer_scene():
    runner, target, grasp, previous = scene()
    runner.skill = SimpleNamespace(empty_width=.003, width_tolerance=.008,
                                   _read_width=Mock(return_value=.05),
                                   place_held=Mock(return_value=PickResult(
                                       True, stage='done', grasp_width=.05,
                                       detail={'holding': False, 'steps': []})))
    runner._placed, runner._placed_labels = {}, {}
    runner._save_held_recovery(target, grasp, previous, PickResult(
        False, error='rejected', grasp_width=.05,
        detail={'holding': True, 'steps': []}))
    temporary = Zone('temporary_stage', np.array([.3, -.15, 0.]), (.2, .2))
    return runner, target, grasp, temporary


def plan(runner, target, temporary, neighbors=()):
    return runner.plan_held_transfer(target, neighbors, temporary,
                                     temporary.corners_xy(), exact_xy=True)


def test_transfer_plan_uses_saved_grip_and_exact_object_center_without_commands():
    runner, target, _grasp, temporary = transfer_scene()
    context = runner._held_recovery_context
    context['grasp'].T_grasp[0, 3] += .012
    original_grip = context['grasp'].T_grasp.copy()
    bundle, why = plan(runner, target, temporary)
    assert bundle is not None, why
    assert runner._held_recovery_context is context
    assert context['zone'] == 'zone_red'  # Pure planning must not replace the goal.
    np.testing.assert_array_equal(bundle['grasp'].T_grasp, original_grip)
    R = bundle['place'].T_release[:3, :3] @ original_grip[:3, :3].T
    released_center = bundle['place'].T_release[:3, 3] + R @ (
        np.r_[target.center_base[:2], .025] - original_grip[:3, 3])
    np.testing.assert_allclose(released_center[:2], temporary.center[:2], atol=.0002)
    runner.skill.place_held.assert_not_called()


@pytest.mark.parametrize('width', [None, float('nan'), 0., .2, .065])
def test_transfer_refuses_missing_changed_or_invalid_held_width(width):
    runner, target, _grasp, temporary = transfer_scene()
    runner.skill._read_width.return_value = width
    bundle, why = plan(runner, target, temporary)
    assert bundle is None and 'feedback' in why
    runner.skill.place_held.assert_not_called()


@pytest.mark.parametrize('changed', ['target', 'aborted', 'unhealthy', 'context'])
def test_transfer_refuses_wrong_target_or_unavailable_arm(changed):
    runner, target, _grasp, temporary = transfer_scene()
    if changed == 'target':
        target = copy.deepcopy(target)
        target.center_base[0] += .03
    elif changed == 'context':
        runner._held_recovery_context = None
    elif changed == 'aborted':
        runner._aborted = lambda: True
    else:
        runner._healthy = lambda: False
    bundle, why = plan(runner, target, temporary)
    assert bundle is None and why
    runner.skill.place_held.assert_not_called()


def test_transfer_cannot_move_an_occupied_fixed_point_to_another_center():
    runner, target, _grasp, temporary = transfer_scene()
    obstacle = Detection('green', 1., center_base=temporary.center + [0., 0., .05],
                         length=.05, width=.05, height=.05, top_z=.05, yaw=0.)
    bundle, why = plan(runner, target, temporary, [obstacle])
    assert bundle is None and why == 'no_clear_supported_release'
    runner.skill.place_held.assert_not_called()


def test_transfer_requires_full_footprint_support():
    runner, target, _grasp, temporary = transfer_scene()
    support = Zone('support', temporary.center, (.02, .02)).corners_xy()
    bundle, why = runner.plan_held_transfer(target, [], temporary, support, exact_xy=True)
    assert bundle is None and why == 'no_clear_supported_release'
    runner.skill.place_held.assert_not_called()


def test_transfer_requires_clear_open_fingers_even_when_object_footprint_fits():
    runner, target, _grasp, temporary = transfer_scene()
    obstacles = [Detection(
        'unknown', 1., center_base=temporary.center + [.049 * np.cos(angle),
                                                      .049 * np.sin(angle), .05],
        length=.012, width=.012, height=.05, top_z=.05, yaw=0.)
        for angle in np.linspace(0., 2 * np.pi, 8, endpoint=False)]
    # The small surrounding objects do not overlap the target's footprint,
    # but leave no direction in which its release fingers can safely open.
    bundle, why = plan(runner, target, temporary, obstacles)
    assert bundle is None and why == 'no_clear_supported_release'
    runner.skill.place_held.assert_not_called()


def test_transfer_cannot_offer_an_unreachable_release_pose():
    runner, target, _grasp, temporary = transfer_scene()
    runner.arm.check_reachable = Mock(side_effect=lambda pose:
        (False, 'IK rejected') if np.linalg.norm(pose[:2, 3] - temporary.center[:2]) < .03
        else (True, ''))
    bundle, why = plan(runner, target, temporary)
    assert bundle is None and why
    assert runner.arm.check_reachable.called
    runner.skill.place_held.assert_not_called()


def test_transfer_cannot_reanchor_saved_grip_for_ik(monkeypatch):
    runner, target, _grasp, temporary = transfer_scene()
    original = runner._held_recovery_context['grasp'].T_grasp.copy()

    def reanchor(_arm, gp, _pp, *_args, **_kwargs):
        gp.T_grasp[0, 3] += .001
        return True, '', {}

    monkeypatch.setattr('cube_task.task.reach.adjust_plan_poses', reanchor)
    bundle, why = plan(runner, target, temporary)
    assert bundle is None and why
    np.testing.assert_array_equal(runner._held_recovery_context['grasp'].T_grasp, original)
    runner.skill.place_held.assert_not_called()


@pytest.mark.parametrize('changed', ['width', 'grip', 'context', 'aborted', 'unhealthy'])
def test_transfer_execution_rechecks_context_grip_feedback_and_abort(changed):
    runner, target, _grasp, temporary = transfer_scene()
    bundle, why = plan(runner, target, temporary)
    assert bundle is not None, why
    if changed == 'width':
        runner.skill._read_width.return_value = .07
    elif changed == 'grip':
        bundle['grasp'].T_grasp[0, 3] += .001
    elif changed == 'context':
        runner._held_recovery_context = copy.deepcopy(runner._held_recovery_context)
    elif changed == 'aborted':
        runner._aborted = lambda: True
    else:
        runner._healthy = lambda: False
    result, why = runner.execute_held_transfer(bundle)
    assert result is None and why
    runner.skill.place_held.assert_not_called()


def test_transfer_executes_only_place_held_and_expires_context_after_release():
    runner, target, _grasp, temporary = transfer_scene()
    bundle, why = plan(runner, target, temporary)
    assert bundle is not None, why
    result, why = runner.execute_held_transfer(bundle)
    assert result.ok and not why
    executed_grip, executed_place, width = runner.skill.place_held.call_args.args
    np.testing.assert_array_equal(executed_grip.T_grasp, bundle['grasp'].T_grasp)
    assert executed_place.zone == temporary.name and width == .05
    assert runner._held_recovery_context is None
    assert not runner._placed_labels  # Intermediate success is not original-goal success.
    replay, why = runner.execute_held_transfer(bundle)
    assert replay is None and why
    assert runner.skill.place_held.call_count == 1


def test_failed_transfer_keeps_the_held_anchor_and_strict_temporary_destination():
    runner, target, _grasp, temporary = transfer_scene()
    runner.skill.place_held.return_value = PickResult(
        False, stage='transit', error='rejected', grasp_width=.05,
        detail={'holding': True, 'steps': []})
    bundle, why = plan(runner, target, temporary)
    assert bundle is not None, why
    result, why = runner.execute_held_transfer(bundle)
    assert not result.ok and not why
    context = runner._held_recovery_context
    assert context['zone'] == temporary.name and context['exact_xy']
    np.testing.assert_array_equal(context['grasp'].T_grasp, bundle['grasp'].T_grasp)
    np.testing.assert_array_equal(context['det'].center_base, target.center_base)
    np.testing.assert_array_equal(context['placement_polygon'], temporary.corners_xy())


def test_existing_same_zone_retry_refuses_ik_that_reanchors_held_grip(monkeypatch):
    runner, target, _grasp, _temporary = transfer_scene()

    def reanchor(_arm, gp, _pp, *_args):
        gp.T_grasp[0, 3] += .001
        return True, '', {}

    monkeypatch.setattr('cube_task.task.reach.adjust_plan_poses', reanchor)
    result, why = runner.retry_held_placement(target, 'zone_red')
    assert result is None and 'held grasp transform' in why
    runner.skill.place_held.assert_not_called()
