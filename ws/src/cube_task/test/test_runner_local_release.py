"""Stationary release replanning uses measured clutter and preserves the goal."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.core.config import load_config
from cube_task.core.transforms import top_down_T
from cube_task.core.types import Detection, GraspPlan, PickResult, Zone
from cube_task.perception.grasp import PlacePlanner
from cube_task.task.runner import TaskRunner


def scene():
    runner = object.__new__(TaskRunner)
    runner.cfg = load_config(use_overlays=False)
    runner.placer = PlacePlanner(runner.cfg)
    runner.arm = SimpleNamespace()
    runner.nominal = {'red': (.05, .05), 'green': (.05, .05)}
    runner.max_width = .1
    runner.skill = SimpleNamespace(empty_width=.003)
    runner.release_lower = {}
    runner.free_tilt = 20.
    runner.zones = {'zone_red': Zone('zone_red', np.array([.4, 0., 0.]), (.2, .2))}
    runner.class_to_zone = {'red': 'zone_red'}
    runner._aborted = lambda: False
    runner._healthy = lambda: True
    runner._floors = lambda *args: {}
    runner.events = []
    runner._event = lambda *args, **fields: runner.events.append(fields)
    target = Detection('red', 1., center_base=np.array([.2, .15, .05]), top_z=.05,
                       height=.05, length=.05, width=.05, yaw=0.)
    grasp = GraspPlan(target, top_down_T(.2, .15, .15, 0.),
                      top_down_T(.2, .15, .02, 0.), top_down_T(.2, .15, .2, 0.),
                      .06, 0., .05, 0.)
    place = runner.placer.plan(runner.zones['zone_red'], .02, .2,
                               held=dict(length=.05, width=.05, elongated=False,
                                         grip_width=.05, grip_z=.02),
                               strict=True)
    assert place.feasible
    return runner, target, grasp, place


def test_new_neighbor_at_old_landing_point_moves_release_to_clear_region_edge():
    runner, target, grasp, previous = scene()
    obstacle = Detection('green', 1., center_base=np.array([.4, 0., .05]), top_z=.05,
                         height=.05, length=.05, width=.05, yaw=0.)
    before_grasp, before_place = copy.deepcopy(grasp), copy.deepcopy(previous)
    updated, why = runner.replan_release(target, grasp, previous, .05, [target, obstacle])
    assert updated is not None, why
    assert np.linalg.norm(updated.T_release[:2, 3] - [.4, 0.]) > .05
    assert runner.events[-1]['action'] == 'replan_release'
    assert runner.events[-1]['changed']
    # This seam plans only; the callback decides when to move and copy the plan.
    np.testing.assert_array_equal(grasp.T_grasp, before_grasp.T_grasp)
    np.testing.assert_array_equal(previous.T_release, before_place.T_release)


def test_fixed_point_occupied_returns_no_plan_instead_of_another_center():
    runner, target, grasp, previous = scene()
    obstacle = Detection('green', 1., center_base=np.array([.4, 0., .05]), top_z=.05,
                         height=.05, length=.05, width=.05, yaw=0.)
    updated, why = runner.replan_release(target, grasp, previous, .05, [target, obstacle],
                                         placement_zone=runner.zones['zone_red'], exact_xy=True)
    assert updated is None and why == 'no_clear_supported_release'
    assert not runner.events


def test_region_final_tilt_failure_retries_another_center():
    runner, target, grasp, previous = scene()
    visited = []

    def final_check(det, gp, pp, neighbors, polygon, desired, held, **kwargs):
        visited.append(tuple(desired))
        return (False, 'leaned finger blocked') if len(visited) == 1 else (True, '')

    runner._adjust_strict_release = final_check
    updated, why = runner.replan_release(target, grasp, previous, .05, [target])
    assert updated is not None, why
    assert len(visited) == 2 and visited[0] != visited[1]


def test_interrupted_release_search_resumes_completed_rejections_in_same_measurement(monkeypatch):
    runner, target, grasp, previous = scene()
    visited, stopped = [], [False]
    state = {}

    def final_check(det, gp, pp, neighbors, polygon, desired, held, **kwargs):
        visited.append(tuple(desired))
        return False, 'leaned finger blocked'

    def progress(fields):
        # One complete rejection, then the next candidate is interrupted.
        if fields['attempt'] == 2:
            stopped[0] = True

    runner._adjust_strict_release = final_check
    updated, why = runner.replan_release(target, grasp, previous, .05, [target],
        cancelled=lambda: stopped[0], progress=progress, search_state=state)
    assert updated is None and why == 'placement_planning_budget_exceeded'
    assert visited == [(.4, 0.)]
    assert state['avoided'] == [(.4, 0.)]

    stopped[0] = False
    runner._adjust_strict_release = lambda *a, **k: (True, '')
    updated, why = runner.replan_release(target, grasp, previous, .05, [target],
        search_state=state)
    assert updated is not None, why
    assert tuple(runner.placer.last_info['xy']) != visited[0]


def test_interrupted_candidate_is_not_saved_as_a_completed_rejection(monkeypatch):
    runner, target, grasp, previous = scene()
    state, stopped = {}, [False]

    def interrupted_reach(*args, **kwargs):
        stopped[0] = True
        return False, 'aborted', {}

    monkeypatch.setattr('cube_task.task.reach.adjust_plan_poses', interrupted_reach)
    updated, why = runner.replan_release(target, grasp, previous, .05, [target],
        cancelled=lambda: stopped[0], search_state=state)
    assert updated is None and why == 'placement_planning_budget_exceeded'
    assert not state.get('avoided') and not state.get('avoided_poses')


def test_point_final_tilt_failure_retries_angle_at_the_same_center():
    runner, target, grasp, previous = scene()
    visited = []

    def final_check(det, gp, pp, neighbors, polygon, desired, held, **kwargs):
        visited.append((tuple(desired), pp.T_release[:3, :3].copy()))
        return (False, 'leaned finger blocked') if len(visited) == 1 else (True, '')

    runner._adjust_strict_release = final_check
    updated, why = runner.replan_release(target, grasp, previous, .05, [target], exact_xy=True)
    assert updated is not None, why
    assert len(visited) == 2 and visited[0][0] == visited[1][0] == (.4, 0.)
    assert not np.allclose(visited[0][1], visited[1][1])


@pytest.mark.parametrize('width', [None, float('nan'), 0., .2])
def test_invalid_held_feedback_refuses_without_planning(width):
    runner, target, grasp, previous = scene()
    updated, why = runner.replan_release(target, grasp, previous, width, [target])
    assert updated is None and 'width' in why
    assert not runner.events


def test_unknown_neighbor_geometry_cannot_prove_free_space():
    runner, target, grasp, previous = scene()
    obstacle = Detection('unknown', .2, center_base=np.array([.4, 0., .05]), valid=False)
    updated, why = runner.replan_release(target, grasp, previous, .05, [target, obstacle])
    assert updated is None and 'neighbor footprint' in why


def test_replanning_preserves_offset_grasp_object_center_at_requested_point():
    runner, target, grasp, previous = scene()
    grasp.T_grasp[0, 3] += .012  # TCP is away from the object's footprint center.
    updated, why = runner.replan_release(target, grasp, previous, .055, [target], exact_xy=True)
    assert updated is not None, why
    R = updated.T_release[:3, :3] @ grasp.T_grasp[:3, :3].T
    source_center = np.r_[target.center_base[:2], .025]
    released_center = updated.T_release[:3, 3] + R @ (source_center - grasp.T_grasp[:3, 3])
    np.testing.assert_allclose(released_center[:2], [.4, 0.], atol=.0002)
    assert updated.open_width >= .055


def test_restore_yaw_restricts_the_release_to_the_original_object_orientation():
    runner, target, grasp, previous = scene()
    target.yaw = .3
    target.extra['_restore_yaw'] = .9
    updated, why = runner.replan_release(target, grasp, previous, .05, [target], exact_xy=True)
    assert updated is not None, why
    closing = np.arctan2(updated.T_release[1, 1], updated.T_release[0, 1])
    assert closing == pytest.approx(.6)


@pytest.mark.parametrize('error', ['scene_changed', 'placement_changed'])
def test_scene_change_while_holding_retains_a_bounded_recovery_context(error):
    runner, target, grasp, previous = scene()
    result = PickResult(False, stage='inspect_place', error=error, grasp_width=.05,
                        detail={'holding': True, 'steps': []})
    runner._save_held_recovery(target, grasp, previous, result)
    assert runner._held_recovery_context['place'].zone == previous.zone
    result.error = 'arbitrary_error'
    runner._save_held_recovery(target, grasp, previous, result)
    assert runner._held_recovery_context is None


def test_missing_fresh_visibility_does_not_allow_blind_release():
    runner, target, grasp, previous = scene()
    updated, why = runner.replan_release(target, grasp, previous, .05, [target],
                                         observation_filter=lambda *args, **kwargs: False)
    assert updated is None and why == 'no_clear_supported_release'


def test_stop_during_visibility_search_does_not_scan_the_remaining_grid():
    runner, target, grasp, previous = scene()
    stopped, calls = [], []
    runner._aborted = lambda: bool(stopped)

    def visibility(*args, **kwargs):
        calls.append(args)
        stopped.append(True)
        return False

    updated, why = runner.replan_release(target, grasp, previous, .05, [target],
                                         observation_filter=visibility)
    assert updated is None and why == 'aborted'
    assert len(calls) == 1
    assert not runner.events


def test_stop_during_reach_check_cannot_publish_a_release(monkeypatch):
    runner, target, grasp, previous = scene()
    stopped = []
    runner._aborted = lambda: bool(stopped)

    def reach(*args, **kwargs):
        stopped.append(True)
        return True, '', {}

    monkeypatch.setattr('cube_task.task.reach.adjust_plan_poses', reach)
    updated, why = runner.replan_release(target, grasp, previous, .05, [target])
    assert updated is None and why == 'aborted'
    assert not runner.events


def test_held_release_checks_only_future_poses_and_never_replans_the_grasp():
    from cube_task.task.reach import adjust_plan_poses
    runner, target, grasp, previous = scene()
    before = copy.deepcopy(grasp)
    checked = []

    def reachable(pose):
        checked.append(pose.copy())
        # The old source is no longer reachable; the grasp has already happened.
        return (bool(pose[0, 3] > .3), 'old grasp is not a future motion')

    arm = SimpleNamespace(check_reachable=reachable)
    ok, why, tilts = adjust_plan_poses(arm, grasp, previous, held_only=True)
    assert ok, why
    assert set(tilts) == {'release', 'pre_place', 'retreat'}
    assert len(checked) == 3
    for attr in ('T_grasp', 'T_pre', 'T_lift'):
        np.testing.assert_array_equal(getattr(grasp, attr), getattr(before, attr))
