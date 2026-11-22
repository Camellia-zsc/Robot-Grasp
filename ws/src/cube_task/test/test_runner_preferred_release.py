"""An existing measured landing is a search preference, never release authority."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.core.transforms import rot_y, top_down_T
from cube_task.core.types import Detection


def edge_scene():
    from test_runner_local_release import scene
    runner, target, grasp, previous = scene()
    for pose, z in ((previous.T_pre, .2), (previous.T_release, .03), (previous.T_retreat, .12)):
        pose[:] = top_down_T(.34, -.06, z, 0.)
    return runner, target, grasp, previous


def blocker(xy):
    return Detection('green', 1., center_base=np.r_[xy, .05], top_z=.05,
                     height=.05, length=.05, width=.05, yaw=0.)


def test_existing_edge_slot_is_checked_before_a_blocked_center_or_the_grid(monkeypatch):
    runner, target, grasp, previous = edge_scene()
    visited = []
    original = runner._strict_place_filter

    def factory(*args, **kwargs):
        allowed = original(*args, **kwargs)

        def checked(x, y, closing, **options):
            visited.append((x, y))
            return allowed(x, y, closing, **options)

        return checked

    monkeypatch.setattr(runner, '_strict_place_filter', factory)
    before_zone = copy.deepcopy(runner.zones['zone_red'])
    before_grasp = grasp.T_grasp.copy()
    updated, why = runner.replan_release(target, grasp, previous, .05,
        [target, blocker([.4, 0.])], max_attempts=1)
    assert updated is not None, why
    np.testing.assert_allclose(runner.placer.last_info['xy'], [.34, -.06])
    assert visited and len(visited) <= 3
    np.testing.assert_allclose(visited, np.tile([.34, -.06], (len(visited), 1)))
    np.testing.assert_array_equal(grasp.T_grasp, before_grasp)
    np.testing.assert_array_equal(runner.zones['zone_red'].center, before_zone.center)
    np.testing.assert_array_equal(runner.zones['zone_red'].corners_xy(), before_zone.corners_xy())


def test_new_obstacle_at_the_preferred_slot_rejects_it_and_falls_back():
    runner, target, grasp, previous = edge_scene()
    updated, why = runner.replan_release(target, grasp, previous, .05,
        [target, blocker([.34, -.06])])
    assert updated is not None, why
    assert np.linalg.norm(np.asarray(runner.placer.last_info['xy']) - [.34, -.06]) > .05


def test_unseen_preferred_slot_cannot_bypass_visibility_and_uses_a_visible_alternative():
    runner, target, grasp, previous = edge_scene()
    checks = []

    def visible(x, y, closing, **kwargs):
        checks.append(((x, y), bool(kwargs.get('gripper_pose') is not None)))
        return not np.allclose([x, y], [.34, -.06], atol=1e-7)

    updated, why = runner.replan_release(target, grasp, previous, .05, [target],
        observation_filter=visible)
    assert updated is not None, why
    assert np.allclose(checks[0][0], [.34, -.06])
    assert not np.allclose(runner.placer.last_info['xy'], [.34, -.06])
    assert any(final for _, final in checks)
    updated, why = runner.replan_release(target, grasp, previous, .05, [target],
        observation_filter=lambda *args, **kwargs: False)
    assert updated is None and why == 'no_clear_supported_release'


@pytest.mark.parametrize('failure', ['ik', 'final_geometry', 'anchor'])
def test_preference_still_runs_final_checks_and_advances_after_a_rejected_pose(monkeypatch, failure):
    runner, target, grasp, previous = edge_scene()
    visited = []
    original_final = runner._adjust_strict_release

    def adjusted(arm, gp, pp, *args, **kwargs):
        visited.append(pp.T_release.copy())
        if len(visited) == 1:
            if failure == 'ik':
                return False, 'unreachable', {}
            if failure == 'anchor':
                gp.T_grasp[0, 3] += .01
        return True, '', {}

    def final(*args, **kwargs):
        if failure == 'final_geometry' and len(visited) == 1:
            return False, 'final leaned fingers blocked'
        return original_final(*args, **kwargs)

    monkeypatch.setattr('cube_task.task.reach.adjust_plan_poses', adjusted)
    monkeypatch.setattr(runner, '_adjust_strict_release', final)
    updated, why = runner.replan_release(target, grasp, previous, .05, [target], max_attempts=2)
    assert updated is not None, why
    assert len(visited) >= 2
    np.testing.assert_allclose(visited[0][:2, 3], [.34, -.06])
    assert not np.array_equal(updated.T_release, visited[0])


@pytest.mark.parametrize('cloud_kind', ['complete', 'asymmetric', 'short', 'nonfinite', 'missing'])
def test_preference_uses_measured_cloud_or_measured_box_through_the_actual_grasp(monkeypatch, cloud_kind):
    runner, target, grasp, previous = edge_scene()
    grasp.T_grasp = top_down_T(.188, .147, .015, 0.)
    tilt = math.radians(30.)
    previous.T_release[:3, :3] = rot_y(tilt) @ grasp.T_grasp[:3, :3]
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    cloud = np.repeat(signs * [.02, .015, .03] + [.215, .165, .03], 5, axis=0)
    if cloud_kind == 'asymmetric':
        cloud = np.repeat([[.19, .14, 0.], [.235, .14, .01],
                           [.20, .17, .055], [.215, .175, .03]], 10, axis=0)
    elif cloud_kind == 'short':
        cloud = cloud[:39]
    elif cloud_kind == 'nonfinite':
        cloud[0, 0] = np.nan
    if cloud_kind != 'missing':
        target.extra['_placement_points_base'] = cloud
    captured = []

    def plan(*args, **kwargs):
        captured.append(kwargs.get('preferred_xy', ()))
        return SimpleNamespace(feasible=False, reason='probe only')

    monkeypatch.setattr(runner.placer, 'plan', plan)
    updated, why = runner.replan_release(target, grasp, previous, .05, [target], max_attempts=1)
    assert updated is None and why == 'probe only'
    # These are measured offsets from the physical grasp, not world truth or
    # the old release TCP's XY. Lean also shifts the projected landing in X.
    dx, dy, dz = ({'complete': (.027, .018, .015),
                  'asymmetric': (.0245, .0105, -.010)}.get(cloud_kind, (.012, .003, .010)))
    expected = [.34 + dx * math.cos(tilt) + dz * math.sin(tilt), -.06 + dy]
    np.testing.assert_allclose([points[0] for points in captured], [expected])
    assert len(captured[0]) == 9  # measured anchor, then broad region alternatives


def test_preferred_xy_does_not_change_an_exact_destination():
    runner, target, grasp, previous = edge_scene()
    updated, why = runner.replan_release(target, grasp, previous, .05, [target], exact_xy=True)
    assert updated is not None, why
    np.testing.assert_allclose(runner.placer.last_info['xy'], [.4, 0.])


def test_planner_preference_is_optional_and_respects_avoid_and_region_boundary():
    runner, target, grasp, previous = edge_scene()
    search, why = runner._release_search_geometry(target, grasp, previous, .05, [target])
    assert search is not None, why
    kwargs = dict(held=search['shape'], strict=True)
    zone = search['zone']
    for preferred, avoid, expected in (((), (), [.4, 0.]),
            (([.34, -.06],), (), [.34, -.06]),
            (([.34, -.06],), ([.34, -.06],), [.4, 0.]),
            (([.9, .9], [math.nan, .1]), (), [.4, 0.])):
        plan = runner.placer.plan(zone, search['held'], search['transit'],
                                 preferred_xy=preferred, avoid=avoid, **kwargs)
        assert plan.feasible
        np.testing.assert_allclose(runner.placer.last_info['xy'], expected)


def test_a_preference_cannot_move_an_occupied_exact_destination():
    runner, target, grasp, previous = edge_scene()
    search, why = runner._release_search_geometry(target, grasp, previous, .05,
                                                  [target, blocker([.4, 0.])])
    assert search is not None, why
    result = runner.placer.plan(search['zone'], search['held'], search['transit'],
        held=search['shape'], occupied=search['occupied'], strict=True, fixed_xy=True,
        preferred_xy=([.34, -.06],))
    assert not result.feasible and result.reason == 'no_clear_supported_release'


def test_ordinary_non_strict_plans_ignore_the_preference():
    runner, target, grasp, previous = edge_scene()
    search, why = runner._release_search_geometry(target, grasp, previous, .05, [target])
    assert search is not None, why
    kwargs = dict(held=search['shape'], strict=False)
    original = runner.placer.plan(search['zone'], search['held'], search['transit'], **kwargs)
    preferred = runner.placer.plan(search['zone'], search['held'], search['transit'],
                                  preferred_xy=([.34, -.06],), **kwargs)
    np.testing.assert_array_equal(preferred.T_release, original.T_release)
