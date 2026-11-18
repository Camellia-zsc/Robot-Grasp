"""A bounded placement search covers the region before refining tight gaps."""
import math

import numpy as np
import pytest

from cube_task.core.types import Zone
from cube_task.perception.grasp import PlacePlanner
from test_task_fakes import make_cfg


def setup_search(yaw=0.):
    planner = PlacePlanner(make_cfg(robot__gripper__max_width=.1))
    zone = Zone('pick_area', np.array([.26, 0., 0.]), (.28, .50), yaw=yaw)
    shape = dict(length=.04, width=.03, grip_width=.03, grip_z=.015,
                 elongated=False)
    return planner, zone, shape


@pytest.mark.parametrize('yaw', [0., math.pi / 5])
def test_bounded_search_reaches_distant_open_support_before_dense_center_refinement(yaw):
    planner, zone, shape = setup_search(yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    visited = []

    def supported(x, y, closing):
        visited.append((x, y))
        dx, dy = x - .26, y
        local = (c * dx + s * dy, -s * dx + c * dy)
        # The central band is occupied/unobserved. The region has a broad
        # usable patch farther away, as in the apple incident.
        return abs(local[0]) < .04 and .145 < local[1] < .195

    result = planner.plan(zone, .015, .18, held=shape, strict=True,
        pose_filter=supported, cancelled=lambda: len(visited) >= 800)
    assert result.feasible, (result.reason, len(visited))
    assert len(visited) < 800
    assert supported(*planner.last_info['xy'], planner.last_info['closing_angle'])


def test_dense_fallback_still_finds_a_gap_between_coarse_samples():
    planner, zone, shape = setup_search()
    shape['_required_closing'] = .3
    visited = []

    def narrow(x, y, closing):
        visited.append((x, y, closing))
        return abs(x - .295) < 1e-8 and abs(y - .115) < 1e-8

    result = planner.plan(zone, .015, .18, held=shape, strict=True, pose_filter=narrow)
    assert result.feasible
    assert planner.last_info['xy'] == pytest.approx([.295, .115])
    assert all(closing == .3 for _, _, closing in visited)


def test_small_support_keeps_nearby_fine_slots_ahead_of_distant_coarse_slots():
    planner, _zone, shape = setup_search()
    zone = Zone('small_support', np.array([.26, 0., 0.]), (.20, .20))
    shape['_required_closing'] = .3
    result = planner.plan(zone, .015, .18, held=shape, strict=True,
        pose_filter=lambda x, y, closing: math.hypot(x-.26, y) > 1e-8)
    assert result.feasible
    # Small colour regions need nearby alternatives for tight finger clearance;
    # spreading their first attempts by 40 mm can skip a reachable narrow slot.
    assert np.linalg.norm(np.asarray(planner.last_info['xy']) - zone.center[:2]) == pytest.approx(.005)


def test_exact_release_rejects_center_without_trying_preferred_or_other_slots():
    planner, zone, shape = setup_search()
    visited = []

    def only_far_slot(x, y, closing):
        visited.append((x, y))
        return y > .14

    result = planner.plan(zone, .015, .18, held=shape, strict=True,
        fixed_xy=True, preferred_xy=([.26, .16],), pose_filter=only_far_slot)
    assert not result.feasible
    assert visited
    assert all(np.allclose(xy, zone.center[:2]) for xy in visited)


def test_preferred_then_center_are_preserved_and_shared_grid_points_are_not_repeated():
    planner, zone, shape = setup_search()
    shape['_required_closing'] = .3
    visited = []

    def rejected(x, y, closing):
        visited.append((round(x, 9), round(y, 9)))
        return False

    result = planner.plan(zone, .015, .18, held=shape, strict=True,
        preferred_xy=([.26, .16], [.26, .16]), pose_filter=rejected)
    assert not result.feasible
    assert visited[:2] == [(.26, .16), (.26, 0.)]
    assert len(visited) == len(set(visited))
