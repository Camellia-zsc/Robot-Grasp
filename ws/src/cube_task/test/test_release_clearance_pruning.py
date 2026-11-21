"""Keep the original segment math and prune only proven capsule collisions."""
import math

import numpy as np
import pytest

from cube_task.core.config import load_config
from cube_task.core.types import Zone
from cube_task.perception.grasp import PlacePlanner, _seg_dist


def original_numpy_segment_distance(p1, q1, p2, q2):
    """The previous implementation is the independent numerical oracle."""
    def pt_seg(p, a, b):
        d = b - a
        L2 = float(d @ d)
        t = 0.0 if L2 < 1e-18 else min(1.0, max(0.0, float((p - a) @ d) / L2))
        return float(np.linalg.norm(p - (a + t * d)))
    d1, d2, r = q1 - p1, q2 - p2, p2 - p1
    den = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(den) > 1e-15:
        t = float(r[0] * d2[1] - r[1] * d2[0]) / den
        u = float(r[0] * d1[1] - r[1] * d1[0]) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0
    return min(pt_seg(p1, p2, q2), pt_seg(q1, p2, q2),
               pt_seg(p2, p1, q1), pt_seg(q2, p1, q1))


@pytest.mark.parametrize('points,expected', [
    ([[0, 0], [0, 0], [3, 4], [3, 4]], 5.),
    ([[0, 0], [1, 0], [0, 1], [1, 1]], 1.),
    ([[0, 0], [1, 1], [0, 1], [1, 0]], 0.),
    ([[0, 0], [1, 0], [1, 0], [2, 0]], 0.),
    ([[0, 0], [1, 0], [2, 0], [3, 0]], 1.),
    ([[0, 0], [1, 0], [0, 1], [1, 1 + 1e-16]], 1.),
    ([[0, 0], [1, 0], [0, 1], [1, 1 + 2e-15]], 1.),
])
def test_segment_distance_crossings_degeneracy_and_near_parallel(points, expected):
    points = np.asarray(points, float)
    assert _seg_dist(*points) == pytest.approx(expected, abs=1e-15)
    assert _seg_dist(*points) == original_numpy_segment_distance(*points)


def test_segment_distance_matches_original_numpy_oracle_across_scales():
    rng = np.random.default_rng(2731)
    segments = rng.uniform(-.6, .6, (4000, 4, 2))
    segments[:100, 1] = segments[:100, 0]
    segments[100:200, 3] = segments[100:200, 2]
    # Preserve the 1e-18 squared-length and 1e-15 determinant decisions,
    # including tiny segments and nearly parallel positive/negative slopes.
    segments[200:400, 1] = (segments[200:400, 0]
                            + rng.uniform(-1e-9, 1e-9, (200, 2)))
    segments[400:600, 3] = (segments[400:600, 2]
                            + rng.uniform(-1e-8, 1e-8, (200, 2)))
    for points in segments:
        assert math.isclose(_seg_dist(*points), original_numpy_segment_distance(*points),
                            rel_tol=1e-14, abs_tol=1e-15)


def release_scene():
    planner = PlacePlanner(load_config(use_overlays=False))
    zone = Zone('test', np.array([.4, 0., 0.]), (.2, .2))
    held = dict(length=.05, width=.05, elongated=False, grip_width=.05, grip_z=.02)
    occupied = [(.4, 0., .05, .05, .05, .05, 0.)]
    return planner, zone, held, occupied


def test_exact_occupied_point_skips_expensive_pose_filter():
    planner, zone, held, occupied = release_scene()
    def expensive(*args):
        pytest.fail('a known capsule collision must be rejected before pose evidence')
    result = planner.plan(zone, .02, .2, held=held, occupied=occupied,
                          strict=True, fixed_xy=True, pose_filter=expensive)
    assert not result.feasible and result.reason == 'no_clear_supported_release'


def test_legal_preferred_points_keep_order_and_final_pose_filter_authority():
    planner, zone, held, occupied = release_scene()
    first, second = (.46, .06), (.34, .06)
    calls = []
    def measured_pose(x, y, closing):
        calls.append((x, y, closing))
        return (x, y) == second
    result = planner.plan(zone, .02, .2, held=held, occupied=occupied,
                          strict=True, preferred_xy=[(.4, 0.), first, second],
                          pose_filter=measured_pose)
    assert result.feasible
    np.testing.assert_array_equal(result.T_release[:2, 3], second)
    assert tuple(planner.last_info['xy']) == second
    assert calls and all((x, y) == first for x, y, _ in calls[:-1])
    assert calls[-1][:2] == second and calls[0][:2] == first
