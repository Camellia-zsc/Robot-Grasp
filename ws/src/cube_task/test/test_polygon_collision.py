"""Optimized footprint SAT must preserve the former collision decisions."""
import cv2
import numpy as np
import pytest

from cube_task.perception.polygon_collision import prepare_polygon, polygons_overlap


def previous_overlap(a, b):
    # Frozen production predicate, retained for differential validation.
    for polygon in (a, b):
        for edge in np.roll(polygon, -1, axis=0) - polygon:
            normal = np.array([-edge[1], edge[0]])
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            pa, pb = a @ normal, b @ normal
            if max(pa.min(), pb.min()) >= min(pa.max(), pb.max()) + .002:
                return False
    return True


def test_same_collision_decisions_for_random_convex_footprints():
    random = np.random.default_rng(741)
    for _ in range(800):
        a = cv2.convexHull(random.normal(0, .025, (10, 2)).astype(np.float32)).reshape(-1, 2).astype(float)
        b = cv2.convexHull(random.normal(0, .025, (10, 2)).astype(np.float32)).reshape(-1, 2).astype(float)
        b += random.uniform(-.10, .10, 2)
        expected = previous_overlap(a, b)
        actual = polygons_overlap(prepare_polygon(a), prepare_polygon(b))
        assert actual == expected
        assert polygons_overlap(prepare_polygon(b), prepare_polygon(a)) == expected


@pytest.mark.parametrize('gap', [-.01, 0., .001, .002, .002001, .01])
def test_original_two_millimetre_separation_boundary_and_duplicate_vertices(gap):
    a = np.array([[0., 0.], [.05, 0.], [.05, .05], [0., .05], [0., .05]])
    b = a + [.05 + gap, 0.]
    assert polygons_overlap(prepare_polygon(a), prepare_polygon(b)) == previous_overlap(a, b)


def test_prepared_obstacle_is_an_owned_readonly_measurement():
    points = np.array([[0., 0.], [.05, 0.], [.05, .05], [0., .05]])
    obstacle = prepare_polygon(points)
    points[:] += 1.
    assert obstacle.points.max() == pytest.approx(.05)
    with pytest.raises(ValueError):
        obstacle.points[0, 0] = 1.
