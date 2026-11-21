"""Convex footprint separation with reusable edge axes and projections."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PreparedPolygon(NamedTuple):
    points: np.ndarray
    axes: np.ndarray
    low: np.ndarray
    high: np.ndarray


def prepare_polygon(points):
    """Own the polygon so cached axes cannot outlive a caller's mutation."""
    points = np.array(points, dtype=float, copy=True)
    edges = np.roll(points, -1, axis=0) - points
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    axes /= np.maximum(np.linalg.norm(axes, axis=1), 1e-12)[:, None]
    projections = points @ axes.T
    low, high = projections.min(axis=0), projections.max(axis=0)
    for array in (points, axes, low, high):
        array.setflags(write=False)
    return PreparedPolygon(points, axes, low, high)


def polygons_overlap(a, b, clearance=.002):
    """Keep the original SAT rule: a separating gap must reach 2 mm.

    Each polygon supplies every edge axis, including zero-length edges. Only
    invariant projections are cached; no clearance or geometry is approximated.
    """
    b_on_a = b.points @ a.axes.T
    if np.any(np.maximum(a.low, b_on_a.min(axis=0)) >=
              np.minimum(a.high, b_on_a.max(axis=0)) + clearance):
        return False
    a_on_b = a.points @ b.axes.T
    return not np.any(np.maximum(b.low, a_on_b.min(axis=0)) >=
                      np.minimum(b.high, a_on_b.max(axis=0)) + clearance)
