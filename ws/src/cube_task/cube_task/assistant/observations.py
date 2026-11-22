"""Consolidate camera observations without consulting the simulated world.

A clipped silhouette can contain surfaces from several already observed objects.
Its fitted center is then too far away for the normal distance merge. Only an
invalid observation whose measured geometry is explained by other views may be
removed; incomplete, unsupported observations remain available for diagnosis.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..perception.geometry import points_in_polygon
from ..task.views import MERGE_DIST, merge_view_detections, view_of

_CELL = 0.002
_MARGIN = 0.003
_SURFACE_DISTANCE = 0.012
_SUPPORT = 0.98
_RESIDUAL_GAP = 0.006


class GoalVerification(dict):
    """Keep each check's camera identity for the subsequent physical audit."""

    def __init__(self, result, targets):
        super().__init__(result)
        self.targets = tuple(targets)


def _capture_of(det):
    # The transient camera slot can hold several poses during one local scan.
    # Retain its public view index for future aiming, but distinguish captures
    # when comparing independent geometry evidence within this scan.
    return view_of(det), (det.extra or {}).get('_capture_index')


def _prefer_pear_neck_views(merged, views, dist, partial):
    """A denser close view of the bulb can still hide this library pear's neck."""
    def geometry(det):
        try:
            values = np.asarray([det.height, det.length, det.width], dtype=float)
            xy = np.asarray(det.center_base, dtype=float)[:2]
        except (TypeError, ValueError, IndexError):
            return None
        if (xy.shape != (2,) or not np.all(np.isfinite(np.r_[values, xy]))
                or np.any(values <= 0)):
            return None
        return values, xy

    # The underlying merge guarantees finite base XYs for these anchors.
    centers = np.asarray([np.asarray(d.center_base, dtype=float)[:2] for d in merged])
    preferred = []
    for det in merged:
        base = geometry(det) if det.valid and det.label == 'pear' else None
        best, best_height = det, 0. if base is None else base[0][0]
        if base is not None:
            dims, xy = base
            for items in views.values():
                for candidate in items:
                    if (not candidate.valid or candidate.label != det.label
                            or _capture_of(candidate) == _capture_of(det)
                            or ((det.extra or {}).get('support_observed') is True
                                and (candidate.extra or {}).get('support_observed') is not True)
                            or (partial is not None and partial(candidate))):
                        continue
                    measured = geometry(candidate)
                    if measured is None:
                        continue
                    size, point = measured
                    # A view between two targets is not a unique association.
                    # Reusing it would publish the same measured object twice.
                    if np.count_nonzero(np.linalg.norm(centers - point, axis=1) <= dist) != 1:
                        continue
                    # Require one extra 5 mm profile bin, retaining the bulb's
                    # footprint. A tall cropped sliver is not a better view.
                    if (np.linalg.norm(point - xy) <= dist and size[0] >= dims[0] + .005
                            and size[1] * size[2] >= .9 * dims[1] * dims[2]
                            and size[0] > best_height):
                        best, best_height = candidate, size[0]
        preferred.append(best)
    return preferred


def _cloud(det):
    try:
        points = np.asarray((det.extra or {}).get('_points_base'), dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 40
            or not np.all(np.isfinite(points))):
        return None
    return np.ascontiguousarray(points)


def _uniform(points):
    # Equal spatial weight prevents a dense nearby surface from hiding a small
    # but distinct unknown object in the same image component.
    _, indices = np.unique(np.floor(points / _CELL).astype(np.int64),
                           axis=0, return_index=True)
    return np.ascontiguousarray(points[np.sort(indices)])


def _distances(points, surface):
    distance, _ = cv2.batchDistance(points, surface, cv2.CV_32F,
                                   normType=cv2.NORM_L2SQR, K=1)
    return distance[:, 0]


def _separate_residual(points, supported):
    """A small disconnected cluster is evidence, even below the 2% allowance."""
    residual = points[~supported]
    if len(residual) < 3:
        return False
    to_supported = _distances(residual, points[supported])
    adjacent = np.sum((residual[:, None] - residual[None, :]) ** 2, axis=2)
    todo = set(range(len(residual)))
    while todo:
        pending = [todo.pop()]
        component = []
        while pending:
            index = pending.pop()
            component.append(index)
            neighbors = {i for i in todo if adjacent[index, i] <= _RESIDUAL_GAP ** 2}
            todo.difference_update(neighbors)
            pending.extend(neighbors)
        if (len(component) >= 3
                and np.min(to_supported[component]) > _RESIDUAL_GAP ** 2):
            return True
    return False


def merge_observations(views, dist=MERGE_DIST, partial=None):
    """Return (objects, diagnostics), retaining all original per-view data.

    A 3 mm expanded measured hull/height envelope limits depth noise. Surface
    support additionally requires a measured point within 12 mm, allowing views
    of different sides without filling the empty interior of that envelope.
    98% of 2 mm spatial samples must be supported, with no separate residual
    cluster. Valid objects are never pruned; class counts and world coordinates
    are not inputs to the test.
    All views here are fresh captures from one dashboard scan.
    """
    merged = _prefer_pear_neck_views(
        merge_view_detections(views, dist=dist, partial=partial), views, dist, partial)
    anchors = []
    for detections in views.values():
        for det in detections:
            if (not det.valid or det.label == 'unknown' or view_of(det) is None
                    or (partial is not None and partial(det))):
                continue
            cloud = _cloud(det)
            if cloud is None:
                continue
            hull = cv2.convexHull(cloud[:, :2].copy()).reshape(-1, 2)
            if len(hull) >= 3:
                anchors.append((det, cloud, hull, cloud[:, 2].min(), cloud[:, 2].max()))
    kept, diagnostics = [], []
    for det in merged:
        cloud = None if det.valid or view_of(det) is None else _cloud(det)
        if cloud is None:
            kept.append(det)
            continue
        points = _uniform(cloud)
        supported = np.zeros(len(points), dtype=bool)
        sources = []
        for anchor, surface, hull, low, high in anchors:
            if _capture_of(anchor) == _capture_of(det):
                continue
            inside = ((points[:, 2] >= low - _MARGIN)
                      & (points[:, 2] <= high + _MARGIN)
                      & points_in_polygon(points[:, :2], hull, margin=_MARGIN))
            if inside.any():
                # The same observed object must supply BOTH the envelope and
                # nearby surface. Combining these tests across different
                # anchors would wrongly fill empty spaces between objects.
                explained = inside & (_distances(points, surface) <= _SURFACE_DISTANCE ** 2)
                if explained.any():
                    supported |= explained
                    sources.append(anchor)
        ratio = float(np.mean(supported))
        if ratio < _SUPPORT or _separate_residual(points, supported):
            kept.append(det)
            continue
        diagnostics.append(dict(det.summary(), view=view_of(det),
                                reason='supported_duplicate_or_mixed_observation',
                                support_ratio=round(ratio, 4),
                                supporting_views=sorted({view_of(a) for a in sources}),
                                supporting_labels=sorted({a.label for a in sources})))
    return kept, diagnostics
