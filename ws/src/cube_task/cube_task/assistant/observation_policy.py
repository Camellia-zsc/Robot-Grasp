"""Small observation decisions based solely on measured camera geometry.

These helpers do not move the arm or inspect the simulation. Cached objects
outside a new image remain useful as obstacles. A grasp needs current source
evidence, supplied by a new image or an individually monitored unchanged body.
Replacing only a view's detections is not
enough: a moved object may also survive in a different, older view.
"""
from __future__ import annotations

import math

import numpy as np

from ..perception.geometry import project_points


def _xy(det):
    try:
        xy = np.asarray(det.center_base, float)[:2]
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    return xy if xy.shape == (2,) and np.all(np.isfinite(xy)) else None


def _same_layer(a, b):
    """A measured upper body does not replace its differently named support."""
    if a.label == b.label:
        return True
    try:
        atop, btop = float(a.top_z), float(b.top_z)
        abottom, bbottom = atop-float(a.height), btop-float(b.height)
        if np.all(np.isfinite([atop, btop, abottom, bbottom])):
            # Touching stack surfaces can form one taller silhouette whose
            # inferred bottom is the tabletop. It cannot disprove a complete
            # known lower body merely by occupying the same projected XY.
            # Actual background depth or explicit motion still removes it.
            for lower, higher in ((a, b), (b, a)):
                if (lower.valid and lower.extra.get('catalog_complete') is True
                        and float(higher.top_z)-float(lower.top_z) > .02
                        and float(higher.height)-float(lower.height) > .02):
                    return False
            return not (bbottom >= atop-.005 or abottom >= btop-.005)
    except (TypeError, ValueError):
        pass
    return True


def rank_views(cameras, intrinsics, target_xy, preferred=None, half_extent=.04,
               target_z=.04, required_targets=()):
    """Rank calibrated views by complete target coverage, then image centrality.

    ``preferred`` is the last measured target view. It wins among cameras that
    cover the target, but cannot outrank a covering camera when it sees none.
    ``required_targets`` adds (XY, half-extent, height) measurements, such as
    a stacking base; a camera covering the whole group outranks a partial view.
    Return all indices so callers may search until a reliable target is found.
    """
    xy = np.asarray(target_xy, float)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError('target_xy must contain two finite coordinates')
    half_extent = max(0., float(half_extent))
    points = np.array([[xy[0] + dx, xy[1] + dy, float(target_z)]
                       for dx, dy in ((0., 0.), (-half_extent, -half_extent),
                                      (-half_extent, half_extent),
                                      (half_extent, -half_extent),
                                      (half_extent, half_extent))])
    for other_xy, other_extent, other_z in required_targets:
        point = np.asarray(other_xy, float)
        extent = float(other_extent)
        if (point.shape != (2,) or not np.all(np.isfinite(point))
                or not math.isfinite(extent) or extent < 0 or not math.isfinite(float(other_z))):
            raise ValueError('required camera targets need finite measured geometry')
        points = np.vstack([points, [[point[0] + dx, point[1] + dy, float(other_z)]
            for dx, dy in ((0., 0.), (-extent, -extent), (-extent, extent),
                           (extent, -extent), (extent, extent))]])
    scores = []
    w, h = int(intrinsics.width), int(intrinsics.height)
    for view, T in enumerate(cameras):
        uv = project_points(points, T, intrinsics)
        inside = (np.all(np.isfinite(uv), axis=1) & (uv[:, 0] >= 8)
                  & (uv[:, 0] <= w - 9) & (uv[:, 1] >= 8) & (uv[:, 1] <= h - 9))
        center = uv[0]
        distance = (float(np.linalg.norm((center - [w / 2, h / 2]) / [w, h]))
                    if np.all(np.isfinite(center)) else math.inf)
        scores.append((not bool(np.all(inside)), not bool(inside[0]),
                       view != preferred, distance, view))
    return [score[-1] for score in sorted(scores)]


def unique_target(detections, label, reference_xy=None, max_distance=.065, partial=None):
    """Find one valid measured target, refusing ambiguous same-class matches.

    With a reference position only candidates in its neighbourhood qualify.
    A distant sole candidate must be explicitly rebound by the caller after
    a search; silently selecting it could swap two objects of the same class.
    """
    candidates = [d for d in detections
                  if d.label == label and d.valid and _xy(d) is not None
                  and not (partial is not None and partial(d))]
    if reference_xy is not None:
        point = np.asarray(reference_xy, float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError('reference_xy must contain two finite coordinates')
        candidates = [d for d in candidates
                      if np.linalg.norm(_xy(d) - point) <= max_distance]
    return candidates[0] if len(candidates) == 1 else None


def merge_local_detections(previous, fresh, visible, association_dist=.04,
                           forget=(), singleton_labels=(), preserve_incomplete=False):
    """Replace measured visible space; preserve the unobserved cached scene.

    ``fresh`` must already be consolidated across this local scan. ``visible``
    confirms image and depth coverage, e.g. :func:`measurement_visible`. Old
    objects in that space disappear even when they produced no new detection.
    ``forget`` contains previous detections deliberately picked from that space.
    A close new measurement supersedes old geometry regardless of label; class
    disagreement must not leave two selectable versions of one silhouette.
    ``singleton_labels`` may explicitly rebind a one-per-class catalog object
    after a complete fresh camera measurement, even when it moved beyond the
    ordinary association radius. Ambiguous same-class fresh instances never
    trigger this override. A fresh ungraspable object still replaces its ghost.
    With unchanged monitored physics, ``preserve_incomplete`` can retain a
    complete old source outside full image coverage beside a new crop. It is
    still subject to source invalidation and contact checks before grasping.
    Inputs are returned by reference and are never mutated.
    """
    fresh = list(fresh)
    rebound = {label for label in singleton_labels
               if sum(d.label == label and _xy(d) is not None for d in fresh) == 1}
    forgotten_xy = [(d.label, p) for d in forget if (p := _xy(d)) is not None]
    kept = []
    for det in previous:
        xy = _xy(det)
        if xy is None:
            continue
        if det.label in rebound:
            continue
        if any(det.label == label and np.linalg.norm(xy - p) <= association_dist
               for label, p in forgotten_xy):
            continue
        nearby = [d for d in fresh if _xy(d) is not None
                  and np.linalg.norm(xy-_xy(d)) <= association_dist
                  and _same_layer(det, d)]
        complete_nearby = any(d.valid and d.extra.get('catalog_complete', True)
                              for d in nearby)
        if (preserve_incomplete and det.valid and nearby
                and not complete_nearby):
            # Seeing the old footprint is not proof that a cropped new
            # segmentation (whose class can also change at a crop) is a
            # better complete body measurement. Callers
            # still invalidate any body whose physical pose has changed.
            kept.append(det)
            continue
        if visible(det):
            continue
        if nearby:
            continue
        kept.append(det)
    return kept + fresh


def measurement_visible(det, frame, T_base_camera, margin_px=8,
                        occlusion_margin=.015):
    """Whether a new RGB-D image can disprove an old object's continued presence.

    Require the full old footprint and its top samples inside the image, finite
    depth, and no measured surface substantially nearer than those samples.
    Border crops, holes and foreground occlusion leave old obstacles cached.
    A retained obstacle is not a freshly confirmed target.
    """
    xy = _xy(det)
    if xy is None or frame is None or frame.depth is None:
        return False
    try:
        length, width = float(det.length), float(det.width)
        top, height = float(det.top_z), float(det.height)
        yaw = float(det.yaw or 0.)
        if (not np.all(np.isfinite([length, width, top, height, yaw]))
                or min(length, width, height) <= 0.):
            return False
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        offsets = np.array([[dx * length / 2, dy * width / 2]
                            for dx in (-1., 0., 1.) for dy in (-1., 0., 1.)])
        footprint = offsets @ R.T + xy
        points = np.vstack([np.column_stack([footprint, np.full(9, z)])
                            for z in (top, top - height)])
        intr = frame.intrinsics
        uv = project_points(points, T_base_camera, intr)
        if (not np.all(np.isfinite(uv)) or np.any(uv[:, 0] < margin_px)
                or np.any(uv[:, 0] > intr.width - 1 - margin_px)
                or np.any(uv[:, 1] < margin_px)
                or np.any(uv[:, 1] > intr.height - 1 - margin_px)):
            return False
        # Bottom projections prove image coverage; top projections test whether
        # nearer surfaces conceal the previous object. Bottom is naturally
        # occluded by the object's own surface and must not be used for that.
        pixels = np.rint(uv[:9]).astype(int)
        observed = np.asarray(frame.depth)[pixels[:, 1], pixels[:, 0]]
        T_cb = np.linalg.inv(np.asarray(T_base_camera, float))
        expected = points[:9] @ T_cb[2, :3] + T_cb[2, 3]
        return bool(np.all(np.isfinite(observed)) and np.all(observed > 0)
                    and np.all(observed >= expected - occlusion_margin))
    except (TypeError, ValueError, AttributeError, IndexError, np.linalg.LinAlgError):
        return False


def measurement_absent(det, frame, T_base_camera):
    """Missing segmentation is not absence: old surface rays must see behind it.

    Use actual previously measured upper-body points, rather than empty box
    corners. Occluded rays, missing depth and an object still at the old depth
    all retain the previous obstacle. Explicit motion invalidation is separate.
    """
    if not measurement_visible(det, frame, T_base_camera):
        return False
    try:
        points = np.asarray(det.extra.get('_placement_points_base',
                            det.extra.get('_points_base')), float)
        if points.ndim != 2 or points.shape[1:] != (3,):
            return False
        points = points[np.all(np.isfinite(points), axis=1)]
        points = points[points[:, 2] >= float(det.top_z)-float(det.height)*.5]
        if len(points) < 8:
            return False
        points = points[::max(1, len(points)//64)][:64]
        # Previously measured side faces can lie just outside a body after a
        # millimetre-scale seating change. Also require the old volume's core
        # to be empty; edge rays alone must not erase a still-present support.
        c, s = math.cos(float(det.yaw or 0)), math.sin(float(det.yaw or 0))
        offsets = np.array([[x*float(det.length), y*float(det.width)]
                            for x in (-.3,-.15,0.,.15,.3)
                            for y in (-.3,-.15,0.,.15,.3)])
        xy = offsets @ np.array([[c,s],[-s,c]]) + np.asarray(det.center_base[:2])
        core = np.column_stack([xy, np.full(len(xy), float(det.top_z)-float(det.height)*.5)])
        points = np.vstack([points, core])
        uv = project_points(points, T_base_camera, frame.intrinsics)
        intr = frame.intrinsics
        if (not np.all(np.isfinite(uv)) or np.any(uv < 0)
                or np.any(uv[:,0] > intr.width-1) or np.any(uv[:,1] > intr.height-1)):
            return False
        pixels = np.rint(uv).astype(int)
        observed = np.asarray(frame.depth)[pixels[:,1], pixels[:,0]]
        T_cb = np.linalg.inv(np.asarray(T_base_camera, float))
        expected = points @ T_cb[2,:3] + T_cb[2,3]
        return bool(np.all(np.isfinite(observed)) and np.all(expected > 0)
                    and np.all(observed > expected+.015))
    except (TypeError, ValueError, AttributeError, IndexError, np.linalg.LinAlgError):
        return False
