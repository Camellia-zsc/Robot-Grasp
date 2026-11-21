"""Camera-point fits for a known spherical section's translation anchor.

The caller owns semantic identity, frame freshness and coordinate transforms.
No body orientation, bottom point or simulator pose is inferred here.
"""
from __future__ import annotations

import math

import numpy as np


_INLIER_DISTANCE = .0012
_MIN_POINTS = 80
_MAX_POINTS = 1500
_HYPOTHESIS_SPREAD = .0005


def _quality(points, center, radius):
    residual = np.linalg.norm(points-center, axis=1)-radius
    mask = np.abs(residual) < _INLIER_DISTANCE
    if int(mask.sum()) < _MIN_POINTS:
        return None
    delta = points[mask]-center
    normals = delta/np.maximum(np.linalg.norm(delta, axis=1)[:, None], 1e-12)
    eigen = np.linalg.eigvalsh(normals.T@normals)/len(normals)
    rmse = float(np.sqrt(np.mean(residual[mask]**2)))
    return dict(points=len(points), inliers=int(mask.sum()),
                inlier_fraction=float(mask.mean()), rmse=rmse,
                normal_eigenvalues=eigen.tolist(),
                normal_condition=float(eigen[-1]/max(eigen[0], 1e-12)),
                uncertainty_2sigma=float(2*rmse/math.sqrt(max(len(normals)*eigen[0], 1e-12))))


def _acceptable(quality):
    try:
        eigen = np.asarray(quality['normal_eigenvalues'], float)
        if eigen.shape != (3,):
            return False
        values = np.asarray([quality['inlier_fraction'], quality['rmse'],
                             *eigen, quality['normal_condition'],
                             quality['uncertainty_2sigma']], float)
        return bool(np.all(np.isfinite(values)) and .60 <= quality['inlier_fraction'] <= 1.
                    and 0 <= quality['rmse'] <= .0006 and min(eigen) >= .015
                    and 1. <= quality['normal_condition'] <= 80.
                    and 0 <= quality['uncertainty_2sigma'] <= .001)
    except (KeyError, TypeError, ValueError):
        return False


def _refine(points, center, radius):
    """Three-dimensional robust Gauss-Newton; radius is never optimized."""
    center = center.copy()
    for _ in range(3):
        mask = np.abs(np.linalg.norm(points-center, axis=1)-radius) < _INLIER_DISTANCE
        selected = points[mask]
        if len(selected) < _MIN_POINTS:
            return None
        for _ in range(20):
            delta = center-selected
            distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-12)
            residual = distance-radius
            jacobian = delta/distance[:, None]
            weight = (1+(residual/.0004)**2)**(-.25)
            try:
                step = np.linalg.lstsq(jacobian*weight[:, None], -residual*weight, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            length = float(np.linalg.norm(step))
            if not np.all(np.isfinite(step)):
                return None
            if length < 1e-10:
                break
            step *= min(1., radius/4/max(length, 1e-12))
            cost = np.sum(np.sqrt(1+(residual/.0004)**2)-1)
            for scale in (1., .5, .25, .125, .0625):
                candidate = center+scale*step
                updated = np.linalg.norm(selected-candidate, axis=1)-radius
                if np.sum(np.sqrt(1+(updated/.0004)**2)-1) <= cost:
                    center = candidate
                    break
            else:
                break
    return center


def _hypothesis(points, radius, prior, seed):
    random = np.random.default_rng(seed)
    best_center, best_count = prior.copy(), -1
    # Three measured points give at most two centres for a fixed radius.
    # Degenerate triangles and distant solutions cannot become candidates.
    candidates = [prior]
    for _ in range(300):
        a, b, c = points[random.choice(len(points), 3, replace=False)]
        u = b-a
        size = float(np.linalg.norm(u))
        if size < .1*radius:
            continue
        u /= size
        v = c-a
        vx = float(v@u)
        v -= vx*u
        height = float(np.linalg.norm(v))
        if height < .06*radius:
            continue
        v /= height
        x = size/2
        y = (float(np.sum((c-a)**2))/2-x*vx)/height
        if x*x+y*y > radius*radius:
            continue
        circle = a+x*u+y*v
        offset = math.sqrt(max(0., radius*radius-x*x-y*y))*np.cross(u, v)
        candidates.extend(center for center in (circle+offset, circle-offset)
                          if np.linalg.norm(center-prior) <= max(2.5*radius, .04))
    best_score = (-1, -math.inf)
    for center in candidates:
        distance = np.abs(np.linalg.norm(points-center, axis=1)-radius)
        mask = distance < _INLIER_DISTANCE
        score = (int(mask.sum()), -float(np.median(distance[mask])) if mask.any() else -math.inf)
        if score > best_score:
            best_center, best_count, best_score = center, int(mask.sum()), score
    return None if best_count < _MIN_POINTS else _refine(points, best_center, radius)


def fit_spherical_anchor(points, radius, prior_center=None):
    """Fit only a known-radius sphere centre from an already identified cloud.

    A source cloud should include the independently observed body surfaces,
    such as its fused placement cloud. Cropped necks and narrow surface patches
    fail the coverage gates. All returned distances use metres.
    """
    try:
        radius = float(radius)
        data = np.asarray(points, float)
        if not math.isfinite(radius) or not 0 < radius <= 1.:
            return dict(ok=False, error='invalid_sphere_radius')
        if data.ndim != 2 or data.shape[1] != 3 or len(data) < _MIN_POINTS:
            return dict(ok=False, error='insufficient_sphere_points')
        finite = np.all(np.isfinite(data), axis=1)
        if finite.mean() < .9 or finite.sum() < _MIN_POINTS:
            return dict(ok=False, error='invalid_sphere_points')
        data = data[finite]
        prior = np.median(data, axis=0) if prior_center is None else np.asarray(prior_center, float)
        if prior.shape != (3,) or not np.all(np.isfinite(prior)):
            return dict(ok=False, error='invalid_sphere_prior')
        inside = np.linalg.norm(data-prior, axis=1) <= max(3*radius, .04)
        if inside.mean() < .6 or inside.sum() < _MIN_POINTS:
            return dict(ok=False, error='sphere_outside_measured_neighbourhood')
        data = data[inside]
        if len(data) > _MAX_POINTS:
            data = data[np.linspace(0, len(data)-1, _MAX_POINTS, dtype=int)]
        centers = [_hypothesis(data, radius, prior, seed) for seed in (0, 7, 29)]
        if any(center is None for center in centers):
            return dict(ok=False, error='sphere_fit_failed')
        centers = np.asarray(centers)
        spread = float(np.linalg.norm(centers[:, None]-centers[None, :], axis=2).max())
        if spread > _HYPOTHESIS_SPREAD:
            return dict(ok=False, error='sphere_hypotheses_disagree', hypothesis_spread=spread)
        center = centers.mean(axis=0)
        quality = _quality(data, center, radius)
        if quality is None or not _acceptable(quality):
            return dict(ok=False, error='sphere_surface_coverage_unconfirmed', **(quality or {}))
        return dict(ok=True, center=center.tolist(), radius=radius,
                    hypothesis_spread=spread, **quality)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return dict(ok=False, error='invalid_sphere_measurement')


def validate_anchor_pair(first, second, stamp_first, stamp_second):
    """Agree two independent captures after caller transforms centres to TCP.

    The caller must also bind both captures to the same object and unchanged
    environment. This function cannot assert sensor identity or world freshness.
    """
    try:
        first_stamp, second_stamp = float(stamp_first), float(stamp_second)
        if not (math.isfinite(first_stamp) and math.isfinite(second_stamp)
                and 0 <= first_stamp < second_stamp):
            return dict(ok=False, error='anchor_capture_not_new')
        spreads = [float(first.get('hypothesis_spread', math.inf)),
                   float(second.get('hypothesis_spread', math.inf))]
        if (first.get('ok') is not True or second.get('ok') is not True
                or not _acceptable(first) or not _acceptable(second)
                or not all(math.isfinite(spread) and 0 <= spread <= _HYPOTHESIS_SPREAD
                           for spread in spreads)):
            return dict(ok=False, error='anchor_fit_unconfirmed')
        if not math.isclose(float(first['radius']), float(second['radius']), rel_tol=1e-6, abs_tol=1e-9):
            return dict(ok=False, error='anchor_radius_changed')
        a, b = np.asarray(first['center'], float), np.asarray(second['center'], float)
        if a.shape != (3,) or b.shape != (3,) or not np.all(np.isfinite(np.r_[a, b])):
            return dict(ok=False, error='invalid_anchor_center')
        change = float(np.linalg.norm(a-b))
        if change > .001:
            return dict(ok=False, error='held_anchor_changed_between_captures', center_change=change)
        return dict(ok=True, center=((a+b)/2).tolist(), radius=float(first['radius']),
                    center_change=change, translation_only=True)
    except (AttributeError, KeyError, TypeError, ValueError):
        return dict(ok=False, error='invalid_anchor_pair')
