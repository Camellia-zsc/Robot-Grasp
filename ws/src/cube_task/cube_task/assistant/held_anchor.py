"""Translation feedback from an observable spherical part of a known object.

The source and held RGB-D clouds supply centers. Catalog geometry supplies only
the sphere radius. Neither simulator object poses nor inertia are read here.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from ..core.transforms import inv_T, is_rotation


def spherical_model(target):
    """Return a class prior only when its source resting frame is known."""
    from .catalog import make_catalog_object
    from ..perception.grasp import detection_table_z

    if target.label == 'pear' and target.extra.get('resting_pose') != 'upright':
        return None
    try:
        model = make_catalog_object(target.label, 'anchor-prior', [0., 0.])
        if model.shape == 'sphere':
            radius = float(model.size[0])
            local = np.array([0., 0., radius])
        elif model.shape == 'compound':
            spheres = [part for part in model.size if part['shape'] == 'sphere']
            if not spheres:
                return None
            part = max(spheres, key=lambda value: float(value['size'][0]))
            radius = float(part['size'][0])
            local = np.asarray(part['pos'], float)
        else:
            return None
        floor = detection_table_z(target)
        center = np.asarray(target.center_base, float).copy()
        if (floor is None or center.shape != (3,) or local.shape != (3,)
                or not np.all(np.isfinite(np.r_[center, local, radius, floor]))
                or radius < .015):
            return None
        yaw = float(target.yaw or 0.)
        c, s = math.cos(yaw), math.sin(yaw)
        center[:2] += np.array([[c, -s], [s, c]]) @ local[:2]
        center[2] = float(floor) + local[2]
        return dict(radius=radius, prior_center=center)
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def seating_requires_feedback(grasp, result, width):
    """A large measured seating change cannot establish an unchanged grip."""
    # Seating can shift a curved body in a table grasp as well as a stack
    # grasp. Bind every measurable spherical source to its actual grip before
    # carrying it with the old rigid transform.
    if spherical_model(grasp.target) is None:
        return False
    closes = [step.get('width') for step in result.detail.get('steps', [])
              if step.get('stage') == 'close' and step.get('ok')]
    try:
        closed = float(closes[-1] if closes else grasp.target.extra['_closed_grip_width'])
        held = float(width)
        if not math.isfinite(closed) or not math.isfinite(held) or min(closed, held) <= 0:
            return True
        return abs(closed-held) > max(.003, .05*closed)
    except (KeyError, IndexError, TypeError, ValueError):
        # Missing original feedback cannot certify an unchanged recovered grip.
        return True


def source_anchor(grasp):
    from .spherical_anchor import fit_spherical_anchor

    prior = spherical_model(grasp.target)
    if prior is None:
        return dict(ok=False, error='spherical_source_prior_unavailable')
    points = grasp.target.extra.get('_placement_points_base')
    if points is None:
        return dict(ok=False, error='fused_source_cloud_unavailable')
    fitted = fit_spherical_anchor(points, prior['radius'], prior['prior_center'])
    if fitted.get('ok'):
        return fitted
    # The spherical section is only part of a compound fruit. At a raised
    # support the visible neck can outnumber the bulb. Select a fixed band
    # around the class prior's equator, without selecting by fitted residuals
    # or relaxing any fit/coverage/ambiguity gate.
    try:
        data = np.asarray(points, float)
        radius, center = prior['radius'], prior['prior_center']
        band = data[(np.abs(data[:, 2]-center[2]) <= .55*radius)
                    & (np.linalg.norm(data[:, :2]-center[:2], axis=1) <= radius+.012)]
        selected = fit_spherical_anchor(band, radius, center)
        if selected.get('ok'):
            return dict(selected, source_section='equatorial_body_band')
    except (ValueError, TypeError, IndexError):
        pass
    return fitted


def held_anchor(detections, grasp, source, tcp):
    """Keep semantic uniqueness even when the whole held silhouette is cropped."""
    from .spherical_anchor import fit_spherical_anchor

    if not source.get('ok'):
        return dict(ok=False, error='source_anchor_unconfirmed')
    T = np.asarray(tcp, float)
    if (T.shape != (4, 4) or not np.all(np.isfinite(T))
            or not is_rotation(T[:3, :3], 1e-3)
            or not np.allclose(T[3], [0., 0., 0., 1.])):
        return dict(ok=False, error='held_tcp_unconfirmed')
    reference = np.r_[np.asarray(source['center'], float), 1.]
    predicted = (T @ inv_T(grasp.T_grasp) @ reference)[:3]
    candidates, masks = [], []
    for detection in detections:
        if detection.label != grasp.target.label:
            continue
        points = np.asarray(detection.extra.get('_points_base', []), float)
        if points.ndim != 2 or points.shape[1:] != (3,):
            continue
        points = points[np.all(np.isfinite(points), axis=1)]
        # The fit may move its centre by at most 50 mm. A second radius here
        # admits the yellow support beneath a similarly coloured pear.
        near = points[np.linalg.norm(points-predicted, axis=1) <= float(source['radius'])+.05]
        if len(near) >= 100:
            candidates.append(near)
            masks.append(detection.mask)
    if not candidates:
        return dict(ok=False, error='held_spherical_identity_unconfirmed')
    if len(candidates) > 1:
        # A tracked pear can be split into disjoint image components beside a
        # same-colour support. Every component must explain the SAME sphere;
        # never choose one semantic instance and silently discard another.
        if not grasp.target.extra.get('_held_translation_anchor'):
            return dict(ok=False, error='held_spherical_identity_unconfirmed')
        if any(mask is None or np.asarray(mask).ndim != 2
               or np.asarray(mask).dtype != np.bool_ for mask in masks):
            return dict(ok=False, error='held_spherical_fragment_masks_unconfirmed')
        masks = [np.asarray(mask) for mask in masks]
        for i, mask in enumerate(masks):
            if not np.any(mask):
                return dict(ok=False, error='held_spherical_fragment_masks_unconfirmed')
            for other in masks[:i]:
                if mask.shape != other.shape or np.any(mask & other):
                    return dict(ok=False, error='held_spherical_fragments_overlap')
        parts = [fit_spherical_anchor(points, source['radius'], predicted) for points in candidates]
        if not all(part.get('ok') for part in parts):
            return dict(ok=False, error='held_spherical_fragment_fit_unconfirmed')
        centers = np.asarray([part['center'] for part in parts], float)
        if np.max(np.linalg.norm(centers[:, None]-centers[None, :], axis=2)) > .001:
            return dict(ok=False, error='held_spherical_fragments_disagree')
    fitted = fit_spherical_anchor(np.vstack(candidates), source['radius'], predicted)
    if fitted.get('ok') and len(candidates) > 1:
        fractions = [float(np.mean(np.abs(np.linalg.norm(points-np.asarray(fitted['center']), axis=1)
                                         -float(source['radius'])) < .0012)) for points in candidates]
        if min(fractions) < .60:
            return dict(ok=False, error='held_spherical_fragments_disagree')
        fitted['fragment_inlier_fractions'] = fractions
    if fitted.get('ok'):
        fitted = copy.deepcopy(fitted)
        fitted['center_base'] = copy.deepcopy(fitted['center'])
        fitted['center'] = (inv_T(T) @ np.r_[fitted['center'], 1.])[:3].tolist()
    return fitted


def apply_translation_anchor(grasp, source, pair, revision):
    """Update the carried reference; retain the executed command for auditing."""
    if not source.get('ok') or not pair.get('ok') or pair.get('translation_only') is not True:
        return dict(ok=False, error='held_anchor_pair_unconfirmed')
    try:
        source_center = np.asarray(source['center'], float)
        local_center = np.asarray(pair['center'], float)
        if (source_center.shape != (3,) or local_center.shape != (3,)
                or not np.all(np.isfinite(np.r_[source_center, local_center]))):
            raise ValueError
        original = grasp.T_grasp.copy()
        updated = source_center-original[:3, :3]@local_center
        correction = updated-original[:3, 3]
        # This is a bounded correction to the already measured grip, never a
        # replacement target identity or an unconstrained pose estimate.
        if np.linalg.norm(correction) > .05:
            return dict(ok=False, error='held_anchor_change_exceeds_bound')
        moved = bool(np.linalg.norm(correction) > .00025)
        if moved:
            grasp.T_grasp = original.copy()
            grasp.T_grasp[:3, 3] = updated
        extra = grasp.target.extra
        if '_executed_grasp_T' not in extra:
            extra['_executed_grasp_T'] = original.copy()
        extra['_held_translation_anchor'] = dict(
            source_center=source_center.tolist(), center_tcp=local_center.tolist(),
            environment_revision=int(revision), translation_only=True,
            correction_m=correction.tolist(), pair=copy.deepcopy(pair))
        return dict(ok=True, changed=moved, correction_mm=(correction*1000).tolist(),
                    translation_only=True)
    except (KeyError, TypeError, ValueError):
        return dict(ok=False, error='held_anchor_data_invalid')
