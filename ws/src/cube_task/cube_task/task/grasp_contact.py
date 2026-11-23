"""Select measured supported grasps with balanced contact after wrist leaning."""
from __future__ import annotations

import copy
import math

import numpy as np

from .reach import accepts_kwarg, reach_adjust


def supported_grasp(planner, arm, detection, baseline, *, free_tilt=None):
    """Reduce the two fingers' height difference on a measured profile grip.

    Reach correction may tilt a formerly vertical profile grasp. Closing along
    that lean puts one pad lower on a tapered body and can eject it on lifting.
    A measured nearly round contact section permits an orthogonal/tangential
    axis; every alternative reruns the full existing planner and IK checks.
    The source height, depth limits, friction and held-feedback rules stay the
    same. No improvement, missing support evidence, or an asymmetric measured
    section keeps the original behavior.
    """
    extra = detection.extra or {}
    from ..perception.grasp import measured_tapered_contact
    supported = (extra.get('support_observed') is True and extra.get('support_label')
                 and extra.get('support_z') is not None)
    tapered = measured_tapered_contact(detection)
    solid_square = (extra.get('catalog_shape') in ('box','cube')
                    and (extra.get('grasp') or {}).get('kind') == 'square')
    if (not detection.valid or not baseline.feasible or not (supported or tapered or solid_square)
            or (extra.get('grasp') or {}).get('mode') != 'profile'
            or not accepts_kwarg(planner.plan, 'closing_angle')):
        return baseline
    try:
        from ..perception.grasp import detection_table_z
        floor = extra['support_z'] if supported else detection_table_z(detection)
        if floor is None or not math.isfinite(float(floor)):
            return baseline
        width = float(baseline.expected_width)
    except (TypeError, ValueError):
        return baseline
    if not math.isfinite(width) or width <= 0.:
        return baseline
    def gap(plan):
        pose, _tilt, _reason = reach_adjust(arm, plan.T_grasp)
        if pose is None:
            return None
        values = [abs(float(pose[2, 1])) * float(plan.expected_width)]
        # A vertical table grasp can require a leaned lift. Check it before
        # closing: the first held feedback arrives after this potential slip.
        lifted, _tilt, _reason = reach_adjust(arm, plan.T_lift, max_tilt_deg=free_tilt)
        if lifted is not None:
            values.append(abs(float(lifted[2, 1])) * float(plan.expected_width))
        return max(values)
    first_gap = gap(baseline)
    if first_gap is None:
        return baseline
    if first_gap < .002:
        return baseline
    best, best_gap = baseline, first_gap
    x, y = baseline.T_grasp[:2, 3]
    tangent = math.atan2(float(y), float(x)) + math.pi / 2
    angles = []
    candidates = ((float(baseline.closing_angle) + math.pi / 2,) if solid_square else
                  (tangent, float(baseline.closing_angle) + math.pi / 2))
    for angle in candidates:
        if not any(abs(math.remainder(angle - old, math.pi)) < math.radians(2) for old in angles):
            angles.append(angle)
    for angle in angles:
        event = getattr(arm, 'abort_event', None)
        if event is not None and event.is_set():
            return baseline
        candidate = planner.plan(copy.deepcopy(detection), closing_angle=angle)
        if not candidate.feasible or (candidate.target.extra.get('grasp') or {}).get('mode') != 'profile':
            continue
        try:
            ratio = float(candidate.expected_width) / width
        except (TypeError, ValueError):
            continue
        if (not math.isfinite(ratio) or not .8 <= ratio <= 1.25
                or np.linalg.norm(candidate.T_grasp[:2, 3] - baseline.T_grasp[:2, 3]) > .005):
            continue
        neighbors = candidate.target.extra.get('_neighbor_xy')
        if neighbors is not None:
            sweep = getattr(planner, '_sweep_mask', None)
            if (sweep is None or np.any(sweep(neighbors, candidate.closing_angle,
                    *candidate.T_grasp[:2, 3], candidate.expected_width, candidate.open_width))):
                # An explicit alternate axis must retain this check even when
                # the ordinary planner's adaptive-neighbor option is disabled.
                continue
        candidate_gap = gap(candidate)
        if candidate_gap is None:
            continue
        if candidate_gap + .002 < best_gap:
            best, best_gap = candidate, candidate_gap
    if best is baseline:
        return baseline
    info = copy.deepcopy(best.target.extra['grasp'])
    info['supported_contact'] = dict(
        strategy='balanced_finger_height', before_gap_mm=round(first_gap * 1000, 2),
        after_gap_mm=round(best_gap * 1000, 2),
        closing_adjustment_deg=round(math.degrees(math.remainder(
            best.closing_angle - baseline.closing_angle, math.pi)), 1))
    detection.extra['grasp'] = info
    best.target = detection
    return best
