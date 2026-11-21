"""Select measured supported grasps with balanced contact after wrist leaning."""
from __future__ import annotations

import copy
import math

import numpy as np

from .reach import accepts_kwarg, reach_adjust


def supported_grasp(planner, arm, detection, baseline):
    """Reduce the two fingers' height difference on a supported upper object.

    Reach correction may tilt a formerly vertical profile grasp. Closing along
    that lean puts one pad lower on a tapered body and can eject it on lifting.
    A measured nearly round contact section permits an orthogonal/tangential
    axis; every alternative reruns the full existing planner and IK checks.
    The source height, depth limits, friction and held-feedback rules stay the
    same. No improvement, missing support evidence, or an asymmetric measured
    section keeps the original behavior.
    """
    extra = detection.extra or {}
    if (not detection.valid or not baseline.feasible or extra.get('support_observed') is not True
            or not extra.get('support_label') or extra.get('support_z') is None
            or (extra.get('grasp') or {}).get('mode') != 'profile'
            or not accepts_kwarg(planner.plan, 'closing_angle')):
        return baseline
    try:
        if not math.isfinite(float(extra['support_z'])):
            return baseline
        width = float(baseline.expected_width)
    except (TypeError, ValueError):
        return baseline
    original, tilt, _reason = reach_adjust(arm, baseline.T_grasp)
    if original is None or tilt < .5:
        return baseline
    if not math.isfinite(width) or width <= 0.:
        return baseline
    first_gap = abs(float(original[2, 1])) * width
    if first_gap < .002:
        return baseline
    best, best_gap = baseline, first_gap
    x, y = baseline.T_grasp[:2, 3]
    tangent = math.atan2(float(y), float(x)) + math.pi / 2
    angles = []
    for angle in (tangent, float(baseline.closing_angle) + math.pi / 2):
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
        pose, _candidate_tilt, _reason = reach_adjust(arm, candidate.T_grasp)
        if pose is None:
            continue
        gap = abs(float(pose[2, 1])) * float(candidate.expected_width)
        if gap + .002 < best_gap:
            best, best_gap = candidate, gap
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
