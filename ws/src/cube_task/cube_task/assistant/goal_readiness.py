"""Camera-only work ordering hints, never grasp or completion authority."""
from __future__ import annotations

import numpy as np

from .batch import whole_footprint_at_goal


def _complete_geometry(target):
    try:
        center = np.asarray(target.get('center_base'), float)
        dimensions = np.asarray([target.get(k) for k in ('length', 'width', 'height')], float)
        top, yaw = float(target.get('top_z')), float(target.get('yaw'))
        return bool(center.shape == (3,) and np.all(np.isfinite(np.r_[center, dimensions, top, yaw]))
                    and np.all(dimensions > 0.))
    except (TypeError, ValueError, OverflowError):
        return False


def goal_worklist(scene, goals):
    """Keep every goal while deferring unavailable source measurements.

    A measured full outline at its destination is only an ordering hint. New
    camera, support and execution checks still decide actual completion.
    Invalid fragments do not obscure one complete identity; two valid objects
    remain ambiguous even if only one has usable geometry.
    """
    if not isinstance(scene, dict) or not isinstance(goals, (list, tuple)):
        raise ValueError('目标工作清单需要有效场景和最终目标。')
    objects = scene.get('objects', ())
    if not isinstance(objects, (list, tuple)):
        raise ValueError('目标工作清单缺少相机观测列表。')
    result = []
    for index, goal in enumerate(goals):
        label = goal.get('label') if isinstance(goal, dict) else None
        if not isinstance(label, str) or not label or len(label) > 100:
            raise ValueError('目标工作清单中的物品类别无效。')
        matches = [o for o in objects if isinstance(o, dict) and o.get('label') == label]
        valid = [o for o in matches if o.get('valid') is True]
        status = 'deferred'
        if scene.get('valid') is False or scene.get('executable') is False:
            reason = 'scene_unavailable'
        elif not valid:
            reason = 'target_invalid' if matches else 'target_not_observed'
        elif len(valid) != 1:
            reason = 'target_ambiguous'
        elif not _complete_geometry(valid[0]):
            reason = 'target_geometry_incomplete'
        else:
            at_goal = whole_footprint_at_goal(valid[0], scene, goal)
            status = 'observed_at_goal' if at_goal else 'observed_pending'
            reason = 'camera_outline_at_goal' if at_goal else 'goal_not_observed_satisfied'
        result.append(dict(goal_index=index, label=label, status=status, reason=reason))
    return result
