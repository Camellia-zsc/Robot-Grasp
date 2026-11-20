"""Camera-derived stacking geometry; simulator contacts are evaluation only."""
from __future__ import annotations

import copy
import math

import cv2
import numpy as np

from ..core.types import Zone
from ..perception.grasp import detection_table_z
from .catalog import catalog_entries
from .observation_policy import unique_target


def measured_support(detections, label):
    base = unique_target(detections, label)
    if base is None:
        raise ValueError('相机尚未完整确认指定底座，请先观察底座')
    entry = next((e for e in catalog_entries() if e['label'] == label), {})
    if entry.get('shape') not in ('cube', 'box'):
        raise ValueError('当前叠放规划需要可靠平顶底座；曲面物品不能作为已确认的稳定支撑')
    values = [base.length, base.width, base.yaw, base.top_z, base.height]
    if (base.center_base is None or any(v is None or not math.isfinite(float(v)) for v in values)
            or min(float(base.length), float(base.width), float(base.height)) <= .005):
        raise ValueError('底座缺少有效的实测顶面与轮廓')
    return dict(label=label, xy=base.center_base[:2].tolist(), top_z=float(base.top_z),
                length=float(base.length), width=float(base.width), yaw=float(base.yaw))


def planning_options(session, destination, detections=None):
    if getattr(session, '_physics', None) is None:
        raise ValueError('物品叠放需要 MuJoCo 重力与接触仿真，请切换物理后端')
    support = measured_support(session._observed_detections if detections is None else detections,
                               destination['label'])
    table = session._placement['table_bounds']
    polygon = np.array([[table['x'][0], table['y'][0]], [table['x'][1], table['y'][0]],
                        [table['x'][1], table['y'][1]], [table['x'][0], table['y'][1]]])
    # The upper footprint can overhang a smaller flat base. Its measured center
    # is constrained separately to the base's interior; real contacts decide
    # whether that release actually remains stable.
    zone = Zone('on_object_' + support['label'], np.r_[support['xy'], support['top_z']],
                (.30, .30), release_height=.004, label=support['label'] + '顶面')
    # Search small offsets too: a fully supported center need not coincide with
    # the top-face center. The runner still checks center_supported(), the whole
    # upper body, real finger clearance and reachable release orientation.
    return dict(placement_zone=zone, placement_polygon=polygon, exact_xy=False), support


def annotate(target, support):
    target.extra['_stack_support'] = copy.deepcopy(support)


def center_supported(xy, support, margin=.004):
    delta = np.asarray(xy, float) - support['xy']
    c, s = math.cos(support['yaw']), math.sin(support['yaw'])
    local = np.array([[c, s], [-s, c]]) @ delta
    return bool(abs(local[0]) <= support['length']/2-margin
                and abs(local[1]) <= support['width']/2-margin)


def camera_confirmed(session, label, destination):
    """Require a newly measured upper object and the specified visible support."""
    if (session._observation_revision != session._environment_revision
            or getattr(session, '_interventions', ()) or session.world.held is not None):
        return False
    upper = unique_target(session._last_fresh, label)
    if upper is None or upper.center_base is None:
        return False
    lower = unique_target(session._last_fresh, destination['label'])
    if lower is not None:
        try:
            support = measured_support(session._last_fresh, destination['label'])
        except ValueError:
            return False
        floor = detection_table_z(upper)
        return bool(floor is not None and abs(float(floor)-support['top_z']) <= .012
                    and center_supported(upper.center_base[:2], support, margin=.001))
    evidence = upper.extra
    polygon = np.asarray(evidence.get('support_polygon', []), float)
    floor = detection_table_z(upper)
    if (evidence.get('support_observed') is not True
            or evidence.get('support_label') != destination['label']
            or floor is None or evidence.get('support_z') is None
            or abs(float(floor)-float(evidence['support_z'])) > .012
            or polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3
            or not np.all(np.isfinite(polygon))):
        return False
    # Contact is occluded by the upper body. A visible adjacent top patch is
    # camera evidence of support identity/height, not a reconstructed full base
    # footprint. Independent contact/stability must additionally pass audit().
    reach = max(float(upper.length or 0), float(upper.width or 0))/2 + .008
    return cv2.pointPolygonTest(polygon.astype(np.float32),
        tuple(float(v) for v in upper.center_base[:2]), True) >= -reach


def audit(session, obj, destination):
    physics = getattr(session, '_physics', None)
    if physics is None or obj is None or obj is session.world.held:
        return dict(success=False, stable=False, reason='physical_support_unavailable')
    physics.sync()
    return physics.support_status(obj.name, destination['label'])
