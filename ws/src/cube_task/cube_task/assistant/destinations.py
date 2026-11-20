"""Configured tabletop destinations, independent of simulator object truth."""
from __future__ import annotations

import math

import numpy as np

from ..perception.geometry import points_in_polygon

POINT_TOLERANCE = 0.015
PLACEMENT_MARGIN = 0.002


def placement_metadata(cfg, pick_polygon):
    from ..sim.render import TABLE_CENTER, TABLE_SIZE
    return {'bounds': {'x': list(cfg.get('robot.workspace.x')),
                       'y': list(cfg.get('robot.workspace.y')),
                       'r_min': float(cfg.get('robot.workspace.r_min')),
                       'r_max': float(cfg.get('robot.workspace.r_max'))},
            'table_bounds': {axis: [float(c - size / 2), float(c + size / 2)]
                             for axis, c, size in zip(('x', 'y'), TABLE_CENTER, TABLE_SIZE)},
            'pick_polygon': [[float(v) for v in p] for p in pick_polygon],
            'point_tolerance': POINT_TOLERANCE, 'margin': PLACEMENT_MARGIN}


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def validate_destination(value, scene, *, require_observed=True, known_labels=()):
    """Normalize destinations; object supports remain semantic camera references.

    Only fixed existing goals or observation requests may retain a temporarily
    occluded known support. Actual new placements require its current measure.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('放置目标必须是对象')
    if value.get('type') == 'on_object':
        label = value.get('label')
        if (set(value) != {'type', 'label'} or not isinstance(label, str)
                or not label.strip() or label != label.strip() or len(label) > 100):
            raise ValueError('叠放支撑物只能使用类别引用，不能指定坐标或高度')
        objects = [obj for obj in scene.get('objects', ())
                   if isinstance(obj, dict) and obj.get('label') == label and obj.get('valid') is True]
        known = set(scene.get('class_to_zone', {})) | set(known_labels)
        known.update(obj.get('label') for obj in scene.get('objects', ())
                     if isinstance(obj, dict) and obj.get('valid') is True)
        if label not in known:
            raise ValueError('叠放支撑物尚未配置或测量')
        if require_observed:
            center = objects[0].get('center_base') if len(objects) == 1 else None
            if (not isinstance(center, (list, tuple, np.ndarray)) or len(center) != 3
                    or not all(_number(v) for v in center)):
                raise ValueError('叠放支撑物必须是当前唯一有效的三维相机测量')
        return {'type': 'on_object', 'label': label}
    placement = scene.get('placement', {})
    if value.get('type') == 'region':
        if set(value) != {'type', 'name'} or value.get('name') != 'pick_area':
            raise ValueError('未知放置区域')
        if not placement.get('pick_polygon'):
            raise ValueError('当前场景没有抓取区配置')
        polygon = np.asarray(placement.get('pick_polygon'), dtype=float)
        if polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3 or not np.all(np.isfinite(polygon)):
            raise ValueError('抓取区配置不可用')
        return {'type': 'region', 'name': 'pick_area'}
    if value.get('type') != 'point' or set(value) != {'type', 'xy'}:
        raise ValueError('放置目标字段不符合约定')
    xy = value['xy']
    if (not isinstance(xy, (list, tuple)) or len(xy) != 2
            or not all(_number(v) for v in xy)):
        raise ValueError('放置点必须包含两个有限数值坐标')
    x, y = (float(v) for v in xy)
    bounds = placement.get('bounds', {})
    try:
        margin = float(placement.get('margin', PLACEMENT_MARGIN))
        legal = all(float(bounds[a][0]) + margin <= p <= float(bounds[a][1]) - margin
                    for a, p in (('x', x), ('y', y)))
        legal = legal and (float(bounds['r_min']) + margin <= math.hypot(x, y)
                           <= float(bounds['r_max']) - margin)
        table = placement['table_bounds']
        legal = legal and all(float(table[a][0]) + margin <= p <= float(table[a][1]) - margin
                              for a, p in (('x', x), ('y', y)))
    except (KeyError, TypeError, ValueError, IndexError):
        raise ValueError('当前场景缺少有效的放置边界') from None
    if not legal:
        raise ValueError('放置点超出当前配置的桌面工作范围')
    return {'type': 'point', 'xy': [x, y]}


def destination_label(destination):
    if destination is None:
        return ''
    if destination['type'] == 'region':
        return '抓取区'
    if destination['type'] == 'on_object':
        names = {'red': '红色方块', 'green': '绿色方块', 'yellow': '黄色方块',
                 'apple': '苹果', 'banana': '香蕉', 'pear': '梨'}
        return names.get(destination['label'], destination['label']) + '顶面'
    return '桌面指定点（%.3f，%.3f 米）' % tuple(destination['xy'])


def footprint_corners(xy, length, width, yaw):
    local = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], float) * [length / 2, width / 2]
    c, s = math.cos(yaw), math.sin(yaw)
    return local @ np.array([[c, -s], [s, c]]).T + np.asarray(xy, float)


def footprint_in_destination(corners, scene, destination):
    """Full footprint must stay on the configured tabletop, and in a region."""
    if destination['type'] == 'on_object':
        return False  # A 2-D tabletop boundary cannot prove height or support.
    placement = scene['placement']
    margin = float(placement.get('margin', PLACEMENT_MARGIN))
    table = placement['table_bounds']
    if any(np.any(corners[:, i] < table[a][0] + margin)
           or np.any(corners[:, i] > table[a][1] - margin)
           for i, a in enumerate(('x', 'y'))):
        return False
    if destination['type'] == 'region':
        polygon = np.asarray(placement['pick_polygon'], float)
        # Negative margins are unsupported by points_in_polygon. Require a
        # small box around every corner inside the region as the edge margin.
        points = np.concatenate([corners + offset for offset in
                                 ((margin, margin), (margin, -margin),
                                  (-margin, margin), (-margin, -margin))])
        return bool(np.all(points_in_polygon(points, polygon)))
    return True


def completion_outline(target, length, width, yaw):
    """Conservative nominal outline without inventing corners on round fruit."""
    from ..perception.geometry import object_footprint
    return object_footprint(np.asarray(target['center_base'], float)[:2], length, width, yaw,
                            label=target.get('label'), resting_pose=target.get('resting_pose'))


def whole_footprint_at_destination(target, scene, destination):
    """Conservative camera-only completion predicate for sequential queues."""
    from .catalog import catalog_entries
    if target.get('label') == 'pear' and target.get('resting_pose') in ('side', 'uncertain'):
        return False
    try:
        destination = validate_destination(destination, scene)
        if destination['type'] == 'on_object':
            return False
        center = np.asarray(target['center_base'], float)
        length, width, yaw = (float(target[k]) for k in ('length', 'width', 'yaw'))
        if center.shape != (3,) or not np.all(np.isfinite(np.r_[center, length, width, yaw])):
            return False
        nominal = next(e['dimensions'] for e in catalog_entries() if e['label'] == target['label'])
        length, width = max(length, max(nominal[:2])), max(width, min(nominal[:2]))
        corners = completion_outline(target, length, width, yaw)
        if destination['type'] == 'point' and np.linalg.norm(center[:2] - destination['xy']) > scene['placement'].get('point_tolerance', POINT_TOLERANCE):
            return False
        return footprint_in_destination(corners, scene, destination)
    except (KeyError, TypeError, ValueError, StopIteration):
        return False
