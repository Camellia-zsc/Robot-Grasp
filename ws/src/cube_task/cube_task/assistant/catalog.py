"""Fixed rigid object library and atomic edits of a tabletop simulation.

The editor owns world geometry only. Its labels and coordinates must never be
used as camera detections. Session locking and scene revisions belong to the
caller, which must invalidate its old observations after a successful edit.
"""
from __future__ import annotations

import copy
import math
from numbers import Real
from typing import Any

import numpy as np

from ..perception.geometry import parse_polygon, points_in_polygon
from ..sim.scenarios import CATALOG, official_cube, official_model
from ..sim.world import SimObject

MAX_OBJECTS = 6
MIN_OBJECT_GAP = 0.008
_DESCRIPTIONS = {
    'red': ('红色方块', 'zone_red', '50 mm 方块'),
    'green': ('绿色方块', 'zone_green', '50 mm 方块'),
    'yellow': ('黄色方块', 'zone_yellow', '50 mm 方块'),
    'apple': ('苹果', 'zone_red', '直径约 62 mm 的球形苹果'),
    'banana': ('香蕉', 'zone_yellow', '带果柄的弯曲香蕉，长约 180 mm'),
    'pear': ('梨', 'zone_green', '带颈部与果柄的梨，腹部直径约 85 mm'),
}


def _number(value: Any, field: str) -> float:
    try:
        valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise ValueError('%s 必须是有限数值。' % field)
    return float(value)


def _xy(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ValueError('xy 必须包含两个桌面坐标。')
    return _number(value[0], 'x'), _number(value[1], 'y')


def make_catalog_object(catalog_id: str, name: str, xy, yaw: float = 0.0,
                        table_z: float = 0.0) -> SimObject:
    """Construct one library object without changing a world."""
    if not isinstance(catalog_id, str) or catalog_id not in _DESCRIPTIONS:
        raise ValueError('请选择物品库中已有的物品。')
    if not isinstance(name, str) or not name or len(name) > 100:
        raise ValueError('物品编号无效。')
    x, y = _xy(xy)
    angle = math.remainder(_number(yaw, 'yaw'), math.tau)
    floor = _number(table_z, 'table_z')
    if catalog_id in ('red', 'green', 'yellow'):
        obj = official_cube(catalog_id, edge=0.050, name=name)
    elif catalog_id == 'apple':
        _category, shape, size, color = CATALOG['apple']
        obj = SimObject(name, catalog_id, shape, size, color, np.eye(4),
                        mass=0.100, material='plastic')
    else:
        obj = official_model(name, catalog_id)
    obj.place_upright(x, y, angle, floor)
    return obj


def catalog_entries() -> list[dict]:
    """Return JSON metadata, including axis-aligned dimensions at zero yaw."""
    entries = []
    for label, (name, zone, description) in _DESCRIPTIONS.items():
        obj = make_catalog_object(label, 'catalog-' + label, [0.0, 0.0])
        lo, hi = obj.aabb()
        entries.append({'id': label, 'label': label, 'name': name,
                        'default_zone': zone, 'dimensions': (hi - lo).tolist(),
                        'description': description, 'shape': obj.shape,
                        'color_rgb': list(reversed(obj.color_bgr))})
    return entries


def _segment_in_rectangle(a, b, lo, hi) -> bool:
    """Whether a polygon boundary enters an AABB's open interior."""
    lower, upper = 0.0, 1.0
    for axis in (0, 1):
        delta = float(b[axis] - a[axis])
        minimum, maximum = float(lo[axis]) + 1e-9, float(hi[axis]) - 1e-9
        if abs(delta) <= 1e-14:
            if not minimum < a[axis] < maximum:
                return False
        else:
            start, end = (minimum - a[axis]) / delta, (maximum - a[axis]) / delta
            lower, upper = max(lower, min(start, end)), min(upper, max(start, end))
            if upper <= lower:
                return False
    return upper > lower


class CatalogEditor:
    """Validate edits in isolation, then replace the object list once.

    AABB separation is conservative for curved objects. It guarantees an 8 mm
    gap even when objects are rotated; it does not claim a grasp is reachable.
    The execution layer still performs its normal measured planning/IK checks.
    """

    def __init__(self, world, pick_polygon):
        polygon = parse_polygon(pick_polygon)
        if polygon is None:
            raise ValueError('摆场区域不能为空。')
        self.world = world
        self.pick_polygon = polygon.copy()
        self._metadata = {'catalog': catalog_entries(), 'pick_polygon': self.pick_polygon.tolist(),
                          'max_objects': MAX_OBJECTS, 'min_gap_m': MIN_OBJECT_GAP,
                          'one_per_class': True}

    def metadata(self) -> dict:
        return copy.deepcopy(self._metadata)

    def _check_placement(self, candidate, objects) -> None:
        lo, hi = candidate.aabb()
        corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]],
                            [hi[0], hi[1]], [hi[0], lo[1]]])
        if (not np.all(points_in_polygon(corners, self.pick_polygon, margin=1e-9))
                or any(_segment_in_rectangle(a, b, lo, hi)
                       for a, b in zip(self.pick_polygon, np.roll(self.pick_polygon, -1, axis=0)))):
            raise ValueError('物品完整轮廓必须位于抓取区内，请向区域中央移动。')
        for other in objects:
            a, b = other.aabb()
            delta = np.maximum(np.maximum(a[:2] - hi[:2], lo[:2] - b[:2]), 0.0)
            if float(np.linalg.norm(delta)) < MIN_OBJECT_GAP - 1e-9:
                raise ValueError('物品之间至少需要 8 mm 间隙，请换一个位置。')

    def apply(self, request: dict) -> dict:
        """Apply add/move/remove/clear or raise ValueError without partial edits."""
        if not isinstance(request, dict):
            raise ValueError('摆场请求必须是对象。')
        allowed = {'scene_version', 'action', 'catalog_id', 'object_id', 'xy', 'yaw'}
        if set(request) - allowed:
            raise ValueError('摆场请求含有不支持的字段。')
        action = request.get('action')
        if action not in ('add', 'move', 'remove', 'clear'):
            raise ValueError('不支持此摆场操作。')
        fields = {'add': {'catalog_id', 'xy', 'yaw'},
                  'move': {'object_id', 'xy', 'yaw'},
                  'remove': {'object_id'}, 'clear': set()}[action]
        if set(request) - fields - {'action', 'scene_version'}:
            raise ValueError('此摆场操作含有不适用的字段。')
        if self.world.held is not None:
            raise ValueError('夹持物品时不能编辑场景。')
        old = list(self.world.objects)
        summary = {'action': action, 'object_id': None, 'count': len(old)}
        if action == 'clear':
            new = []
        elif action == 'add':
            if len(old) >= MAX_OBJECTS:
                raise ValueError('场景最多放置六件物品。')
            label = request.get('catalog_id')
            if not isinstance(label, str) or label not in _DESCRIPTIONS:
                raise ValueError('请选择物品库中已有的物品。')
            if any(o.label == label for o in old):
                raise ValueError('每类物品最多放置一件，请移动已有物品。')
            index = 1
            names = {o.name for o in old}
            while 'catalog-%s-%d' % (label, index) in names:
                index += 1
            candidate = make_catalog_object(label, 'catalog-%s-%d' % (label, index),
                                            request.get('xy'), request.get('yaw', 0.0),
                                            self.world.table_z)
            self._check_placement(candidate, old)
            new = old + [candidate]
            summary['object_id'] = candidate.name
        else:
            object_id = request.get('object_id')
            matches = [o for o in old if o.name == object_id]
            if not isinstance(object_id, str) or len(matches) != 1:
                raise ValueError('该物品已不存在，请刷新场景。')
            target = matches[0]
            summary['object_id'] = target.name
            others = [o for o in old if o is not target]
            if action == 'remove':
                new = others
            else:
                if 'xy' not in request and 'yaw' not in request:
                    raise ValueError('移动操作需要坐标或朝向。')
                position = _xy(request.get('xy', target.center_xy()))
                yaw = _number(request.get('yaw', target.yaw()), 'yaw')
                candidate = target.copy()
                candidate.place_upright(*position, math.remainder(yaw, math.tau), self.world.table_z)
                self._check_placement(candidate, others)
                new = [candidate if o is target else o for o in old]
        # No geometry mutation occurs before this single assignment.
        self.world.objects = new
        summary['count'] = len(new)
        return summary
