"""Measured identities and destinations for a finite sequential grasp queue."""
from __future__ import annotations

import math

import numpy as np

MAX_TARGET_DRIFT = 0.04


def idle_batch():
    return {'status': 'idle', 'total': 0, 'completed': 0, 'current_index': None,
            'items': [], 'message': ''}


def build_items(decision, scene):
    from .llm import validate_decision

    decision = validate_decision(decision, scene)
    if decision['action'] != 'pick_batch':
        raise ValueError('需要多物品抓取计划')
    observed = {o['object_id']: o for o in scene['objects']}
    return [dict(index=index, label=observed[item['object_id']]['label'],
                 reference_xy=list(observed[item['object_id']]['center_base'][:2]),
                 zone_name=item['zone_name'], object_id=None, status='pending', message='等待执行')
            for index, item in enumerate(decision['items'], 1)]


def _in_completed_zone(center, scene, completed_zones):
    for name in completed_zones:
        zones = [z for z in scene.get('zones', []) if z.get('name') == name]
        if len(zones) != 1:
            raise ValueError('已完成物品的区域信息无效，无法确认下一件身份')
        zone = zones[0]
        try:
            origin, size = np.asarray(zone['center'], float), np.asarray(zone['size'], float)
            yaw = float(zone.get('yaw', 0.))
            if (origin.shape != (3,) or size.shape != (2,) or np.any(size <= 0)
                    or not np.all(np.isfinite(np.r_[origin, size, yaw]))):
                raise ValueError
        except (KeyError, ValueError, TypeError):
            raise ValueError('已完成物品的区域信息无效，无法确认下一件身份') from None
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        if np.all(np.abs((center[:2] - origin[:2]) @ rotation) <= size / 2 + MAX_TARGET_DRIFT):
            return True
    return False


def matching_objects(scene, label, reference_xy, completed_zones=()):
    """A label alone cannot substitute a different object after a new scan."""
    reference = np.asarray(reference_xy, dtype=float)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)):
        raise ValueError('队列目标的原始观测位置无效')
    matches = []
    for obj in scene.get('objects', []):
        if obj.get('valid') is not True or obj.get('label') != label:
            continue
        try:
            center = np.asarray(obj.get('center_base'), dtype=float)
        except (ValueError, TypeError):
            continue
        if (center.shape == (3,) and np.all(np.isfinite(center))
                and np.linalg.norm(center[:2] - reference) <= MAX_TARGET_DRIFT):
            # A completed object can appear near a later item's original pose.
            # With no persistent visual identity, do not claim it as a new item.
            if not _in_completed_zone(center, scene, completed_zones):
                matches.append(obj)
    return matches


def bind_item(item, scene, completed_zones=()):
    from .llm import validate_decision

    matches = matching_objects(scene, item['label'], item['reference_xy'], completed_zones)
    if not matches:
        raise ValueError('下一件物品未被可靠观察到、位置变化过大或无法与已完成同类区分，已停止后续任务')
    if len(matches) != 1:
        raise ValueError('下一件物品无法唯一确认，已停止后续任务')
    target = matches[0]
    decision = validate_decision(
        {'scene_version': scene['scene_version'], 'action': 'pick',
         'object_id': target['object_id'], 'zone_name': item['zone_name'],
         'message': '执行第 %d 件物品' % item['index']}, scene)
    return decision, target


def whole_footprint_in_zone(target, scene, zone_name):
    """Conservative measured-footprint check; a center in a zone is insufficient."""
    from .catalog import catalog_entries

    zones = [z for z in scene.get('zones', []) if z.get('name') == zone_name]
    if len(zones) != 1:
        return False
    zone = zones[0]
    try:
        length, width, yaw = [float(target[k]) for k in ('length', 'width', 'yaw')]
        center = np.asarray(target['center_base'], float)
        zcenter = np.asarray(zone['center'], float)
        zsize = np.asarray(zone['size'], float)
        zyaw = float(zone.get('yaw', 0.0))
        if (center.shape != (3,) or zcenter.shape != (3,) or zsize.shape != (2,)
                or not np.all(np.isfinite(np.r_[center, zcenter, zsize, length, width, yaw, zyaw]))
                or min(length, width, *zsize) <= 0):
            return False
        # Oblique views can underestimate the far side. Use library dimensions
        # as conservative class bounds, never a live object's true pose.
        nominal = next((e['dimensions'] for e in catalog_entries() if e['label'] == target['label']), None)
        if nominal is None:
            return False
        length = max(length, max(nominal[:2]))
        width = max(width, min(nominal[:2]))
        local = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * [length / 2, width / 2]
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        corners = local @ rotation.T + center[:2]
        zrotation = np.array([[math.cos(zyaw), -math.sin(zyaw)], [math.sin(zyaw), math.cos(zyaw)]])
        in_zone = (corners - zcenter[:2]) @ zrotation
        return bool(np.all(np.abs(in_zone) <= zsize / 2 - .005))
    except (KeyError, ValueError, TypeError, StopIteration):
        return False
