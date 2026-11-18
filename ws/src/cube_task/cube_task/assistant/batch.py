"""Measured identities and destinations for a finite sequential grasp queue."""
from __future__ import annotations

import copy
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
    items = []
    for index, item in enumerate(decision['items'], 1):
        queued = dict(index=index, label=observed[item['object_id']]['label'],
                      reference_xy=list(observed[item['object_id']]['center_base'][:2]),
                      zone_name=item['zone_name'], object_id=None, status='pending', message='等待执行')
        if 'destination' in item:
            queued['destination'] = copy.deepcopy(item['destination'])
        items.append(queued)
    return items


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


def matching_objects(scene, label, reference_xy, completed_zones=(), completed_positions=()):
    """A label alone cannot substitute a different object after a new scan."""
    reference = np.asarray(reference_xy, dtype=float)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)):
        raise ValueError('队列目标的原始观测位置无效')
    completed = []
    for position in completed_positions:
        try:
            point = np.asarray(position, dtype=float)
            if point.shape != (2,) or not np.all(np.isfinite(point)):
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise ValueError('已完成物品的观测位置无效，无法确认下一件身份') from None
        completed.append(point)
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
            if (not _in_completed_zone(center, scene, completed_zones)
                    and not any(np.linalg.norm(center[:2] - p) <= MAX_TARGET_DRIFT for p in completed)):
                matches.append(obj)
    return matches


def bind_item(item, scene, completed_zones=(), completed_positions=()):
    from .llm import validate_decision

    matches = matching_objects(scene, item['label'], item['reference_xy'],
                               completed_zones, completed_positions)
    if not matches:
        raise ValueError('下一件物品未被可靠观察到、位置变化过大或无法与已完成同类区分，已停止后续任务')
    if len(matches) != 1:
        raise ValueError('下一件物品无法唯一确认，已停止后续任务')
    target = matches[0]
    proposed = {'scene_version': scene['scene_version'], 'action': 'pick',
                'object_id': target['object_id'], 'zone_name': item['zone_name'],
                'message': '执行第 %d 件物品' % item['index']}
    if 'destination' in item:
        proposed['destination'] = copy.deepcopy(item['destination'])
    decision = validate_decision(proposed, scene)
    return decision, target


def whole_footprint_at_goal(target, scene, item):
    # Camera-confirmed side pears remain selectable for recovery, but have not
    # fulfilled the requested upright placement even when their XY is correct.
    if target.get('label') == 'pear' and target.get('resting_pose') in ('side', 'uncertain'):
        return False
    if item.get('destination') is not None:
        from .destinations import whole_footprint_at_destination
        return whole_footprint_at_destination(target, scene, item['destination'])
    return whole_footprint_in_zone(target, scene, item['zone_name'])


def completed_position(target, scene, item, result):
    """Locate a completed extended goal using observations and its planned release."""
    if result.get('skipped'):
        return list(target['center_base'][:2])
    try:
        release = np.asarray(result.get('placement_center'), dtype=float)
        if release.shape not in ((2,), (3,)) or not np.all(np.isfinite(release)):
            return None
        matches = [obj for obj in matching_objects(scene, item['label'], release[:2])
                   if whole_footprint_at_goal(obj, scene, item)]
        if len(matches) == 1:
            return list(matches[0]['center_base'][:2])
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def whole_footprint_in_zone(target, scene, zone_name):
    """Conservative measured-footprint check; a center in a zone is insufficient."""
    from .catalog import catalog_entries

    if target.get('label') == 'pear' and target.get('resting_pose') in ('side', 'uncertain'):
        return False

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
        from .destinations import completion_outline
        corners = completion_outline(target, length, width, yaw)
        zrotation = np.array([[math.cos(zyaw), -math.sin(zyaw)], [math.sin(zyaw), math.cos(zyaw)]])
        in_zone = (corners - zcenter[:2]) @ zrotation
        return bool(np.all(np.abs(in_zone) <= zsize / 2 - .005))
    except (KeyError, ValueError, TypeError, StopIteration):
        return False
