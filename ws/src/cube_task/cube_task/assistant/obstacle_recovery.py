"""Camera-only, bounded obstacle relocation with an explicit restoration obligation.

The model selects an opaque proposal ID. Coordinates and all three operations are
created locally from measured detections and checked by the normal planners.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import numpy as np

from ..perception.grasp import (annotate_grasp_neighbors, detection_table_z,
                                held_offset_of, transit_height)
from ..task.reach import adjust_plan_poses, release_workspace_reason
from ..task.runner import held_shape
from .destinations import footprint_corners, validate_destination

RESTORE_XY_TOLERANCE = .015
RESTORE_YAW_TOLERANCE = math.radians(10)
_LABEL_NAMES = {'red': '红色方块', 'green': '绿色方块', 'yellow': '黄色方块',
                'apple': '苹果', 'banana': '香蕉', 'pear': '梨'}


def _measured(det):
    if det.center_base is None:
        return False
    xyz = np.asarray(det.center_base, float)
    return xyz.shape == (3,) and bool(np.all(np.isfinite(xyz)))


def _angle_error(a, b):
    # RGB-D principal axes are measured modulo pi, rather than a signed heading.
    return abs(math.remainder(float(a) - float(b), math.pi))


def _grasp(session, det, neighbors):
    candidate = copy.deepcopy(det)
    peers = [candidate] + [copy.deepcopy(other) for other in neighbors if other is not det]
    annotate_grasp_neighbors(candidate, peers)
    return candidate, session.runner.planner.plan(candidate)


def _translated(det, xy, yaw=None):
    candidate = copy.deepcopy(det)
    origin = np.asarray(det.center_base[:2], float)
    delta = np.asarray(xy, float) - origin
    candidate.center_base[:2] += delta
    candidate.yaw = float(det.yaw or 0) if yaw is None else float(yaw)
    turn = candidate.yaw - float(det.yaw or 0)
    c, s = math.cos(turn), math.sin(turn)
    rotation = np.array([[c, -s], [s, c]])
    for key in ('_points_base', '_placement_points_base'):
        points = candidate.extra.get(key)
        if points is not None:
            points = np.asarray(points, float).copy()
            if points.ndim == 2 and points.shape[1] == 3:
                points[:, :2] = (points[:, :2] - origin) @ rotation.T + np.asarray(xy)
                candidate.extra[key] = points
    points_xy = candidate.extra.get('_points_xy')
    if points_xy is not None:
        points_xy = np.asarray(points_xy, float)
        if points_xy.ndim == 2 and points_xy.shape[1] == 2:
            candidate.extra['_points_xy'] = (points_xy - origin) @ rotation.T + np.asarray(xy)
    for key in ('cap_center_xy', 'stem_xy'):
        point = candidate.extra.get(key)
        if point is not None:
            point = np.asarray(point, float)
            if point.shape == (2,):
                candidate.extra[key] = ((point - origin) @ rotation.T + np.asarray(xy)).tolist()
    for key in ('_grasp_neighbors', '_neighbor_xy', 'grasp'):
        candidate.extra.pop(key, None)
    return candidate


def _checked_placement(session, det, destination, neighbors, desired_yaw=None):
    """Use the real strict support/finger/reach gates without sending motion."""
    runner = session.runner
    target, gp = _grasp(session, det, neighbors)
    if not gp.feasible:
        return None
    if destination.get('type') == 'named_region':
        zone = runner.zones[destination['name']]
        polygon = footprint_corners(zone.center[:2], *zone.size, float(zone.yaw))
        options = {'exact_xy': False}
    else:
        options = session._placement_options(destination)
        zone, polygon = options['placement_zone'], options['placement_polygon']
    shape = held_shape(target, gp, runner.nominal.get(target.label))
    if shape is None:
        return None
    shape['center_offset'] = np.asarray(target.center_base[:2]) - gp.T_grasp[:2, 3]
    shape['grasp_closing'] = gp.closing_angle
    if desired_yaw is not None:
        shape['_required_closing'] = gp.closing_angle + desired_yaw - float(target.yaw or 0)
    others = [target] + [copy.deepcopy(d) for d in neighbors if d is not det]
    strict_filter = runner._strict_place_filter(target, gp, others, polygon)

    def pose_filter(x, y, closing):
        predicted_yaw = float(target.yaw or 0) + closing - gp.closing_angle
        return ((desired_yaw is None or _angle_error(predicted_yaw, desired_yaw) <= RESTORE_YAW_TOLERANCE)
                and strict_filter(x, y, closing))

    offset = held_offset_of(gp)
    tops = [float(d.top_z) for d in others if d is not target and d.top_z is not None]
    transit = transit_height(runner.cfg, list(runner.zones.values()), detection_table_z(target),
                             offset, max(tops) if tops else None)
    placer = copy.copy(runner.placer)
    pp = placer.plan(zone, offset, transit, footprint=runner._footprint(target),
                     held=shape, strict=True, fixed_xy=options['exact_xy'],
                     candidate_filter=lambda x, y: not release_workspace_reason(runner.arm, (x, y)),
                     pose_filter=pose_filter, lower=runner.release_lower.get(target.label, 0.))
    if not pp.feasible:
        return None
    previous = getattr(runner, '_neighbors', None)
    runner._neighbors = others
    try:
        ok, _why, _tilts = adjust_plan_poses(runner.arm, gp, pp,
                                            runner._floors(target, gp, pp, offset), runner.free_tilt)
        if not ok:
            return None
        ok, _why = runner._adjust_strict_release(target, gp, pp, others, polygon,
                                                placer.last_info['xy'], offset)
        if not ok:
            return None
    finally:
        if previous is None:
            del runner._neighbors
        else:
            runner._neighbors = previous
    rotation = pp.T_release[:3, :3] @ gp.T_grasp[:3, :3].T
    axis = rotation @ [math.cos(float(target.yaw or 0)), math.sin(float(target.yaw or 0)), 0]
    yaw = math.atan2(float(axis[1]), float(axis[0]))
    if desired_yaw is not None and _angle_error(yaw, desired_yaw) > RESTORE_YAW_TOLERANCE:
        return None
    return {'xy': list(placer.last_info['xy']), 'yaw': yaw}


def _goal_destination(session, goal):
    if goal.get('destination') is not None:
        return validate_destination(goal['destination'], session.scene)
    zone = session.runner.zones.get(goal.get('zone_name'))
    if zone is None:
        return None
    return {'type': 'named_region', 'name': zone.name}


def propose_clearance(session, target, goal):
    """Offer at most one measured blocker relocation per proposal, never a guess."""
    session._clearance_proposals = {}
    if (not isinstance(goal, Mapping) or goal.get('label') != target.label
            or not _measured(target) or not target.valid):
        return []
    observed = [d for d in session._observed_detections if _measured(d)]
    same = [d for d in observed if d.label == target.label and d.valid]
    if len(same) != 1:
        return []
    target = same[0]
    initial, grasp = _grasp(session, target, observed)
    if grasp.feasible:
        return []  # Narrower openings and alternative grasps take priority.
    reason = str(getattr(grasp, 'reason', ''))
    blockers = (initial.extra.get('grasp') or {}).get('blockers', [])
    if not blockers and 'blocked_by_neighbor' not in reason:
        return []
    goal_destination = _goal_destination(session, goal)
    if goal_destination is None:
        return []
    polygon = np.asarray(session._placement['pick_polygon'], float)
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    grid = [[float(x), float(y)] for x in np.linspace(low[0] + .055, high[0] - .055, 3)
            for y in np.linspace(low[1] + .065, high[1] - .065, 3)]
    options = []
    for obstacle in observed:
        if obstacle is target or not obstacle.valid or obstacle.yaw is None:
            continue
        if sum(d.label == obstacle.label and d.valid for d in observed) != 1:
            continue
        if blockers and not any(b.get('label') == obstacle.label for b in blockers):
            continue
        if np.linalg.norm(obstacle.center_base[:2] - target.center_base[:2]) > .20:
            continue
        remaining = [d for d in observed if d is not obstacle]
        _, without = _grasp(session, target, remaining)
        if not without.feasible:
            continue
        _, obstacle_grasp = _grasp(session, obstacle, observed)
        if not obstacle_grasp.feasible:
            continue
        origin = [float(v) for v in obstacle.center_base[:2]]
        origin_yaw = float(obstacle.yaw)
        candidates = sorted(grid, key=lambda p: -min(np.linalg.norm(np.asarray(p)-target.center_base[:2]),
                                                    np.linalg.norm(np.asarray(p)-origin)))
        tried = 0
        for point in candidates:
            if min(np.linalg.norm(np.asarray(point)-target.center_base[:2]),
                   np.linalg.norm(np.asarray(point)-origin)) < .14:
                continue
            tried += 1
            if tried > 3:
                break
            temporary = {'type': 'point', 'xy': point}
            try:
                validate_destination(temporary, session.scene)
                validate_destination({'type': 'point', 'xy': origin}, session.scene)
            except ValueError:
                continue
            temp_pose = _checked_placement(session, obstacle, temporary, observed, origin_yaw)
            if temp_pose is None:
                continue
            moved = _translated(obstacle, point, temp_pose['yaw'])
            after_move = remaining + [moved]
            main_pose = _checked_placement(session, target, goal_destination, after_move)
            if main_pose is None:
                continue
            completed_target = _translated(target, main_pose['xy'], main_pose['yaw'])
            restoration_neighbors = [d for d in after_move if d is not target] + [completed_target]
            original_destination = {'type': 'point', 'xy': origin}
            if _checked_placement(session, moved, original_destination,
                                  restoration_neighbors, origin_yaw) is None:
                continue
            option_id = 'clear_restore_' + obstacle.label
            option = {'id': option_id, 'strategy': 'clear_restore', 'label': obstacle.label,
                      'description': '先把邻近的%s临时移到已检查的空位，再完成原目标，最后将%s放回原位置和朝向；任何一步异常都会明确报告复位情况。'
                                     % (_LABEL_NAMES.get(obstacle.label, obstacle.label),
                                        _LABEL_NAMES.get(obstacle.label, obstacle.label)),
                      '_clearance': {'goal': copy.deepcopy(dict(goal)),
                                     'obstacle_label': obstacle.label,
                                     'original_xy': origin, 'original_yaw': origin_yaw,
                                     'temporary_destination': temporary,
                                     'environment_revision': getattr(session, '_environment_revision', None)}}
            options.append(option)
            session._clearance_proposals[option_id] = copy.deepcopy(option)
            break
    return options


def _cancelled(session, external):
    return bool((external is not None and external.is_set())
                or (getattr(session, 'cancel_event', None) is not None and session.cancel_event.is_set()))


def _bind(session, label):
    scene = session.scene
    matches = [d for d in scene.get('objects', []) if d.get('label') == label and d.get('valid')]
    if len(matches) != 1:
        raise ValueError('目标%s缺失或存在歧义，不能继续移障事务' % label)
    return scene, matches[0]


def _holding(session):
    try:
        return bool(session.snapshot().get('held_object'))
    except Exception:
        # Missing feedback never authorizes an automatic restoration motion.
        return True


def _at_origin(obj, plan):
    try:
        return (np.linalg.norm(np.asarray(obj['center_base'][:2]) - plan['original_xy']) <= RESTORE_XY_TOLERANCE
                and _angle_error(obj['yaw'], plan['original_yaw']) <= RESTORE_YAW_TOLERANCE)
    except (KeyError, TypeError, ValueError):
        return False


def execute_clearance(session, option, cancel_event=None):
    """Execute a registered local proposal in order and report any unpaid restore."""
    proposed = getattr(session, '_clearance_proposals', {})
    option_id = option.get('id') if isinstance(option, dict) else str(option)
    if option_id not in proposed:
        return {'ok': False, 'stage': 'clearance_validation', 'error': 'unknown_clearance_option',
                'pending_restore': None, 'message': '移障方案已过期或未经本地检查'}
    plan = copy.deepcopy(proposed[option_id]['_clearance'])
    # Consume it before motion: repeated model selections cannot replay an old transaction.
    session._clearance_proposals = {}
    if (plan.get('environment_revision') is not None and plan['environment_revision']
            != getattr(session, '_environment_revision', None)):
        return {'ok': False, 'stage': 'clearance_validation', 'error': 'stale_clearance_option',
                'pending_restore': None, 'message': '移障规划后现场发生变化，请重新观察规划'}
    goal = plan['goal']
    label = plan['obstacle_label']
    steps = []
    obligation = False
    main = None

    def controller_halted(outcome):
        return bool(outcome and (outcome.get('fault') or outcome.get('aborted')))

    def perform(which, object_label, destination=None, zone_name=None, restore_yaw=None):
        if _cancelled(session, cancel_event):
            result = {'ok': False, 'stage': which, 'error': 'aborted', 'aborted': True}
        else:
            dispatched = False
            try:
                scene, obj = _bind(session, object_label)
                decision = {'action': 'pick', 'scene_version': scene['scene_version'],
                            'object_id': obj['object_id'], 'zone_name': zone_name}
                if destination is not None:
                    decision['destination'] = copy.deepcopy(destination)
                if restore_yaw is not None:
                    decision['_restore_yaw'] = float(restore_yaw)
                dispatched = True
                result = session.execute(decision, cancel_event=cancel_event)
            except Exception as exc:
                result = {'ok': False, 'stage': which, 'error': 'clearance_step_refused',
                          'message': str(exc), 'fault': dispatched,
                          'aborted': _cancelled(session, cancel_event)}
        steps.append({'step': which, 'label': object_label, 'result': copy.deepcopy(result)})
        if hasattr(session, '_emit'):
            names = {'move_obstacle': '临时移开遮挡物', 'original_task': '完成原抓取任务',
                     'restore_obstacle': '将遮挡物放回原位'}
            session._emit('clearance_step', {'message': '%s：%s' % (names.get(which, which), '完成' if result.get('ok') else '未完成'),
                                             'step': which, 'label': object_label, 'ok': bool(result.get('ok'))})
        return result

    displaced = perform('move_obstacle', label, plan['temporary_destination'], restore_yaw=plan['original_yaw'])
    try:
        _, current = _bind(session, label)
        obligation = not _at_origin(current, plan)
    except ValueError:
        obligation = True
    if displaced.get('holding') or _holding(session):
        # A held blocker may still have its old cached table observation. That
        # stale XY must never erase the obligation to put it back.
        obligation = True
    if displaced.get('ok') or controller_halted(displaced):
        obligation = True
    if displaced.get('ok') and not controller_halted(displaced):
        main = perform('original_task', goal['label'], goal.get('destination'), goal.get('zone_name'))
    holding = _holding(session)
    restore = None
    if (obligation and not holding and not _cancelled(session, cancel_event)
            and not controller_halted(main or displaced)):
        restore = perform('restore_obstacle', label, {'type': 'point', 'xy': plan['original_xy']},
                          restore_yaw=plan['original_yaw'])
        if restore.get('ok') and not controller_halted(restore):
            try:
                _, current = _bind(session, label)
                obligation = not _at_origin(current, plan)
            except ValueError:
                obligation = True
    pending = ({'label': label, 'destination': {'type': 'point', 'xy': plan['original_xy']},
                'yaw': plan['original_yaw'], 'message': '临时挪开的物品尚未确认回到原位置和朝向'}
               if obligation else None)
    result = copy.deepcopy(main or displaced)
    result.update(clearance_steps=steps, pending_restore=pending,
                  clearance_option_id=option_id, target_result=copy.deepcopy(main),
                  holding=_holding(session),
                  fault=any(step['result'].get('fault') for step in steps),
                  aborted=bool(_cancelled(session, cancel_event) or
                               any(step['result'].get('aborted') for step in steps)))
    if pending is not None:
        result.update(ok=False, stage='obstacle_restore', error='pending_restore',
                      message='原任务%s；临时移开的物品尚未复位，请查看恢复记录。' % ('已完成' if main and main.get('ok') else '未完成'))
    elif main and main.get('ok') and restore and restore.get('ok'):
        result.update(ok=True, message='原目标已完成，临时移开的物品已放回原位置和朝向。')
        if main.get('verified_revision') != getattr(session, '_environment_revision', None):
            # The user may move the completed target while the obstacle is being
            # restored. Recheck only on such a change, rather than rescanning an
            # unchanged destination after every transaction.
            try:
                verification = session.verify_goals([copy.deepcopy(goal)])
            except Exception as exc:
                verification = {'ok': False, 'items': [], 'message': str(exc), 'fault': True}
            result['final_goal_verification'] = copy.deepcopy(verification)
            result['fault'] = bool(result['fault'] or verification.get('fault'))
            result['aborted'] = bool(result['aborted'] or verification.get('aborted')
                                     or _cancelled(session, cancel_event))
            if not verification.get('ok') or controller_halted(result):
                result.update(ok=False, stage='clearance_final_verify',
                              error='goal_changed_after_restore', camera_verified=False,
                              message=verification.get('message') or
                              '遮挡物已复位，但原目标在此期间发生变化，尚未确认仍满足任务。')
            else:
                result['verified_revision'] = getattr(session, '_environment_revision', None)
    return result
