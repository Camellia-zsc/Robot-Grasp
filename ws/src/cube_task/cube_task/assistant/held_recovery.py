"""Measured, confirmed transactions: stage a held target, rearrange, then regrasp.

An incomplete edge detection can justify looking again after staging; it never
authorizes grasping that obstacle. All actual moves require fresh unique targets.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .batch import whole_footprint_at_goal
from .obstacle_recovery import _checked_placement, _translated

NAMES = {'red': '红色方块', 'green': '绿色方块', 'yellow': '黄色方块',
         'apple': '苹果', 'banana': '香蕉', 'pear': '梨'}


def _destination(session, goal):
    return copy.deepcopy(goal.get('destination') or
                         {'type': 'named_region', 'name': goal['zone_name']})


def _center(session, destination):
    if destination['type'] == 'point':
        return np.asarray(destination['xy'], float)
    if destination['type'] == 'region' and destination['name'] == 'pick_area':
        return np.mean(session._placement['pick_polygon'], axis=0)
    return np.asarray(session.runner.zones[destination['name']].center[:2], float)


def propose_held_recovery(session, goal, protected_goals=()):
    """Precheck safe staging; expose bounded conditional rearrangement to humans."""
    session._held_proposals = {}
    target = session._recovery_target
    if target is None or not session.runner.has_held_recovery:
        return []
    neighbors = [d for d in session._recovery_neighbors
                 if d.center_base is not None and d.label != target.label]
    candidates = [target.center_base[:2].tolist(), [.26, -.12], [.26, .12], [.20, -.10]]
    bundle = None
    for xy in candidates:
        destination = {'type': 'point', 'xy': list(xy)}
        try:
            options = session._placement_options(destination)
            bundle, _why = session.runner.plan_held_transfer(target, neighbors, **options)
        except (ValueError, KeyError):
            continue
        if bundle is not None:
            break
    if bundle is None:
        return []
    goal_destination = _destination(session, goal)
    center = _center(session, goal_destination)
    possible = [d for d in neighbors if d.label != 'unknown'
                and np.linalg.norm(d.center_base[:2] - center) < .18]
    labels = list(dict.fromkeys(d.label for d in possible))[:2]
    options = []
    # A regrasp alone can repair an awkward held orientation. When occupancy is
    # measured near the destination, also offer a conditional same-region move.
    for obstacle in labels + [None]:
        option_id = 'stage_rearrange_' + obstacle if obstacle else 'stage_regrasp'
        target_name = NAMES.get(target.label, target.label)
        steps = ['将%s安全暂放到已检查的桌面位置' % target_name,
                 '空爪重新观察目的区域，确认目标与障碍的完整轮廓']
        if obstacle:
            steps.append('仅在严格检查通过时调整%s；保持其所在区域，位置会改变且不承诺原位复位' % NAMES.get(obstacle, obstacle))
        steps.append('重新抓取%s，完成原目标并核验相关物品' % target_name)
        option = dict(id=option_id, strategy='stage_rearrange' if obstacle else 'stage_regrasp',
            label=target.label, requires_confirmation=True,
            question='是否先暂放%s%s，再重新抓取完成原任务？' %
                     (target_name, '并允许调整%s在原区域内的位置' % NAMES.get(obstacle, obstacle) if obstacle else ''),
            steps=steps, description='；'.join(steps) + '。若新观测不足或无可行方案则停止，暂放不算任务完成。')
        options.append(option)
        session._held_proposals[option_id] = dict(bundle=bundle, staging=destination,
            goal=copy.deepcopy(goal), obstacle=obstacle,
            protected_goals=copy.deepcopy(list(protected_goals)),
            revision=session._environment_revision)
    return options


def _fresh_target(session, label, xy, purpose):
    scene = session.observe(target={'label': label, 'xy': list(xy), 'require_grasp': True}, purpose=purpose)
    fresh = [d for d in session._last_fresh if d.valid and d.label == label
             and d.center_base is not None]
    if len(fresh) != 1 or session._observation_revision != session._environment_revision:
        raise ValueError('新相机图像未唯一确认%s，不能搬动' % NAMES.get(label, label))
    measured = fresh[0]
    matches = [o for o in scene['objects'] if o.get('valid') and o['label'] == label
               and np.linalg.norm(np.asarray(o['center_base'][:2]) - measured.center_base[:2]) < .025]
    if len(matches) != 1:
        raise ValueError('目标身份存在歧义')
    return measured, matches[0]


def _pick(session, label, xy, destination, cancel):
    if session.cancel_event.is_set() or (cancel is not None and cancel.is_set()):
        raise RuntimeError('任务已停止')
    _measured, obj = _fresh_target(session, label, xy, 'rearrange_target')
    if session.cancel_event.is_set() or (cancel is not None and cancel.is_set()):
        raise RuntimeError('任务已停止')
    decision = dict(action='pick', scene_version=session.scene['scene_version'],
                    object_id=obj['object_id'], zone_name=None)
    if destination['type'] == 'named_region':
        decision['zone_name'] = destination['name']
    else:
        decision['destination'] = destination
    return session.execute(decision, cancel_event=cancel)


def _shift_candidates(session, obstacle, target, destination, protected):
    observed = [d for d in session._observed_detections if d.center_base is not None]
    peers = [d for d in observed if d.label not in (obstacle.label, target.label)] + [target, obstacle]
    zones = [name for name in session.runner.zones
             if whole_footprint_at_goal(obstacle.summary(), session.scene, {'zone_name': name})]
    if len(zones) != 1:
        return []  # A permanent adjustment must preserve an observed named region.
    zone_goal = {'label': obstacle.label, 'zone_name': zones[0]}
    constraints = [zone_goal] + [g for g in protected if g.get('label') == obstacle.label]
    origin = obstacle.center_base[:2]
    candidates = []
    # Small moves first. Try near corners too, retaining the complete footprint.
    offsets = [(dx, dy) for radius in (.04, .06, .08, .10)
               for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius),
                              (radius, radius), (radius, -radius), (-radius, radius), (-radius, -radius))]
    zone = session.runner.zones[zones[0]]
    offsets += [(float(x-origin[0]), float(y-origin[1]))
                for x in (zone.center[0]-.065, zone.center[0]+.065)
                for y in (zone.center[1]-.065, zone.center[1]+.065)]
    for dx, dy in offsets:
        if session.cancel_event.is_set():
            raise RuntimeError('任务已停止')
        xy = origin + [dx, dy]
        point = {'type': 'point', 'xy': xy.tolist()}
        predicted = _translated(obstacle, xy)
        if not all(whole_footprint_at_goal(predicted.summary(), session.scene, g) for g in constraints):
            continue
        # Cheap grasp/footprint rejection runs before the more expensive IK.
        moved_neighbors = [d for d in peers if d is not obstacle] + [predicted]
        main = _checked_placement(session, target, destination, moved_neighbors)
        if main is None:
            continue
        checked = _checked_placement(session, obstacle, point, peers)
        if checked is None:
            continue
        actual = _translated(obstacle, xy, checked['yaw'])
        if not all(whole_footprint_at_goal(actual.summary(), session.scene, g) for g in constraints):
            continue
        candidates.append(dict(destination=point, origin=origin.tolist(),
                               zone=zones[0], predicted=actual))
        break
    return candidates


def execute_held_recovery(session, option_id, cancel_event=None, confirmed=False):
    if confirmed is not True:
        raise ValueError('多步持物恢复尚未获得人工同意')
    plan = getattr(session, '_held_proposals', {}).get(option_id)
    session._held_proposals = {}
    if plan is None or plan['revision'] != session._environment_revision:
        raise ValueError('持物恢复方案已过期，请重新观察')
    original_target = copy.deepcopy(session._recovery_target)
    original_goal = copy.deepcopy(plan['goal'])
    evaluation = session._recovery_evaluation
    destination = _destination(session, original_goal)
    steps, rearranged = [], []
    staged = False
    phase = 'staging'
    target_result = None
    final_verification = None
    result = dict(ok=False, stage='held_recovery', holding=True)

    def stopped():
        return session.cancel_event.is_set() or (cancel_event is not None and cancel_event.is_set())

    def record(name, outcome):
        steps.append(dict(step=name, result=copy.deepcopy(outcome)))
        session._emit('rearrangement_step', {'message': name + ('：完成' if outcome.get('ok') else '：未完成')})
        if not outcome.get('ok'):
            raise RuntimeError(outcome.get('message') or name + '未通过核验')
        if stopped():
            raise RuntimeError('任务已停止')

    try:
        if stopped():
            raise RuntimeError('任务已停止')
        options = session._placement_options(plan['staging'])
        session._active_placement = (original_target, plan['staging'], options['placement_zone'].name, options)
        session._active_revision = session._environment_revision
        session._status = 'executing'
        pick_result, why = session.runner.execute_held_transfer(plan['bundle'])
        result = session._finish_attempt(pick_result, why, 'staged-held-target',
            options['placement_zone'].name, original_target, evaluation, plan['staging'])
        staged = bool(result.get('ok'))
        record('安全暂放原目标', result)
        phase = 'staged'
        target, _obj = _fresh_target(session, original_goal['label'], plan['staging']['xy'], 'confirm_staged_target')
        obstacle_label = plan['obstacle']
        if obstacle_label:
            # Staging frees the camera. The old cropped obstacle is never a
            # grasp input; reacquire its complete footprint now.
            obstacle, _obj = _fresh_target(session, obstacle_label, _center(session, destination), 'inspect_blocker')
            # Local destination observations retain the measured staged target.
            peers = [d for d in session._observed_detections
                     if d.center_base is not None and d.label != target.label] + [target]
            direct = _checked_placement(session, target, destination, peers)
            if direct is None:
                shifts = _shift_candidates(session, obstacle, target, destination, plan['protected_goals'])
                if not shifts:
                    raise RuntimeError('新观测下没有保持已完成目标的可行重排位置，目标已安全暂放')
                shift = shifts[0]
                adjustment = dict(label=obstacle_label, original_xy=shift['origin'],
                    approved_final_destination=shift['destination'], preserved_zone=shift['zone'],
                    status='attempting')
                rearranged.append(adjustment)
                phase = 'adjusting'
                result = _pick(session, obstacle_label, obstacle.center_base[:2], shift['destination'], cancel_event)
                adjustment['status'] = 'verified' if result.get('ok') else 'unconfirmed'
                record('调整目的区域障碍', result)
                phase = 'staged'
        phase = 'regrasping'
        result = _pick(session, original_goal['label'], plan['staging']['xy'], destination, cancel_event)
        target_result = copy.deepcopy(result)
        record('重新抓取并完成原目标', result)
        phase = 'goal_requires_verification'
        goals = [original_goal] + list(plan['protected_goals'])
        goals += [{'label': item['label'], 'zone_name': item['preserved_zone']} for item in rearranged]
        verified = session.verify_goals(goals)
        final_verification = copy.deepcopy(verified)
        if not verified['ok']:
            raise RuntimeError('重排后部分原定目标未通过相机核验')
        result.update(ok=True, final_goal_verification=verified,
                      verified_revision=session._environment_revision,
                      message='已完成暂放、必要的区域内调整及原目标抓放，并核验相关物品')
    except Exception as exc:
        holding = bool(session.snapshot().get('held_object'))
        location = ('held' if holding else 'release_requires_verification') if phase == 'staging' else (
            'confirmed_staging' if phase in ('staged', 'adjusting') else
            'held' if phase == 'regrasping' and holding else
            'requires_localization' if phase == 'regrasping' else 'goal_requires_verification')
        result = dict(ok=False, stage='held_rearrangement', label=original_goal['label'],
            zone=original_goal.get('zone_name'), controller_ok=False, camera_verified=False,
            truth={'success': False}, target_result=target_result,
            final_goal_verification=final_verification,
            holding=holding, aborted=stopped(),
            message=str(exc), error='held_rearrangement_incomplete',
            pending_rearrangement=dict(goal=original_goal, target_staged=staged,
                staging_destination=plan['staging'], adjustment_attempts=copy.deepcopy(rearranged),
                target_location_status=location))
    finally:
        # Intermediate executes replace these fields; the outer obligation is
        # always the human's original target, never the temporary destination.
        session._recovery_target = original_target
        session._recovery_goal = original_goal
        session._recovery_evaluation = evaluation
        session._publish_snapshot()
    return dict(result, rearrangement_steps=steps, rearranged_objects=rearranged,
                original_goal=original_goal)
