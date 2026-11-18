"""Camera-grounded execution of individual semantic manipulation skills.

The model selects labels and destinations. Measured coordinates, grasp anchors,
free-space searches and safety checks remain local to the existing task layer.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import numpy as np

from ..task.runner import footprint_dims, pick_summary
from .destinations import validate_destination
from .observation_policy import unique_target


class _PlanChanged(RuntimeError):
    pass


class SkillExecutor:
    """Execute one controller step without changing the user's final goals."""

    def __init__(self, session, originals=()):
        self.session = session
        self._expected_revision = None
        self.originals = copy.deepcopy(originals)
        self._free_failures = getattr(session, '_skill_free_failures', {})
        session._skill_free_failures = self._free_failures

    def _stopped(self, cancel):
        return self.session.cancel_event.is_set() or (cancel is not None and cancel.is_set())

    def _held(self):
        return bool(self.session.runner.has_held_recovery)

    def _holding(self):
        snapshot = getattr(self.session, 'snapshot', lambda: {})()
        return bool(self._held() or snapshot.get('held_object'))

    def _failure(self, step, error, message):
        return dict(ok=False, skill=step['skill'], label=step.get('label'),
                    stage=step['skill'], error=error, message=message,
                    holding=self._holding(), controller_ok=False, camera_verified=False)

    def _plan_current(self):
        if self._expected_revision is None:
            return True
        session = self.session
        return (getattr(session, '_environment_revision', None) == self._expected_revision
                and not getattr(session, '_interventions', ())
                and not session.snapshot().get('editor', {}).get('intervention_pending', 0))

    def _check_plan(self):
        if not self._plan_current():
            raise _PlanChanged('现场变化使原技能方案失效，请重新规划')

    def execute(self, step, goals, protected_goals=(), cancel_event=None, expected_revision=None):
        """Run observe/pick/place/move/verify; ask/finish/stop belong to the controller."""
        step = dict(step)
        session = self.session
        with session._operation_lock:
            session._check_open()
            session.cancel_event.external = cancel_event
            previous_revision = self._expected_revision
            self._expected_revision = expected_revision
            arm_skill = getattr(session.runner, 'skill', None)
            previous_guard = getattr(arm_skill, 'execution_guard', None)

            def guard(stage):
                if not self._plan_current():
                    return 'scene_changed'
                # The existing guard may itself apply a queued intervention.
                error = previous_guard(stage) if previous_guard is not None else None
                return error or (None if self._plan_current() else 'scene_changed')

            if arm_skill is not None and expected_revision is not None:
                arm_skill.execution_guard = guard
            try:
                if self._stopped(cancel_event):
                    return self._failure(step, 'aborted', '任务已停止')
                self._check_plan()
                skill = step.get('skill')
                if skill == 'observe':
                    return self._observe(step)
                if skill == 'pick':
                    return self._pick(step, goals, cancel_event)
                if skill == 'place':
                    return self._place(step, goals, protected_goals, cancel_event)
                if skill == 'move':
                    return self._move(step, goals, protected_goals, cancel_event)
                if skill == 'verify':
                    selected = goals
                    if step.get('goal_index') is not None:
                        selected = [self._goal(goals, step['goal_index'])]
                    if self._holding():
                        return self._failure(step, 'holding_object', '请先放置当前物品，再核验最终目标')
                    result = session.verify_goals(copy.deepcopy(list(selected)))
                    camera_ok = bool(result.get('ok'))
                    for item in result.get('items', ()):
                        goal = item['goal']
                        measured = unique_target(session._observed_detections, goal['label'])
                        evaluation = session._evaluation_object(measured) if measured is not None else None
                        truth = None
                        if evaluation is not None:
                            if goal.get('destination') is not None:
                                truth = session._verify_destination(evaluation, goal['destination'])
                            else:
                                zone = session.world.zone_of(evaluation)
                                truth = bool(evaluation is not session.world.held and zone is not None
                                             and zone.name == goal.get('zone_name'))
                        item['truth'] = {'success': truth}
                        item['ok'] = bool(item['ok'] and truth is True)
                    items = result.get('items', ())
                    result['ok'] = bool(items and len(items) == len(selected)
                                        and all(item['ok'] for item in items))
                    truths = [item['truth']['success'] for item in result.get('items', ())]
                    result['truth'] = {'success': False if False in truths else
                                      True if truths and all(t is True for t in truths) else None}
                    if not result['ok']:
                        result.update(error='verification_incomplete',
                                      message='相机与独立核验未全部确认原定目标')
                    return dict(result, skill=skill, holding=False,
                                camera_verified=camera_ok)
                return self._failure(step, 'invalid_skill', '该步骤不是可执行的基本技能')
            except _PlanChanged as exc:
                return self._failure(step, 'scene_changed', str(exc))
            except (ValueError, KeyError) as exc:
                return self._failure(step, 'invalid_step', str(exc))
            finally:
                if arm_skill is not None and expected_revision is not None:
                    arm_skill.execution_guard = previous_guard
                self._expected_revision = previous_revision
                session.cancel_event.external = None

    @staticmethod
    def _goal(goals, index):
        if type(index) is not int or index < 0 or index >= len(goals):
            raise ValueError('目标编号超出任务范围')
        return goals[index]

    def _reference(self, label):
        """Use only task-start or current camera summaries as observation hints."""
        scene = self.session.scene
        current = [o for o in scene.get('objects', ())
                   if o.get('valid') and o.get('label') == label and o.get('center_base') is not None]
        if len(current) == 1:
            return copy.deepcopy(current[0])
        original = self.originals.values() if isinstance(self.originals, Mapping) else self.originals
        matches = [o for o in original if o.get('label') == label and o.get('center_base') is not None]
        return copy.deepcopy(matches[0]) if len(matches) == 1 else {'label': label}

    def _fresh_target(self, label, purpose):
        session = self.session
        reference = self._reference(label)
        reference.update(label=label, require_grasp=True)
        scene = session.observe(target=reference, purpose=purpose)
        det = unique_target(session._last_fresh, label,
                            partial=lambda d: session._target_partial(d, reference))
        if det is None or session._observation_revision != session._environment_revision:
            raise ValueError('新相机图像未唯一确认目标，未执行旧坐标抓取')
        matches = [obj for obj in scene.get('objects', ())
                   if obj.get('valid') and obj.get('label') == label
                   and obj.get('center_base') is not None]
        if (len(matches) != 1 or
                np.linalg.norm(np.asarray(matches[0]['center_base'][:2]) - det.center_base[:2]) >= .025):
            raise ValueError('相机目标身份存在歧义')
        return det, matches[0]

    def _stationary(self):
        """Refresh held-state surroundings without an unchecked camera move."""
        session = self.session
        from ..core.extrinsics import camera_pose_in_base
        from .observation_policy import measurement_visible, merge_local_detections

        session._apply_interventions()
        revision = session._environment_revision
        fresh = session._stationary_recovery_observation()
        flange = session.arm.get_flange_pose()
        pose = camera_pose_in_base(session.runner.ext, flange.value, session.runner.T_flange_tcp)
        frame = session.camera._last
        target = session.runner._held_recovery_context['det']
        # The held target uses its original measured anchor. Its current image
        # may be cropped by the gripper and must never replace that anchor.
        fresh = [d for d in fresh if d.center_base is not None and d.label != target.label]
        neighbors = merge_local_detections(
            session._observed_detections, fresh,
            lambda d: measurement_visible(d, frame, pose), forget=[target])
        session._last_fresh = fresh
        session._observation_revision = revision
        session._observed_detections = copy.deepcopy(neighbors)
        session._recovery_neighbors = copy.deepcopy(neighbors)
        session._publish_scene(neighbors, [], [], session.runner._last_stamp)
        return neighbors

    def _observe(self, step):
        session = self.session
        if self._held():
            self._stationary()
            scene = session.scene
        elif self._holding():
            return self._failure(step, 'held_context_missing', '当前仍有持物反馈，缺少安全续接记录，已保持当前位置')
        else:
            label = step.get('label')
            scene = session.observe(target=self._reference(label) if label else None,
                                    purpose='skill_observe')
        session._status = 'ready'
        session._publish_snapshot()
        return dict(ok=True, skill='observe', stage='observed', holding=self._held(),
                    scene=copy.deepcopy(scene), camera_verified=True,
                    message='已获得新的相机观测')

    def _pick(self, step, goals, cancel):
        session = self.session
        label = step.get('label')
        if self._held() or session.snapshot().get('held_object'):
            return self._failure(step, 'holding_object', '请先放置当前物品，再抓取其他物品')
        if label not in session.runner.class_to_zone:
            raise ValueError('物品类别不在当前配置中')
        det, obj = self._fresh_target(label, 'skill_pick')
        self._check_plan()
        if self._stopped(cancel):
            return self._failure(step, 'aborted', '任务已停止')
        target = copy.deepcopy(det)
        neighbors = [target] + [copy.deepcopy(d) for d in session._observed_detections
                                if d.center_base is not None and not (
                                    d.label == label and
                                    np.linalg.norm(d.center_base[:2] - det.center_base[:2]) < .04)]
        session._active_revision = session._observation_revision
        session._recovery_target = copy.deepcopy(det)
        session._recovery_neighbors = copy.deepcopy(neighbors[1:])
        original_goal = next((g for g in goals if g.get('label') == label), None)
        session._recovery_goal = copy.deepcopy(original_goal)
        # Independent association is for result evaluation only.
        session._recovery_evaluation = session._evaluation_object(det)
        session._recovery_context = None
        session._feedback = {'grasp_plan': {}, 'checks': []}
        session._status = 'executing'
        session._publish_snapshot()
        result, why = session.runner.pick_only_detection(target, neighbors=neighbors)
        summary = (pick_summary(result) if result is not None else
                   dict(ok=False, stage='plan', error=why, label=label))
        summary.update(skill='pick', object_id=obj['object_id'],
                       holding=bool(self._held() or (result and result.detail.get('holding'))),
                       controller_ok=bool(result and result.ok),
                       camera_verified=True, source_xy=det.center_base[:2].tolist(),
                       scene=copy.deepcopy(session.scene))
        # A pick-only result requires the saved measured anchor and live width,
        # even if the simulated world happens to report an object in the gripper.
        if summary.get('ok'):
            context, _width, feedback_why = session.runner._held_transfer_feedback(det)
            summary['ok'] = context is not None
            if context is None:
                summary['error'] = feedback_why
        summary['aborted'] = self._stopped(cancel)
        if summary['aborted']:
            summary['ok'], summary['error'] = False, 'aborted'
        summary['message'] = ('已抓取并保持夹持' if summary['ok'] else
                              '抓取步骤未完成，已保留执行反馈')
        session._invalidate_scene('pick-only changed the scene')
        session._status = 'ready' if not summary['aborted'] else 'stopped'
        session._publish_snapshot()
        session._emit('execution_result', summary)
        return summary

    def _resolve(self, value, goals):
        if not isinstance(value, Mapping):
            raise ValueError('该技能需要放置目标')
        kind = value.get('type')
        if kind == 'goal':
            goal = self._goal(goals, value.get('index'))
            if goal.get('destination') is not None:
                return validate_destination(copy.deepcopy(goal['destination']), self.session.scene), None
            value = {'type': 'zone', 'name': goal.get('zone_name')}
            kind = 'zone'
        if kind in ('zone', 'named_region'):
            if value.get('name') not in self.session.runner.zones:
                raise ValueError('放置区域不在当前配置中')
            return {'type': 'named_region', 'name': value['name']}, None
        if kind == 'original':
            originals = self.originals.values() if isinstance(self.originals, Mapping) else self.originals
            matches = [o for o in originals if o.get('label') == value.get('label')]
            if len(matches) != 1:
                raise ValueError('任务开始时没有唯一测量的原位置')
            item = matches[0]
            xy = item.get('center_base')
            yaw = item.get('yaw')
            if xy is None or yaw is None or not math.isfinite(float(yaw)):
                raise ValueError('原位置缺少完整相机坐标或朝向')
            return validate_destination({'type': 'point', 'xy': list(xy[:2])}, self.session.scene), float(yaw)
        if kind == 'auto_free':
            return dict(value), None
        return validate_destination(dict(value), self.session.scene), None

    def _options(self, destination):
        if destination['type'] == 'named_region':
            zone = self.session.runner.zones[destination['name']]
            return dict(placement_zone=copy.deepcopy(zone), placement_polygon=zone.corners_xy(),
                        exact_xy=False)
        return self.session._placement_options(destination)

    def _candidates(self, target, specification, neighbors):
        """Sample the configured tabletop; no fixed staging anchors or truth."""
        revision = self.session._environment_revision
        # A real scene intervention permits reconsidering all candidates. During
        # ordinary replanning, a failed release must not select the same point.
        for key in list(self._free_failures):
            if key[1] != revision:
                del self._free_failures[key]
        failed = self._free_failures.get((target.label, revision), set())
        placement = self.session._placement
        table, bounds = placement['table_bounds'], placement['bounds']
        dims = footprint_dims(target, self.session.runner.nominal.get(target.label))
        if dims is None:
            return []
        margin = max(.025, max(dims) / 2 + .012)
        low = np.array([max(table[a][0], bounds[a][0]) + margin for a in ('x', 'y')])
        high = np.array([min(table[a][1], bounds[a][1]) - margin for a in ('x', 'y')])
        if np.any(high <= low):
            return []
        spacing = max(.025, min(.05, min(dims) + .01))
        axes = [np.linspace(lo, hi, max(2, int(math.ceil((hi - lo) / spacing)) + 1))
                for lo, hi in zip(low, high)]
        away_labels = set(specification.get('away_from') or ())
        away = [d.center_base[:2] for d in neighbors
                if d.center_base is not None and d.label in away_labels]
        away += [target.center_base[:2]]
        occupied = [d.center_base[:2] for d in neighbors if d.center_base is not None]
        points = []
        for x in axes[0]:
            for y in axes[1]:
                destination = {'type': 'point', 'xy': [float(x), float(y)]}
                if tuple(round(float(v), 4) for v in destination['xy']) in failed:
                    continue
                try:
                    validate_destination(destination, self.session.scene)
                except ValueError:
                    continue
                clearance = min((np.linalg.norm(np.array([x, y]) - p) for p in occupied), default=.25)
                separation = min(np.linalg.norm(np.array([x, y]) - p) for p in away)
                points.append((min(clearance, .18) + min(separation, .25), destination))
        points.sort(key=lambda item: (-item[0], item[1]['xy']))
        return [destination for _score, destination in points]

    def _remember_failed_free(self, label, destination):
        key = (label, self.session._environment_revision)
        point = tuple(round(float(v), 4) for v in destination['xy'])
        self._free_failures.setdefault(key, set()).add(point)

    def _place(self, step, goals, protected, cancel):
        session = self.session
        context = session.runner._held_recovery_context
        if context is None or context['det'].label != step.get('label'):
            return self._failure(step, 'no_held_target', '当前没有该物品的已测量持物记录')
        destination, yaw = self._resolve(step.get('destination'), goals)
        automatic = destination['type'] == 'auto_free'
        target = copy.deepcopy(context['det'])
        neighbors = self._stationary()
        self._check_plan()
        if self._stopped(cancel):
            return self._failure(step, 'aborted', '任务已停止')
        candidates = (self._candidates(target, destination, neighbors)
                      if destination['type'] == 'auto_free' else [destination])
        previous_yaw = context['det'].extra.get('_restore_yaw')
        if yaw is not None:
            context['det'].extra['_restore_yaw'] = yaw
            target.extra['_restore_yaw'] = yaw
        bundle, why, options = None, '没有通过严格检查的桌面空位', None
        try:
            for candidate in candidates:
                if self._stopped(cancel):
                    return self._failure(step, 'aborted', '任务已停止')
                options = self._options(candidate)
                bundle, why = session.runner.plan_held_transfer(target, neighbors, **options)
                if bundle is not None:
                    destination = candidate
                    break
            if bundle is None:
                return self._failure(step, 'placement_infeasible', why)
            target = bundle['det']
            zone_name = options['placement_zone'].name
            external_destination = None if destination['type'] == 'named_region' else destination
            session._active_placement = (target, external_destination, zone_name, options)
            session._active_revision = session._environment_revision
            session._placement_center = None
            session._status = 'executing'
            session._publish_snapshot()
            self._check_plan()
            result, why = session.runner.execute_held_transfer(bundle)
            outcome = session._finish_attempt(result, why, 'skill-held-target', zone_name,
                                             target, session._recovery_evaluation,
                                             external_destination)
            if automatic and result is not None and not outcome.get('ok'):
                self._remember_failed_free(target.label, destination)
            return dict(outcome, skill='place', resolved_destination=copy.deepcopy(destination))
        finally:
            # Restore the metadata on a retained grasp without changing its
            # measured transform. Final goals always remain controller-owned.
            held = session.runner._held_recovery_context
            if held is not None:
                if previous_yaw is None:
                    held['det'].extra.pop('_restore_yaw', None)
                else:
                    held['det'].extra['_restore_yaw'] = previous_yaw

    def _move(self, step, goals, protected, cancel):
        session = self.session
        if self._held() or session.snapshot().get('held_object'):
            return self._failure(step, 'holding_object', '请先放置当前物品，再搬动物品')
        destination, yaw = self._resolve(step.get('destination'), goals)
        automatic = destination['type'] == 'auto_free'
        det, obj = self._fresh_target(step.get('label'), 'recover_queued_target')
        self._check_plan()
        if self._stopped(cancel):
            return self._failure(step, 'aborted', '任务已停止')
        if destination['type'] == 'auto_free':
            from .obstacle_recovery import _checked_placement
            selected = None
            for candidate in self._candidates(det, destination, session._observed_detections):
                if self._stopped(cancel):
                    return self._failure(step, 'aborted', '任务已停止')
                if _checked_placement(session, det, candidate, session._observed_detections) is not None:
                    selected = candidate
                    break
            if selected is None:
                return self._failure(step, 'placement_infeasible', '没有通过严格检查的桌面空位')
            destination = selected
        decision = dict(action='pick', scene_version=session.scene['scene_version'],
                        object_id=obj['object_id'], zone_name=None)
        if destination['type'] == 'named_region':
            decision['zone_name'] = destination['name']
        else:
            decision['destination'] = destination
        if yaw is not None:
            decision['_restore_yaw'] = yaw
        self._check_plan()
        outcome = session.execute(decision, cancel_event=cancel)
        if automatic and not outcome.get('ok'):
            self._remember_failed_free(det.label, destination)
        return dict(outcome, skill='move', resolved_destination=copy.deepcopy(destination))
