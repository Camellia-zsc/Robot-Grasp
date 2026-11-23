"""Camera-grounded execution of individual semantic manipulation skills.

The model selects labels and destinations. Measured coordinates, grasp anchors,
free-space searches and safety checks remain local to the existing task layer.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np

from ..task.runner import footprint_dims, pick_summary
from .destinations import validate_destination
from .observation_policy import unique_target


class _PlanChanged(RuntimeError):
    pass


class _TargetUnavailable(ValueError):
    """No grasp was dispatched because fresh camera identity is unresolved."""


class SkillExecutor:
    """Execute one controller step without changing the user's final goals."""

    def __init__(self, session, originals=()):
        self.session = session
        self._expected_revision = None
        self.originals = copy.deepcopy(originals)
        self._free_failures = getattr(session, '_skill_free_failures', {})
        session._skill_free_failures = self._free_failures

    def capabilities(self):
        physical = bool(getattr(self.session, '_physics', None))
        return dict(skills=['observe', 'pick', 'place', 'move', 'verify', 'ask', 'finish'],
            observe_modes=['stationary', 'active'],
            observe_destination_types=['goal', 'zone', 'region', 'original', 'on_object'],
            destination_types=['goal', 'zone', 'region', 'auto_free', 'original', 'on_object'],
            holding_allowed_skills=['observe', 'place', 'ask'],
            active_observe_requires_geometry=True, multi_object_rearrangement=True,
            restores_original_pose=True, stacking=physical, physical_stacking=physical,
            incremental_observation=bool(getattr(self.session, '_motion_observer', None)),
            automatic_righting=False,
            gravity_contacts=physical, physics_backend='mujoco' if physical else 'sim')

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

    @contextmanager
    def _guarded_operation(self, cancel_event, expected_revision):
        session = self.session
        with session._operation_lock:
            session._check_open()
            previous_external = getattr(session.cancel_event, 'external', None)
            session.cancel_event.external = cancel_event
            previous_revision = self._expected_revision
            self._expected_revision = expected_revision
            arm_skill = getattr(session.runner, 'skill', None)
            previous_guard = getattr(arm_skill, 'execution_guard', None)

            def guard(stage):
                # The existing guard may itself apply a queued intervention.
                error = previous_guard(stage) if previous_guard is not None else None
                if (error is None and stage == 'held_transit'
                        and getattr(session, '_motion_replan_required', False)
                        and getattr(session, '_active_revision', None) == session._environment_revision):
                    self._expected_revision = session._environment_revision
                return error or (None if self._plan_current() else 'scene_changed')

            if arm_skill is not None and expected_revision is not None:
                arm_skill.execution_guard = guard
            try:
                yield
            finally:
                if arm_skill is not None and expected_revision is not None:
                    arm_skill.execution_guard = previous_guard
                self._expected_revision = previous_revision
                session.cancel_event.external = previous_external

    def check_completed(self, step, goals, cancel_event=None, expected_revision=None):
        """Sense a resolved move's postcondition without dispatching any grasp.

        The controller may also use this for an explicit pick/place pair whose
        placement postcondition is known. A generic pick has no such condition.
        """
        step = dict(step)
        with self._guarded_operation(cancel_event, expected_revision):
            try:
                if self._stopped(cancel_event):
                    return self._failure(step, 'aborted', '任务已停止')
                self._check_plan()
                if self._holding():
                    return None
                destination, yaw = self._resolve(step.get('destination'), goals)
                return self._completed_move(step, destination, yaw, cancel_event)
            except _PlanChanged as exc:
                return self._failure(step, 'scene_changed', str(exc))
            except (ValueError, KeyError) as exc:
                return self._failure(step, 'invalid_step', str(exc))

    @staticmethod
    def _independent_goal_move(step, goals):
        """Only a final single-object move can be postponed independently."""
        if step.get('skill') != 'move':
            return None
        destination = step.get('destination') or {}
        for index, goal in enumerate(goals):
            if goal.get('label') != step.get('label'):
                continue
            if destination.get('type') == 'goal' and destination.get('index') == index:
                return index
            if (destination.get('type') == 'zone' and goal.get('destination') is None
                    and destination.get('name') == goal.get('zone_name')):
                return index
            if destination == goal.get('destination') and destination.get('type') == 'region':
                return index
        return None

    def _can_defer(self, step, goals):
        session = self.session
        if (self._independent_goal_move(step, goals) is None or self._holding()
                or self._stopped(None) or not self._plan_current()):
            return False
        health_fn = getattr(getattr(session, 'arm', None), 'health', None)
        health = health_fn() if health_fn is not None else None
        return health is not None and health.ok

    def _deferred_target(self, step, reason):
        return dict(self._failure(step, 'target_not_observed',
            '该目标尚未获得唯一可靠测量，暂缓搬运；保留最终目标，先处理其他可确认物品'),
            deferred=True, manipulation_started=False, changed=False, reason=str(reason)[:200])

    def defer_unobserved(self, step, goals):
        """Refuse an unresolved final move without scanning or commanding a grasp."""
        if not self._can_defer(step, goals):
            return None
        from .goal_readiness import goal_worklist
        index = self._independent_goal_move(step, goals)
        item = goal_worklist(self.session.scene, goals)[index]
        return self._deferred_target(step, item['reason']) if item['status'] == 'deferred' else None

    def execute(self, step, goals, protected_goals=(), cancel_event=None, expected_revision=None):
        """Run observe/pick/place/move/verify; ask/finish/stop belong to the controller."""
        step = dict(step)
        session = self.session
        with self._guarded_operation(cancel_event, expected_revision):
            try:
                if self._stopped(cancel_event):
                    return self._failure(step, 'aborted', '任务已停止')
                self._check_plan()
                skill = step.get('skill')
                deferred = self.defer_unobserved(step, goals)
                if deferred is not None:
                    return deferred
                if skill == 'observe':
                    return self._observe(step, goals)
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
                    targets = getattr(result, 'targets', ())
                    for index, item in enumerate(result.get('items', ())):
                        # verify_goals owns the camera-only verdict here.
                        # Keep each camera result before combining it with
                        # independent physical truth, including partial audits.
                        item['camera_verified'] = (item.get('ok') is True
                                                   and item.get('camera_verified', True) is True)
                        goal = item['goal']
                        # Each destination can require a different camera view.
                        # A later crop must not replace the earlier full target
                        # before its independent physical check.
                        measured = (targets[index] if len(targets) == len(result['items'])
                                    else unique_target(session._observed_detections, goal['label']))
                        evaluation = session._evaluation_object(measured) if measured is not None else None
                        truth = None
                        if evaluation is not None:
                            if goal.get('destination') is not None:
                                truth = session._verify_destination(evaluation, goal['destination'])
                            elif hasattr(session, '_verify_zone'):
                                truth = session._verify_zone(evaluation, goal['zone_name'])
                            else:
                                zone = session.world.zone_of(evaluation)
                                truth = bool(evaluation is not session.world.held and zone is not None
                                             and zone.name == goal.get('zone_name'))
                                verify_zone = getattr(session, '_verify_zone', None)
                                if verify_zone is not None:
                                    truth = bool(verify_zone(evaluation, goal['zone_name']))
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
            except _TargetUnavailable as exc:
                if self._can_defer(step, goals):
                    return self._deferred_target(step, str(exc))
                return self._failure(step, 'invalid_step', str(exc))
            except (ValueError, KeyError) as exc:
                return self._failure(step, 'invalid_step', str(exc))

    @staticmethod
    def _goal(goals, index):
        if type(index) is not int or index < 0 or index >= len(goals):
            raise ValueError('目标编号超出任务范围')
        return goals[index]

    def _reference(self, label, *, allow_partial=False):
        """Use only task-start or current camera summaries as observation hints."""
        scene = self.session.scene
        current = [o for o in scene.get('objects', ())
                   if o.get('valid') and o.get('label') == label and o.get('center_base') is not None]
        if len(current) == 1:
            return copy.deepcopy(current[0])
        if allow_partial:
            hints = [o for o in scene.get('objects', ())
                     if o.get('label') == label and o.get('center_base') is not None]
            if len(hints) == 1:
                # A cropped silhouette can aim a new camera view, but it never
                # becomes an executable detection or a replacement grasp.
                hint = copy.deepcopy(hints[0])
                hint['_observation_hint_only'] = True
                return hint
        original = self.originals.values() if isinstance(self.originals, Mapping) else self.originals
        original = list(original)
        matches = [o for o in original if o.get('label') == label and o.get('center_base') is not None]
        if len(matches) == 1:
            return copy.deepcopy(matches[0])
        if allow_partial:
            # An upper object's measured support patch can aim a camera at
            # the hidden base after the upper object has been moved away.
            # The patch supplies no complete base outline or grasp authority.
            hints = []
            for upper in list(scene.get('objects', ())) + original:
                if upper.get('support_observed') is not True or upper.get('support_label') != label:
                    continue
                try:
                    polygon = np.asarray(upper['support_polygon'], float)
                    top = float(upper['support_z'])
                except (KeyError, TypeError, ValueError):
                    continue
                if (polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3
                        or not np.all(np.isfinite(polygon)) or not math.isfinite(top)):
                    continue
                hints.append(np.r_[polygon.mean(axis=0), top])
            if hints and max(np.linalg.norm(point - hints[0]) for point in hints) <= .04:
                return dict(label=label, valid=False, center_base=hints[0].tolist(),
                    top_z=float(hints[0][2]), reject_reason='remembered_support_aiming_hint',
                    _observation_hint_only=True)
            editor = getattr(self.session, 'editor', None)
            if editor is not None and editor.metadata().get('one_per_class'):
                complete = [o for o in scene.get('objects', ()) if o.get('valid')
                            and o.get('center_base') is not None]
                unexplained = []
                for fragment in scene.get('objects', ()):
                    point = fragment.get('center_base')
                    if fragment.get('valid') or point is None:
                        continue
                    if any(np.linalg.norm(np.asarray(point[:2])-np.asarray(o['center_base'][:2])) < .065
                           for o in complete):
                        continue
                    if not any(np.linalg.norm(np.asarray(point[:2])-np.asarray(o['center_base'][:2])) < .04
                               for o in unexplained):
                        unexplained.append(fragment)
                if len(unexplained) == 1:
                    # Cropping can change the classifier's label. Reinspect
                    # the sole unexplained fragment; never relabel it as a
                    # graspable object or assert it is the missing target.
                    hint = copy.deepcopy(unexplained[0])
                    hint.update(label=label, valid=False, _observation_hint_only=True,
                                reject_reason='unexplained_fragment_aiming_hint')
                    return hint
        return {'label': label}

    def _fresh_target(self, label, purpose, support_label=None):
        session = self.session
        if support_label is not None:
            from .stacking import require_flat_support
            require_flat_support(support_label)
        reference = self._reference(label, allow_partial=True)
        reference.update(label=label, require_grasp=True)
        if reference.get('_observation_hint_only'):
            reference['_observe_source'] = True
        if support_label is not None:
            reference['required_support'] = self._reference(support_label)
        reusable = getattr(session, '_reusable_target', lambda target: False)(reference)
        if reusable:
            scene = session.scene
            session._last_observation_purpose = purpose
        else:
            # A complete silhouette can still lack the two views/contact
            # geometry needed for a grasp (especially above another object).
            # Request useful source views, not the same fixed table cameras.
            reference['_observe_source'] = True
            scene = session.observe(target=reference, purpose=purpose)
        det = unique_target(session._last_fresh, label,
                            partial=lambda d: session._target_partial(d, reference))
        if session._observation_revision != session._environment_revision:
            raise _PlanChanged('新相机观测期间现场变化，未执行旧坐标抓取')
        if det is None:
            raise _TargetUnavailable('新相机图像未唯一确认目标，未执行旧坐标抓取')
        if support_label is not None and not session._support_observed(
                reference, session._last_fresh, session._observation_revision):
            raise ValueError('新相机观测未同时完整确认抓取物品与指定底座，未执行抓取')
        matches = [obj for obj in scene.get('objects', ())
                   if obj.get('valid') and obj.get('label') == label
                   and obj.get('center_base') is not None]
        if (len(matches) != 1 or
                np.linalg.norm(np.asarray(matches[0]['center_base'][:2]) - det.center_base[:2]) >= .025):
            raise _TargetUnavailable('相机目标身份存在歧义')
        return det, matches[0]

    def _stationary(self):
        """Refresh held-state surroundings without an unchecked camera move."""
        session = self.session
        from ..core.extrinsics import camera_pose_in_base
        from .observation_policy import measurement_absent, measurement_visible, merge_local_detections

        session._apply_interventions()
        revision = session._environment_revision
        fresh = session._stationary_recovery_observation()
        captured = fresh
        flange = session.arm.get_flange_pose()
        pose = camera_pose_in_base(session.runner.ext, flange.value, session.runner.T_flange_tcp)
        frame = session.camera._last
        context = session.runner._held_recovery_context
        target = context['det'] if context is not None else None
        # The held target uses its original measured anchor. Its current image
        # may be cropped by the gripper and must never replace that anchor.
        fresh = [d for d in fresh if d.center_base is not None
                 and (target is None or d.label != target.label)]
        neighbors = merge_local_detections(
            session._observed_detections, fresh,
            lambda d: (measurement_absent if getattr(session, '_motion_observer', None) is not None
                       else measurement_visible)(d, frame, pose),
            forget=[target] if target is not None else [],
            preserve_incomplete=getattr(session, '_motion_observer', None) is not None)
        session._last_fresh = fresh
        session._observation_revision = revision
        session._observed_detections = copy.deepcopy(neighbors)
        session._recovery_neighbors = copy.deepcopy(neighbors)
        session._publish_scene(neighbors, [], [], session.runner._last_stamp)
        from .carried_observation import CarriedDetections
        return CarriedDetections(neighbors, getattr(captured, 'anchor_detections', captured))

    def _observation_focus(self, step, goals):
        specification = step.get('destination')
        if specification is None and not step.get('label'):
            goal = getattr(self.session, '_recovery_goal', None)
            if goal:
                specification = (goal.get('destination') or
                                 {'type': 'zone', 'name': goal.get('zone_name')})
        if specification is not None:
            destination, _yaw = self._resolve(specification, goals)
            if destination['type'] == 'auto_free':
                raise ValueError('主动观察需要一个已有目的区域或物品位置')
            if destination['type'] == 'on_object':
                reference = self._reference(destination['label'])
                if reference.get('center_base') is None:
                    raise ValueError('缺少底座的相机位置线索，请先扫描寻找底座')
                xy = np.asarray(reference['center_base'][:2], float)
                # Observation can revisit a temporarily occluded measured base;
                # execution still requires a new complete support measurement.
                return xy, np.array([[-.07, -.07], [-.07, .07], [.07, .07], [.07, -.07]]) + xy
            options = self._options(destination)
            zone, polygon = options['placement_zone'], options['placement_polygon']
            xy = np.asarray(zone.center[:2], float)
            if destination['type'] != 'point':
                return xy, np.asarray(polygon, float)
        elif step.get('label'):
            reference = self._reference(step['label'])
            if reference.get('center_base') is None:
                raise ValueError('缺少该物品的实测观察位置，请指定已有目的区域')
            xy = np.asarray(reference['center_base'][:2], float)
        else:
            raise ValueError('主动观察需要已有目标区域或物品位置')
        return xy, np.array([[-.10, -.10], [-.10, .10], [.10, .10], [.10, -.10]]) + xy

    def _active_observe(self, step, goals):
        from .active_observation import inspect_held
        xy, polygon = self._observation_focus(step, goals)
        # Camera inspection moves the carried target just like transport.
        # Physics must still invalidate moved obstacles, while the commanded
        # held body is checked by the normal grip-feedback guards instead.
        self.session._status = 'executing'
        def stationary():
            try:
                return self._stationary()
            finally:
                # Publishing a fresh scene normally marks it ready. This
                # camera capture is inside an unfinished held-motion action.
                self.session._status = 'executing'
        try:
            result = inspect_held(self.session, xy, polygon, stationary, self._check_plan)
        finally:
            self.session._status = 'stopped' if self.session.cancel_event.is_set() else 'ready'
            self.session._publish_snapshot()
        return dict(result, skill='observe', scene=copy.deepcopy(self.session.scene))

    def _observe(self, step, goals):
        session = self.session
        from .active_observation import measurements_changed
        before = copy.deepcopy(session.scene.get('objects', []))
        if self._held():
            if step.get('mode') == 'active':
                return self._active_observe(step, goals)
            self._stationary()
            scene = session.scene
        elif self._holding():
            return self._failure(step, 'held_context_missing', '当前仍有持物反馈，缺少安全续接记录，已保持当前位置')
        elif step.get('mode') == 'stationary':
            self._stationary()
            scene = session.scene
        else:
            label = step.get('label')
            reference = self._reference(label, allow_partial=True) if label else None
            source_hint = bool(reference and (reference.get('center_base') is not None
                                             or reference.get('xy') is not None))
            if source_hint:
                # An object search aims at its measured source. A placement
                # destination is only a fallback when its location is unknown.
                reference['_observe_source'] = True
            elif step.get('destination') is not None:
                xy, _polygon = self._observation_focus(step, goals)
                destination = {'type': 'point', 'xy': xy.tolist()}
                session._ensure_destination_view(destination)
                reference = {'xy': xy.tolist()}
                if label:
                    reference['label'] = label
            from .simulation import _SourceObservationUnavailable
            try:
                scene = session.observe(target=reference, purpose='skill_observe')
            except _SourceObservationUnavailable as exc:
                return self._failure(step, 'no_safe_observation_path', str(exc))
        session._status = 'ready'
        session._publish_snapshot()
        return dict(ok=True, skill='observe', stage='observed', holding=self._held(),
                    changed=measurements_changed(before, scene.get('objects', [])),
                    observation_mode='stationary' if self._held() or step.get('mode') == 'stationary' else 'active_local',
                    reason='camera_measurement_refresh',
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
                return validate_destination(copy.deepcopy(goal['destination']), self.session.scene,
                    require_observed=goal['destination'].get('type') != 'on_object'), None
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
            from .catalog import restoration_yaw_period
            period = restoration_yaw_period(item['label'],item.get('resting_pose'))
            if period is None and xy is not None:
                return validate_destination({'type':'point','xy':list(xy[:2])},self.session.scene),None
            if xy is None or yaw is None or not math.isfinite(float(yaw)):
                raise ValueError('原位置缺少完整相机坐标或朝向')
            return validate_destination({'type': 'point', 'xy': list(xy[:2])}, self.session.scene), float(yaw)
        if kind == 'auto_free':
            return dict(value), None
        if kind == 'on_object':
            return validate_destination(dict(value), self.session.scene, require_observed=False), None
        return validate_destination(dict(value), self.session.scene), None

    def _options(self, destination):
        if destination['type'] == 'named_region':
            zone = self.session.runner.zones[destination['name']]
            return dict(placement_zone=copy.deepcopy(zone), placement_polygon=zone.corners_xy(),
                        exact_xy=False)
        return self.session._placement_options(destination)

    def _current_measurement(self, label, *, fresh=True):
        """A completion claim needs fresh camera data and one scene identity."""
        session = self.session
        revision = getattr(session, '_environment_revision', None)
        observer = getattr(session, '_motion_observer', None)
        if ((observer is None and getattr(session, '_last_observation_purpose', '') == 'consumed_for_pick') or
                getattr(session, '_interventions', ()) or
                session.snapshot().get('editor', {}).get('intervention_pending', 0)):
            return None
        if fresh and observer is None and getattr(session, '_observation_revision', None) != revision:
            return None
        detections = (observer.current(label) if observer is not None else
                      getattr(session, '_last_fresh' if fresh else '_observed_detections', ()))
        if not fresh:
            # Older views may prove that a grasp is still needed, but may not
            # prove completion after a later robot motion in the same revision.
            detections = [d for d in detections
                          if d.extra.get('_observed_revision') == revision]
        det = unique_target(detections, label)
        matches = [obj for obj in session.scene.get('objects', ())
                   if obj.get('valid') and obj.get('label') == label
                   and obj.get('center_base') is not None]
        if (det is None or len(matches) != 1 or
                np.linalg.norm(np.asarray(matches[0]['center_base'][:2]) - det.center_base[:2]) >= .025):
            return None
        return det, matches[0]

    def _at_destination(self, det, destination, yaw):
        """Completion is full supported geometry, including requested restore yaw."""
        from ..perception.grasp import detection_table_z
        from .batch import whole_footprint_in_zone
        from .destinations import whole_footprint_at_destination

        session = self.session
        measured = det.summary()
        footprint_ok = (whole_footprint_in_zone(measured, session.scene, destination['name'])
                        if destination['type'] == 'named_region' else
                        whole_footprint_at_destination(measured, session.scene, destination))
        if not footprint_ok:
            return False
        floor = (session.runner.zones[destination['name']].center[2]
                 if destination['type'] == 'named_region' else
                 session.cfg.get('perception.table.expected_z', None))
        support = detection_table_z(det)
        if (floor is None or support is None or not math.isfinite(float(floor)) or
                not math.isfinite(float(support)) or abs(support - float(floor)) > .01 or
                det.height is None or not math.isfinite(float(det.height)) or det.height <= 0):
            return False
        if yaw is not None:
            from .catalog import restoration_yaw_period
            period = restoration_yaw_period(det.label,det.extra.get('resting_pose'))
            if period is None:
                return True
            if (det.yaw is None or not math.isfinite(float(det.yaw)) or
                    abs(math.remainder(float(det.yaw) - yaw, period)) > math.radians(10)):
                return False
        return True

    def _completed_move(self, step, destination, yaw, cancel, *, observe=True):
        """Prove this step's postcondition before spending work on its grasp."""
        session = self.session
        if destination['type'] == 'on_object':
            from .stacking import camera_confirmed, audit
            if self._holding() or not camera_confirmed(session, step['label'], destination):
                return None
            det = unique_target(session._last_fresh, step['label'])
            truth = audit(session, session._evaluation_object(det), destination)
            return dict(ok=truth.get('success') is True, skipped=True, changed=False,
                skill=step['skill'], label=step['label'], stage='already_satisfied',
                camera_verified=True, controller_ok=True, holding=False,
                verified_revision=session._environment_revision, truth=truth,
                resolved_destination=copy.deepcopy(destination),
                message='相机与物理接触确认已叠放，跳过重复抓取' if truth.get('success') else
                        '已看到叠放目标，物理支撑或落稳尚未确认，未重复抓取')
        if (destination['type'] == 'auto_free' or self._holding() or
                not getattr(session, 'scene', None)):
            return None
        label = step.get('label')
        current = self._current_measurement(label)
        if current is None:
            known = self._current_measurement(label, fresh=False)
            if known is not None and not self._at_destination(known[0], destination, yaw):
                return None
            if not observe or not hasattr(session, 'observe'):
                return None
            if destination['type'] == 'named_region':
                xy = session.runner.zones[destination['name']].center[:2]
            elif destination['type'] == 'point':
                session._ensure_destination_view(destination)
                xy = destination['xy']
            else:
                polygon = np.asarray(session.scene['placement']['pick_polygon'], float)
                xy = (polygon.min(axis=0) + polygon.max(axis=0)) / 2
            editor = getattr(session, 'editor', None)
            one_per_class = bool(editor is not None and editor.metadata().get('one_per_class'))
            # A human can move the sole class instance outside the old view.
            # Its previous-revision source memory must not become a second
            # phantom identity when the destination camera finds it again.
            # Same-revision duplicates remain ambiguous and are never erased.
            forget = [d for d in getattr(session, '_observed_detections', ())
                      if one_per_class and d.label == label
                      and d.extra.get('_observed_revision') != session._environment_revision]
            self._check_plan()
            session.observe(target={'xy': list(xy), 'label': label}, purpose='verify_place', forget=forget)
            self._check_plan()
            if self._stopped(cancel):
                return self._failure(step, 'aborted', '任务已停止')
            current = self._current_measurement(label)
        if current is None or not self._at_destination(current[0], destination, yaw):
            return None
        det, obj = current
        revision = session._environment_revision
        # Camera geometry decides to refrain from grasping. Simulator truth is
        # only an independent audit; disagreement must not trigger a re-grasp.
        evaluation = session._evaluation_object(det)
        truth = None
        if evaluation is not None:
            if destination['type'] == 'named_region':
                zone = session.world.zone_of(evaluation)
                floor = float(session.runner.zones[destination['name']].center[2])
                truth = bool(evaluation is not session.world.held and zone is not None
                             and zone.name == destination['name']
                             and abs(evaluation.bottom_z() - floor) <= .01)
                verify_zone = getattr(session, '_verify_zone', None)
                if verify_zone is not None:
                    truth = bool(verify_zone(evaluation, destination['name']))
            else:
                truth = bool(session._verify_destination(evaluation, destination))
            physics = getattr(session, '_physics', None)
            if physics is not None:
                truth = truth and physics.resting_status(evaluation.name)['success']
        self._check_plan()
        if self._stopped(cancel):
            return self._failure(step, 'aborted', '任务已停止')
        if (session._environment_revision != revision or self._holding() or
                self._current_measurement(label) is None):
            raise _PlanChanged('完成核验后现场发生变化，请重新规划')
        goal = (dict(label=label, zone_name=destination['name'])
                if destination['type'] == 'named_region' else
                dict(label=label, zone_name=None, destination=copy.deepcopy(destination)))
        item = dict(label=label, goal=goal, ok=truth is True,
                    verified_revision=revision, truth={'success': truth}, yaw_verified=True)
        result = dict(ok=truth is True, skipped=True, changed=False, skill=step['skill'],
                      stage='already_satisfied', label=label, object_id=obj['object_id'],
                      resolved_destination=copy.deepcopy(destination), controller_ok=True,
                      camera_verified=True, holding=False, verified_revision=revision,
                      truth={'success': truth}, yaw_verified=True, items=[item],
                      message='相机确认该步骤已完成，已跳过重复抓取')
        if truth is not True:
            result.update(error='completion_not_verified', message='相机显示已完成，但独立核验未确认，已跳过重复抓取')
        return result

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
                radius = math.hypot(x, y)
                room = min(x - bounds['x'][0], bounds['x'][1] - x,
                    y - bounds['y'][0], bounds['y'][1] - y,
                    radius - bounds['r_min'], bounds['r_max'] - radius)
                points.append((min(clearance, .18) + min(separation, .25), room, destination))
        # Saturated clearance scores used to tie-break at the smallest x/y,
        # repeatedly searching workspace corners before equally clear interior
        # points. Keep clearance priority, then reserve mechanical reach room.
        points.sort(key=lambda item: (-item[0], -item[1], item[2]['xy']))
        return [destination for _score, _room, destination in points]

    def _remember_failed_free(self, label, destination):
        key = (label, self.session._environment_revision)
        point = tuple(round(float(v), 4) for v in destination['xy'])
        self._free_failures.setdefault(key, set()).add(point)

    def _place(self, step, goals, protected, cancel):
        import time
        from .simulation import _PlacementBudget, _PlacementInterrupted

        session = self.session
        context = session.runner._held_recovery_context
        if context is None or context['det'].label != step.get('label'):
            return self._failure(step, 'no_held_target', '当前没有该物品的已测量持物记录')
        destination, yaw = self._resolve(step.get('destination'), goals)
        automatic = destination['type'] == 'auto_free'
        target = copy.deepcopy(context['det'])
        context['det'].extra.pop('_stack_support', None)
        saved_target = getattr(context.get('grasp'), 'target', None)
        if saved_target is not None:
            saved_target.extra.pop('_stack_support', None)
        target.extra.pop('_stack_support', None)
        neighbors = self._stationary()
        calibrate = getattr(session, '_calibrate_carried_anchor', None)
        if callable(calibrate):
            from ..core.types import PickResult
            confirmed_context, width, reason = session.runner._held_transfer_feedback(context['det'])
            if confirmed_context is None:
                return self._failure(step, 'held_feedback_changed', reason)
            feedback = PickResult(False, grasp_width=width, detail={'holding': True})
            confirmed, reason, _fresh = calibrate(context['grasp'], feedback, width, neighbors)
            if not confirmed:
                return dict(self._failure(step, 'held_anchor_unconfirmed',
                            '夹持位置尚未得到可靠测量，保持夹持'), diagnostics=reason)
        if destination['type'] == 'on_object':
            from .stacking import planning_options, annotate
            _options, support = planning_options(session, destination)
            annotate(context['det'], support)
            if saved_target is not None:
                annotate(saved_target, support)
            annotate(target, support)
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
        budget = _PlacementBudget(getattr(session, 'arm', None), session.cancel_event, seconds=15.)
        emit = getattr(session, '_emit', lambda *_args: None)
        last_progress = [-float('inf')]

        def interrupted():
            return budget.cancelled() or self._stopped(cancel) or not self._plan_current()

        def progress(fields):
            now = time.monotonic()
            if self._stopped(cancel) or now - last_progress[0] < 1.:
                return
            last_progress[0] = now
            emit('placement_progress', dict(fields,
                message='正在检查持物的安全放置方案，仍保持夹持'))

        def search():
            nonlocal bundle, why, options, destination
            for candidate in candidates:
                self._check_plan()
                if interrupted():
                    raise _PlacementInterrupted()
                options = self._options(candidate)
                bundle, why = session.runner.plan_held_transfer(
                    target, neighbors, **options, cancelled=interrupted,
                    progress=progress)
                self._check_plan()
                if interrupted():
                    raise _PlacementInterrupted()
                if bundle is not None:
                    destination = candidate
                    break

        try:
            emit('placement_planning', dict(message='正在检查持物的安全放置点，仍保持夹持'))
            try:
                budget.run(search)
            except _PlacementInterrupted:
                stopped = self._stopped(cancel)
                reason = 'aborted' if stopped else 'placement_planning_budget_exceeded'
                message = ('任务已停止，仍保持当前夹持' if stopped else
                           '本轮持物放置搜索已达计算预算，仍保持夹持，可调整观察后重新规划')
                if not stopped:
                    emit('placement_changed', dict(message=message, reason=reason))
                return dict(self._failure(step, reason, message), diagnostics=reason,
                            aborted=stopped, fault=False)
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
        completed = self._completed_move(step, destination, yaw, cancel)
        if completed is not None:
            return completed
        if destination['type'] == 'on_object':
            det, obj = self._fresh_target(step.get('label'), 'recover_queued_target',
                                          support_label=destination['label'])
        else:
            det, obj = self._fresh_target(step.get('label'), 'recover_queued_target')
        self._check_plan()
        if self._stopped(cancel):
            return self._failure(step, 'aborted', '任务已停止')
        completed = self._completed_move(step, destination, yaw, cancel, observe=False)
        if completed is not None:
            return completed
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
            # auto_free grants a choice of safe support, not this provisional
            # point. Seat and measure the grip before binding its real landing.
            picked = self._pick(dict(step, skill='pick'), goals, cancel)
            if not picked.get('ok') or not picked.get('holding'):
                return dict(picked, skill='move')
            placed = self._place(dict(step, skill='place'), goals, protected, cancel)
            return dict(placed, skill='move')
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
