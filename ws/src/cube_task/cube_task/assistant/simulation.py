"""Camera-to-motion adapter and editable tabletop for the local assistant.

Only the existing kinematic simulator can be constructed here. Perception
observations are separate from the ground truth published to the viewer; model
decisions resolve exclusively to detections from the latest observation.
"""
from __future__ import annotations

import copy
import logging
import math
import threading
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping, Optional

import numpy as np

from ..camera.factory import make_camera
from ..core.clock import SimClock
from ..core.extrinsics import extrinsics_from_cfg
from ..core.logger import RunLogger, to_jsonable
from ..core.transforms import inv_T
from ..core.zones import parse_zones
from ..robot.factory import make_arm
from ..task.operator import Operator
from ..task.runner import TaskRunner, pick_summary
from ..task.views import tag_view
from ..tools.common import load_cfg_for
from .catalog import CatalogEditor
from .catalog_perception import configure_sandbox, make_sandbox_perception
from .observations import merge_observations

_LOG = logging.getLogger(__name__)
_STAGE_MESSAGES = {
    'start': '开始抓放', 'open': '张开夹爪', 'pre': '移动到目标上方',
    'descend': '下降到抓取位置', 'close': '闭合夹爪抓取物品',
    'lift': '提起物品', 'transit': '移动到放置区域上方',
    'release': '下降到放置位置', 'open_release': '张开夹爪放下物品',
    'retreat': '抬起并离开放置区域', 'recover_open': '重新张开夹爪',
    'recover_retreat': '抓取未完成，抬起机械臂', 'recover_up': '放置未完成，先向上抬起',
    'putback_lift': '抬起方块并返回原处', 'putback_down': '将方块降回原处',
    'putback_open': '松开夹爪放回方块', 'putback_retreat': '放回方块后向上离开',
    'hover_grasp': '移动到抓取预览位置', 'up': '向上抬起机械臂',
    'hover_release': '移动到放置预览位置', 'done': '抓放动作结束',
    'verify_close': '检查是否夹住方块', 'dropped': '检查抬升后的夹持状态',
    'dropped_in_transit': '检查移动后的夹持状态',
}


class _Cancellation(threading.Event):
    """Latch an optional caller event into the simulator's shared abort event."""

    external: Optional[threading.Event] = None

    def is_set(self) -> bool:
        if self.external is not None and self.external.is_set():
            self.set()
        return super().is_set()

    def wait(self, timeout=None) -> bool:
        # Poll the caller event too, without a watcher thread per instruction.
        import time
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while not self.is_set():
            remaining = 0.02 if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                return False
            super().wait(min(0.02, remaining))
        return True


class _ViewerClock(SimClock):
    """Advance the normal simulation clock, optionally pacing its real states."""

    def __init__(self, cancel_event: _Cancellation, realtime: bool):
        super().__init__()
        self.cancel_event = cancel_event
        self.realtime = bool(realtime)
        self.paced = False  # initialization must not animate a long startup move
        self.speed = 1.0
        self.on_tick: Optional[Callable[[], None]] = None

    def sleep(self, dt: float) -> None:
        remaining = float(dt)
        while remaining > 1e-9:
            step = min(remaining, 0.05)
            if self.on_tick is not None:
                self.on_tick()
            if self.realtime and self.paced and not self.cancel_event.is_set():
                self.cancel_event.wait(step / self.speed)
            super().sleep(step)
            remaining -= step


class _EventLogger:
    """Preserve normal task logs and forward their structured stage events."""

    def __init__(self, session, base_dir):
        self.session = session
        self.inner = RunLogger(base_dir, 'assistant_L1', session.clock, save_images=False)
        self.dir = self.inner.dir

    def event(self, kind: str, **fields: Any) -> None:
        self.inner.event(kind, **fields)
        if kind == 'skill_step' and fields.get('stage') in _STAGE_MESSAGES:
            fields = dict(fields, message=_STAGE_MESSAGES[fields['stage']])
        self.session._record_feedback(kind, fields)
        self.session._emit(kind, fields)

    def save_json(self, name, value):
        return self.inner.save_json(name, value)

    def save_image(self, name, value):
        return None

    def close(self):
        self.inner.close()


class SimulationSession:
    """Own one L1 simulation and the latest selectable camera observation.

    ``prepare``, ``observe`` and ``execute`` are serialized. ``snapshot``,
    ``frame_jpeg``, ``set_speed`` and ``stop`` can be called while a move runs.
    After a stop the abort remains latched; create a new session to restart.
    """

    def __init__(self, seed: int = 0, log_dir: Optional[Path] = None,
                 on_event: Optional[Callable[[str, dict], None]] = None,
                 realtime: bool = True):
        self.seed = int(seed)
        self.on_event = on_event
        self.cancel_event = _Cancellation()
        self.clock = _ViewerClock(self.cancel_event, realtime)
        self._operation_lock = threading.RLock()
        self._cache_lock = threading.Lock()
        self._snapshot: dict = {}
        self._jpeg: Optional[bytes] = None
        self._observation_views: dict[int, dict] = {}
        self._observation_jpegs: dict[int, bytes] = {}
        self._scene: dict = {}
        self._detections: Mapping[str, Any] = MappingProxyType({})
        self._scene_version = 0
        self._started_at = self.clock.now()
        self._prepared = False
        self._closed = False
        self._status = 'initializing'
        self._stage = ''
        self._feedback = {'grasp_plan': {}, 'checks': []}
        self._recovery_target = None
        self._recovery_neighbors = []
        self._recovery_goal = None
        self._recovery_evaluation = None
        self._recovery_context = None
        args = SimpleNamespace(mode='sim', task='L1', seed=self.seed,
                               mount='eye_in_hand', difficulty='normal', layout='A',
                               site=False, config=[], set=[], extrinsics_error=(0.0, 0.0),
                               log_dir=None if log_dir is None else str(log_dir))
        self.cfg, self.world, self.info = load_cfg_for(args, 'L1')
        configure_sandbox(self.cfg)
        self.editor = CatalogEditor(self.world, self.info['pick_polygon'])
        # Fail closed even if a future scenario overlay changes its backend.
        if self.cfg.get('robot.backend') != 'sim' or self.cfg.get('camera.backend') != 'sim':
            raise ValueError('the assistant only supports the kinematic simulation')
        # Competition rounds only search the pick polygon and hide placed cubes.
        # Conversation instructions may move a cube again, so this session's
        # RGB-D segmentation also covers the configured placement workspace.
        zones = parse_zones(self.cfg)
        region = list(self.cfg.get('perception.roi.pick_polygon', None) or [])
        for zone in zones.values():
            region.extend(zone.corners_xy().tolist())
        if region:
            lo, hi = np.min(region, axis=0), np.max(region, axis=0)
            self.cfg.set('perception.roi.pick_polygon', [
                [float(lo[0]), float(lo[1])], [float(hi[0]), float(lo[1])],
                [float(hi[0]), float(hi[1])], [float(lo[0]), float(hi[1])]])
        self.cfg.set('perception.roi.exclude_zones', False)
        self.arm = make_arm(self.cfg, self.clock, sim_world=self.world)
        self.operator = Operator(enabled=False)
        self.operator.abort_event = self.cancel_event
        self.arm.set_abort_event(self.cancel_event)
        self.camera = make_camera(self.cfg, self.clock, sim_world=self.world, arm=self.arm)
        self.logger = _EventLogger(self, log_dir)
        self.runner = TaskRunner(self.cfg, 'Sandbox', self.arm, self.camera,
                                 make_sandbox_perception(self.cfg), self.clock, self.logger,
                                 operator=self.operator, mode='sim')
        self._configure_observation_views()
        self.runner.skill.on_stage = self._on_stage
        try:
            self.arm.connect()
            if not self.arm.enable():
                raise RuntimeError('simulation arm did not enable')
            self.camera.start()
            self.clock.on_tick = self._publish_snapshot
            self._publish_snapshot()
        except BaseException:
            self.close()
            raise

    def _emit(self, kind: str, fields: dict) -> None:
        if self.on_event is not None:
            try:
                self.on_event(kind, to_jsonable(fields))
            except Exception:
                _LOG.exception('assistant event listener failed')

    def _on_stage(self, stage: str) -> None:
        self._stage = str(stage)
        self._publish_snapshot()
        self._emit('stage', {'stage': stage,
                             'message': _STAGE_MESSAGES.get(stage, '正在执行机械臂动作')})

    def _record_feedback(self, kind: str, fields: dict) -> None:
        """Expose existing measured checks; viewer feedback never commands motion."""
        if kind == 'decision' and fields.get('action') == 'pick':
            target = fields.get('target') or {}
            self._feedback['grasp_plan'] = {
                'label': fields.get('label'),
                **{key: target.get(key) for key in ('length', 'width', 'height')},
                **{key: fields.get(key) for key in
                   ('expected_width', 'open_width', 'close_width')},
                'closing_angle_deg': math.degrees(fields.get('closing_angle', 0.0)),
                'kind': (fields.get('grasp') or {}).get('kind'),
                'mode': (fields.get('grasp') or {}).get('mode'),
            }
        elif kind == 'skill_step' and fields.get('stage') == 'close':
            width = fields.get('width')
            ok = bool(fields.get('ok') and width is not None
                      and width >= self.runner.skill.empty_width)
            self._append_check('close', ok, '闭合后检测到夹持' if ok else '闭合后未确认夹持',
                               measured_width=width,
                               expected_width=self._feedback['grasp_plan'].get('expected_width'))
        elif kind == 'grasp_feedback':
            names = {'dropped': '抬升后夹持检查', 'dropped_in_transit': '搬运后夹持检查',
                     'dropped_before_retry': '恢复前夹持检查'}
            self._append_check(fields['stage'], bool(fields.get('ok')),
                               names.get(fields['stage'], '夹持反馈检查'),
                               measured_width=fields.get('measured_width'))
        else:
            return
        self._publish_snapshot()

    def _append_check(self, stage, ok, message, **extra):
        self._feedback['checks'].append(dict(stage=stage, ok=bool(ok), message=message, **extra))
        self._feedback['checks'] = self._feedback['checks'][-20:]

    def edit_scene(self, request: dict) -> dict:
        """Commit one validated layout edit and expire all camera/planning state."""
        with self._operation_lock:
            self._check_open()
            if not isinstance(request, dict):
                raise ValueError('摆场请求必须是 JSON 对象')
            if (type(request.get('scene_version')) is not int
                    or request['scene_version'] != self._scene_version):
                raise ValueError('布局版本已变化，请等待界面刷新后重试')
            if (not self._prepared or self.cancel_event.is_set()
                    or self._status not in ('ready', 'error')):
                raise ValueError('当前不能编辑场景，请先完成任务或重置')
            if self.world.held is not None:
                raise ValueError('夹爪仍夹着物品，不能编辑场景')
            change = self.editor.apply(request)
            self.runner._placed.clear()
            self.runner._placed_labels.clear()
            self.runner._tracks.clear()
            self.runner._held_recovery_context = None
            self.runner._live_dets = []
            self.runner._live_mark += 1
            self._recovery_target = self._recovery_goal = self._recovery_evaluation = None
            self._recovery_context = None
            self._recovery_neighbors = []
            self._recovery_retry = None
            self._feedback = {'grasp_plan': {}, 'checks': []}
            self._invalidate_scene('布局已改变，执行下一条指令时将重新扫描')
            with self._cache_lock:
                self._observation_views.clear()
                self._observation_jpegs.clear()
                self._jpeg = None
            self._status, self._stage = 'ready', ''
            self._publish_snapshot()
            self._emit('scene_edit', {'message': '布局已更新，旧观测已清除', 'change': change})
            return self.scene

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError('simulation session is closed')

    def _configure_observation_views(self) -> None:
        """Aim one reachable wrist view at each taught zone, without object truth."""
        from ..sim.scenarios import look_at

        self._view_labels = ['抓取区视角 %d' % (i + 1)
                             for i in range(len(self.runner.views))]
        self.runner.ext = extrinsics_from_cfg(self.cfg)
        ext = self.runner.ext
        if ext is None:
            raise ValueError('the assistant wrist camera needs calibrated extrinsics')
        T_fc = ext.T if ext.parent_frame == 'flange' else self.runner.T_flange_tcp @ ext.T
        seed = self.runner.views[len(self.runner.views) // 2] if self.runner.views else None
        colors = {'red': '红色', 'green': '绿色', 'yellow': '黄色'}
        for zone in self.runner.zones.values():
            target = np.asarray(zone.center, dtype=float) + [0.0, 0.0, 0.025]
            radial = target[:2] / max(float(np.linalg.norm(target[:2])), 1e-6)
            eye = np.r_[target[:2] - 0.13 * radial, target[2] + 0.255]
            T_bc = look_at(eye, target)
            q, _ep, _er = self.arm.kin.ik(T_bc @ inv_T(T_fc), q_seed=seed)
            if q is None:
                raise ValueError('no reachable observation pose for ' + zone.name)
            self.runner.views.append(q.tolist())
            self._view_labels.append(colors.get(zone.label, zone.label or zone.name) + '放置区')
        self.runner.multi_view = len(self.runner.views) > 1

    def prepare(self) -> dict:
        """Ready the existing task runner, then observe through the RGB-D camera."""
        with self._operation_lock:
            self._check_open()
            if not self._prepared:
                ok, why = self.runner.ready_for_pick(go_observe=True)
                if not ok:
                    raise RuntimeError('simulation preparation failed: ' + why)
                self._prepared = True
            scene = self.observe()
            self.clock.paced = True
            return scene

    def observe(self) -> dict:
        """Capture fresh detections, assigning new IDs for this observation only."""
        with self._operation_lock:
            self._check_open()
            if not self._prepared:
                return self.prepare()
            if self.cancel_event.is_set():
                raise RuntimeError('simulation is stopped; reset it before observing')
            self._status = 'observing'
            self._publish_snapshot()
            with self._cache_lock:
                self._observation_views = {}
                self._observation_jpegs = {}
            self._emit_scan_view(0)
            result, why = self.runner.observe_once()
            if result is None:
                self._invalidate_scene(why)
                self._status = 'error'
                self._publish_snapshot()
                raise RuntimeError('simulation observation failed: ' + why)
            self._cache_observation(0, result, getattr(self.camera, '_last', None))
            views = {0: tag_view(result.detections, 0)}
            warnings = list(result.warnings)
            # The official L1 scene uses several taught wrist views. Scan with
            # the runner's same camera/perception path, then use its existing
            # merge rule, then explain clipped mixed silhouettes using only
            # measured geometry from the other views of this same scan.
            for view in range(1, len(self.runner.views)):
                self._emit_scan_view(view)
                status, next_result, frame = self.runner._observe(view)
                if status != 'ok' or next_result is None:
                    why = '%s: %s' % (status, self.runner._last_obs_failure)
                    self._invalidate_scene(why)
                    self._status = 'error'
                    self._publish_snapshot()
                    raise RuntimeError('simulation observation failed: ' + why)
                result = next_result
                self._cache_observation(view, result, frame)
                views[view] = tag_view(result.detections, view)
                warnings.extend(result.warnings)
            detections, diagnostics = merge_observations(
                views, dist=self.runner.view_merge,
                partial=lambda d: not d.valid or self.runner._partial(d))
            self._scene_version += 1
            mapping = {}
            objects = []
            for index, det in enumerate(detections, 1):
                # Retain measured in-zone objects for later instructions;
                # zone membership uses measured geometry, never object truth.
                zone = None
                if det.center_base is not None:
                    x, y = [float(v) for v in det.center_base[:2]]
                    zone = next((z.name for z in self.runner.zones.values()
                                 if z.contains_xy(x, y)), None)
                object_id = 'v%d-object-%d' % (self._scene_version, index)
                measured = copy.deepcopy(det)
                if measured.center_base is not None:
                    measured.center_base.setflags(write=False)
                if measured.mask is not None:
                    measured.mask.setflags(write=False)
                mapping[object_id] = measured
                objects.append(dict(measured.summary(), object_id=object_id,
                                    in_zone=zone is not None, zone=zone,
                                    view=measured.extra.get('view', 0)))
            self._detections = MappingProxyType(mapping)
            self._scene = {'scene_version': self._scene_version, 'objects': objects,
                           'zones': self._zones(),
                           'class_to_zone': dict(self.runner.class_to_zone),
                           'task': 'Sandbox', 'mode': 'sim',
                           'observation_diagnostics': diagnostics,
                           'observed_at': float(result.frame_stamp),
                           'warnings': warnings}
            self._status = 'ready'
            self._stage = ''
            self._publish_snapshot()
            self._emit('scene', self._scene)
            return copy.deepcopy(self._scene)

    def _emit_scan_view(self, view: int) -> None:
        total = len(self.runner.views)
        label = self._view_labels[view]
        self._emit('scan_view', {'view': view, 'total': total, 'label': label,
                                 'message': '正在扫描 %d/%d：%s' % (view + 1, total, label)})

    def _cache_observation(self, view: int, result, frame, label=None) -> None:
        """Freeze each processed camera view; gallery reads never render or move."""
        bgr = result.debug_image if result.debug_image is not None else (
            None if frame is None else frame.color)
        if bgr is None:
            return
        import cv2
        ok, encoded = cv2.imencode('.jpg', bgr)
        if not ok:
            return
        metadata = {'id': view, 'view_id': view, 'label': label or self._view_labels[view],
                    'detected_labels': sorted({d.label for d in result.detections if d.valid}),
                    'stamp': float(result.frame_stamp),
                    'sim_time': float(result.frame_stamp) - self._started_at,
                    'width': int(bgr.shape[1]), 'height': int(bgr.shape[0])}
        with self._cache_lock:
            self._jpeg = encoded.tobytes()
            self._observation_views[view] = metadata
            self._observation_jpegs[view] = self._jpeg

    def _invalidate_scene(self, reason: str) -> None:
        self._scene_version += 1
        self._detections = MappingProxyType({})
        self._scene = {'scene_version': self._scene_version, 'objects': [],
                       'zones': self._zones(), 'class_to_zone': dict(self.runner.class_to_zone),
                       'task': 'Sandbox', 'mode': 'sim', 'warnings': [reason]}

    def execute(self, decision, cancel_event: Optional[threading.Event] = None) -> dict:
        """Resolve a current object ID and use TaskRunner.pick_detection unchanged."""
        with self._operation_lock:
            self._check_open()
            self.cancel_event.external = cancel_event
            try:
                get = decision.get if isinstance(decision, Mapping) else (
                    lambda key, default=None: getattr(decision, key, default))
                object_id = get('object_id')
                version = get('scene_version', self._scene_version)
                zone_name = get('zone_name') or get('zone')
                if self.cancel_event.is_set():
                    return self._refusal('aborted', 'execution cancelled', object_id, zone_name)
                if get('action', 'pick') != 'pick':
                    return self._refusal('invalid_action', 'only a pick decision can move the arm',
                                         object_id, zone_name)
                if version != self._scene_version or object_id not in self._detections:
                    return self._refusal('stale_scene', 'target is not in the current observation',
                                         object_id, zone_name)
                det = self._detections[object_id]
                if not det.valid or det.center_base is None:
                    return self._refusal('invalid_target', 'target has no valid measured geometry',
                                         object_id, zone_name)
                if det.label not in self.runner.class_to_zone:
                    return self._refusal('invalid_target', 'target is not a configured catalog class',
                                         object_id, zone_name)
                zone_name = zone_name or self.runner.class_to_zone[det.label]
                if zone_name not in self.runner.zones:
                    return self._refusal('invalid_zone', 'target zone does not exist',
                                         object_id, zone_name)
                # Pass the exact measured target among the valid neighbors; this
                # preserves the runner's identity checks for transit clearance.
                target = copy.deepcopy(det)
                neighbors = [target if d is det else copy.deepcopy(d)
                             for d in self._detections.values()
                             if d.valid and d.center_base is not None]
                # Independent evaluation only: this association is never used
                # by the motion planner or the model's selectable scene.
                evaluation_object = self._evaluation_object(det)
                self._recovery_target = copy.deepcopy(det)
                self._recovery_neighbors = copy.deepcopy(neighbors)
                self._recovery_goal = {'label': det.label, 'zone_name': zone_name}
                self._recovery_evaluation = evaluation_object
                self._recovery_context = None
                self._feedback = {'grasp_plan': {}, 'checks': []}
                self._refresh_placement_occupancy(det)
                self._status = 'executing'
                self._publish_snapshot()
                result, why = self.runner.pick_detection(target, zone_name=zone_name,
                                                         neighbors=neighbors)
                return self._finish_attempt(result, why, object_id, zone_name,
                                            det, evaluation_object)
            except Exception:
                # An unexpected adapter/planner failure must expire every
                # selectable target and retain a held cube for an explicit reset.
                self.stop()
                self._invalidate_scene('execution failed')
                self._publish_snapshot()
                raise
            finally:
                self.cancel_event.external = None

    def _finish_attempt(self, result, why, object_id, zone_name, det, evaluation_object):
        """Use the same independent placement check for initial and recovery actions."""
        summary = (pick_summary(result) if result is not None else
                   {'ok': False, 'stage': 'plan', 'error': why, 'label': det.label,
                    'zone': zone_name, 'duration': 0.0})
        if result is not None:
            summary['diagnostics'] = str(result.detail.get('last_motion_detail') or
                                         result.detail.get('reason') or why or '')[:500]
        summary['object_id'] = object_id
        summary['aborted'] = self.cancel_event.is_set()
        summary['holding'] = self.world.held is not None
        self._recovery_context = None
        self._invalidate_scene('observation invalidated after execution')
        if not summary['aborted'] and not summary.get('fault') and not summary['holding']:
            try:
                self.observe()
            except RuntimeError as exc:
                summary['observation_error'] = str(exc)
        else:
            self._status = 'stopped' if summary['aborted'] else 'error'
        summary['scene'] = copy.deepcopy(self._scene)
        summary['score'] = self.world.score(self.runner.class_to_zone)
        actual_zone = None if evaluation_object is None else self.world.zone_of(evaluation_object)
        truth_success = bool(evaluation_object is not None
                             and evaluation_object is not self.world.held
                             and actual_zone is not None and actual_zone.name == zone_name)
        summary['controller_ok'] = bool(summary['ok'])
        summary['truth'] = {'success': truth_success,
                            'object_name': None if evaluation_object is None else evaluation_object.name,
                            'expected_zone': zone_name,
                            'actual_zone': None if actual_zone is None else actual_zone.name}
        summary['ok'] = summary['controller_ok'] and truth_success
        if summary['controller_ok'] and not truth_success:
            summary['error'] = 'placement_not_verified'
        colors = {'red': '红色', 'green': '绿色', 'yellow': '黄色'}
        if summary['aborted']:
            summary['message'] = ('任务已停止，夹爪保持夹持，机械臂保持当前位置'
                                  if summary['holding'] else '任务已停止，机械臂保持当前位置')
        elif summary['ok']:
            zone_label = self.runner.zones[zone_name].label
            names = {'red': '红色方块', 'green': '绿色方块', 'yellow': '黄色方块',
                     'apple': '苹果', 'banana': '香蕉', 'pear': '梨子'}
            summary['message'] = '已将%s放到%s区域' % (
                names.get(det.label, det.label), colors.get(zone_label, zone_label or zone_name))
        elif summary['controller_ok']:
            summary['message'] = ('抓放动作已结束，但无法唯一核对目标物品'
                                  if evaluation_object is None else
                                  '抓放动作已结束，但目标物品没有位于指定区域')
        else:
            summary['message'] = '本次抓取未完成，请查看执行记录并重新观察场景'
        self._append_check('placement', truth_success,
                           '已独立核验物品落入指定区域' if truth_success else '落位核验未通过')
        self.logger.save_json('last_result.json', summary)
        self._publish_snapshot()
        self._emit('execution_result', summary)
        return summary

    def prepare_recovery(self, failure, goal, attempt, history):
        """Offer goal-preserving local repairs, observing without moving a held cube."""
        with self._operation_lock:
            self._check_open()
            if self.cancel_event.is_set() or failure.get('aborted'):
                raise RuntimeError('任务已停止，不能自动恢复')
            if goal != self._recovery_goal or self._recovery_target is None:
                raise ValueError('恢复目标与原任务不一致')
            health = self.arm.health()
            healthy = health is not None and health.ok
            holding = self.world.held is not None
            options = []
            observations = []
            observation_mode = 'unavailable'
            self._recovery_retry = None
            if healthy and holding:
                observation_mode = 'stationary_with_last_scan'
                fresh = self._stationary_recovery_observation()
                # The last full scan records the neighbors before this grasp.
                # Replace only visible measured neighbors; the carried target
                # must never become an occupied placement slot.
                neighbors = [copy.deepcopy(d) for d in self._recovery_neighbors
                             if d.label != goal['label']]
                for det in fresh:
                    if det.valid and det.center_base is not None and det.label != goal['label']:
                        neighbors = [d for d in neighbors if d.label != det.label]
                        neighbors.append(copy.deepcopy(det))
                self._recovery_neighbors = neighbors
                observations = [d.summary() for d in neighbors]
                if getattr(self.runner, 'has_held_recovery', False):
                    options.append({'id': 'place_held_alternate', 'description':
                        '保持当前夹持，依据相机测得的已有物体重新选择原目标区域中的空位，'
                        '排除之前失败的放置点，抬升后移动、下降、放下并核验；'
                        '不重复抓取，不在空中松爪。本次是第 %d 次重新选位。' % attempt})
            elif healthy:
                observation_mode = 'full_scan'
                scene = self.observe()
                observations = scene['objects']
                matches = [o for o in observations if o.get('valid') and
                           o.get('label') == goal['label']]
                if len(matches) == 1:
                    target = matches[0]
                    self._recovery_retry = {
                        'scene_version': scene['scene_version'], 'action': 'pick',
                        'object_id': target['object_id'], 'zone_name': goal['zone_name']}
                    options.append({'id': 'reobserve_retry', 'description':
                        '根据刚完成的六视角扫描，重新定位同一个目标，重新规划抓取并放入原指定区域。'})
            else:
                self._scene_version += 1
            if holding:
                self._scene_version += 1
            context = {'scene_version': self._scene_version, 'attempt': int(attempt),
                       'max_attempts': 3, 'goal': copy.deepcopy(goal),
                       'failure': dict(failure, holding=holding),
                       'observations': observations, 'options': options,
                       'observation_mode': observation_mode,
                       'history': copy.deepcopy(history)}
            self._recovery_context = copy.deepcopy(context)
            self._publish_snapshot()
            return context

    def _stationary_recovery_observation(self):
        """Capture and process a new RGB-D frame without commanding any joint."""
        from ..camera.utils import capture_averaged
        from ..core.extrinsics import camera_pose_in_base

        newer = float(getattr(self.runner, '_last_stamp', None) or 0.0)
        frame = capture_averaged(self.camera, self.runner.n_avg,
                                 newer_than=newer, timeout=self.runner.frame_timeout)
        if frame is None:
            raise RuntimeError('恢复时未获得新的相机图像')
        flange = self.arm.get_flange_pose()
        if flange is None or self.runner.ext is None:
            raise RuntimeError('恢复时无法读取相机坐标变换')
        pose = camera_pose_in_base(self.runner.ext, flange.value, self.runner.T_flange_tcp)
        result = self.runner.perception.process(frame, pose, self.runner.zones)
        self.runner._last_stamp = float(frame.stamp)
        self._cache_observation(6, result, frame, label='恢复时静止观测')
        self._emit('recovery_observation', {'message': '已保持夹持并完成静止相机观测',
                                           'detections': [d.summary() for d in result.detections]})
        return result.detections

    def execute_recovery(self, decision, cancel_event=None):
        """Execute only a current locally offered action, preserving the original goal."""
        from .recovery import validate_recovery_decision

        with self._operation_lock:
            self._check_open()
            context = self._recovery_context
            if context is None:
                raise ValueError('恢复方案已过期，请重新观察')
            if context['scene_version'] != self._scene_version:
                raise ValueError('恢复场景已变化，请重新观察')
            chosen = validate_recovery_decision(decision, context)
            if chosen['action'] != 'recover':
                raise ValueError('该恢复决策不执行动作')
            self._recovery_context = None
            self.cancel_event.external = cancel_event
            try:
                if self.cancel_event.is_set():
                    raise RuntimeError('任务已停止')
                if chosen['option_id'] == 'reobserve_retry':
                    if self.world.held is not None or self._recovery_retry is None:
                        raise ValueError('当前状态不能重复抓取')
                    return self.execute(self._recovery_retry, cancel_event=cancel_event)
                if chosen['option_id'] != 'place_held_alternate' or self.world.held is None:
                    raise ValueError('当前状态不能续接放置')
                target = self._recovery_target
                goal = self._recovery_goal
                self._refresh_placement_occupancy(target, self._recovery_neighbors)
                self._status = 'executing'
                result, why = self.runner.retry_held_placement(
                    target, goal['zone_name'], neighbors=self._recovery_neighbors)
                return self._finish_attempt(result, why, 'held-recovery', goal['zone_name'],
                                            target, self._recovery_evaluation)
            finally:
                self.cancel_event.external = None

    def _refresh_placement_occupancy(self, target, detections=None) -> None:
        """Replace the round's placement history with measured current neighbors.

        The target will leave its current slot. Excluding it also lets an
        instruction place it again in the same zone without a ghost collision.
        """
        occupied, labels = {}, {}
        for det in (self._detections.values() if detections is None else detections):
            if det is target or not det.valid or det.center_base is None:
                continue
            x, y = [float(v) for v in det.center_base[:2]]
            zone = next((z for z in self.runner.zones.values() if z.contains_xy(x, y)), None)
            footprint = self.runner._footprint(det)
            if zone is None or footprint is None:
                continue
            occupied.setdefault(zone.name, []).append(
                (x, y, footprint, det.height or 0.0, det.length, det.width, det.yaw))
            labels.setdefault(zone.name, []).append(det.label)
        self.runner._placed = occupied
        self.runner._placed_labels = labels

    def _evaluation_object(self, det):
        """Associate one measured target for scoring; ambiguity never counts as success."""
        # 30 mm is the normal track match tolerance and less than one L1 cube
        # edge. A detection farther away or near two same-color cubes cannot
        # provide a unique ground-truth identity for evaluation.
        xy = np.asarray(det.center_base[:2], dtype=float)
        matches = [obj for obj in self.world.objects if obj.label == det.label
                   and float(np.linalg.norm(obj.center_xy() - xy)) <= 0.03]
        return matches[0] if len(matches) == 1 else None

    def _refusal(self, error, detail, object_id, zone_name) -> dict:
        if error == 'aborted':
            self._status = 'stopped'
            self._invalidate_scene('execution cancelled')
            self._publish_snapshot()
        messages = {'aborted': '任务已停止，机械臂保持当前位置',
                    'stale_scene': '观察结果已更新，请重新选择目标',
                    'invalid_target': '当前目标没有可用的抓取信息',
                    'invalid_zone': '场景中没有这个放置区域',
                    'invalid_action': '当前指令不需要执行抓取'}
        summary = {'ok': False, 'stage': 'validation', 'error': error, 'detail': detail,
                   'object_id': object_id, 'zone': zone_name,
                   'message': messages.get(error, '当前无法执行抓取'),
                   'aborted': error == 'aborted', 'scene': copy.deepcopy(self._scene)}
        self._emit('execution_result', summary)
        return summary

    def _zones(self) -> list:
        return [dict(z.to_dict(), color_rgb=list(reversed(z.color_bgr)))
                for z in self.world.zones]

    @property
    def scene(self) -> dict:
        """Return a detached copy of the latest model-visible observation."""
        return copy.deepcopy(self._scene)

    def set_speed(self, speed: float) -> None:
        """Change viewer playback speed without changing the motion planner."""
        speed = float(speed)
        if not math.isfinite(speed) or speed not in (0.5, 1.0, 2.0, 4.0):
            raise ValueError('simulation speed must be 0.5, 1, 2 or 4')
        self.clock.speed = speed
        with self._cache_lock:
            if self._snapshot:
                self._snapshot = dict(self._snapshot, speed=speed)

    def snapshot(self) -> dict:
        """Read cached real simulation state without waiting for a motion."""
        with self._cache_lock:
            return copy.deepcopy(self._snapshot)

    def frame_jpeg(self) -> Optional[bytes]:
        """Return the most recent perception image; polling has no side effects."""
        with self._cache_lock:
            return self._jpeg

    def observation_views(self) -> list:
        """Return metadata for the frozen frames in the latest observation scan."""
        with self._cache_lock:
            return copy.deepcopy(list(self._observation_views.values()))

    def observation_frame_jpeg(self, view_id: int) -> Optional[bytes]:
        """Read one cached observation image without changing simulation time."""
        with self._cache_lock:
            return self._observation_jpegs.get(view_id)

    def stop(self) -> None:
        """Latch the normal task abort and hold; never move home or release."""
        self.cancel_event.set()
        self.arm.stop()
        with self._cache_lock:
            if self._snapshot:
                self._snapshot = dict(self._snapshot, status='stopped')

    def _publish_snapshot(self) -> None:
        joints = self.arm.get_joints()
        tcp = self.arm.get_tcp_pose()
        width = self.arm.gripper_width()
        if joints is None or tcp is None:
            return
        q = np.asarray(joints.value, float)
        T = np.asarray(tcp.value, float)
        frames = self.arm.kin.fk_all(q)
        objects = []
        for obj in self.world.objects:
            pose = obj.T_base_obj
            lo, hi = obj.aabb()
            zone = self.world.zone_of(obj)
            objects.append({'id': obj.name, 'label': obj.label, 'shape': obj.shape,
                            'center': ((lo + hi) / 2).tolist(), 'size': list(obj.size),
                            'dimensions': (hi - lo).tolist(),
                            'yaw': obj.yaw(), 'transform': pose.tolist(),
                            'color_rgb': list(reversed(obj.color_bgr)),
                            'held': obj is self.world.held, 'placed': zone is not None,
                            'zone': None if zone is None else zone.name})
        editor = dict(self.editor.metadata(), scene_version=self._scene_version,
                      editable=(self._prepared and not self.cancel_event.is_set()
                                and self.world.held is None and self._status in ('ready', 'error')))
        snap = {'mode': 'sim', 'simulation_kind': 'kinematic', 'seed': self.seed,
                'status': 'stopped' if self.cancel_event.is_set() else self._status,
                'stage': self._stage, 'scene_version': self._scene_version,
                'sim_time': self.clock.now() - self._started_at,
                'camera_available': self._jpeg is not None,
                'speed': self.clock.speed, 'joints': q.tolist(),
                'tcp_xyz': T[:3, 3].tolist(), 'tcp_transform': T.tolist(),
                'gripper_width': None if width is None else float(width.value),
                'link_points': [[0.0, 0.0, 0.0]] + [f[:3, 3].tolist() for f in frames]
                               + [T[:3, 3].tolist()],
                'link_frames': [f.tolist() for f in frames], 'objects': objects,
                'zones': self._zones(), 'table_z': self.world.table_z,
                'pick_polygon': to_jsonable(self.info['pick_polygon']),
                'held_object': None if self.world.held is None else self.world.held.name,
                'editor': editor, 'feedback': copy.deepcopy(self._feedback),
                'score': self.world.score(self.runner.class_to_zone),
                'class_to_zone': dict(self.runner.class_to_zone)}
        with self._cache_lock:
            self._snapshot = snap

    def close(self) -> None:
        """Stop before disconnecting the simulator and closing its run log."""
        self.stop()
        with self._operation_lock:
            if self._closed:
                return
            self.clock.on_tick = None
            self.camera.stop()
            self.arm.close()
            self.operator.close()
            self.logger.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
