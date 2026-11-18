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
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping, Optional

import numpy as np

from ..camera.factory import make_camera
from ..core.clock import SimClock
from ..core.extrinsics import extrinsics_from_cfg
from ..core.logger import RunLogger, to_jsonable
from ..core.transforms import inv_T
from ..core.types import Zone
from ..core.zones import parse_zones
from ..robot.factory import make_arm
from ..task.operator import Operator
from ..task.runner import TaskRunner, pick_summary
from ..task.views import tag_view
from ..tools.common import load_cfg_for
from .catalog import CatalogEditor
from .catalog_perception import configure_sandbox, make_sandbox_perception
from .observations import merge_observations
from .destinations import (destination_label, footprint_in_destination, placement_metadata,
                           validate_destination)

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
    'inspect_lift': '保持夹持并抬升到观察高度', 'inspect_place': '从侧上方观察目标区域空位',
    'placement_planning': '保持夹持，正在检查安全放置点',
}


class _PlacementInterrupted(Exception):
    pass


class _PlacementBudget:
    """Bound computation only; camera motions do not consume this budget."""

    def __init__(self, arm, stop, seconds=45.):
        self.arm, self.stop = arm, stop
        self.remaining = float(seconds)
        self.deadline = None

    def cancelled(self):
        return (self.stop.is_set() or self.remaining <= 0 or
                (self.deadline is not None and time.monotonic() >= self.deadline))

    def run(self, fn, *args, **kwargs):
        if self.cancelled():
            raise _PlacementInterrupted()
        started = time.monotonic()
        self.deadline = started + self.remaining
        scope = getattr(self.arm, 'planning_scope', None)
        try:
            with scope(self.cancelled) if scope is not None else nullcontext():
                value = fn(*args, **kwargs)
                if self.cancelled():
                    raise _PlacementInterrupted()
                return value
        finally:
            self.remaining -= time.monotonic() - started
            self.deadline = None


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

    targeted_observation = True

    def __init__(self, seed: int = 0, log_dir: Optional[Path] = None,
                 on_event: Optional[Callable[[str, dict], None]] = None,
                 realtime: bool = True):
        self.seed = int(seed)
        self.on_event = on_event
        self.cancel_event = _Cancellation()
        self.clock = _ViewerClock(self.cancel_event, realtime)
        self._operation_lock = threading.RLock()
        self._intervention_lock = threading.Lock()
        self._interventions = deque()
        self._intervention_id = 0
        self._environment_revision = 0
        self._active_revision = 0
        self._observed_detections = []
        self._scan_stats = {'frames': 0, 'full_scans': 0, 'local_scans': 0}
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
        self._placement = placement_metadata(self.cfg, self.info['pick_polygon'])
        self._placement_center = None
        self._destination_views = {}
        table = self._placement['table_bounds']
        table_polygon = [[table['x'][0], table['y'][0]], [table['x'][1], table['y'][0]],
                         [table['x'][1], table['y'][1]], [table['x'][0], table['y'][1]]]
        self.editor = CatalogEditor(self.world, self.info['pick_polygon'], edit_polygon=table_polygon)
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
        xb, yb = self._placement['bounds']['x'], self._placement['bounds']['y']
        region.extend([[x, y] for x in xb for y in yb])
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
        self._base_view_count = len(self.runner.views)
        self.runner.skill.on_stage = self._on_stage
        self.runner.skill.execution_guard = self._execution_guard
        self.runner.skill.before_release = self._before_release
        try:
            self.arm.connect()
            if not self.arm.enable():
                raise RuntimeError('simulation arm did not enable')
            self.camera.start()
            self.clock.on_tick = self._tick
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

    def _tick(self):
        self._apply_interventions()
        self._publish_snapshot()

    def intervene_scene(self, request):
        """Queue an edit; only the simulation owner mutates world geometry."""
        if not isinstance(request, dict) or request.get('action') != 'move':
            raise ValueError('运行中干预仅支持移动或旋转未夹持物品')
        if set(request) - {'action', 'object_id', 'xy', 'yaw', 'scene_version'}:
            raise ValueError('干预请求含有不支持的字段')
        snap = self.snapshot()
        if self._closed or self.cancel_event.is_set() or not self._prepared:
            raise ValueError('当前不能干预，请先准备或重置仿真')
        obj = next((o for o in snap.get('objects', []) if o['id'] == request.get('object_id')), None)
        if obj is None or obj.get('held'):
            raise ValueError('请选择桌面上未夹持的物品')
        if type(request.get('scene_version')) is not int or request['scene_version'] != snap['scene_version']:
            raise ValueError('场景版本已变化，请刷新后重试')
        # Validate syntax now; geometry is checked again at the application boundary.
        for key, size in (('xy', 2), ('yaw', None)):
            if key not in request:
                continue
            value = request[key]
            values = value if size else [value]
            if (size and (not isinstance(value, list) or len(value) != size)) or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or abs(v) > 1e6 or not math.isfinite(v)
                    for v in values):
                raise ValueError('干预坐标与角度必须为有限数字')
        if 'xy' not in request and 'yaw' not in request:
            raise ValueError('干预需要坐标或角度')
        with self._intervention_lock:
            if len(self._interventions) >= 12:
                raise ValueError('等待生效的干预过多，请稍后重试')
            self._intervention_id += 1
            request_id = self._intervention_id
            self._interventions.append((request_id, copy.deepcopy(request)))
        self._emit('scene_intervention_queued', {'request_id': request_id,
                   'message': '干预已排队，将在下一仿真时刻生效'})
        # Idle sessions have no ticking worker. Nonblocking ownership avoids a race.
        if self._operation_lock.acquire(blocking=False):
            try:
                self._apply_interventions()
                self._publish_snapshot()
            finally:
                self._operation_lock.release()
        return {'queued': True, 'request_id': request_id, 'message': '干预已接收'}

    def _apply_interventions(self):
        with self._intervention_lock:
            pending = list(self._interventions)
            self._interventions.clear()
        for request_id, request in pending:
            try:
                obj = next((o for o in self.world.objects if o.name == request['object_id']), None)
                if obj is None or obj is self.world.held:
                    raise ValueError('物品已被夹持或已不存在，干预未生效')
                candidate = obj.copy()
                xy = request.get('xy', obj.center_xy())
                candidate.place_upright(float(xy[0]), float(xy[1]),
                                       float(request.get('yaw', obj.yaw())), self.world.table_z)
                self.editor._check_placement(
                    candidate, [o for o in self.world.objects if o is not obj and not
                                (o is self.world.held and
                                 float(o.aabb()[0][2]) > float(candidate.aabb()[1][2]) + .008)])
                # Keep object identity: held/evaluation references must remain meaningful.
                obj.T_base_obj = candidate.T_base_obj.copy()
                self._environment_revision += 1
                self._emit('scene_intervention_applied', {'request_id': request_id,
                           'message': '人为干预已生效，后续动作将通过相机重新确认'})
            except ValueError as exc:
                self._emit('scene_intervention_rejected', {'request_id': request_id,
                           'message': str(exc)})

    def _execution_guard(self, stage):
        self._apply_interventions()
        if (stage in ('descend', 'close') and
                self._environment_revision != self._active_revision):
            return 'scene_changed'
        if (stage in ('release', 'open_release') and
                self._environment_revision != self._active_revision):
            return 'scene_changed'
        return None

    def _before_release(self, grasp, place, width, result):
        if self.cancel_event.is_set():
            return None
        budget = _PlacementBudget(self.arm, self.cancel_event)
        try:
            return self._refine_release(grasp, place, width, result, budget)
        except _PlacementInterrupted:
            reason = ('aborted' if self.cancel_event.is_set()
                      else 'placement_planning_budget_exceeded')
            result.detail['placement_planning_reason'] = reason
            if not self.cancel_event.is_set():
                self._emit('placement_changed', {
                    'message': '本轮放置规划计算已达时间预算，仍保持夹持，将反馈后重新规划',
                    'reason': reason})
            return None

    def _refine_release(self, grasp, place, width, result, budget):
        """At the destination hover, measure occupancy before choosing a final slot."""
        target, destination, zone_name, options = self._active_placement
        self._apply_interventions()
        self._on_stage('placement_planning')
        self._emit('inspect_place', {'message': '已到目标区域上方，正在局部确认空位'})
        from .observation_policy import merge_local_detections, measurement_visible
        from .placement_observation import (candidate_visibility_filter, iter_safe_inspection_poses,
                                             safe_inspection_route)
        from ..core.extrinsics import camera_pose_in_base
        neighbors = list(self._observed_detections)
        last_progress = [0.]
        def progress(fields):
            now = time.monotonic()
            if self.cancel_event.is_set() or now - last_progress[0] < 1.:
                return
            last_progress[0] = now
            count = fields.get('checked')
            message = ('正在检查安全放置点，仍保持夹持' if count is None else
                       '正在检查观察视角，已检查 %d 个，仍保持夹持' % count)
            self._emit('placement_progress', dict(fields, message=message))
        self._emit('placement_planning', {'message': '正在检查安全放置点，仍保持夹持'})
        def measure():
            nonlocal neighbors
            revision = self._environment_revision
            fresh = self._stationary_recovery_observation()
            frame = self.camera._last
            flange = self.arm.get_flange_pose()
            pose = camera_pose_in_base(self.runner.ext, flange.value, self.runner.T_flange_tcp)
            fresh = [d for d in fresh if d.center_base is not None and d.label != target.label]
            neighbors = merge_local_detections(neighbors, fresh,
                         lambda d: measurement_visible(d, frame, pose), forget=[target])
            self._recovery_neighbors = copy.deepcopy(neighbors)
            self._refresh_placement_occupancy(target, neighbors, destination)
            visible = candidate_visibility_filter(target, grasp, frame, pose,
                        nominal=self.runner.nominal.get(target.label),
                        max_open_width=self.runner.placer.open_width)
            updated, why = self.runner.replan_release(target, grasp, place, width, neighbors,
                              observation_filter=visible, cancelled=budget.cancelled,
                              progress=progress, **options)
            return updated, why, revision
        updated, why, revision = budget.run(measure)
        if updated is None:
            # Extra views only help when the measurements admit a reachable
            # release. Known occupancy can already rule every candidate out;
            # raising the held object again cannot resolve that obstruction.
            capacity, capacity_why = budget.run(self.runner.replan_release,
                target, grasp, place, width, neighbors, cancelled=budget.cancelled,
                progress=progress, **options)
            if capacity is None:
                why = '当前测量下无可确认的放置方案：' + str(capacity_why)
            else:
                ext = self.runner.ext
                T_fc = ext.T if ext.parent_frame == 'flange' else self.runner.T_flange_tcp @ ext.T
                table = float(self.cfg.get('perception.table.expected_z', 0) or 0)
                xy = (options['placement_zone'].center[:2] if options else
                      self.runner.zones[zone_name].center[:2])
                # The pose generator bounds rotations above the table. Actual
                # obstacles are checked against each full swept route below,
                # so a distant tall item cannot mandate an impossible lift.
                poses = iter_safe_inspection_poses(self.arm, grasp, T_fc,
                                 self.runner.T_flange_tcp, xy, table,
                                 table_z=table, free_tilt_deg=self.runner.free_tilt,
                                 nominal=self.runner.nominal.get(target.label),
                                 intrinsics=self.camera.intrinsics(),
                                 cancelled=budget.cancelled, progress=progress)
                observed_views = 0
                sentinel = object()
                while True:
                    inspection = budget.run(next, poses, sentinel)
                    if inspection is sentinel:
                        break
                    current = self.arm.get_tcp_pose().value.copy()
                    route, route_why = budget.run(safe_inspection_route,
                        self.arm, grasp, current, inspection, neighbors, width,
                        table_z=table, nominal=self.runner.nominal.get(target.label),
                        neighbor_nominal=self.runner.nominal,
                        free_tilt_deg=self.runner.free_tilt,
                        cancelled=budget.cancelled, progress=progress)
                    if not route:
                        why = route_why
                        continue
                    observed_views += 1
                    route_revision = self._environment_revision
                    for waypoint in route:
                        if self.cancel_event.is_set():
                            raise _PlacementInterrupted()
                        self._apply_interventions()
                        if self._environment_revision != route_revision:
                            break
                        self.runner.skill._move(result, 'inspect_place', waypoint, True)
                        self.runner.skill._verify_held(result, 'dropped_in_transit')
                    self._on_stage('placement_planning')
                    updated, why, revision = budget.run(measure)
                    if updated is not None or observed_views >= 4:
                        break
        self._active_revision = revision
        if updated is None:
            result.detail['placement_planning_reason'] = str(why)
            self._emit('placement_changed', {'message': '当前测量下无法确认安全放置方案，将保持夹持', 'reason': why})
            return None
        self._observed_detections = copy.deepcopy(neighbors)
        info = dict(self.runner.placer.last_info)
        self._placement_center = list(info.get('xy', updated.T_release[:2, 3]))
        self._emit('placement_refined', {'message': '已根据局部观测确认放置位置',
                                         'position': self._placement_center})
        return updated

    def _record_feedback(self, kind: str, fields: dict) -> None:
        """Expose existing measured checks; viewer feedback never commands motion."""
        if kind == 'decision' and fields.get('action') in ('pick', 'retry_held_placement', 'replan_release'):
            self._placement_center = list((fields.get('place') or {}).get('xy')
                                          or (fields.get('release_xyz') or [])[:2])
            if fields.get('action') != 'pick':
                return
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
            self._environment_revision += 1
            self._observed_detections = []
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

    def observe(self, target=None, purpose='search', forget=()) -> dict:
        """Capture fresh detections, assigning new IDs for this observation only."""
        with self._operation_lock:
            self._check_open()
            if not self._prepared:
                return self.prepare()
            if self.cancel_event.is_set():
                raise RuntimeError('simulation is stopped; reset it before observing')
            self._status = 'observing'
            self._publish_snapshot()
            from .observation_policy import (merge_local_detections, measurement_visible,
                                             unique_target)
            from ..core.extrinsics import camera_pose_in_base
            local = target is not None
            require_grasp = bool(local and target.get('require_grasp'))
            grasp_target = None
            self._last_fresh = []
            order = self._view_order(target) if local else list(range(self._base_view_count))
            if local and purpose == 'verify_place':
                order = order[:2]
            self._scan_stats['local_scans' if local else 'full_scans'] += 1
            if not local:
                with self._cache_lock:
                    self._observation_views = {}
                    self._observation_jpegs = {}
            views, warnings, coverage = {}, [], []
            revision = self._environment_revision
            for step, view in enumerate(order, 1):
                self._emit_scan_view(view, step=step, total=len(order), local=local)
                status, next_result, frame = self.runner._observe(view)
                if status != 'ok' or next_result is None:
                    why = '%s: %s' % (status, self.runner._last_obs_failure)
                    self._invalidate_scene(why)
                    self._status = 'error'
                    self._publish_snapshot()
                    raise RuntimeError('simulation observation failed: ' + why)
                result = next_result
                self._scan_stats['frames'] += 1
                self._cache_observation(view, result, frame)
                views[view] = tag_view(result.detections, view)
                warnings.extend(result.warnings)
                flange = self.arm.get_flange_pose()
                pose = camera_pose_in_base(self.runner.ext, flange.value, self.runner.T_flange_tcp)
                coverage.append((frame, pose))
                if local:
                    label = target.get('label')
                    measured = unique_target(views[view], label,
                        partial=lambda d: self._target_partial(d, target)) if label else None
                    # A pear's low bulb view is valid but insufficient for neck grasping.
                    reliable = measured is not None and (label != 'pear' or
                                float(measured.height or 0) >= float(target.get('height', 0) or 0) - .007)
                    geometry_ready = (label not in ('banana', 'pear') or
                                      self._stable_cloud(measured, revision) is not None or
                                      sum(any(d.valid and d.label == label for d in ds)
                                          for ds in views.values()) >= 2)
                    if require_grasp:
                        # A complete silhouette can still hide the contact
                        # section. Seek another target view when the unchanged
                        # grasp planner cannot confirm stable pad contact.
                        from .obstacle_recovery import _grasp
                        if reliable:
                            peers = [d for d in self._observed_detections
                                     if d.label != label]
                            peers += [d for d in views[view] if d.label != label]
                            _candidate, planned = _grasp(self, measured, peers)
                            if planned.feasible:
                                grasp_target = measured
                        if grasp_target is not None and geometry_ready:
                            break
                        continue
                    if (reliable and (geometry_ready or purpose == 'verify_place')) or (not label and len(coverage) >= 1):
                        break
            detections, diagnostics = merge_observations(
                views, dist=self.runner.view_merge,
                partial=lambda d: not d.valid or self.runner._partial(d))
            if require_grasp:
                # Preserve the actual successful profile, rather than letting
                # the generic density ranking choose a denser rejected view.
                label = target.get('label')
                if grasp_target is not None:
                    detections = [d for d in detections if d.label != label]
                    detections.append(copy.deepcopy(grasp_target))
                else:
                    for d in detections:
                        if d.label == label:
                            d.valid = False
                            d.reject_reason = 'stable_grasp_not_observed'
            fresh = list(detections)
            for det in fresh:
                cloud = self._placement_cloud(det, views, fresh)
                previous_cloud = self._stable_cloud(det, revision)
                if previous_cloud is not None:
                    cloud = previous_cloud if cloud is None else np.vstack([previous_cloud, cloud])
                if cloud is not None:
                    det.extra['_placement_points_base'] = cloud
                    det.extra['_cloud_views'] = max(2 if previous_cloud is not None else 0,
                                                     det.extra.get('_placement_view_count', 0))
                det.extra['_observed_revision'] = revision
            if require_grasp and grasp_target is not None:
                final_target = next(d for d in fresh if d.label == target.get('label'))
                _candidate, planned = _grasp(self, final_target, fresh)
                if not planned.feasible:
                    final_target.valid = False
                    final_target.reject_reason = 'stable_grasp_not_observed: ' + str(planned.reason)
            if local:
                detections = merge_local_detections(
                    self._observed_detections, fresh,
                    lambda d: any(measurement_visible(d, f, p) for f, p in coverage),
                    forget=forget)
            self._last_fresh = fresh
            self._observation_revision = revision
            self._last_observation_purpose = purpose
            self._observed_detections = copy.deepcopy(detections)
            self._emit('observation_complete', {'purpose': purpose, 'views': list(views),
                       'message': '局部观测完成' if local else '首次全场观测完成',
                       'stats': copy.deepcopy(self._scan_stats)})
            return self._publish_scene(detections, warnings, diagnostics, result.frame_stamp)

    def _stable_cloud(self, det, revision):
        if det is None or det.center_base is None:
            return None
        previous = [d for d in self._observed_detections if d.label == det.label and d.valid]
        if len(previous) != 1:
            return None
        old = previous[0]
        if det.label in ('banana', 'pear') and old.extra.get('_cloud_views', 0) < 2:
            return None
        if (old.extra.get('_observed_revision') != revision or
                np.linalg.norm(det.center_base[:2] - old.center_base[:2]) > .003):
            return None
        if det.label in ('banana', 'pear') and abs(math.remainder(
                float(det.yaw or 0) - float(old.yaw or 0), math.pi)) > math.radians(4):
            return None
        if any(a is None or b is None or abs(a-b) > .10 * max(a, b)
               for a, b in ((det.length, old.length), (det.width, old.width))):
            return None
        return old.extra.get('_placement_points_base')

    def _target_partial(self, det, reference):
        """A known complete fruit need not match the old competition size prior."""
        if not self.runner._partial(det):
            return False
        return not all(a is not None and b is not None and
                       .9 * float(b) <= float(a) <= 1.15 * float(b)
                       for a, b in ((det.length, reference.get('length')),
                                    (det.width, reference.get('width')),
                                    (det.height, reference.get('height'))))

    def _view_order(self, target):
        from .observation_policy import rank_views
        ext = self.runner.ext
        T_fc = ext.T if ext.parent_frame == 'flange' else self.runner.T_flange_tcp @ ext.T
        cameras = [self.arm.kin.fk(q) @ T_fc for q in self.runner.views]
        point = target.get('center_base') or target.get('xy') or [.26, 0]
        return rank_views(cameras, self.camera.intrinsics(), point[:2],
                          preferred=target.get('view'),
                          half_extent=max(.04, float(target.get('length') or .08) / 2),
                          target_z=float(target.get('top_z') or .04))

    def _publish_scene(self, detections, warnings, diagnostics, stamp):
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
                measured.extra['object_id'] = object_id
                # Keep the visible support from every uniquely associated
                # wrist view. A curved object's rectangle center is not a
                # rigid anchor when it is rotated for a later placement.
                if measured.center_base is not None:
                    measured.center_base.setflags(write=False)
                if measured.mask is not None:
                    measured.mask.setflags(write=False)
                mapping[object_id] = measured
                objects.append(dict(measured.summary(), object_id=object_id,
                                    in_zone=zone is not None, zone=zone,
                                    view=measured.extra.get('view', 0)))
            self._detections = MappingProxyType(mapping)
            self._observed_detections = list(mapping.values())
            self._scene = {'scene_version': self._scene_version, 'objects': objects,
                           'zones': self._zones(),
                           'placement': copy.deepcopy(self._placement),
                           'class_to_zone': dict(self.runner.class_to_zone),
                           'task': 'Sandbox', 'mode': 'sim',
                           'observation_diagnostics': diagnostics,
                           'observed_at': float(stamp),
                           'observation_stats': copy.deepcopy(self._scan_stats),
                           'warnings': warnings}
            self._status = 'ready'
            self._stage = ''
            self._publish_snapshot()
            self._emit('scene', self._scene)
            return copy.deepcopy(self._scene)

    def _placement_cloud(self, det, views, merged):
        if det.center_base is None:
            return None
        peers = [d for d in merged if d.label == det.label and d.valid and d.center_base is not None]
        radius = max(.05, float(self.runner._footprint(det) or 0) * .75)
        clouds, used_views = [], set()
        for view_id, detections in views.items():
            for candidate in detections:
                if candidate.label != det.label or candidate.center_base is None or candidate.score < .5:
                    continue
                matches = [d for d in peers if np.linalg.norm(d.center_base[:2] - candidate.center_base[:2]) <= radius]
                if len(matches) != 1 or matches[0] is not det:
                    continue
                points = np.asarray((candidate.extra or {}).get('_points_base'), dtype=float)
                if points.ndim == 2 and points.shape[1] == 3 and len(points) >= 40 and np.all(np.isfinite(points)):
                    clouds.append(points)
                    used_views.add(view_id)
        if not clouds:
            return None
        points = np.vstack(clouds)
        _, keep = np.unique(np.floor(points / .002).astype(np.int64), axis=0, return_index=True)
        points = points[np.sort(keep)]
        det.extra['_placement_view_count'] = len(used_views)
        points.setflags(write=False)
        return points

    def _emit_scan_view(self, view: int, step=None, total=None, local=False) -> None:
        total = len(self.runner.views) if total is None else total
        label = self._view_labels[view]
        self._emit('scan_view', {'view': view, 'total': total, 'label': label,
                                 'message': ('正在局部观察：' + label if local else
                                             '正在扫描 %d/%d：%s' % (step or view + 1, total, label))})

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
                       'placement': copy.deepcopy(self._placement),
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
                destination = get('destination')
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
                # Reject malformed or out-of-workspace requests before any camera motion.
                if destination is not None:
                    if zone_name is not None:
                        return self._refusal('invalid_destination', '放置区域与桌面目标不能同时指定', object_id, zone_name)
                    try:
                        destination = validate_destination(destination, self._scene)
                    except ValueError as exc:
                        return self._refusal('invalid_destination', str(exc), object_id, None)
                    placement_options = self._placement_options(destination)
                    zone_name = placement_options['placement_zone'].name
                else:
                    placement_options = {}
                    zone_name = zone_name or self.runner.class_to_zone[det.label]
                    if zone_name not in self.runner.zones:
                        return self._refusal('invalid_zone', 'target zone does not exist', object_id, zone_name)
                # Initial intent is stable; coordinates must be confirmed in a new image.
                original = copy.deepcopy(det)
                preconfirmed = (getattr(self, '_last_observation_purpose', '') in
                                ('recover_target', 'recover_queued_target') and
                                self._observation_revision == self._environment_revision and
                                any(d.valid and d.label == original.label and d.center_base is not None and
                                    np.linalg.norm(d.center_base[:2] - original.center_base[:2]) < .003
                                    for d in self._last_fresh))
                if not preconfirmed:
                    self.observe(target=original.summary() | {'view': original.extra.get('view')},
                                 purpose='confirm_pick', forget=[original])
                self._last_observation_purpose = 'consumed_for_pick'
                from .observation_policy import unique_target
                det = unique_target(self._last_fresh, original.label,
                                    partial=lambda d: self._target_partial(d, original.summary()))
                if det is None:
                    return self._refusal('target_lost', '目标未在新相机图像中可靠找到，未执行旧坐标抓取',
                                         object_id, zone_name)
                if (np.linalg.norm(det.center_base[:2] - original.center_base[:2]) > .04
                        and not self.editor.metadata().get('one_per_class')):
                    return self._refusal('target_ambiguous', '目标发生较大移动，无法唯一确认身份', object_id, zone_name)
                self._active_revision = self._observation_revision
                # Pass the exact measured target among the valid neighbors; this
                # preserves the runner's identity checks for transit clearance.
                target = copy.deepcopy(det)
                self.runner._tracks = [t for t in self.runner._tracks if not
                    (t.get('label') == det.label and
                     np.linalg.norm(np.asarray(t['xy']) - det.center_base[:2]) < .04 and
                     str(t.get('reason', '')).startswith('grasp infeasible: blocked_by_neighbor'))]
                if get('_restore_yaw') is not None:
                    yaw = get('_restore_yaw')
                    if isinstance(yaw, bool) or not isinstance(yaw, (float, int)) or not math.isfinite(yaw):
                        raise ValueError('恢复朝向无效')
                    target.extra['_restore_yaw'] = float(yaw)
                neighbors = [target]
                neighbors.extend(copy.deepcopy(d) for d in self._observed_detections
                                 if d.center_base is not None and not (d.label == det.label and
                                 np.linalg.norm(d.center_base[:2] - det.center_base[:2]) < .04))
                if destination is not None and destination['type'] == 'point':
                    # The requested point can lie beyond the old pick/card
                    # views. Measure its surroundings before committing a grasp.
                    view = self._ensure_destination_view(destination)
                    if view is None:
                        return self._refusal('destination_not_observed',
                                             '无法从可达相机视角确认目标点周围空间', object_id, zone_name)
                    self._status = 'observing'
                    self._emit_scan_view(view)
                    self._publish_snapshot()
                    status, local, frame = self.runner._observe(view)
                    if status != 'ok' or local is None:
                        return self._refusal('destination_not_observed',
                                             '目标点观测未完成：' + str(status), object_id, zone_name)
                    self._cache_observation(view, local, frame)
                    additions = [d for d in local.detections if d.center_base is not None
                                 and not (d.label == det.label and
                                          np.linalg.norm(d.center_base[:2] - det.center_base[:2]) < .03)]
                    for other in additions:
                        neighbors = [d for d in neighbors if d is target or d.label != other.label or
                                     np.linalg.norm(d.center_base[:2] - other.center_base[:2]) > .03]
                        neighbors.append(copy.deepcopy(other))
                # Independent evaluation only: this association is never used
                # by the motion planner or the model's selectable scene.
                evaluation_object = self._evaluation_object(det)
                self._recovery_target = copy.deepcopy(det)
                self._recovery_neighbors = copy.deepcopy([d for d in neighbors if d is not target])
                self._recovery_goal = {'label': det.label,
                                       'zone_name': None if destination else zone_name}
                if destination is not None:
                    self._recovery_goal['destination'] = copy.deepcopy(destination)
                self._recovery_evaluation = evaluation_object
                self._recovery_context = None
                self._feedback = {'grasp_plan': {}, 'checks': []}
                self._placement_center = None
                self._refresh_placement_occupancy(target, neighbors, destination=destination)
                # A prior explicit request may have failed only because its
                # destination was blocked. A new destination can be feasible.
                if destination is not None:
                    self.runner._tracks = [t for t in self.runner._tracks
                                           if not (t.get('label') == det.label and
                                                   np.linalg.norm(np.asarray(t['xy']) - det.center_base[:2]) < .03
                                                   and str(t.get('reason', '')).startswith(('place infeasible', 'unreachable')))]
                self._status = 'executing'
                self._active_placement = (target, destination, zone_name, placement_options)
                self._publish_snapshot()
                result, why = self.runner.pick_detection(target, zone_name=zone_name,
                                                         neighbors=neighbors, **placement_options)
                return self._finish_attempt(result, why, object_id, zone_name,
                                            det, evaluation_object, destination)
            except Exception:
                # An unexpected adapter/planner failure must expire every
                # selectable target and retain a held cube for an explicit reset.
                self.stop()
                self._invalidate_scene('execution failed')
                self._publish_snapshot()
                raise
            finally:
                self.cancel_event.external = None

    def _placement_options(self, destination):
        """Resolve configured support into a local planning zone, never a world zone."""
        if destination['type'] == 'region':
            polygon = np.asarray(self._placement['pick_polygon'], float)
            low, high = polygon.min(axis=0), polygon.max(axis=0)
            center, size, name = (low + high) / 2, high - low, 'pick_area'
        else:
            table = self._placement['table_bounds']
            polygon = np.array([[table['x'][0], table['y'][0]], [table['x'][1], table['y'][0]],
                                [table['x'][1], table['y'][1]], [table['x'][0], table['y'][1]]])
            center = np.asarray(destination['xy'], float)
            size = 2 * np.max(np.abs(polygon - center), axis=0) + .01
            name = 'desktop_point'
        floor = self.cfg.get('perception.table.expected_z', 0.0)
        zone = Zone(name=name, center=np.r_[center, float(floor or 0.0)],
                    size=tuple(float(v) for v in size), label=destination_label(destination))
        return dict(placement_zone=zone, placement_polygon=polygon,
                    exact_xy=destination['type'] == 'point')

    def _ensure_destination_view(self, destination):
        """Add a reachable camera view of a requested point for later instructions."""
        if destination['type'] != 'point':
            return
        key = tuple(round(float(v), 3) for v in destination['xy'])
        if key in self._destination_views:
            return self._destination_views[key]
        from ..sim.scenarios import look_at
        ext = self.runner.ext
        T_fc = ext.T if ext.parent_frame == 'flange' else self.runner.T_flange_tcp @ ext.T
        target = np.r_[destination['xy'], float(self.cfg.get('perception.table.expected_z', 0) or 0) + .025]
        radial = target[:2] / max(float(np.linalg.norm(target[:2])), 1e-6)
        for distance, height in ((.13, .255), (.10, .25), (.16, .28), (.08, .22)):
            eye = np.r_[target[:2] - distance * radial, target[2] + height]
            q, _ep, _er = self.arm.kin.ik(look_at(eye, target) @ inv_T(T_fc),
                                          q_seed=self.runner.views[len(self.runner.views)//2])
            if q is not None:
                # Reuse one transient view instead of growing every future scan.
                view = self._base_view_count
                if len(self.runner.views) == view:
                    self.runner.views.append(q.tolist())
                    self._view_labels.append('桌面目标点视角')
                else:
                    self.runner.views[view] = q.tolist()
                self.runner.multi_view = True
                self._destination_views = {key: view}
                return view

    def _verify_destination(self, obj, destination):
        """Independent truth check, after execution only; never supplies a plan."""
        if obj is None or obj is self.world.held or abs(obj.bottom_z() - self.world.table_z) > .01:
            return False
        if (destination['type'] == 'point' and
                np.linalg.norm(obj.center_xy() - destination['xy']) > self._placement['point_tolerance']):
            return False
        return footprint_in_destination(np.vstack(obj.footprint_polygons()),
                                        {'placement': self._placement}, destination)

    def verify_goals(self, goals):
        """Recheck changed completed goals, sharing one local view per destination."""
        from .batch import whole_footprint_at_goal
        groups = {}
        for goal in goals:
            destination = goal.get('destination')
            if destination is None:
                zone = self.runner.zones[goal['zone_name']]
                key, xy = ('zone', zone.name), zone.center[:2].tolist()
            elif destination['type'] == 'point':
                key, xy = ('point', *destination['xy']), destination['xy']
            else:
                key, xy = ('region', destination['name']), [.26, 0.]
            groups.setdefault(key, (xy, []))[1].append(goal)
        items = []
        for xy, group in groups.values():
            scene = self.observe(target={'xy': xy}, purpose='verify_place')
            fresh = list(self._last_fresh)
            for goal in group:
                def confirmed():
                    return (self._observation_revision == self._environment_revision and any(
                        d.valid and d.label == goal['label'] and
                        whole_footprint_at_goal(d.summary(), scene, goal) for d in fresh))
                if not confirmed():
                    scene = self.observe(target={'xy': xy, 'label': goal['label']}, purpose='verify_place')
                    fresh = list(self._last_fresh)
                items.append({'label': goal['label'], 'ok': confirmed(),
                              'verified_revision': self._observation_revision,
                              'goal': copy.deepcopy(goal)})
        for item in items:
            item['ok'] = item['ok'] and item['verified_revision'] == self._environment_revision
        ok = all(item['ok'] for item in items)
        return {'ok': ok, 'items': items, 'verified_revision': self._environment_revision,
                'message': '新相机图像已确认全部目标' if ok else '场景变化后部分目标不再满足原任务'}

    def _finish_attempt(self, result, why, object_id, zone_name, det, evaluation_object,
                        destination=None):
        """Use the same independent placement check for initial and recovery actions."""
        summary = (pick_summary(result) if result is not None else
                   {'ok': False, 'stage': 'plan', 'error': why, 'label': det.label,
                    'zone': zone_name, 'duration': 0.0})
        if result is not None:
            summary['diagnostics'] = str(result.detail.get('last_motion_detail') or
                                         result.detail.get('placement_planning_reason') or
                                         result.detail.get('reason') or why or '')[:500]
        summary['object_id'] = object_id
        summary['source_xy'] = det.center_base[:2].tolist()
        if destination is not None:
            summary['destination'] = copy.deepcopy(destination)
            summary['placement_center'] = copy.deepcopy(self._placement_center)
        summary['aborted'] = self.cancel_event.is_set()
        summary['holding'] = self.world.held is not None
        self._recovery_context = None
        if destination is not None and result is None and not summary['aborted']:
            self._status, self._stage = 'ready', ''
        else:
            self._status = 'ready'
        if result is None and destination is not None and not summary['aborted']:
            pass  # Planning refusal sent no commands and preserves this observation.
        elif not summary['aborted'] and not summary.get('fault') and not summary['holding']:
            try:
                if destination is not None:
                    xy = (self._placement_center or destination.get('xy') or [.26, 0])
                    if destination['type'] == 'point':
                        self._ensure_destination_view(destination)
                    query = {'xy': xy}
                else:
                    query = {'xy': self.runner.zones[zone_name].center[:2].tolist()}
                if result is not None and result.ok:
                    query.update(label=det.label, height=det.height)
                self.observe(target=query, purpose='verify_place',
                             forget=[det] if result is not None and result.ok else [])
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
        if destination is not None:
            truth_success = self._verify_destination(evaluation_object, destination)
        summary['controller_ok'] = bool(summary['ok'])
        summary['truth'] = {'success': truth_success,
                            'object_name': None if evaluation_object is None else evaluation_object.name,
                            'expected_zone': zone_name,
                            'actual_zone': None if actual_zone is None else actual_zone.name}
        if destination is not None:
            summary['truth']['expected_destination'] = copy.deepcopy(destination)
            summary['truth']['expected_zone'] = None
        from .batch import whole_footprint_at_goal
        goal = {'zone_name': zone_name}
        if destination is not None:
            goal['destination'] = destination
        camera_success = any(d.valid and d.label == det.label and
                             whole_footprint_at_goal(d.summary(), self._scene, goal)
                             for d in getattr(self, '_last_fresh', []))
        summary['camera_verified'] = camera_success
        summary['verified_revision'] = getattr(self, '_observation_revision', -1)
        camera_success = camera_success and (getattr(self, '_observation_revision', -1) ==
                                              getattr(self, '_environment_revision', -2))
        summary['camera_verified'] = camera_success
        summary['ok'] = summary['controller_ok'] and truth_success and camera_success
        if summary['controller_ok'] and not truth_success:
            summary['error'] = 'placement_not_verified'
        elif summary['controller_ok'] and not camera_success:
            summary['error'] = 'placement_not_observed'
        colors = {'red': '红色', 'green': '绿色', 'yellow': '黄色'}
        if summary['aborted']:
            summary['message'] = ('任务已停止，夹爪保持夹持，机械臂保持当前位置'
                                  if summary['holding'] else '任务已停止，机械臂保持当前位置')
        elif summary['ok']:
            zone_label = self.runner.zones[zone_name].label if destination is None else ''
            names = {'red': '红色方块', 'green': '绿色方块', 'yellow': '黄色方块',
                     'apple': '苹果', 'banana': '香蕉', 'pear': '梨子'}
            summary['message'] = '已将%s放到%s' % (
                names.get(det.label, det.label), destination_label(destination) if destination else
                colors.get(zone_label, zone_label or zone_name) + '区域')
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

    def prepare_recovery(self, failure, goal, attempt, history, reference_xy=None, completed_zones=(),
                         completed_positions=(), protected_goals=()):
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
                from .observation_policy import merge_local_detections, measurement_visible
                from ..core.extrinsics import camera_pose_in_base
                pose = camera_pose_in_base(self.runner.ext, self.arm.get_flange_pose().value,
                                            self.runner.T_flange_tcp)
                neighbors = merge_local_detections(self._recovery_neighbors,
                    [d for d in fresh if d.center_base is not None and d.label != goal['label']],
                    lambda d: measurement_visible(d, self.camera._last, pose),
                    forget=[self._recovery_target])
                self._recovery_neighbors = neighbors
                observations = [d.summary() for d in neighbors]
                if getattr(self.runner, 'has_held_recovery', False):
                    previous = history or failure.get('recovery_history') or []
                    repeated = sum(item.get('option_id') == 'place_held_alternate' and not item.get('ok') for item in previous)
                    if repeated < 2:
                        options.append({'id': 'place_held_alternate', 'description':
                        '保持当前夹持，依据相机测得的已有物体重新选择原目标区域中的空位，'
                        '排除之前失败的放置点，抬升后移动、下降、放下并核验；'
                        '不重复抓取，不在空中松爪。本次是第 %d 次重新选位。' % attempt})
                    from .held_recovery import propose_held_recovery
                    options.extend(propose_held_recovery(self, goal, protected_goals))
            elif healthy:
                observation_mode = 'target_search'
                scene = self.observe(target=self._recovery_target.summary() |
                                     {'view': self._recovery_target.extra.get('view')},
                                     purpose='recover_target', forget=[self._recovery_target])
                observations = scene['objects']
                matches = [o for o in observations if o.get('valid') and
                           o.get('label') == goal['label']]
                if reference_xy is not None:
                    from .batch import matching_objects
                    matches = matching_objects(scene, goal['label'], reference_xy, completed_zones,
                                               completed_positions)
                    if not matches and self.editor.metadata().get('one_per_class'):
                        matches = [o for o in scene['objects'] if o.get('valid') and o.get('label') == goal['label']]
                if len(matches) == 1:
                    target = matches[0]
                    self._recovery_retry = {
                        'scene_version': scene['scene_version'], 'action': 'pick',
                        'object_id': target['object_id'], 'zone_name': goal['zone_name']}
                    if goal.get('destination') is not None:
                        self._recovery_retry['destination'] = copy.deepcopy(goal['destination'])
                    options.append({'id': 'reobserve_retry', 'description':
                        '根据刚完成的目标优先观测，重新定位同一个目标，使用实测截面与邻物间隙重新规划。'})
                    from .obstacle_recovery import propose_clearance
                    measured = self._detections[target['object_id']]
                    clearance = propose_clearance(self, measured, goal)
                    if clearance:
                        # A direct retry was already geometrically ruled out.
                        options = clearance
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
            if goal.get('destination') is not None:
                context['placement'] = copy.deepcopy(self._placement)
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
        self._scan_stats['frames'] += 1
        self.runner._last_stamp = float(frame.stamp)
        self._cache_observation(6, result, frame, label='恢复时静止观测')
        self._emit('recovery_observation', {'message': '已保持夹持并完成静止相机观测',
                                           'detections': [d.summary() for d in result.detections]})
        return result.detections

    def execute_recovery(self, decision, cancel_event=None, confirmed=False):
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
                self._apply_interventions()
                if chosen['option_id'].startswith('stage_'):
                    from .held_recovery import execute_held_recovery
                    return execute_held_recovery(self, chosen['option_id'], cancel_event, confirmed)
                if chosen['option_id'].startswith('clear_restore_'):
                    from .obstacle_recovery import execute_clearance
                    saved = {key: copy.deepcopy(getattr(self, key)) for key in
                             ('_recovery_target', '_recovery_goal', '_recovery_neighbors')}
                    saved_evaluation = self._recovery_evaluation
                    outcome = execute_clearance(self, chosen['option_id'], cancel_event)
                    if not outcome.get('ok') and not outcome.get('pending_restore') and not outcome.get('holding'):
                        for key, value in saved.items():
                            setattr(self, key, value)
                        self._recovery_evaluation = saved_evaluation
                    return outcome
                if chosen['option_id'] == 'reobserve_retry':
                    if self.world.held is not None or self._recovery_retry is None:
                        raise ValueError('当前状态不能重复抓取')
                    return self.execute(self._recovery_retry, cancel_event=cancel_event)
                if chosen['option_id'] != 'place_held_alternate' or self.world.held is None:
                    raise ValueError('当前状态不能续接放置')
                target = self._recovery_target
                goal = self._recovery_goal
                destination = goal.get('destination')
                options = self._placement_options(destination) if destination is not None else {}
                zone_name = options['placement_zone'].name if destination is not None else goal['zone_name']
                self._refresh_placement_occupancy(target, self._recovery_neighbors, destination)
                self._status = 'executing'
                result, why = self.runner.retry_held_placement(
                    target, zone_name, neighbors=self._recovery_neighbors, **options)
                return self._finish_attempt(result, why, 'held-recovery', zone_name,
                                            target, self._recovery_evaluation, destination)
            finally:
                self.cancel_event.external = None

    def _refresh_placement_occupancy(self, target, detections=None, destination=None) -> None:
        """Replace the round's placement history with measured current neighbors.

        The target will leave its current slot. Excluding it also lets an
        instruction place it again in the same zone without a ghost collision.
        """
        occupied, labels = {}, {}
        for det in (self._detections.values() if detections is None else detections):
            if det is target or det.center_base is None or (not det.valid and destination is None):
                continue
            x, y = [float(v) for v in det.center_base[:2]]
            zone = next((z for z in self.runner.zones.values() if z.contains_xy(x, y)), None)
            footprint = self.runner._footprint(det)
            if destination is not None:
                zone = self._placement_options(destination)['placement_zone']
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
        elif self._status == 'observing':
            self._status = 'ready'
            self._publish_snapshot()
        messages = {'aborted': '任务已停止，机械臂保持当前位置',
                    'stale_scene': '观察结果已更新，请重新选择目标',
                    'invalid_target': '当前目标没有可用的抓取信息',
                    'invalid_zone': '场景中没有这个放置区域',
                    'invalid_action': '当前指令不需要执行抓取',
                    'invalid_destination': '放置目标无效',
                    'destination_not_observed': '当前无法确认目标点周围空间，未执行抓取'}
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
                      intervention_available=self._prepared and not self.cancel_event.is_set(),
                      intervention_pending=len(self._interventions),
                      editable=(self._prepared and not self.cancel_event.is_set()
                                and self.world.held is None and self._status in ('ready', 'error')))
        snap = {'mode': 'sim', 'simulation_kind': 'kinematic', 'seed': self.seed,
                'placement': copy.deepcopy(self._placement),
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
                'observation_stats': copy.deepcopy(self._scan_stats),
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
