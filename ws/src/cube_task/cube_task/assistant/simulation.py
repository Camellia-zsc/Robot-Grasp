"""Camera-to-motion adapter and editable tabletop for the local assistant.

The dashboard supports kinematic and MuJoCo simulation. Perception
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
from ..core.transforms import inv_T, rotation_angle
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
DASHBOARD_ZONE_COLORS_BGR = {
    'zone_red': (188, 192, 234),
    'zone_green': (179, 216, 186),
    'zone_yellow': (160, 230, 248),
}
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


class _SourceObservationUnavailable(ValueError):
    """No source-view motion passed the final current-state path check."""


class _PlacementInterrupted(Exception):
    pass


class _PlacementBudget:
    """Bound computation only; camera motions do not consume this budget."""

    def __init__(self, arm, stop, seconds=45.):
        self.arm, self.stop = arm, stop
        self.remaining = float(seconds)
        self.deadline = None
        self.wall_deadline = None
        self.now = getattr(arm, 'planning_time', time.monotonic)

    def cancelled(self):
        return (self.stop.is_set() or self.remaining <= 0 or
                (self.deadline is not None and self.now() >= self.deadline) or
                (self.wall_deadline is not None and time.monotonic() >= self.wall_deadline))

    def run(self, fn, *args, **kwargs):
        if self.cancelled():
            raise _PlacementInterrupted()
        started = self.now()
        self.deadline = started + self.remaining
        self.wall_deadline = time.monotonic() + max(30., 6*self.remaining)
        scope = getattr(self.arm, 'planning_scope', None)
        try:
            with scope(self.cancelled) if scope is not None else nullcontext():
                value = fn(*args, **kwargs)
                if self.cancelled():
                    raise _PlacementInterrupted()
                return value
        finally:
            self.remaining -= self.now() - started
            self.deadline = None
            self.wall_deadline = None


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
            started = time.monotonic()
            if self.on_tick is not None:
                self.on_tick()
            # Physics is work within this playback interval too. Waiting
            # before advancing would add the full solver cost to every frame.
            super().sleep(step)
            if self.realtime and self.paced and not self.cancel_event.is_set():
                delay = step / self.speed - (time.monotonic() - started)
                if delay > 0:
                    self.cancel_event.wait(delay)
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
                 realtime: bool = True, backend: str = 'sim'):
        if backend not in ('sim', 'mujoco'):
            raise ValueError('仿真后端必须为 sim 或 mujoco')
        self.backend = backend
        self._physics = None
        self._physics_reference = {}
        self._idle_stop = threading.Event()
        self._idle_thread = None
        self.seed = int(seed)
        self.on_event = on_event
        self.cancel_event = _Cancellation()
        self.clock = _ViewerClock(self.cancel_event, realtime)
        self._operation_lock = threading.RLock()
        self._intervention_lock = threading.Lock()
        self._interventions = deque()
        self._intervention_id = 0
        self._environment_revision = 0
        self._last_physical_change_revision = -1
        self._active_revision = 0
        self._observed_detections = []
        self._scan_stats = {'frames': 0, 'full_scans': 0, 'local_scans': 0}
        self._cache_lock = threading.Lock()
        self._snapshot: dict = {}
        self._last_snapshot_wall = float('-inf')
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
        # Dashboard cards have room for multiple library objects. Competition
        # layouts and the physical robot's taught zones are not changed.
        for zone in self.world.zones:
            zone.size = tuple(max(.30, float(v)) for v in zone.size)
            if zone.name in DASHBOARD_ZONE_COLORS_BGR:
                zone.color_bgr = DASHBOARD_ZONE_COLORS_BGR[zone.name]
            self.cfg.set('zones.' + zone.name + '.size', list(zone.size))
            self.info['zones'][zone.name].size = zone.size
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
        if backend == 'mujoco':
            from .physics import DashboardPhysics
            self._physics = DashboardPhysics(self.cfg, self.world, self.info, self.clock,
                                             on_event=self._emit)
            self.cfg = self._physics.cfg
            self.arm, physics_camera = self._physics.arm, self._physics.camera
        else:
            self.arm = make_arm(self.cfg, self.clock, sim_world=self.world)
        self.arm.planning_time = time.thread_time
        self.operator = Operator(enabled=False)
        self.operator.abort_event = self.cancel_event
        self.arm.set_abort_event(self.cancel_event)
        self.camera = (physics_camera if self._physics is not None else
                       make_camera(self.cfg, self.clock, sim_world=self.world, arm=self.arm))
        self.logger = _EventLogger(self, log_dir)
        self.runner = TaskRunner(self.cfg, 'Sandbox', self.arm, self.camera,
                                 make_sandbox_perception(self.cfg), self.clock, self.logger,
                                 operator=self.operator, mode=backend)
        from .motion_observation import MotionObservation
        self._motion_observer = MotionObservation(self)
        self._configure_observation_views()
        self._base_view_count = len(self.runner.views)
        self.runner.skill.on_stage = self._on_stage
        self.runner.skill.execution_guard = self._execution_guard
        self.runner.skill.before_release = self._before_release
        self.runner.skill.before_open = self._before_open
        if self._physics is not None:
            self.runner.skill.before_carry = self._before_carry
            self.runner.skill.before_held_transit = self._before_held_transit
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
        observer = getattr(self, '_motion_observer', None)
        # The stage callback belongs to motion primitives; gripper commands
        # do not call it. Lift is the first callback after successful closure.
        if observer is not None and stage in ('lift', 'retry_lift') and self._recovery_target is not None:
            # Only the body in our measured grasp starts moving. Earlier
            # background depth remains valid (the old source body can only
            # occlude it conservatively). Completion evidence is still revoked.
            observer.invalidate([self._recovery_target.label], clear_views=False)
        if observer is not None and stage == 'retreat':
            observer.frames.clear()  # A released body now occupies new space.
        self._publish_snapshot()
        self._emit('stage', {'stage': stage,
                             'message': _STAGE_MESSAGES.get(stage, '正在执行机械臂动作')})

    def _tick(self):
        if getattr(self, '_physics', None) is not None:
            self._physics.sync()
            # Contacts already exclude held bodies. A resting stack elsewhere
            # still needs accurate integration while another target is handled.
            self._physics.set_rest_contact_precision(True)
            self._check_physical_changes()
        revision = self._environment_revision
        self._apply_interventions()
        observer = getattr(self, '_motion_observer', None)
        if observer is not None and self._status in ('executing', 'settling'):
            observer.capture()
        # Keep every intervention/control tick, but bound display work in wall
        # time. Stage/observation publications and human edits remain immediate.
        if (revision != self._environment_revision or
                time.monotonic() - self._last_snapshot_wall >= .15):
            self._publish_snapshot()

    def _remember_physics(self):
        self._physics_reference = {o.name: o.T_base_obj.copy() for o in self.world.objects}

    def _check_physical_changes(self):
        """Unexpected body motion invalidates camera proofs, including while idle."""
        if not self._prepared or not self._physics_reference:
            self._remember_physics()
            return
        context = getattr(getattr(self, 'runner', None), '_held_recovery_context', None)
        active = getattr(self, '_recovery_target', None)
        active_label = (getattr(context.get('det'), 'label', None) if context else
                        getattr(active, 'label', None))
        self._physics.expected_labels = ({active_label} if active_label and
            self._status in ('executing', 'settling') else set())
        moving = []
        for obj in self.world.objects:
            old = self._physics_reference.get(obj.name)
            if old is None:
                continue
            # The currently commanded target can move without invalidating its
            # own plan. Other bodies remain obstacles with measured positions.
            if self._status in ('executing', 'settling') and obj.label == active_label:
                self._physics_reference[obj.name] = obj.T_base_obj.copy()
                continue
            distance = float(np.linalg.norm(obj.T_base_obj[:3, 3] - old[:3, 3]))
            angle = math.acos(float(np.clip((np.trace(old[:3, :3].T @ obj.T_base_obj[:3, :3])-1)/2, -1, 1)))
            if distance > .006 or angle > math.radians(8):
                moving.append(obj.label)
        if moving:
            self._environment_revision += 1
            self._last_physical_change_revision = self._environment_revision
            observer = getattr(self, '_motion_observer', None)
            if observer is not None:
                observer.invalidate(moving)
            self._remember_physics()
            self._emit('physics_scene_changed', dict(labels=moving,
                message='物理接触使物品位置发生变化，旧观测和完成记录已失效'))

    def _start_idle_physics(self):
        if self._physics is None or self._idle_thread is not None or not self.clock.realtime:
            return
        def idle():
            while not self._idle_stop.is_set():
                started = time.monotonic()
                if not self._operation_lock.acquire(blocking=False):
                    self._idle_stop.wait(.025)
                    continue
                try:
                    if (self._prepared and not self._closed and not self.cancel_event.is_set()
                            and self._status in ('ready', 'error')):
                        self.clock.sleep(.025)
                finally:
                    self._operation_lock.release()
                # The viewer clock has already paced its physics interval.
                # Only idle/backoff work needs an additional interruptible wait.
                self._idle_stop.wait(max(.001, .025 / self.clock.speed -
                                         (time.monotonic() - started)))
        self._idle_thread = threading.Thread(target=idle, name='dashboard-physics', daemon=True)
        self._idle_thread.start()

    def intervene_scene(self, request):
        """Queue an edit; only the simulation owner mutates world geometry."""
        if not isinstance(request, dict) or request.get('action') != 'move':
            raise ValueError('运行中干预仅支持移动或旋转未夹持物品')
        if set(request) - {'action', 'object_id', 'xy', 'yaw', 'scene_version', 'environment_revision'}:
            raise ValueError('干预请求含有不支持的字段')
        snap = self.snapshot()
        if self._closed or self.cancel_event.is_set() or not self._prepared:
            raise ValueError('当前不能干预，请先准备或重置仿真')
        obj = next((o for o in snap.get('objects', []) if o['id'] == request.get('object_id')), None)
        if obj is None or obj.get('held'):
            raise ValueError('请选择桌面上未夹持的物品')
        marker = 'environment_revision' if 'environment_revision' in request else 'scene_version'
        expected = self._environment_revision if marker == 'environment_revision' else snap['scene_version']
        if (('environment_revision' in request and 'scene_version' in request)
                or type(request.get(marker)) is not int or request[marker] != expected):
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
                previous_pose = obj.T_base_obj.copy()
                obj.T_base_obj = candidate.T_base_obj.copy()
                if self._physics is not None:
                    try:
                        self._physics.apply_scene_changes(changed_names=[obj.name])
                    except ValueError:
                        obj.T_base_obj = previous_pose
                        raise
                    self._remember_physics()
                if getattr(self, '_released_observation_obstacle', None) is not None and self._released_observation_obstacle.label == obj.label:
                    self._released_observation_obstacle = None
                getattr(self, '_released_observation_hints', {}).pop(obj.label, None)
                self._environment_revision += 1
                observer = getattr(self, '_motion_observer', None)
                if observer is not None:
                    observer.invalidate([obj.label])
                self._emit('scene_intervention_applied', {'request_id': request_id,
                           'message': '人为干预已生效，后续动作将通过相机重新确认'})
            except ValueError as exc:
                self._emit('scene_intervention_rejected', {'request_id': request_id,
                           'message': str(exc)})

    def _execution_guard(self, stage):
        self._apply_interventions()
        # An edit to another object during lift is handled at this stopped,
        # held boundary. The ensuing carry callback must certify a NEW route.
        # Descent, release and an already executing transit never waive a change.
        if (stage == 'held_transit' and self._stage == 'lift'
                and self._environment_revision != self._active_revision
                and getattr(self, '_last_physical_change_revision', -1) <= self._active_revision
                and getattr(self, '_motion_observer', None) is not None):
            observer = self._motion_observer
            observer.capture(force=True)
            target = self._recovery_target
            if target is not None and not self.cancel_event.is_set():
                self._active_revision = self._environment_revision
                self._motion_replan_required = True
        if (stage in ('descend', 'close') and
                self._environment_revision != self._active_revision):
            return 'scene_changed'
        if (stage in ('release', 'open_release', 'held_transit', 'release_clearance', 'retreat') and
                self._environment_revision != self._active_revision):
            return 'scene_changed'
        return None

    def _before_held_transit(self, grasp, place, width, result):
        """Precheck the actual held entry path after an arbitrary inspected pose."""
        budget = _PlacementBudget(self.arm, self.cancel_event, seconds=15.)
        uncertainty = .001
        def reject(reason):
            result.detail['held_transit_planning_reason'] = reason
            return None
        try:
            self._apply_interventions()
            revision = self._environment_revision
            if revision != self._active_revision:
                return reject('scene_changed')
            try:
                fresh = budget.run(self._stationary_recovery_observation)
            except RuntimeError as exc:
                # Missing camera evidence is a reason to hold and reobserve,
                # not to discard an otherwise valid measured held grasp.
                return reject('fresh_observation_unavailable: ' + str(exc))
            self._apply_interventions()
            if revision != self._environment_revision:
                return reject('scene_changed')
            context, measured_width, why = self.runner._held_transfer_feedback(grasp.target)
            if context is None:
                return reject(why)
            if abs(measured_width - width) > uncertainty:
                return reject('held_opening_changed')
            confirmed, why, last_capture = self._calibrate_carried_anchor(
                grasp, result, measured_width, fresh)
            if not confirmed:
                return reject(why)
            if last_capture is not None:
                fresh = last_capture
            route, why = self._plan_carried_entry(grasp, place, width, result,
                                                 fresh, revision, budget)
            if route is None:
                return reject(why)
            return route
        except _PlacementInterrupted:
            return reject('aborted' if self.cancel_event.is_set()
                          else 'held_transit_planning_budget_exceeded')

    def _plan_carried_entry(self, grasp, place, width, result, fresh, revision, budget):
        """Check a carried path from the current, visually refined reference."""
        from .observation_policy import merge_local_detections, measurement_visible
        from .placement_observation import safe_inspection_route
        from ..core.extrinsics import camera_pose_in_base
        fresh, why = self._observe_changed_carried(grasp, place, width, result,
                                                 fresh, revision, budget)
        if why:
            return None, why
        current = self.arm.get_tcp_pose()
        if (current is None or current.value is None or
                self.clock.now()-float(current.stamp) > self.runner.skill.feedback_max_age):
            return None, 'no_fresh_pose_feedback'
        frame = self.camera._last
        pose = camera_pose_in_base(self.runner.ext, self.arm.get_flange_pose().value,
                                   self.runner.T_flange_tcp)
        neighbors = merge_local_detections(self._observed_detections,
            [d for d in fresh if d.center_base is not None and d.label != grasp.target.label],
            lambda d: self._observation_replaced(d, frame, pose), forget=[grasp.target],
            preserve_incomplete=getattr(self, '_motion_observer', None) is not None)
        observer = getattr(self, '_motion_observer', None)
        if observer is not None:
            neighbors += observer.uncertainty_obstacles(exclude=[grasp.target.label])
        self._emit('placement_progress', {'message': '正在按当前夹持姿态检查搬运路径，保持夹持'})
        route, why = budget.run(safe_inspection_route, self.arm, grasp,
            np.asarray(current.value).copy(), place.T_pre, neighbors, width,
            table_z=float(self.cfg.get('perception.table.expected_z', 0.) or 0.),
            nominal=self.runner.nominal.get(grasp.target.label),
            neighbor_nominal=self.runner.nominal, free_tilt_deg=self.runner.free_tilt,
            opening_uncertainty=.001, cancelled=budget.cancelled, allow_joint_route=True,
            allow_table_departure=bool(result.detail.get('held_anchor_replan_required')))
        self._apply_interventions()
        if revision != self._environment_revision:
            return None, 'scene_changed'
        if not route:
            return None, why
        self._observed_detections = copy.deepcopy(
            [d for d in neighbors if not d.label.startswith('unobserved_')])
        result.detail['held_transit_opening_uncertainty'] = .001
        return route, ''

    def _observe_changed_carried(self, grasp, place, width, result, fresh, revision, budget):
        """Locate changed bodies before committing to a destination entry path.

        A whole-table uncertainty volume must stay in every swept-path check.
        Use measured fragments only to aim the camera; only a new complete
        RGB-D detection may remove that volume. This also works during the
        first lift, before a held-recovery context has been saved.
        """
        observer = getattr(self, '_motion_observer', None)
        if observer is None or not observer.invalidated - {grasp.target.label}:
            return fresh, ''
        from .placement_observation import (iter_safe_inspection_poses,
            safe_inspection_route, inspection_history, remember_inspection)
        ext = self.runner.ext
        T_fc = ext.T if ext.parent_frame == 'flange' else self.runner.T_flange_tcp @ ext.T
        table = float(self.cfg.get('perception.table.expected_z', 0.) or 0.)
        self._emit('active_observation', {
            'message': '物品位置已变化，先保持夹持并从局部视角确认，再继续搬运'})
        remember_inspection(self, grasp, self.arm.get_tcp_pose().value)
        missing = observer.invalidated - {grasp.target.label}
        hints = [d for d in self._observed_detections if d.label in missing
                 and d.center_base is not None
                 and d.extra.get('_observed_revision') == revision]
        xy = (max(hints, key=lambda d: d.n_points).center_base[:2]
              if hints else place.T_pre[:2, 3])
        obstacles = [d for d in self._observed_detections if d.label != grasp.target.label]
        obstacles += observer.uncertainty_obstacles(exclude=[grasp.target.label])
        highest = max([table] + [float(d.top_z) for d in obstacles
                                if d.top_z is not None and math.isfinite(float(d.top_z))])
        poses = iter_safe_inspection_poses(self.arm, grasp, T_fc,
            self.runner.T_flange_tcp, xy, highest, table_z=table,
            free_tilt_deg=self.runner.free_tilt,
            nominal=self.runner.nominal.get(grasp.target.label),
            intrinsics=self.camera.intrinsics(), cancelled=budget.cancelled,
            exclude_poses=inspection_history(self, grasp),
            orientation_clearance=True,
            focus_z=table+max(observer.height_priors.get(label, .22) for label in missing)/2)
        views, reason = 0, 'changed_object_not_observed'
        sentinel = object()
        while True:
            pose = budget.run(next, poses, sentinel)
            if pose is sentinel:
                break
            self._apply_interventions()
            if self.cancel_event.is_set():
                raise _PlacementInterrupted()
            if revision != self._environment_revision:
                return fresh, 'scene_changed'
            neighbors = [d for d in self._observed_detections
                         if d.label != grasp.target.label and not d.label.startswith('unobserved_')]
            neighbors += observer.uncertainty_obstacles(exclude=[grasp.target.label])
            current = self.arm.get_tcp_pose().value.copy()
            route, reason = budget.run(safe_inspection_route, self.arm, grasp,
                current, pose, neighbors, width,
                table_z=table, nominal=self.runner.nominal.get(grasp.target.label),
                neighbor_nominal=self.runner.nominal, free_tilt_deg=self.runner.free_tilt,
                cancelled=budget.cancelled, opening_uncertainty=.001,
                allow_joint_route=True, endpoint_clearance=.02, bend_reference=current)
            if not route:
                continue
            for waypoint in route:
                self._apply_interventions()
                if revision != self._environment_revision:
                    return fresh, 'scene_changed'
                measured = self.runner.skill._verify_held(result, 'dropped_before_observation')
                if measured is not None and abs(measured-width) > .001:
                    return fresh, 'held_opening_changed'
                if not self._move_inspection_waypoint(result, waypoint, **(
                        {'joint_route': route} if hasattr(route, 'end_joints') else {})):
                    break
            fresh = self._stationary_recovery_observation(grasp=grasp, width=width)
            if revision != self._environment_revision:
                return fresh, 'scene_changed'
            remember_inspection(self, grasp, self.arm.get_tcp_pose().value)
            views += 1
            if not observer.invalidated - {grasp.target.label}:
                return fresh, ''
            if views >= 4:
                break
        return fresh, 'changed_object_not_observed: ' + str(reason)

    def _calibrate_carried_anchor(self, grasp, result, width, first_detections=None):
        """Confirm seating before using the carried reference for another path."""
        from .held_anchor import (apply_translation_anchor, seating_requires_feedback,
                                  source_anchor)
        if not seating_requires_feedback(grasp, result, width):
            return True, '', None
        try:
            width = float(width)
            if not math.isfinite(width) or width <= 0:
                raise ValueError
        except (TypeError, ValueError):
            return False, 'held_anchor_unconfirmed: held_width_unavailable', None
        reference = grasp.target.extra.get('_held_translation_anchor') or {}
        try:
            if abs(float(width)-float(reference['grip_width'])) <= .003:
                return True, '', None
        except (KeyError, TypeError, ValueError):
            pass
        source = source_anchor(grasp)
        if not source.get('ok'):
            return False, 'source_anchor_unconfirmed: '+str(source.get('error')), None
        pair, fresh = self._measure_held_anchor(grasp, source, first_detections)
        if not pair.get('ok'):
            return False, 'held_anchor_unconfirmed: '+str(pair.get('error')), fresh
        calibrated = apply_translation_anchor(grasp, source, pair, self._environment_revision)
        if not calibrated.get('ok'):
            return False, 'held_anchor_unconfirmed: '+str(calibrated.get('error')), fresh
        grasp.target.extra['_held_translation_anchor']['grip_width'] = float(width)
        result.detail['held_anchor_feedback'] = calibrated
        self._emit('grasp_anchor_refined', dict(message='已用两帧深度图修正夹持位置', **calibrated))
        return True, '', fresh

    def _before_carry(self, grasp, place, result):
        replan = getattr(self, '_motion_replan_required', False)
        self._motion_replan_required = False
        original = grasp.T_grasp.copy()
        revision = self._environment_revision
        measured = self.runner.skill._verify_held(result, 'dropped_before_carry')
        if measured is not None:
            result.grasp_width = measured
        confirmed, reason, fresh = self._calibrate_carried_anchor(grasp, result, result.grasp_width)
        if not confirmed:
            result.detail['held_transit_planning_reason'] = reason
            return False, reason, None
        observer = getattr(self, '_motion_observer', None)
        unknown = observer is not None and observer.invalidated - {grasp.target.label}
        if np.array_equal(original, grasp.T_grasp) and not replan and not unknown:
            return True, '', None
        # An externally moved body needs a checked new view as well as the
        # carry route. Give that combined operation the normal placement
        # allowance; a small route-only slice causes repeated model recovery.
        budget = _PlacementBudget(self.arm, self.cancel_event, seconds=45. if replan or unknown else 15.)
        try:
            if replan:
                # An edit can occupy the provisional landing/hover itself.
                # Searching more paths to that obsolete endpoint cannot help.
                # Reuse the normal RGB-D landing planner before committing to
                # entry; it also certifies a route from the actual held pose.
                updated = self._refine_release(grasp, place, result.grasp_width, result, budget)
                if updated is None:
                    reason = result.detail.get('placement_planning_reason', 'placement_changed')
                    return False, reason, None
                route = getattr(updated, 'inspection_entry_route', None)
                if route is None:
                    route, reason = self._plan_carried_entry(grasp, updated, result.grasp_width,
                        result, self._last_fresh, revision, budget)
                    if not route:
                        result.detail['held_transit_planning_reason'] = reason
                        return False, reason, None
                place.__dict__.update(copy.deepcopy(updated.__dict__))
                result.detail['held_transit_opening_uncertainty'] = .001
                return True, '', route
            if fresh is None:
                fresh = self._stationary_recovery_observation(grasp=grasp, width=result.grasp_width)
            route, reason = self._plan_carried_entry(grasp, place, result.grasp_width,
                                                    result, fresh, revision, budget)
        except _PlacementInterrupted:
            result.detail['held_transit_planning_reason'] = 'held_transit_planning_budget_exceeded'
            return False, 'held_transit_planning_budget_exceeded', None
        if not route:
            result.detail['held_transit_planning_reason'] = reason
        return bool(route), reason, route

    def _measure_held_anchor(self, grasp, source, first_detections=None):
        """Two actual stationary captures, bound to one unchanged grip/scene."""
        from .held_anchor import held_anchor
        from .spherical_anchor import validate_anchor_pair
        revision = self._environment_revision
        captures, fits, stamps, poses = [], [], [], []
        try:
            for index in range(2):
                self._apply_interventions()
                if self.cancel_event.is_set() or revision != self._environment_revision:
                    return dict(ok=False, error='scene_changed'), []
                fresh = (first_detections if index == 0 and first_detections is not None
                         else self._stationary_recovery_observation(grasp=grasp))
                self._apply_interventions()
                if self.cancel_event.is_set() or revision != self._environment_revision:
                    return dict(ok=False, error='scene_changed'), fresh
                frame = self.camera._last
                sample = self.arm.get_tcp_pose()
                age = self.clock.now()-float(sample.stamp) if sample is not None else math.inf
                frame_age = self.clock.now()-float(frame.stamp) if frame is not None else math.inf
                max_age = getattr(self.runner.skill, 'feedback_max_age', .5)
                if not math.isfinite(frame_age) or not -.05 <= frame_age <= max_age:
                    return dict(ok=False, error='held_camera_not_fresh'), fresh
                if (sample is None or not math.isfinite(age) or age < -.05
                        or age > max_age or abs(float(sample.stamp)-float(frame.stamp)) > max_age):
                    return dict(ok=False, error='held_tcp_not_fresh'), fresh
                pose = np.asarray(sample.value, float)
                # Environment segmentation removes the carried body before
                # merging obstacles. Seating feedback still needs that same
                # capture's original surfaces, never a previous frame's cache.
                fitted = held_anchor(getattr(fresh, 'anchor_detections', fresh), grasp, source, pose)
                if not fitted.get('ok'):
                    return fitted, fresh
                fits.append(fitted)
                stamps.append(float(frame.stamp))
                poses.append(pose.copy())
                captures.append(fresh)
            if (np.linalg.norm(poses[1][:3, 3]-poses[0][:3, 3]) > .001
                    or rotation_angle(poses[0][:3, :3].T@poses[1][:3, :3]) > math.radians(1.)):
                return dict(ok=False, error='held_camera_moved_between_captures'), captures[-1]
            pair = validate_anchor_pair(fits[0], fits[1], stamps[0], stamps[1])
            if pair.get('ok'):
                pair['capture_stamps'] = stamps
                pair['environment_revision'] = revision
            return pair, captures[-1]
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return dict(ok=False, error='held_anchor_capture_unavailable'), []

    def _before_open(self, grasp, place, result):
        """Recheck a calibrated translation at the actual release, before opening."""
        from .held_anchor import apply_translation_anchor, seating_requires_feedback, source_anchor
        measured = self.runner.skill._verify_held(result, 'dropped_before_open')
        if measured is not None:
            result.grasp_width = measured
        reference = grasp.target.extra.get('_held_translation_anchor')
        if reference is None:
            if seating_requires_feedback(grasp, result, result.grasp_width):
                return False, 'held_anchor_unconfirmed: 夹内位置已变化，需在上方重新观测后规划'
            return True, ''
        source = source_anchor(grasp)
        pair, _fresh = self._measure_held_anchor(grasp, source)
        if not pair.get('ok'):
            return False, 'held_anchor_unconfirmed: '+str(pair.get('error'))
        delta = np.asarray(pair['center'], float)-np.asarray(reference['center_tcp'], float)
        result.detail['held_anchor_before_open'] = dict(
            translation_only=True, drift_mm=(delta*1000).tolist(),
            capture_stamps=pair['capture_stamps'])
        if np.linalg.norm(delta) > .003:
            apply_translation_anchor(grasp, source, pair, self._environment_revision)
            result.detail['held_anchor_replan_required'] = True
            return False, '夹内位置再次变化超过 3 毫米，已保留夹持，需要按新测量重规划'
        return True, ''

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

    def _move_inspection_waypoint(self, result, waypoint, *, joint_route=None):
        """Use the actual held-camera pose after a stopped simulation motion.

        MuJoCo's driver holds its measured joints on timeout/stall. The desired
        inspection pose is not a release target: a fresh image may already have
        sufficient coverage. Never claim that the waypoint was reached, or
        continue its unexecuted route. All subsequent moves are replanned.
        """
        from ..task.skills import _StepFailed
        reached = True
        try:
            if joint_route is not None:
                joints = self.arm.get_joints()
                if (joints is None or self.clock.now()-float(joints.stamp) > self.runner.skill.feedback_max_age
                        or np.max(np.abs(np.asarray(joints.value)-joint_route.start_joints)) > .002):
                    raise _StepFailed('inspect_place', 'placement_changed', 'checked joint route start changed')
                self.runner.skill._move(result, 'inspect_place', waypoint, False, joints=joint_route.end_joints)
            else:
                self.runner.skill._move(result, 'inspect_place', waypoint, True)
        except _StepFailed as exc:
            if self._physics is None or exc.fault or exc.error not in ('timeout', 'stalled'):
                raise
            self.runner.skill._check_abort('inspect_place')
            if not self.runner.skill._healthy():
                raise
            pose = self.arm.get_tcp_pose()
            if (pose is None or pose.value is None or
                    self.clock.now() - float(pose.stamp) > self.runner.skill.feedback_max_age):
                raise
            reached = False
            self._emit('inspection_motion_incomplete', {
                'message': '观察移动已停在实际位置，将先读取这里的相机画面再重新规划',
                'error': exc.error, 'diagnostics': exc.detail,
                'actual_tcp': np.asarray(pose.value)[:3, 3].tolist()})
        self.runner.skill._verify_held(result, 'dropped_in_transit')
        return reached

    def _refine_release(self, grasp, place, width, result, budget):
        """At the destination hover, measure occupancy before choosing a final slot."""
        target, destination, zone_name, options = self._active_placement
        self._apply_interventions()
        self._on_stage('placement_planning')
        self._emit('inspect_place', {'message': '正在结合实时观测确认目标区域空位'})
        from .observation_policy import merge_local_detections, measurement_visible
        from .placement_observation import (candidate_visibility_filter, iter_safe_inspection_poses,
                                             safe_inspection_route, inspection_history,
                                             remember_inspection)
        from ..core.extrinsics import camera_pose_in_base
        neighbors = list(self._observed_detections)
        checks = result.detail['placement_checks'] = dict(
            release_rejection='', capacity=None, capacity_reason='', inspection_rejection='')
        observer = getattr(self, '_motion_observer', None)
        support_views = observer.support_views() if observer is not None else []
        support_revision = self._environment_revision if support_views else None
        last_measured = []
        inspection_focus = None
        support_current = True
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
        def phase(function, *args, seconds, **kwargs):
            # A difficult first release must leave computation for a new view.
            # This deadline is local: it never consumes or latches robot stop.
            deadline = budget.now() + min(float(seconds), budget.remaining)
            def interrupted():
                return budget.cancelled() or budget.now() >= deadline
            scope = getattr(self.arm, 'planning_scope', None)
            with scope(interrupted) if scope is not None else nullcontext():
                return function(*args, cancelled=interrupted, **kwargs)

        def measure():
            nonlocal neighbors, support_revision, options, support_current, width, last_measured, inspection_focus
            support_current = True
            checks.update(capacity=None, capacity_reason='')
            revision = self._environment_revision
            observer = getattr(self, '_motion_observer', None)
            fresh = self._stationary_recovery_observation(grasp=grasp, width=width)
            if revision != self._environment_revision:
                support_views.clear()
                return None, 'scene_changed', revision
            measured = self.runner.skill._verify_held(result, 'dropped_before_release')
            if measured is not None:
                width = result.grasp_width = measured
            confirmed, why, last_capture = self._calibrate_carried_anchor(grasp, result, width, fresh)
            if not confirmed:
                return None, why, revision
            if last_capture is not None:
                fresh = last_capture
            last_measured = fresh
            remember_inspection(self, grasp, self.arm.get_tcp_pose().value)
            frame = self.camera._last
            flange = self.arm.get_flange_pose()
            pose = camera_pose_in_base(self.runner.ext, flange.value, self.runner.T_flange_tcp)
            if support_revision != revision or revision != self._environment_revision:
                support_views.clear()
                support_revision = revision
            # Only this release cycle's newly captured, unchanged-scene RGB-D
            # evidence can complement the current wrist view's self-occlusion.
            if revision == self._environment_revision:
                if observer is not None:
                    # Wrist motion can have supplied clearer views than the
                    # stopped endpoint. Use the owner's current-revision rays,
                    # including those captured during this inspection route.
                    stamp = getattr(frame, 'stamp', None)
                    support_views[:] = [view for view in observer.support_views()
                        if stamp is None or getattr(view[0], 'stamp', None) != stamp]
                support_views.append((copy.deepcopy(frame), pose.copy()))
                del support_views[:-6]
            fresh = [d for d in fresh if d.center_base is not None and d.label != target.label]
            # A complete current singleton measurement can supersede an old
            # cropped centroid beyond normal association distance. Ambiguous
            # fresh instances and incomplete/invalid geometry remain obstacles.
            editor = getattr(self, 'editor', None)
            singleton_labels = ({d.label for d in fresh
                                 if d.valid and d.extra.get('catalog_complete')}
                                if editor is not None and editor.metadata().get('one_per_class')
                                else ())
            neighbors = merge_local_detections(
                         [d for d in self._observed_detections if not d.label.startswith('unobserved_')], fresh,
                         lambda d: self._observation_replaced(d, frame, pose), forget=[target],
                         singleton_labels=singleton_labels,
                         preserve_incomplete=getattr(self, '_motion_observer', None) is not None)
            if observer is not None:
                observer.remember(fresh, revision)
                neighbors += observer.uncertainty_obstacles(exclude=[target.label])
            self._recovery_neighbors = copy.deepcopy(neighbors)
            if observer is not None and observer.invalidated - {target.label}:
                # No release search can pass the unresolved body's exclusion
                # volume. Obtain its missing pose instead of burning a complete
                # grid/IK budget and asking the model to repeat that search.
                support_current = False
                return None, 'changed_object_not_observed', revision
            if destination is not None and destination['type'] == 'on_object':
                from .stacking import planning_options, annotate
                support_current = False
                try:
                    # A carried object commonly hides its own landing base.
                    # The monitor retains complete RGB-D geometry until that
                    # body changes; a cropped endpoint view must not force an
                    # otherwise unchanged base to be scanned all over again.
                    support_measurements = (observer.current(destination['label'])
                        if observer is not None else fresh)
                    options, support = planning_options(self, destination, support_measurements)
                except ValueError as exc:
                    return None, str(exc), revision
                support_current = True
                annotate(target, support)
                annotate(grasp.target, support)
                self._active_placement = (target, destination, zone_name, options)
            self._refresh_placement_occupancy(target, neighbors, destination)
            landing_support = (target.extra or {}).get('_stack_support')
            if landing_support is not None:
                landing_z = float(landing_support['top_z'])
            else:
                landing_zone = options.get('placement_zone')
                if landing_zone is None:
                    landing_zone = self.runner.zones[zone_name]
                landing_z = float(landing_zone.center[2])
            visible = candidate_visibility_filter(target, grasp, frame, pose,
                        nominal=self.runner.nominal.get(target.label),
                        max_open_width=self.runner.placer.open_width,
                        observations=support_views[:-1], landing_z=landing_z)
            # A fresh measurement owns these exclusions. A timed slice must
            # not restart the same expensive, fully rejected edge slots; nor
            # may those rejections survive a new view or changed scene/seed.
            search_state = {}
            updated, why = phase(self.runner.replan_release,
                              target, grasp, place, width, neighbors, seconds=8.,
                              observation_filter=visible, search_state=search_state,
                              progress=progress, **options)
            if (updated is None and why == 'placement_planning_budget_exceeded'
                    and (search_state.get('avoided') or search_state.get('avoided_poses'))
                    and not search_state.get('unseen_xy')
                    and not budget.cancelled()):
                self._apply_interventions()
                if revision != self._environment_revision:
                    support_views.clear()
                    return None, 'scene_changed', revision
                # One continuation rewards completed work while retaining
                # computation for a checked new view if this one is deficient.
                updated, why = phase(self.runner.replan_release,
                    target, grasp, place, width, neighbors, seconds=8.,
                    observation_filter=visible, search_state=search_state,
                    progress=progress, **options)
            checks['release_rejection'] = str(why) if updated is None else ''
            unseen = search_state.get('unseen_xy', ())
            inspection_focus = np.asarray(unseen[0], float) if unseen else None
            return updated, why, revision
        updated, why, revision = budget.run(measure)
        if str(why).startswith('held_anchor_unconfirmed'):
            result.detail['placement_planning_reason'] = why
            self._emit('placement_changed', dict(message='夹内位置尚未得到可靠视觉确认，保持夹持', reason=why))
            return None
        if why == 'scene_changed':
            result.detail['placement_planning_reason'] = why
            return None
        if updated is None:
            # Geometry alone can prove a destination full. IK exhaustion cannot:
            # a different camera pose may reveal another feasible candidate.
            # This preflight returns only capacity, never an executable plan.
            if not support_current:
                capacity, capacity_why = None, why
            else:
                capacity, capacity_why = budget.run(phase, self.runner.has_release_capacity,
                    target, grasp, place, width, neighbors, seconds=5.,
                    progress=progress, **options)
            checks.update(capacity=capacity, capacity_reason=str(capacity_why))
            if capacity is False:
                why = '当前测量下无可确认的放置方案：' + str(capacity_why)
            else:
                ext = self.runner.ext
                T_fc = ext.T if ext.parent_frame == 'flange' else self.runner.T_flange_tcp @ ext.T
                table = float(self.cfg.get('perception.table.expected_z', 0) or 0)
                # Inspect the support under the already planned held release.
                # Region-centred views can repeatedly miss an available edge
                # slot in a cluttered region. This uses the measured grasp's
                # rigid transform, never the simulator object's current pose.
                source = np.asarray(target.center_base, float).copy()
                source[2] = float(target.top_z) - float(target.height) / 2.
                landing = (place.T_release[:3, :3] @ grasp.T_grasp[:3, :3].T
                           @ (source - grasp.T_grasp[:3, 3]) + place.T_release[:3, 3])
                xy = landing[:2]
                if inspection_focus is not None:
                    # Look at an actual clear candidate missing camera rays,
                    # rather than repeatedly looking at the occupied old slot.
                    # This supplies no release authority; the new image must
                    # pass the complete release/path/visibility checks again.
                    xy = inspection_focus
                elif (capacity is True and why != 'placement_planning_budget_exceeded'
                        and destination is not None and destination.get('type') != 'on_object'):
                    # Choose where to LOOK using full reachable geometry when
                    # visibility rejects the old slot. This draft grants no
                    # release authority: the next capture still replans with
                    # the complete support-visibility and collision checks.
                    previous_info = copy.deepcopy(self.runner.placer.last_info)
                    try:
                        draft, _draft_why = budget.run(phase, self.runner.replan_release,
                            target, grasp, place, width, neighbors, seconds=4.,
                            max_attempts=8, search_state={}, progress=progress, **options)
                        hint = np.asarray(self.runner.placer.last_info.get('xy', []), float)
                        if draft is not None and hint.shape == (2,) and np.all(np.isfinite(hint)):
                            xy = hint
                            checks['view_target_xy'] = hint.tolist()
                    finally:
                        self.runner.placer.last_info = previous_info
                # The pose generator bounds rotations above the table. Actual
                # obstacles are checked against each full swept route below,
                # so a distant tall item cannot mandate an impossible lift.
                focus_options = {}
                observer = getattr(self, '_motion_observer', None)
                missing = set() if observer is None else observer.invalidated - {target.label}
                if missing:
                    # A changed tall object's position is needed before a slot
                    # can be certified. A table-centred gaze can crop its neck
                    # in every view. Partial detections only aim this camera;
                    # they never authorize grasping or shrinking the obstacle.
                    hints = [d for d in self._observed_detections if d.label in missing
                             and d.center_base is not None
                             and d.extra.get('_observed_revision') == revision]
                    if hints:
                        hint = max(hints, key=lambda d: d.n_points)
                        xy = np.asarray(hint.center_base[:2], float)
                    height = max(observer.height_priors.get(label, .22) for label in missing)
                    focus_options['focus_z'] = table + height/2
                    focus_options['orientation_clearance'] = True
                highest = max([table] + [float(d.top_z) for d in neighbors
                    if d.top_z is not None and math.isfinite(float(d.top_z))]) if missing else table
                poses = iter_safe_inspection_poses(self.arm, grasp, T_fc,
                                 self.runner.T_flange_tcp, xy, highest,
                                 table_z=table, free_tilt_deg=self.runner.free_tilt,
                                 nominal=self.runner.nominal.get(target.label),
                                 intrinsics=self.camera.intrinsics(),
                                 cancelled=budget.cancelled, progress=progress,
                                 exclude_poses=inspection_history(self, grasp), **focus_options)
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
                        cancelled=budget.cancelled, progress=progress, allow_joint_route=True,
                        endpoint_clearance=.02,
                        bend_reference=current,
                        allow_table_departure=bool(result.detail.get('held_anchor_replan_required')))
                    if not route:
                        checks['inspection_rejection'] = str(route_why)
                        why = route_why
                        continue
                    observed_views += 1
                    route_revision = revision
                    for waypoint in route:
                        if self.cancel_event.is_set():
                            raise _PlacementInterrupted()
                        self._apply_interventions()
                        if self._environment_revision != route_revision:
                            break
                        joint_route = route if hasattr(route, 'end_joints') else None
                        if not self._move_inspection_waypoint(result, waypoint, **(
                                {'joint_route': joint_route} if joint_route is not None else {})):
                            break
                    self._on_stage('placement_planning')
                    # A newly reached checked view starts a new measurement
                    # problem. Reserve enough computation to actually use it;
                    # the bounded four-view loop and visited-pose history still
                    # prevent an unbounded search/revisit cycle.
                    if self._environment_revision == route_revision:
                        budget.remaining = max(budget.remaining, 25.)
                    updated, why, revision = budget.run(measure)
                    if updated is not None or why == 'scene_changed' or observed_views >= 4:
                        break
        self._active_revision = revision
        if updated is None:
            result.detail['placement_planning_reason'] = str(why)
            self._emit('placement_changed', {'message': '当前测量下无法确认安全放置方案，将保持夹持', 'reason': why})
            return None
        current = self.arm.get_tcp_pose().value
        from ..core.transforms import rotation_angle
        if (np.linalg.norm(current[:3, 3] - updated.T_pre[:3, 3]) > .002
                or rotation_angle(current[:3, :3].T @ updated.T_pre[:3, :3]) > math.radians(1.)):
            # The current pick has no saved recovery context yet. Certify its
            # return using this release cycle's actual RGB-D and grasp, rather
            # than invoking a recovery-only callback or an unchecked joint move.
            route, why = self._plan_carried_entry(
                grasp, updated, width, result, last_measured, revision, budget)
            if route is None:
                result.detail['placement_planning_reason'] = why
                return None
            updated.inspection_entry_route = route
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
            if self._physics is not None:
                self._physics.apply_scene_changes()
                self._remember_physics()
            self._environment_revision += 1
            observer = getattr(self, '_motion_observer', None)
            if observer is not None:
                observer.evidence.clear()
                observer.invalidated.clear()
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
            if self._physics is not None:
                self._remember_physics()
                self._start_idle_physics()
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
            grasp_geometry_ready = False
            self._last_fresh = []
            order = self._view_order(target) if local else list(range(self._base_view_count))
            source_views = []
            source_focus = bool(local and target.get('_observe_source'))
            if source_focus:
                source_views = self._source_observation_views(target)
                # Repeating all fixed cameras cannot repair a crop shared by
                # those same cameras. Spend this action on bounded new views.
                order = source_views if source_views else order[:2]
            if local and purpose == 'verify_place':
                order = order[:2]
            self._scan_stats['local_scans' if local else 'full_scans'] += 1
            if not local:
                with self._cache_lock:
                    self._observation_views = {}
                    self._observation_jpegs = {}
            views, warnings, coverage = {}, [], []
            captured_views, attempted_views = [], []
            stack_fallbacks = 0
            verified_merge = None
            revision = self._environment_revision
            for step, view in enumerate(order, 1):
                guarded_view = source_focus or (getattr(self, '_physics', None) is not None
                                               and (local or bool(self._observed_detections)))
                if guarded_view and not self._prepare_observation_motion(
                        self.runner.views[view], self._observed_detections):
                    # Previous observations may have changed the start joints
                    # or the environment. Recheck the entire actual sweep.
                    if step == len(order) and stack_fallbacks < 3:
                        goals = (target or {}).get('verify_goals', ())
                        hint = getattr(self, '_released_observation_obstacle', None)
                        extra_view = self._verification_view(
                            goals, [] if hint is None else [hint], revision, attempted_views)
                        if extra_view is not None and extra_view not in order:
                            order.append(extra_view)
                            stack_fallbacks += 1
                    continue
                self._emit_scan_view(view, step=step, total=len(order), local=local)
                status, next_result, frame = self.runner._observe(view)
                if status != 'ok' or next_result is None:
                    why = '%s: %s' % (status, self.runner._last_obs_failure)
                    self._invalidate_scene(why)
                    self._status = 'error'
                    self._publish_snapshot()
                    raise RuntimeError('simulation observation failed: ' + why)
                result = next_result
                observer = getattr(self, '_motion_observer', None)
                if observer is not None:
                    observer.reconcile(result.detections, revision)
                self._scan_stats['frames'] += 1
                self._cache_observation(view, result, frame)
                current_detections = tag_view(result.detections, view)
                # The transient slot is reused for different camera poses. Keep
                # each capture in this scan so later views can complement it.
                capture_key = view if view not in views else max(views) + 1
                views[capture_key] = current_detections
                captured_views.append(view)
                if hasattr(self.runner, 'views'):
                    attempted_views.append(np.asarray(self.runner.views[view], float).copy())
                warnings.extend(result.warnings)
                flange = self.arm.get_flange_pose()
                pose = camera_pose_in_base(self.runner.ext, flange.value, self.runner.T_flange_tcp)
                coverage.append((frame, pose))
                for measured in current_detections:
                    measured.extra['_capture_index'] = step
                    measured.extra['_capture_stamp'] = float(frame.stamp)
                    measured.extra['_capture_flange'] = np.asarray(flange.value, float).copy()
                observer = getattr(self, '_motion_observer', None)
                if observer is not None:
                    observer.remember_frame(frame, pose, revision)
                if local and target.get('required_support') and (
                        revision != self._environment_revision or getattr(self, '_interventions', ())
                        or self.cancel_event.is_set()):
                    break
                if local:
                    goals = target.get('verify_goals')
                    if goals:
                        verified_merge = merge_observations(views, dist=self.runner.view_merge,
                            partial=lambda d: not d.valid or self.runner._partial(d))
                        if all(self._camera_goal_confirmed(goal, verified_merge[0], revision) for goal in goals):
                            break
                        if (purpose == 'verify_place' and step == len(order)
                                and stack_fallbacks < 3):
                            extra_view = self._verification_view(
                                goals, verified_merge[0], revision, attempted_views)
                            if extra_view is not None:
                                # Bounded alternatives around this same stack;
                                # success stops immediately, never a table scan.
                                order.append(extra_view)
                                stack_fallbacks += 1
                        continue
                    label = target.get('label')
                    measured = unique_target(current_detections, label,
                        partial=lambda d: self._target_partial(d, target)) if label else None
                    # Current completeness includes a separate, cloud-proven
                    # lying-pear pose. A fall must not be compared to the old
                    # upright height, which would prevent every fresh rebind.
                    reliable = measured is not None
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
                            peers += [d for d in current_detections if d.label != label]
                            _candidate, planned = _grasp(self, measured, peers)
                            if planned.feasible:
                                grasp_target = measured
                                grasp_geometry_ready = geometry_ready
                                if geometry_ready:
                                    from .held_anchor import spherical_model, source_anchor
                                    if spherical_model(measured) is not None:
                                        # A curved grip may seat during lift.
                                        # Obtain its body reference while it
                                        # is still supported, not after closing
                                        # has made that source view unavailable.
                                        body = copy.deepcopy(measured)
                                        cloud = self._placement_cloud(measured, views, current_detections)
                                        previous_cloud = self._stable_cloud(measured, revision)
                                        if previous_cloud is not None:
                                            cloud = previous_cloud if cloud is None else np.vstack([previous_cloud, cloud])
                                        if cloud is not None:
                                            body.extra['_placement_points_base'] = cloud
                                        reference = copy.copy(planned)
                                        reference.target = body
                                        grasp_geometry_ready = bool(source_anchor(reference).get('ok'))
                        support_ready = True
                        if target.get('required_support'):
                            joint, _diagnostics = merge_observations(views, dist=self.runner.view_merge,
                                partial=lambda d: not d.valid or self.runner._partial(d))
                            support_ready = self._support_observed(target, joint, revision)
                        if grasp_target is not None and grasp_geometry_ready and support_ready:
                            break
                        if (grasp_target is not None and grasp_geometry_ready
                                and not support_ready):
                            # A complete upper-body grasp stays measured while
                            # another fresh view supplies its missing support.
                            # Look for that remaining entity before unrelated
                            # cards; this only ranks unvisited checked views.
                            support = target['required_support']
                            if (support.get('center_base') is not None
                                    or support.get('xy') is not None):
                                ranked = self._view_order(support)
                                if source_focus:
                                    # Bounded source views can omit the base.
                                    # After proving the upper grasp, allow new
                                    # local support views without rescanning it.
                                    order.extend(candidate for candidate in ranked[:4]
                                                 if candidate not in order)
                                priority = {candidate: rank for rank, candidate in enumerate(ranked)}
                                order[step:] = sorted(order[step:],
                                    key=lambda candidate: priority.get(candidate, len(priority)))
                        continue
                    if (reliable and (geometry_ready or purpose == 'verify_place')) or (not label and len(coverage) >= 1):
                        break
            if not captured_views:
                self._observation_revision = -1
                self._status = 'stopped' if self.cancel_event.is_set() else 'ready'
                self._publish_snapshot()
                if self.cancel_event.is_set():
                    raise RuntimeError('source observation stopped before capture')
                raise _SourceObservationUnavailable(
                    '局部观察视角均未通过当前路径检查，未取得新的相机证据')
            detections, diagnostics = verified_merge or merge_observations(
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
                from .held_anchor import spherical_model, source_anchor
                final_target = next(d for d in fresh if d.label == target.get('label'))
                _candidate, planned = _grasp(self, final_target, fresh)
                if not planned.feasible:
                    final_target.valid = False
                    final_target.reject_reason = 'stable_grasp_not_observed: ' + str(planned.reason)
                elif spherical_model(final_target) is not None:
                    anchor = source_anchor(planned)
                    if not anchor.get('ok'):
                        final_target.valid = False
                        final_target.reject_reason = 'source_anchor_unconfirmed: '+str(anchor.get('error'))
            if local:
                detections = merge_local_detections(
                    self._observed_detections, fresh,
                    lambda d: any(self._observation_replaced(d, f, p) for f, p in coverage),
                    forget=forget,
                    singleton_labels={d.label for d in fresh
                                      if d.extra.get('catalog_complete')},
                    preserve_incomplete=getattr(self, '_motion_observer', None) is not None)
            self._last_fresh = fresh
            self._observation_revision = revision
            self._last_observation_purpose = purpose
            self._observed_detections = copy.deepcopy(detections)
            observer = getattr(self, '_motion_observer', None)
            if observer is not None:
                observer.remember(fresh, revision)
            self._emit('observation_complete', {'purpose': purpose, 'views': captured_views,
                       'message': '局部观测完成' if local else '首次全场观测完成',
                       'stats': copy.deepcopy(self._scan_stats)})
            return self._publish_scene(detections, warnings, diagnostics, result.frame_stamp)

    def _support_observed(self, reference, detections, revision, *, check_age=False):
        """A stack's lower body must be complete in the same observation batch."""
        from .observation_policy import unique_target
        from .stacking import measured_support
        support = reference.get('required_support')
        if not support:
            return True
        if (revision != self._environment_revision or getattr(self, '_interventions', ())
                or self.cancel_event.is_set()):
            return False
        det = unique_target(detections, support.get('label'),
                            partial=lambda d: self._target_partial(d, support))
        if det is None:
            return False
        try:
            measured_support(detections, support['label'])
            if check_age:
                stamp = float(det.extra['_capture_stamp'])
                if not 0 <= self.clock.now() - stamp <= self.runner.max_frame_age:
                    return False
        except (ValueError, TypeError, KeyError):
            return False
        return True

    def _reusable_target(self, reference):
        """Reuse monitored base-frame evidence, or the legacy same-pose capture."""
        observer = getattr(self, '_motion_observer', None)
        if observer is not None:
            if observer.source(reference) is None:
                return False
            self._last_fresh = observer.current()
            self._observation_revision = self._environment_revision
            self._last_observation_purpose = 'motion_observation'
            observer.stats['source_reuses'] += 1
            return True
        from .observation_policy import unique_target
        # A release-verification view proves a completed placement, not a new
        # grasp observation. Keep that boundary even if its frame is recent.
        if (getattr(self, '_last_observation_purpose', '') not in
                ('skill_observe', 'skill_pick', 'recover_target', 'recover_queued_target', 'confirm_pick') or
                self._observation_revision != self._environment_revision or
                getattr(self, '_interventions', ()) or self.cancel_event.is_set()):
            return False
        label = reference.get('label')
        det = unique_target(self._last_fresh, label,
                            partial=lambda d: self._target_partial(d, reference))
        if det is None or (label in ('banana', 'pear') and det.extra.get('_cloud_views', 0) < 2):
            return False
        if reference.get('required_support') and not self._support_observed(
                reference, self._last_fresh, self._observation_revision, check_age=True):
            return False
        stamp, pose = det.extra.get('_capture_stamp'), det.extra.get('_capture_flange')
        flange = self.arm.get_flange_pose()
        if (stamp is None or pose is None or flange is None or
                not 0 <= self.clock.now() - float(stamp) <= self.runner.max_frame_age or
                not np.allclose(flange.value, pose, atol=1e-5, rtol=0)):
            return False
        from .obstacle_recovery import _grasp
        peers = [d for d in self._observed_detections if d.label != label]
        _candidate, plan = _grasp(self, det, peers)
        return bool(plan.feasible)

    def _camera_goal_confirmed(self, goal, fresh, revision):
        """Evaluate the whole visual postcondition while a local scan is in progress."""
        from .batch import whole_footprint_at_goal
        from .observation_policy import unique_target
        if revision != self._environment_revision or getattr(self, '_interventions', ()):
            return False
        if (goal.get('destination') or {}).get('type') == 'on_object':
            from .stacking import camera_confirmed
            evidence = SimpleNamespace(_last_fresh=fresh, _observation_revision=revision,
                _environment_revision=self._environment_revision,
                _interventions=getattr(self, '_interventions', ()), world=self.world)
            return camera_confirmed(evidence, goal['label'], goal['destination'])
        det = unique_target(fresh, goal['label'])
        return bool(det is not None and whole_footprint_at_goal(det.summary(), self._scene, goal))

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
        if det.label in ('banana', 'pear'):
            from .catalog import restoration_yaw_period
            period = restoration_yaw_period(det.label, det.extra.get('resting_pose'))
            if period is not None and abs(math.remainder(
                    float(det.yaw or 0) - float(old.yaw or 0), period)) > math.radians(4):
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

    def _observation_replaced(self, det, frame, pose):
        from .observation_policy import measurement_absent, measurement_visible
        predicate = (measurement_absent if getattr(self, '_motion_observer', None) is not None
                     else measurement_visible)
        return predicate(det, frame, pose)

    def _view_order(self, target):
        from .observation_policy import rank_views
        ext = self.runner.ext
        T_fc = ext.T if ext.parent_frame == 'flange' else self.runner.T_flange_tcp @ ext.T
        cameras = [self.arm.kin.fk(q) @ T_fc for q in self.runner.views]
        point = target.get('center_base') or target.get('xy') or [.26, 0]
        # A fully covering current named view wins over an older recorded view.
        # rank_views still puts complete coverage ahead of this preference.
        preferred = (self.runner._at_view if self.runner._at_observe else target.get('view'))
        support = target.get('required_support') or {}
        support_xy = support.get('center_base') or support.get('xy')
        required = []
        if support_xy is not None:
            required.append((support_xy[:2], max(.04, float(support.get('length') or .08) / 2,
                                                float(support.get('width') or .08) / 2),
                             float(support.get('top_z') or .04)))
        return rank_views(cameras, self.camera.intrinsics(), point[:2],
                          preferred=preferred,
                          half_extent=max(.04, float(target.get('length') or .08) / 2),
                          target_z=float(target.get('top_z') or .04), required_targets=required)

    def _publish_scene(self, detections, warnings, diagnostics, stamp, *, preserve_status=False):
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
                observer = getattr(self, '_motion_observer', None)
                changed = observer is not None and measured.label in observer.invalidated
                if changed:
                    measured.valid = False
                    measured.reject_reason = 'changed_not_reobserved'
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
                                    view=measured.extra.get('view', 0),
                                    measurement_state='changed' if changed else
                                        'updated' if measured.extra.get('_capture_stamp') == float(stamp) else 'retained',
                                    measured_at=measured.extra.get('_capture_stamp')))
                for key in ('support_z', 'support_label', 'support_polygon', 'support_observed'):
                    if key in measured.extra:
                        objects[-1][key] = copy.deepcopy(measured.extra[key])
            self._detections = MappingProxyType(mapping)
            self._observed_detections = list(mapping.values())
            self._scene = {'scene_version': self._scene_version, 'objects': objects,
                           'zones': self._zones(),
                           'placement': copy.deepcopy(self._placement),
                           'class_to_zone': dict(self.runner.class_to_zone),
                           'task': 'Sandbox', 'mode': self.backend,
                           'observation_diagnostics': diagnostics,
                           'observed_at': float(stamp),
                           'incremental': bool(preserve_status),
                           'observation_stats': copy.deepcopy(self._scan_stats),
                           'warnings': warnings}
            if not preserve_status:
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
        from .placement_observation import forget_inspection
        forget_inspection(self)
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
                stack_reference = None
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
                        destination = validate_destination(destination, self._scene,
                            require_observed=not (isinstance(destination, dict)
                                                  and destination.get('type') == 'on_object'))
                        if destination['type'] == 'on_object':
                            from .stacking import require_flat_support
                            require_flat_support(destination['label'])
                    except ValueError as exc:
                        return self._refusal('invalid_destination', str(exc), object_id, None)
                    if destination['type'] == 'on_object':
                        supports = [d for d in self._observed_detections
                                    if d.label == destination['label'] and d.center_base is not None]
                        support_hint = (supports[0].summary() if len(supports) == 1
                                        else {'label': destination['label']})
                        stack_reference = det.summary() | {
                            'view': det.extra.get('view'), 'require_grasp': True,
                            'required_support': support_hint}
                        placement_options = {}
                        zone_name = 'on_object_' + destination['label']
                    else:
                        placement_options = self._placement_options(destination)
                        zone_name = placement_options['placement_zone'].name
                else:
                    zone_name = zone_name or self.runner.class_to_zone[det.label]
                    if zone_name not in self.runner.zones:
                        return self._refusal('invalid_zone', 'target zone does not exist', object_id, zone_name)
                    zone = self.runner.zones[zone_name]
                    placement_options = dict(placement_zone=copy.deepcopy(zone),
                                             placement_polygon=zone.corners_xy(), exact_xy=False)
                # Initial intent is stable; coordinates must be confirmed in a new image.
                original = copy.deepcopy(det)
                preconfirmed = self._reusable_target(stack_reference or
                    (original.summary() | {'view': original.extra.get('view')})) or (
                                getattr(self, '_last_observation_purpose', '') in
                                ('recover_target', 'recover_queued_target') and
                                self._observation_revision == self._environment_revision and
                                not getattr(self, '_interventions', ()) and
                                any(d.valid and d.label == original.label and d.center_base is not None and
                                    np.linalg.norm(d.center_base[:2] - original.center_base[:2]) < .003
                                    for d in self._last_fresh))
                if stack_reference is not None:
                    preconfirmed = preconfirmed and self._support_observed(
                        stack_reference, self._last_fresh, self._observation_revision)
                if not preconfirmed:
                    self.observe(target=stack_reference or
                                 (original.summary() | {'view': original.extra.get('view')}),
                                 purpose='confirm_pick', forget=[original])
                self._last_observation_purpose = 'consumed_for_pick'
                if self.cancel_event.is_set():
                    return self._refusal('aborted', 'execution cancelled', object_id, zone_name)
                if (self._observation_revision != self._environment_revision
                        or getattr(self, '_interventions', ())):
                    return self._refusal('stale_scene', '现场已变化，联合观测已失效', object_id, zone_name)
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
                if destination is not None and destination['type'] == 'on_object':
                    from .stacking import planning_options, annotate
                    if not self._support_observed(stack_reference, self._last_fresh,
                                                  self._observation_revision):
                        return self._refusal('destination_not_observed',
                            '新相机观测未完整确认指定底座，未执行抓取', object_id, zone_name)
                    try:
                        placement_options, support = planning_options(self, destination, self._last_fresh)
                    except ValueError as exc:
                        return self._refusal('destination_not_observed', str(exc), object_id, zone_name)
                    annotate(target, support)
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
        if destination['type'] == 'on_object':
            from .stacking import planning_options
            return planning_options(self, destination)[0]
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

    def _source_observation_views(self, reference):
        """Aim bounded, checked local views using a partial camera hint only.

        The measured silhouette gives an aiming envelope, not grasp evidence.
        Side baselines expose support surfaces hidden by a tall upper object;
        new images must still pass every existing completeness/support gate.
        """
        from ..perception.geometry import check_intrinsics, project_points
        from ..sim.scenarios import look_at

        try:
            center = np.asarray(reference.get('center_base') or reference.get('xy'), float)
            top = float(reference['top_z'])
            measured = np.asarray([reference['length'], reference['width']], float)
            table = float(self.cfg.get('perception.table.expected_z', 0.) or 0.)
            if (center.shape not in ((2,), (3,))
                    or not np.all(np.isfinite(np.r_[center, measured, top, table]))
                    or np.any(measured <= 0.) or not table < top < table + .45):
                return []
            xy = center[:2]
            intrinsics = self.camera.intrinsics()
            check_intrinsics(intrinsics)
            ext = self.runner.ext
            T_fc = ext.T if ext.parent_frame == 'flange' else self.runner.T_flange_tcp @ ext.T
            seeds = [self.arm.get_joints().value, self.runner.views[len(self.runner.views) // 2]]
        except (KeyError, ValueError, TypeError, AttributeError):
            return []
        half_extent = max(.04, float(np.max(measured)) / 2.)
        radial = xy / max(float(np.linalg.norm(xy)), 1e-6)
        lateral = np.array([-radial[1], radial[0]])
        distance, baseline = 4 * half_extent, 2 * half_extent
        eye_z = max(table + .30, top + .16)
        # Include the support height and the uncertain silhouette's top inside
        # the calibrated image, leaving room for the missing cropped part.
        outline = np.array([[xy[0] + dx, xy[1] + dy, z]
                            for dx in (-half_extent, half_extent)
                            for dy in (-half_extent, half_extent)
                            for z in (table, top + .015)])
        accepted = []
        revision = self._environment_revision
        for fraction in (.75, .5):
            gaze = np.r_[xy, table + fraction * (top - table)]
            for side in (1., -1.):
                if (self.cancel_event.is_set() or self._interventions
                        or revision != self._environment_revision):
                    return []
                eye = np.r_[xy - distance * radial + side * baseline * lateral, eye_z]
                pose = look_at(eye, gaze)
                pixels = project_points(outline, pose, intrinsics)
                if (not np.all(np.isfinite(pixels)) or np.any(pixels < 12)
                        or np.any(pixels[:, 0] >= intrinsics.width - 12)
                        or np.any(pixels[:, 1] >= intrinsics.height - 12)):
                    continue
                for seed in seeds:
                    q, _ep, _er = self.arm.kin.ik(pose @ inv_T(T_fc), q_seed=seed)
                    if q is None:
                        continue
                    # IK tolerance must not turn a predicted in-frame envelope
                    # into an actually cropped source at the solved joints.
                    actual = project_points(outline, self.arm.kin.fk(q) @ T_fc, intrinsics)
                    covered = (np.all(np.isfinite(actual)) and np.all(actual >= 12)
                               and np.all(actual[:, 0] < intrinsics.width - 12)
                               and np.all(actual[:, 1] < intrinsics.height - 12))
                    if (covered and self._safe_observation_joints(q, self._observed_detections)
                            and not any(np.max(np.abs(np.asarray(q) - old)) < math.radians(2)
                                        for old in accepted)):
                        accepted.append(np.asarray(q, float))
                        break
        if not accepted:
            return []
        # Reserve one destination slot plus at most four source-view slots.
        # Full scans continue to use only the original fixed view count.
        first = self._base_view_count + 1
        while len(self.runner.views) < first:
            self.runner.views.append(list(self.runner.views[0]))
            self._view_labels.append('桌面目标点视角')
        indices = []
        for offset, q in enumerate(accepted):
            index = first + offset
            if index == len(self.runner.views):
                self.runner.views.append(q.tolist())
                self._view_labels.append('目标局部补充视角')
            else:
                self.runner.views[index] = q.tolist()
                self._view_labels[index] = '目标局部补充视角'
                if self.runner._at_view == index:
                    self.runner._at_observe = False
            indices.append(index)
        if indices:
            self.runner.multi_view = True
        return indices

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
                    self._view_labels[view] = '桌面目标点视角'
                    if self.runner._at_view == view:
                        self.runner._at_observe = False
                self.runner.multi_view = True
                self._destination_views = {key: view}
                return view

    def _prepare_observation_motion(self,q,detections):
        """An empty camera can lift through a checked gateway before turning."""
        if self._safe_observation_joints(q,detections):
            return True
        if not hasattr(self.arm,'_ik_tcp') or not hasattr(self.arm,'get_tcp_pose'):
            return False
        revision = self._environment_revision
        current = self.arm.get_tcp_pose().value
        joints = self.arm.get_joints().value
        now = getattr(self.arm,'planning_time',time.monotonic)
        deadline = now()+8.
        def via(gateway):
            if not self._safe_observation_joints(gateway,detections,escaping=True):
                return None
            if not self._safe_observation_joints(q,detections,start_joints=gateway):
                return None
            if (revision!=self._environment_revision or self.cancel_event.is_set()
                    or self._interventions):
                return False
            self._emit('observation_route',{'message':'先沿受检路径抬高空夹爪，再移动到局部观察位置'})
            moved = self.arm.move_joints(gateway)
            self.runner._at_observe = False
            return bool(moved.ok and revision==self._environment_revision
                        and not self.cancel_event.is_set() and not self._interventions)
        # Taught FK poses can turn the empty wrist while lifting. Preserving
        # an exact optical attitude can have no IK even with a clear safe path.
        if hasattr(self.arm,'kin'):
            for gateway in getattr(self.runner,'views',()):
                if self.cancel_event.is_set() or self._interventions or now()>=deadline:
                    return False
                pose = self.arm.kin.fk(gateway) @ self.runner.T_flange_tcp
                if float(pose[2,3]) < float(current[2,3])+.025:
                    continue
                reached = via(gateway)
                if reached is not None:
                    return reached
        radial = current[:2,3]/max(np.linalg.norm(current[:2,3]),1e-6)
        for lift,inward in ((.04,0.),(.08,0.),(.12,0.),(.08,.04),(.12,.04)):
            if (self.cancel_event.is_set() or self._interventions
                    or revision!=self._environment_revision or now()>=deadline):
                return False
            pose = current.copy()
            pose[2,3] += lift
            pose[:2,3] -= inward*radial
            scope = getattr(self.arm,'planning_scope',None)
            with scope(lambda:self.cancel_event.is_set() or now()>=deadline) if scope is not None else nullcontext():
                gateway,_ep,_eo = self.arm._ik_tcp(pose,joints)
            if gateway is None:
                continue
            reached = via(gateway)
            if reached is not None:
                return reached
        return False

    def _safe_observation_joints(self, q, detections, *, start_joints=None, escaping=False):
        """Check an empty-gripper joint sweep against measured object envelopes."""
        import cv2
        from ..perception.geometry import object_footprint
        from ..perception.polygon_collision import prepare_polygon,polygons_overlap
        from ..task.runner import footprint_dims

        if self.world.held is not None or self.cancel_event.is_set() or self._interventions:
            return False
        joints, width = self.arm.get_joints(), self.arm.gripper_width()
        table_z = self.arm.table_z
        if joints is None or width is None or table_z is None or not math.isfinite(float(table_z)):
            return False
        current = np.asarray(joints.value if start_joints is None else start_joints,float)
        target = np.asarray(q,float)
        if current.shape != (6,) or target.shape != (6,) or not np.all(np.isfinite(np.r_[current, target, width.value])):
            return False
        detections = list(detections)
        released = getattr(self, '_released_observation_obstacle', None)
        if released is not None:
            detections.append(released)
        obstacles = []
        for det in detections:
            values = [det.length, det.width, det.height, det.top_z, det.yaw]
            if (det.center_base is None or any(v is None for v in values) or
                    not np.all(np.isfinite(np.r_[det.center_base, values])) or min(values[:3]) <= 0):
                return False
            length, breadth = footprint_dims(det, getattr(self.runner, 'nominal', {}).get(det.label))
            obstacles.append((prepare_polygon(object_footprint(det.center_base[:2],length,breadth,det.yaw,
                              label=det.label,resting_pose=det.extra.get('resting_pose'))),
                              float(det.top_z)))
        if not obstacles:
            return False
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        fingers = [np.array([0., side * (float(width.value) / 2 + .006), -.025]) +
                   signs * [.010, .006, .025] for side in (-1, 1)]
        # Include the palm/connector above the finger pads as a conservative box.
        tool = fingers + [np.array([0., 0., -.10]) + signs * [.04, .055, .05]]
        count = max(25, int(math.ceil(float(np.max(np.abs(target-current))) / math.radians(1))))
        swept = []
        for sample in np.linspace(current, target, count + 1):
            if self.cancel_event.is_set() or self._interventions:
                return False
            pose = self.arm.kin.fk(sample) @ self.runner.T_flange_tcp
            if not self.arm.guard.check_tcp(pose, self.arm.table_z, check_tilt=False)[0]:
                return False
            swept.append([part @ pose[:3, :3].T + pose[:3, 3] for part in tool])
        for start, end in zip(swept, swept[1:]):
            # Sweep each solid separately. One hull around both fingers and
            # the higher palm falsely fills the open grasp and projects the
            # whole palm down to fingertip height, trapping valid view starts.
            for part,(before,after) in enumerate(zip(start,end)):
                corners = np.vstack([before,after])
                xy = (corners[:, None, :2] + np.array([[-.005, -.005], [-.005, .005],
                                                      [.005, .005], [.005, -.005]])).reshape(-1, 2)
                hull = prepare_polygon(cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2))
                low = float(corners[:, 2].min()) - .005
                if low <= float(table_z) + .005:
                    return False
                for footprint,top in obstacles:
                    if low > top+.015 or not polygons_overlap(hull,footprint,clearance=1e-9):
                        continue
                    # A stopped inspection can already be inside the extra
                    # stand-off, while the measured solid remains clear. A
                    # checked gateway may leave that margin only upward: every
                    # swept part stays clear, never lowers its near surface,
                    # and finishes outside the full original stand-off.
                    initial = float(swept[0][part][:,2].min())-.005
                    upward_exit = (escaping and initial >= top+.005
                        and low >= initial-1e-6
                        and float(after[:,2].min()) >= float(before[:,2].min())-1e-6
                        and float(swept[-1][part][:,2].min())-.005 > top+.015)
                    if not upward_exit:
                        return False
        return True

    def _remember_released_obstacle(self, det, result):
        """Bound the commanded release until new images locate it again.

        Only measured source geometry and measured release TCP are used. This
        conservative hint cannot be selected, grasped or verify any goal.
        """
        from ..core.types import Detection
        from ..perception.geometry import object_prism
        from ..perception.grasp import detection_table_z
        from ..task.runner import footprint_dims
        import cv2
        try:
            transform = np.asarray(result.detail['released_transform'], float)
            dimensions = footprint_dims(det, self.runner.nominal.get(det.label))
            floor = detection_table_z(det)
            if transform.shape != (4, 4) or dimensions is None or floor is None:
                return
            cloud = object_prism(det.center_base[:2], *dimensions, float(det.yaw or 0.),
                floor, float(det.height), label=det.label,
                resting_pose=(det.extra or {}).get('resting_pose'))
            measured = np.asarray(det.extra.get('_placement_points_base', []), float)
            if measured.ndim == 2 and measured.shape[1:] == (3,):
                cloud = np.vstack([cloud, measured])
            carried = cloud @ transform[:3, :3].T + transform[:3, 3]
            if not np.all(np.isfinite(carried)):
                return
            (x,y),(length,width),angle = cv2.minAreaRect(carried[:, :2].astype(np.float32))
            low,high = float(carried[:,2].min()),float(carried[:,2].max())
            self._released_observation_obstacle = Detection(det.label, 0.,
                center_base=np.array([x,y,high]), top_z=high, height=high-low,
                length=length, width=width, yaw=math.radians(angle), valid=False,
                reject_reason='commanded_release_obstacle_only',
                extra={'table_z':low, '_identity_revision':self._environment_revision})
            if not hasattr(self, '_released_observation_hints'):
                self._released_observation_hints = {}
            # Keep per-object aiming hints when another release supersedes the
            # immediate safety envelope. They are never camera/goal evidence.
            self._released_observation_hints[det.label] = copy.deepcopy(self._released_observation_obstacle)
        except (KeyError, ValueError, TypeError):
            return

    def _stack_support_hint(self, goal):
        """Recover a measured aiming hint without publishing it as fresh evidence."""
        from .observation_policy import unique_target
        from ..core.types import Detection
        destination = goal.get('destination') or {}
        base = unique_target(self._observed_detections, destination.get('label'))
        if base is not None:
            return base
        active = getattr(self, '_active_placement', None)
        if (not active or active[0].label != goal.get('label')
                or active[1] != destination):
            return None
        support = active[0].extra.get('_stack_support') or {}
        try:
            xy = np.asarray(support['xy'], float)
            top, length, width, yaw = [float(support[k]) for k in ('top_z', 'length', 'width', 'yaw')]
            table = float(self.cfg.get('perception.table.expected_z', 0.) or 0.)
            if (support['label'] != destination.get('label') or xy.shape != (2,)
                    or not np.all(np.isfinite(np.r_[xy, top, length, width, yaw, table]))
                    or min(length, width, top-table) <= .005):
                return None
        except (KeyError, TypeError, ValueError):
            return None
        return Detection(support['label'], 1., center_base=np.r_[xy, top], top_z=top,
            height=top-table, length=length, width=width, yaw=yaw, valid=False,
            reject_reason='remembered_support_aiming_hint')

    def _verification_view(self, goals, fresh, revision, attempted=()):
        view = self._stack_verification_view(goals,fresh,revision,attempted)
        if view is not None:
            return view
        if revision != self._environment_revision or self.cancel_event.is_set():
            return None
        for goal in goals:
            if (goal.get('destination') or {}).get('type')=='on_object':
                continue
            hints = [d for d in fresh if d.label==goal['label'] and d.center_base is not None]
            if not hints:
                released = getattr(self,'_released_observation_hints',{}).get(goal['label'])
                if released is None:
                    released = getattr(self,'_released_observation_obstacle',None)
                if released is not None and released.label==goal['label']:
                    hints = [released]
            if len(hints)!=1 or hints[0].valid:
                continue
            for index in self._source_observation_views(hints[0].summary()):
                q = np.asarray(self.runner.views[index],float)
                if not any(np.max(np.abs(q-old))<math.radians(2.) for old in attempted):
                    return index
            if hasattr(self.arm,'kin'):
                from ..core.types import Detection
                from ..perception.grasp import detection_table_z
                from ..task.runner import footprint_dims
                upper = copy.deepcopy(hints[0])
                floor = detection_table_z(upper)
                dims = footprint_dims(upper,self.runner.nominal.get(upper.label))
                if floor is None or dims is None:
                    continue
                upper.length,upper.width = dims
                # This lower rectangle frames the measured table, never an
                # invented object or proof of support. Joint proposals still
                # need full fresh-image verification after the camera moves.
                plane = Detection('_table_aim',0.,center_base=np.r_[upper.center_base[:2],floor],
                    top_z=floor,length=upper.length,width=upper.width,yaw=upper.yaw,valid=False)
                q = self._stack_joint_view(upper,plane,self._observed_detections,revision,attempted)
                if q is not None:
                    return self._remember_stack_view(q,stacked=False)
        return None

    def _stack_joint_view(self, upper, base, neighbors, revision, attempted=()):
        """Use nearby taught joint poses when an exact optical pose has no IK."""
        from ..perception.geometry import project_points
        from .destinations import footprint_corners
        ext = self.runner.ext
        T_fc = ext.T if ext.parent_frame == 'flange' else self.runner.T_flange_tcp @ ext.T
        intr = self.camera.intrinsics()
        try:
            corners = footprint_corners(upper.center_base[:2], float(upper.length),
                                        float(upper.width), float(upper.yaw or 0.))
            lower = footprint_corners(base.center_base[:2], float(base.length),
                                      float(base.width), float(base.yaw or 0.))
            # A remembered base can be far behind a newly observed moving
            # stack. Keep it as an obstacle, but do not require one image to
            # cover its old position and the current target simultaneously.
            nearby = np.linalg.norm(upper.center_base[:2] - base.center_base[:2]) <= (
                .5 * (math.hypot(upper.length, upper.width) +
                      math.hypot(base.length, base.width)) + .02)
            points = np.vstack([np.column_stack([corners, np.full(4, z)])
                                for z in (float(base.top_z), float(upper.top_z))] +
                               ([np.column_stack([lower, np.full(4, float(base.top_z))])]
                                if nearby else []))
            if not np.all(np.isfinite(points)):
                return None
        except (TypeError, ValueError):
            return None
        reference = upper.summary()
        if nearby:
            reference['required_support'] = base.summary()
        anchors = [v for v in self._view_order(reference) if v < self._base_view_count]
        # Sample both higher and lower elevations and both sides. Prefer a
        # different line of sight to the measured target, not just different
        # joint angles that can leave the same support surface hidden.
        gaze = np.r_[upper.center_base[:2], (float(base.top_z) + float(upper.top_z)) / 2]
        def direction(q):
            delta = (self.arm.kin.fk(q) @ T_fc)[:3, 3] - gaze
            return delta / max(float(np.linalg.norm(delta)), 1e-9)
        old_directions = [direction(q) for q in attempted]
        candidates = []
        for view in anchors:
            for shoulder in (-.48, -.32, -.16, .08, .16, .32):
                for swivel in (-.30, .30, -.15, .15, 0., -.12, .12):
                    if (self.cancel_event.is_set() or self._interventions
                            or revision != self._environment_revision):
                        return None
                    q = np.asarray(self.runner.views[view], float).copy()
                    q[0] += swivel
                    q[1] += shoulder
                    if any(np.max(np.abs(q - old)) < .03 for old in attempted):
                        continue
                    if not self.arm.kin.within_limits(q):
                        continue
                    uv = project_points(points, self.arm.kin.fk(q) @ T_fc, intr)
                    if (not np.all(np.isfinite(uv)) or np.any(uv[:, 0] < 8)
                            or np.any(uv[:, 0] > intr.width-9) or np.any(uv[:, 1] < 8)
                            or np.any(uv[:, 1] > intr.height-9)):
                        continue
                    novelty = (min(1. - float(np.clip(direction(q) @ old, -1., 1.))
                                   for old in old_directions) if old_directions else 0.)
                    candidates.append((novelty, q))
        for _, q in sorted(candidates, key=lambda item: -item[0]):
            if self.cancel_event.is_set() or self._interventions or revision != self._environment_revision:
                return None
            if self._safe_observation_joints(q, neighbors):
                return q
        return None

    def _remember_stack_view(self, q, *, stacked=True):
        view = self._base_view_count
        if len(self.runner.views) == view:
            self.runner.views.append(q.tolist())
            self._view_labels.append('叠放支撑局部视角')
        else:
            self.runner.views[view] = q.tolist()
            self._view_labels[view] = '叠放支撑局部视角'
        self.runner._at_observe = False
        self.runner.multi_view = True
        self._destination_views = {}
        self._emit('stack_observation', {'message': ('顶面被遮挡，正在从一个安全局部视角确认支撑关系'
            if stacked else '目标被画面边缘裁剪，正在从安全局部视角核验落位')})
        return view

    def _stack_verification_view(self, goals, fresh, revision, attempted=()):
        """Offer one safely reachable local view when the upper body hides its base."""
        from ..sim.scenarios import look_at

        if (revision != self._environment_revision or self.world.held is not None or
                getattr(self, '_interventions', ()) or not hasattr(self.arm, 'kin')):
            return None
        for goal in goals:
            destination = goal.get('destination') or {}
            if destination.get('type') != 'on_object' or self._camera_goal_confirmed(goal, fresh, revision):
                continue
            # A cropped upper body is still a camera aiming hint, never a grasp
            # target or proof of support. Require one unambiguous measured hint.
            upper = [d for d in fresh if d.label == goal['label'] and d.center_base is not None]
            base = self._stack_support_hint(goal)
            if len(upper) != 1 or base is None or base.top_z is None or upper[0].top_z is None:
                continue
            upper = upper[0]
            if upper.top_z <= base.top_z + .01:
                continue
            gaze_z = min(float(base.top_z) + .05, (float(base.top_z) + float(upper.top_z)) / 2)
            gaze = np.r_[upper.center_base[:2], gaze_z]
            radial = gaze[:2] / max(float(np.linalg.norm(gaze[:2])), 1e-6)
            ext = self.runner.ext
            T_fc = ext.T if ext.parent_frame == 'flange' else self.runner.T_flange_tcp @ ext.T
            neighbors = list(fresh) + [d for d in self._observed_detections
                                      if d.label not in {item.label for item in fresh}]
            if not any(d.label == base.label for d in neighbors):
                neighbors.append(base)  # Conservative obstacle only, never fresh evidence.
            # A broad belly can hide the lower object's whole top from nearby
            # high views. Aim a lower, genuinely different side view at the
            # measured support, rather than repeatedly circling a cropped
            # upper-body centroid. Catalogue dimensions bound framing only;
            # fresh RGB-D must still establish the actual support relation.
            wide = (max(float(upper.length or 0.), float(upper.width or 0.)) >
                    1.4 * max(float(base.length or 0.), float(base.width or 0.)))
            nearby = np.linalg.norm(upper.center_base[:2]-base.center_base[:2]) < .10
            if wide and nearby:
                from .catalog import catalog_entries
                from .destinations import footprint_corners
                from ..perception.geometry import project_points
                entry = next((item for item in catalog_entries()
                              if item['label'] == upper.label), None)
                if entry is not None:
                    length, width, height = map(float, entry['dimensions'])
                    low_gaze = np.r_[base.center_base[:2],
                                     float(base.top_z)+min(.06, height/2)]
                    low_radial = low_gaze[:2]/max(float(np.linalg.norm(low_gaze[:2])), 1e-6)
                    top = max(float(upper.top_z), float(base.top_z)+height)
                    points = []
                    for center in (upper.center_base[:2], base.center_base[:2]):
                        corners = footprint_corners(center, max(length, float(upper.length)),
                                                    max(width, float(upper.width)),
                                                    float(upper.yaw or 0.))
                        points.extend(np.column_stack([corners, np.full(4, z)])
                                      for z in (float(base.top_z), top))
                    corners = footprint_corners(base.center_base[:2], float(base.length),
                                                float(base.width), float(base.yaw or 0.))
                    points.append(np.column_stack([corners, np.full(4, float(base.top_z))]))
                    intr = self.camera.intrinsics()
                    for angle in (-45., 45., -65., 65.):
                        if (self.cancel_event.is_set() or self._interventions
                                or revision != self._environment_revision):
                            return None
                        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
                        direction = np.array([[c, -s], [s, c]]) @ low_radial
                        eye = np.r_[low_gaze[:2]-.16*direction, low_gaze[2]+.12]
                        q, _ep, _er = self.arm.kin.ik(look_at(eye, low_gaze) @ inv_T(T_fc),
                            q_seed=self.arm.get_joints().value, cancelled=self.cancel_event.is_set)
                        if (q is None or any(np.max(np.abs(q-old)) < .03 for old in attempted)
                                or not self.arm.kin.within_limits(q)):
                            continue
                        uv = project_points(np.vstack(points), self.arm.kin.fk(q) @ T_fc, intr)
                        if (not np.all(np.isfinite(uv)) or np.any(uv[:, 0] < 8)
                                or np.any(uv[:, 0] > intr.width-9) or np.any(uv[:, 1] < 8)
                                or np.any(uv[:, 1] > intr.height-9)
                                or not self._safe_observation_joints(q, neighbors)):
                            continue
                        if (self.cancel_event.is_set() or self._interventions
                                or revision != self._environment_revision):
                            return None
                        return self._remember_stack_view(q)
            # After one extra optical view, try another viewing direction
            # before spending the remaining budget on nearly identical IK poses.
            if len(attempted) > 2:
                q = self._stack_joint_view(upper, base, neighbors, revision, attempted)
                if q is not None:
                    if self.cancel_event.is_set() or self._interventions or revision != self._environment_revision:
                        return None
                    return self._remember_stack_view(q)
            for angle, distance, height in ((5., .18, .22), (-5., .18, .22),
                                            (5., .16, .22), (-5., .16, .22)):
                if self.cancel_event.is_set() or self._interventions or revision != self._environment_revision:
                    return None
                c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
                direction = np.array([[c, -s], [s, c]]) @ radial
                eye = np.r_[gaze[:2] - distance * direction, gaze_z + height]
                q, _ep, _er = self.arm.kin.ik(look_at(eye, gaze) @ inv_T(T_fc),
                    q_seed=self.arm.get_joints().value, cancelled=self.cancel_event.is_set)
                if (q is None or any(np.max(np.abs(q - old)) < .03 for old in attempted)
                        or not self._safe_observation_joints(q, neighbors)):
                    continue
                if self.cancel_event.is_set() or self._interventions or revision != self._environment_revision:
                    return None
                return self._remember_stack_view(q)
            q = self._stack_joint_view(upper, base, neighbors, revision, attempted)
            if (q is not None and not self.cancel_event.is_set() and not self._interventions
                    and revision == self._environment_revision):
                return self._remember_stack_view(q)
        return None

    def _verify_zone(self, obj, zone_name):
        """Independent full-zone and stable-posture audit after execution."""
        if obj is None or obj is self.world.held:
            return False
        zone = self.runner.zones[zone_name]
        if abs(obj.bottom_z() - float(zone.center[2])) > .01:
            return False
        if self._physics is not None and not self._physics.resting_status(obj.name)['success']:
            return False
        c, s = math.cos(zone.yaw), math.sin(zone.yaw)
        for axis, span in ((np.array([c, s, 0.]), zone.size[0]),
                           (np.array([-s, c, 0.]), zone.size[1])):
            origin = float(np.asarray(zone.center) @ axis)
            limit = float(span) / 2 - .002
            if obj.support(axis) - origin > limit or obj.support(-axis) + origin > limit:
                return False
        return True

    def _verify_destination(self, obj, destination):
        """Independent truth check, after execution only; never supplies a plan."""
        if self._physics is not None and (obj is None or not
                self._physics.resting_status(obj.name)['success']):
            return False
        if destination['type'] == 'on_object':
            from .stacking import audit
            return bool(audit(self, obj, destination).get('success'))
        if obj is None or obj is self.world.held or abs(obj.bottom_z() - self.world.table_z) > .01:
            return False
        if (destination['type'] == 'point' and
                np.linalg.norm(obj.center_xy() - destination['xy']) > self._placement['point_tolerance']):
            return False
        from .destinations import physical_outline
        return footprint_in_destination(physical_outline(obj),
                                        {'placement': self._placement}, destination)

    def verify_goals(self, goals):
        """Recheck changed completed goals, sharing one local view per destination."""
        from .batch import whole_footprint_at_goal
        from .observation_policy import unique_target
        from .observations import GoalVerification
        groups = {}
        for goal in goals:
            destination = goal.get('destination')
            if destination is None:
                zone = self.runner.zones[goal['zone_name']]
                key, xy = ('zone', zone.name), zone.center[:2].tolist()
            elif destination['type'] == 'point':
                key, xy = ('point', *destination['xy']), destination['xy']
            elif destination['type'] == 'on_object':
                key = ('on_object', destination['label'])
                reference = next((d for d in self._observed_detections
                    if d.center_base is not None and d.label == destination['label']), None)
                if reference is None:
                    reference = next((d for d in self._observed_detections
                        if d.center_base is not None and d.label == goal['label']), None)
                xy = reference.center_base[:2].tolist() if reference is not None else [.26, 0.]
            else:
                key, xy = ('region', destination['name']), [.26, 0.]
            groups.setdefault(key, (xy, []))[1].append(goal)
        items, targets = [], []
        for xy, group in groups.values():
            query = {'xy': xy, 'verify_goals': copy.deepcopy(group)}
            if len(group) == 1:
                query['label'] = group[0]['label']
            for goal in group:
                if (goal.get('destination') or {}).get('type') == 'on_object':
                    base = next((d for d in self._observed_detections
                        if d.label == goal['destination']['label'] and d.top_z is not None), None)
                    upper = next((d for d in self._observed_detections
                        if d.label == goal['label'] and d.height is not None), None)
                    if base is not None and upper is not None:
                        query['top_z'] = max(query.get('top_z', 0.), float(base.top_z + upper.height))
            observer = getattr(self, '_motion_observer', None)
            pooled = observer.current() if observer is not None else []
            if pooled and all(self._camera_goal_confirmed(g, pooled,
                                self._environment_revision) for g in group):
                fresh, scene = pooled, self.scene
                self._last_fresh = fresh
                self._observation_revision = self._environment_revision
                observer.stats['verification_reuses'] += len(group)
            else:
                scene = self.observe(target=query, purpose='verify_place')
                fresh = list(self._last_fresh)
            for goal in group:
                targets.append(unique_target(fresh, goal['label']))
                def confirmed():
                    if (goal.get('destination') or {}).get('type') == 'on_object':
                        from .stacking import camera_confirmed, audit
                        from .observation_policy import unique_target
                        upper = unique_target(fresh, goal['label'])
                        obj = self._evaluation_object(upper) if upper is not None else None
                        return bool(camera_confirmed(self, goal['label'], goal['destination']) and
                                    audit(self, obj, goal['destination']).get('success'))
                    def physically_settled(d):
                        obj = self._evaluation_object(d)
                        if self._physics is None:
                            return True
                        if goal.get('destination') is None:
                            return self._verify_zone(obj, goal['zone_name'])
                        return obj is not None and self._verify_destination(obj, goal['destination'])
                    return (self._observation_revision == self._environment_revision and any(
                        d.valid and d.label == goal['label'] and
                        whole_footprint_at_goal(d.summary(), scene, goal) and
                        physically_settled(d) for d in fresh))
                ok = confirmed()
                measured = targets[-1]
                error, message = '', '目标已核验'
                if not ok:
                    if measured is None:
                        error, message = 'target_not_observed', '尚未获得目标的唯一完整测量'
                    elif (goal.get('destination') or {}).get('type') == 'on_object':
                        error, message = 'stack_support_unconfirmed', '叠放支撑或落稳尚未确认'
                    elif not whole_footprint_at_goal(measured.summary(), scene, goal):
                        error, message = 'target_outside_destination', '目标完整轮廓尚未位于指定区域内，需要调整落点'
                    else:
                        error, message = 'placement_not_stable', '目标支撑或稳定性尚未通过检查'
                items.append({'label': goal['label'], 'ok': ok, 'error': error, 'message': message,
                              'verified_revision': self._observation_revision,
                              'goal': copy.deepcopy(goal)})
        for item in items:
            item['ok'] = item['ok'] and item['verified_revision'] == self._environment_revision
        ok = all(item['ok'] for item in items)
        return GoalVerification(
            {'ok': ok, 'items': items, 'verified_revision': self._environment_revision,
             'message': '新相机图像已确认全部目标' if ok else '场景变化后部分目标不再满足原任务'},
            targets)

    def _finish_attempt(self, result, why, object_id, zone_name, det, evaluation_object,
                        destination=None):
        """Use the same independent placement check for initial and recovery actions."""
        summary = (pick_summary(result) if result is not None else
                   {'ok': False, 'stage': 'plan', 'error': why, 'label': det.label,
                    'zone': zone_name, 'duration': 0.0})
        if result is not None:
            summary['diagnostics'] = str(result.detail.get('last_motion_detail') or
                                         result.detail.get('placement_planning_reason') or
                                         result.detail.get('held_transit_planning_reason') or
                                         result.detail.get('reason') or why or '')[:500]
            if 'placement_checks' in result.detail:
                summary['placement_checks'] = copy.deepcopy(result.detail['placement_checks'])
        if (result is not None and self._physics is not None and
                not self.cancel_event.is_set() and self.world.held is None):
            self._status = 'settling'
            self._emit('settling', {'message': '正在等待物品在重力和接触作用下落稳'})
            self._physics.settle(.8)
            self._remember_physics()
        summary['object_id'] = object_id
        summary['source_xy'] = det.center_base[:2].tolist()
        if destination is not None:
            summary['destination'] = copy.deepcopy(destination)
            summary['placement_center'] = copy.deepcopy(self._placement_center)
        summary['aborted'] = self.cancel_event.is_set()
        summary['holding'] = self.world.held is not None
        if result is not None and not result.detail.get('holding'):
            from .placement_observation import forget_inspection
            forget_inspection(self)
        self._recovery_context = None
        if result is None and not summary['aborted']:
            self._status, self._stage = 'ready', ''
        else:
            self._status = 'ready'
        if result is None and not summary['aborted']:
            pass  # Planning refusal sent no commands and preserves this observation.
        elif not summary['aborted'] and not summary.get('fault') and not summary['holding']:
            try:
                if result is not None and result.ok:
                    self._remember_released_obstacle(det, result)
                if destination is not None:
                    xy = (self._placement_center or destination.get('xy') or [.26, 0])
                    support = None
                    if destination['type'] == 'on_object':
                        from .observation_policy import unique_target
                        support = unique_target(self._observed_detections, destination['label'])
                        if self._placement_center is None and support is not None:
                            xy = support.center_base[:2].tolist()
                    if destination['type'] == 'point':
                        self._ensure_destination_view(destination)
                    query = {'xy': xy}
                    if support is not None and support.top_z is not None and det.height is not None:
                        query['top_z'] = float(support.top_z + det.height)
                else:
                    query = {'xy': getattr(self,'_placement_center',None) or self.runner.zones[zone_name].center[:2].tolist()}
                if result is not None and result.ok:
                    query.update(label=det.label, height=det.height)
                    goal = {'label': det.label, 'zone_name': zone_name}
                    if destination is not None:
                        goal['destination'] = destination
                    query['verify_goals'] = [goal]
                observer = getattr(self, '_motion_observer', None)
                pooled = observer.current() if observer is not None else []
                if (result is not None and result.ok and pooled and
                        self._camera_goal_confirmed(query['verify_goals'][0], pooled,
                                                    self._environment_revision)):
                    self._last_fresh = pooled
                    self._observation_revision = self._environment_revision
                    observer.stats['verification_reuses'] += 1
                else:
                    self.observe(target=query, purpose='verify_place',
                                 forget=[det] if result is not None and result.ok else [])
            except RuntimeError as exc:
                summary['observation_error'] = str(exc)
        else:
            self._status = 'stopped' if summary['aborted'] else 'error'
        summary['scene'] = copy.deepcopy(self._scene)
        summary['score'] = self.world.score(self.runner.class_to_zone)
        actual_zone = None if evaluation_object is None else self.world.zone_of(evaluation_object)
        truth_success = self._verify_zone(evaluation_object, zone_name) if destination is None else False
        if destination is not None:
            truth_success = self._verify_destination(evaluation_object, destination)
        resting = (self._physics.resting_status(evaluation_object.name)
                   if self._physics is not None and evaluation_object is not None else None)
        if resting is not None:
            truth_success = truth_success and resting['success']
        summary['controller_ok'] = bool(summary['ok'])
        summary['truth'] = {'success': truth_success,
                            'object_name': None if evaluation_object is None else evaluation_object.name,
                            'expected_zone': zone_name,
                            'actual_zone': None if actual_zone is None else actual_zone.name}
        if resting is not None:
            summary['truth']['resting'] = resting
        if destination is not None:
            summary['truth']['expected_destination'] = copy.deepcopy(destination)
            summary['truth']['expected_zone'] = None
            if destination['type'] == 'on_object':
                from .stacking import audit
                summary['truth']['support'] = audit(self, evaluation_object, destination)
        from .batch import whole_footprint_at_goal
        goal = {'zone_name': zone_name}
        if destination is not None:
            goal['destination'] = destination
        if destination is not None and destination['type'] == 'on_object':
            from .stacking import camera_confirmed
            camera_success = camera_confirmed(self, det.label, destination)
        else:
            camera_success = any(d.valid and d.label == det.label and
                                 whole_footprint_at_goal(d.summary(), self._scene, goal)
                                 for d in getattr(self, '_last_fresh', []))
        summary['camera_verified'] = camera_success
        summary['verified_revision'] = getattr(self, '_observation_revision', -1)
        camera_success = camera_success and (getattr(self, '_observation_revision', -1) ==
                                              getattr(self, '_environment_revision', -2))
        summary['camera_verified'] = camera_success
        # Release success can survive a later retreat failure. Public success
        # must still stop legacy callers before they dispatch another action.
        summary['ok'] = bool(summary['controller_ok'] and truth_success and camera_success
                             and not summary.get('fault') and not summary['aborted'])
        if summary['controller_ok'] and not truth_success:
            summary['error'] = ('pear_not_upright' if resting is not None and
                not resting['posture_ok'] else 'stack_unstable_or_fallen' if destination is not None and
                destination['type'] == 'on_object' else 'placement_not_verified')
        elif summary['controller_ok'] and not camera_success:
            summary['error'] = 'placement_not_observed'
        colors = {'red': '红色', 'green': '绿色', 'yellow': '黄色'}
        if summary['aborted']:
            summary['message'] = ('任务已停止，夹爪保持夹持，机械臂保持当前位置'
                                  if summary['holding'] else '任务已停止，机械臂保持当前位置')
        elif summary.get('fault'):
            summary['message'] = '机械臂执行异常，已停止后续动作并保留现场'
        elif summary['ok']:
            zone_label = self.runner.zones[zone_name].label if destination is None else ''
            names = {'red': '红色方块', 'green': '绿色方块', 'yellow': '黄色方块',
                     'apple': '苹果', 'banana': '香蕉', 'pear': '梨子'}
            summary['message'] = '已将%s放到%s' % (
                names.get(det.label, det.label), destination_label(destination) if destination else
                colors.get(zone_label, zone_label or zone_name) + '区域')
        elif summary['controller_ok']:
            if summary['error'] == 'pear_not_upright':
                summary['message'] = '梨子已放下但未保持直立，需要重新观察并调整放置姿态'
            elif truth_success and not camera_success:
                relation = ('目标物品与指定底座的支撑关系' if destination is not None and
                            destination['type'] == 'on_object' else '目标物品的落位')
                summary['message'] = '抓放动作已完成，物理检查通过，但相机尚未确认' + relation
            elif destination is not None and destination['type'] == 'on_object':
                support = summary['truth'].get('support', {})
                summary['message'] = ('物品已放下，但未检测到与指定底座的接触'
                    if support.get('contact') is False else
                    '物品已放下，但物品或底座仍在运动，尚未落稳'
                    if support.get('stable') is False and support.get('contact') is True else
                    '叠放尚未通过支撑与落稳核验，需要重新确认目标与底座')
            else:
                summary['message'] = ('抓放动作已结束，但无法唯一核对目标物品'
                    if evaluation_object is None else '抓放动作已结束，但目标物品没有位于指定区域')
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
            self._recovery_context = None
            self._apply_interventions()
            revision = self._environment_revision
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
                    lambda d: self._observation_replaced(d, self.camera._last, pose),
                    forget=[self._recovery_target],
                    preserve_incomplete=getattr(self, '_motion_observer', None) is not None)
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
            self._apply_interventions()
            if revision != self._environment_revision:
                self._held_proposals = {}
                self._recovery_retry = None
                raise ValueError('恢复观测期间现场已变化，请重新观察')
            if holding:
                self._scene_version += 1
            context = {'scene_version': self._scene_version,
                       'environment_revision': revision, 'attempt': int(attempt),
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

    def _stationary_recovery_observation(self, *, grasp=None, width=None):
        """Capture and process a new RGB-D frame without commanding any joint."""
        from ..camera.utils import capture_averaged
        from ..core.extrinsics import camera_pose_in_base

        revision = self._environment_revision
        newer = max(float(getattr(self.runner, '_last_stamp', None) or 0.0), self.clock.now())
        frame = capture_averaged(self.camera, self.runner.n_avg,
                                 newer_than=newer, timeout=self.runner.frame_timeout)
        if frame is None:
            raise RuntimeError('恢复时未获得新的相机图像')
        flange = self.arm.get_flange_pose()
        if flange is None or self.runner.ext is None:
            raise RuntimeError('恢复时无法读取相机坐标变换')
        pose = camera_pose_in_base(self.runner.ext, flange.value, self.runner.T_flange_tcp)
        raw_result = self.runner.perception.process(frame, pose, self.runner.zones)
        result = raw_result
        context = getattr(self.runner, '_held_recovery_context', None)
        if context is not None or grasp is not None:
            # Separate self surfaces in 3-D before segmentation: a carried
            # block can merge with a fruit below and acquire a different label.
            # Keep original depth for visibility and the raw surfaces for
            # independent held-anchor feedback from this exact capture.
            from .carried_observation import CarriedDetections, mask_carried_depth
            if grasp is None:
                confirmed, opening, _why = self.runner._held_transfer_feedback(context['det'])
                grasp = context['grasp'] if confirmed is not None else None
            else:
                # During the first move the recovery context is saved only
                # after the skill returns. Its live callback already owns the
                # executed measured grasp and verified opening.
                opening = width if width is not None else self.runner.skill._read_width()
                if not self.runner.skill._healthy():
                    grasp = None
            if grasp is not None:
                tcp = self.arm.get_tcp_pose()
                if (tcp is not None and tcp.value is not None
                        and abs(float(tcp.stamp) - float(frame.stamp)) <= self.runner.skill.feedback_max_age):
                    environment_frame = mask_carried_depth(frame, pose, grasp,
                        tcp.value, opening, nominal=self.runner.nominal.get(grasp.target.label))
                    if environment_frame is not frame:
                        result = self.runner.perception.process(environment_frame, pose, self.runner.zones)
            detections = CarriedDetections(result.detections, raw_result.detections)
        else:
            detections = result.detections
        self._scan_stats['frames'] += 1
        self.runner._last_stamp = float(frame.stamp)
        observer = getattr(self, '_motion_observer', None)
        if observer is not None and revision == self._environment_revision:
            target_label = (grasp.target.label if grasp is not None else
                            context['det'].label if context is not None else None)
            for d in detections:
                d.extra.update(_capture_stamp=float(frame.stamp),
                               _capture_flange=np.asarray(flange.value).copy(),
                               _observed_revision=revision)
            fresh = [d for d in detections if d.label != target_label]
            observer.reconcile(fresh, revision)
            observer.remember(fresh, revision)
            observer.remember_frame(frame, pose, revision)
            from .observation_policy import merge_local_detections, measurement_visible
            singletons = {d.label for d in fresh if d.valid and d.extra.get('catalog_complete')}
            merged = merge_local_detections(self._observed_detections, fresh,
                lambda d: self._observation_replaced(d, frame, pose), singleton_labels=singletons,
                preserve_incomplete=True)
            self._last_fresh = fresh
            self._observation_revision = revision
            self._publish_scene(merged, result.warnings, [], frame.stamp, preserve_status=True)
        self._cache_observation(6, result, frame, label='动作间隙观测')
        self._emit('recovery_observation', {'message': ('已保持夹持并完成静止相机观测'
                    if self.runner.has_held_recovery else '已完成当前视角的静止相机观测'),
                                           'detections': [d.summary() for d in result.detections]})
        return detections

    def execute_recovery(self, decision, cancel_event=None, confirmed=False):
        """Execute only a current locally offered action, preserving the original goal."""
        from .recovery import validate_recovery_decision

        with self._operation_lock:
            self._check_open()
            self._apply_interventions()
            context = self._recovery_context
            if context is None:
                raise ValueError('恢复方案已过期，请重新观察')
            if (context['scene_version'] != self._scene_version
                    or context.get('environment_revision') != self._environment_revision):
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
                if context['environment_revision'] != self._environment_revision:
                    raise ValueError('恢复场景已变化，请重新观察')
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
                self._active_revision = context['environment_revision']
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
            if (destination is not None and destination['type'] == 'on_object'
                    and det.label == destination['label']):
                continue  # Only the designated support is intentional contact.
            x, y = [float(v) for v in det.center_base[:2]]
            zone = next((z for z in self.runner.zones.values() if z.contains_xy(x, y)), None)
            footprint = self.runner._footprint(det)
            if destination is not None:
                zone = (self._active_placement[3]['placement_zone'] if
                    destination['type'] == 'on_object' and getattr(self, '_active_placement', None)
                    and self._active_placement[1] == destination else
                    self._placement_options(destination)['placement_zone'])
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

    def task_inventory(self) -> list:
        """Operator-created identities for task scope; never motion geometry."""
        from collections import Counter
        with self._operation_lock:
            self._check_open()
            counts = Counter(obj.label for obj in self.world.objects)
            return [dict(label=label, count=count) for label, count in counts.items()]

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

    def resume_after_stop(self) -> bool:
        """Rearm an idle, healthy simulation without executing the old route.

        The dashboard calls this only after its previous worker has unwound.
        No homing, finger opening, pose edit or motion command is issued here.
        """
        with self._operation_lock:
            if self._closed or not self._prepared:
                return False
            health = self.arm.health()
            if not getattr(health, 'ok', False) or not getattr(health, 'enabled', False):
                return False
            joints, tcp, width = self.arm.get_joints(), self.arm.get_tcp_pose(), self.arm.gripper_width()
            try:
                now = float(self.clock.now())
                max_age = float(self.runner.skill.feedback_max_age)
                for sample, shape in ((joints, (6,)), (tcp, (4, 4)), (width, ())):
                    value = np.asarray(sample.value, float)
                    age = now - float(sample.stamp)
                    if (value.shape != shape or not np.all(np.isfinite(value))
                            or not math.isfinite(age) or not -1e-6 <= age <= max_age):
                        return False
                transform = np.asarray(tcp.value, float)
                rotation = transform[:3, :3]
                if (not np.allclose(transform[3], [0., 0., 0., 1.], atol=1e-6)
                        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                        or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
                    return False
            except (AttributeError, TypeError, ValueError):
                return False
            with self._intervention_lock:
                pending = list(self._interventions)
                self._interventions.clear()
            for request_id, _request in pending:
                self._emit('scene_intervention_rejected', dict(request_id=request_id,
                    message='任务已停止，尚未生效的旧干预已取消'))
            self.cancel_event.external = None
            self.cancel_event.clear()
            self._environment_revision += 1
            observer = getattr(self, '_motion_observer', None)
            if observer is not None:
                observer.evidence.clear()
                observer.invalidated.clear()
                observer.frames.clear()
            self._observed_detections = []
            self._invalidate_scene('上次任务已停止；下一项操作将重新确认当前现场')
            self._status, self._stage = 'ready', ''
            self._publish_snapshot()
            return True

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
        physics = getattr(self, '_physics', None)
        snap = {'mode': getattr(self, 'backend', 'sim'),
                'simulation_kind': 'mujoco' if physics is not None else 'kinematic', 'seed': self.seed,
                'placement': copy.deepcopy(self._placement),
                'status': 'stopped' if self.cancel_event.is_set() else self._status,
                'stage': self._stage, 'scene_version': self._scene_version,
                'environment_revision': self._environment_revision,
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
        observer = getattr(self, '_motion_observer', None)
        if observer is not None:
            snap['motion_observation'] = dict(observer.stats, pending_labels=sorted(observer.invalidated))
        if physics is not None:
            snap['physics'] = physics.state()
        with self._cache_lock:
            self._snapshot = snap
        self._last_snapshot_wall = time.monotonic()

    def close(self) -> None:
        """Stop before disconnecting the simulator and closing its run log."""
        self.stop()
        self._idle_stop.set()
        if self._idle_thread is not None and self._idle_thread is not threading.current_thread():
            self._idle_thread.join(timeout=2)
        with self._operation_lock:
            if self._closed:
                return
            self.clock.on_tick = None
            self.camera.stop()
            self.arm.close()
            if self._physics is not None:
                self._physics.close()
            self.operator.close()
            self.logger.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
