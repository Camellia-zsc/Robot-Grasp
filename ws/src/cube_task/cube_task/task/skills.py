"""
Pick-and-place skill: one top-down grasp of one detection into one zone.

All poses are TCP poses in the robot base frame (m, rad). Gripper widths in m.
Durations use the injected Clock.

Failure policy (the runner never auto-restarts, see the competition rules):

- Motion errors 'comm' / 'arm_error' / 'aborted', gripper errors 'comm' /
  'fault' / 'aborted', missing gripper feedback and unknown error strings are
  faults: return at once with ``detail['fault'] = True`` and send no more motion.
  Exception: at the release, an opening that stopped short (finger against the
  bin or an earlier object) once open past held width + 10 mm counts as released.
- 'timeout' / 'stalled' / 'unreachable' / 'rejected' are recoverable. If the
  arm is healthy, one recovery is tried and the pick reports failure:
  before the object is held, a linear retreat up to the pre-grasp pose; while
  holding over the pick area or in transit, the object is put back where it was
  (back to T_lift, down to T_grasp, open, up to T_pre). The gripper is never
  opened over a bin except at the release pose. If a held object cannot be put
  down, the result is a fault ('holding_object').
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.types import GraspPlan, MotionResult, PickResult, PlacePlan
from ..core.transforms import is_rotation, rotation_angle
from ..perception.grasp import RELEASE_OPEN_EXTRA, max_gripper_width

_LOG = logging.getLogger('cube_task.task.skills')

FAULT_MOTION_ERRORS = ('comm', 'arm_error', 'aborted')
RECOVERABLE_MOTION_ERRORS = ('timeout', 'stalled', 'unreachable', 'rejected')
FAULT_GRIPPER_ERRORS = ('comm', 'fault', 'aborted')
#: measured opening must exceed the expected object width by this before descending.
OPEN_CLEARANCE = 0.002
#: a stopped release move still counts as "at release" within these limits (m).
RELEASE_XY_TOL = 0.005
RELEASE_Z_ABOVE_TOL = 0.002
RELEASE_Z_BELOW_TOL = 0.002
RELEASE_ROT_TOL = math.radians(3.0)
RELEASE_SETTLE_S = 0.12
RELEASE_MAX_SPEED = 0.01
RELEASE_MAX_ANGULAR_SPEED = math.radians(6.0)
#: at the release the fingers must open this far past the held width (m).
RELEASE_OPEN_MARGIN = 0.01
#: ... but never more than the commanded opening minus this (m): a gripper opened to
#: its full width reports a few um to a few 0.1 mm less (MuJoCo: 0.06995 for 0.07),
#: which must not turn a release into a 'holding_object' fault.
RELEASE_FULL_OPEN_TOL = 0.002
#: detail text of a gripper 'fault' whose opening stopped short (robot/pyagxarm_arm.py).
OPENING_SHORT = 'opening stopped'
#: AGX gripper force range (N); a configured force above it is clamped.
MAX_GRIPPER_FORCE = 3.0
#: Light foam cubes use L1 force; mixed rigid fruit use the model task's
#: 40-100 g force budget. The interactive task must not inherit the foam-only
#: robot default, which lets a tapered neck slip during a real contact lift.
TASK_GRIPPER_FORCE = {'L1': 1.5, 'L2': 2.5, 'Sandbox': 2.5}
#: stages whose target is a free-space pose (may lean up to max_tilt_free_deg).
FREE_STAGES = ('pre', 'lift', 'transit', 'retreat', 'recover_retreat', 'recover_up',
               'putback_lift', 'putback_retreat', 'up', 'retry_lift', 'inspect_lift', 'inspect_place')


def _force_value(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f >= 0.0 else None


def gripper_force_for(cfg, task: Optional[str] = None) -> Optional[float]:
    """
    Return the grasp force (N) for task, clamped to MAX_GRIPPER_FORCE.

    tasks.<task>.gripper_force, else TASK_GRIPPER_FORCE, else
    robot.gripper.force. None when none of them is a number >= 0.
    """
    f = None
    if task:
        f = _force_value(cfg.get('tasks.%s.gripper_force' % task, None))
        if f is None:
            f = _force_value(TASK_GRIPPER_FORCE.get(task))
    if f is None:
        f = _force_value(cfg.get('robot.gripper.force', None))
    if f is None:
        return None
    if f > MAX_GRIPPER_FORCE:
        _LOG.warning('gripper force %.2f N above %.1f N: clamped', f, MAX_GRIPPER_FORCE)
    return min(f, MAX_GRIPPER_FORCE)


class _StepFailed(Exception):
    def __init__(self, stage: str, error: str, detail: str = '', fault: bool = False):
        super().__init__('%s: %s %s' % (stage, error, detail))
        self.stage = stage
        self.error = error or 'unknown'
        self.detail = detail
        self.fault = fault


class _Lost(Exception):
    """The gripper closed on nothing after holding: the object fell."""

    def __init__(self, stage: str, width: float):
        super().__init__(stage)
        self.stage = stage
        self.width = width


def _xyz(T: np.ndarray) -> List[float]:
    return [round(float(v), 4) for v in np.asarray(T)[:3, 3]]


def is_fault_motion_error(error: str) -> bool:
    """Return True when a MotionResult error must stop the run (no more motion)."""
    return error not in RECOVERABLE_MOTION_ERRORS


class PickPlaceSkill:
    """
    Execute GraspPlan + PlacePlan on an ArmInterface.

    logger: core.logger.RunLogger-like object with ``event(kind, **fields)``, or
    None. Config: robot.gripper.*, grasp.linear_descent, grasp.verify_grasp,
    runner.dry_run_hover, robot.motion.feedback_max_age_s. task: grasp force
    from gripper_force_for(cfg, task) (None = robot.gripper.force, clamped);
    force: explicit force (N) that wins over both. Moves to free-space poses
    (FREE_STAGES) pass robot.workspace.max_tilt_free_deg to an arm that takes
    a per-call tilt limit (task.reach.free_tilt_limit).
    """

    def __init__(self, cfg, arm, clock, logger=None, task: Optional[str] = None,
                 force: Optional[float] = None):
        self.arm = arm
        self.clock = clock
        self.logger = logger
        self.max_width = max_gripper_width(cfg)
        ew = cfg.get('robot.gripper.empty_width', None)
        if ew is None or not math.isfinite(float(ew)) or float(ew) < 0.0:
            raise ValueError('robot.gripper.empty_width must be a number >= 0, got %r' % (ew,))
        self.empty_width = float(ew)
        f = _force_value(force)
        self.force = gripper_force_for(cfg, task) if f is None else min(f, MAX_GRIPPER_FORCE)
        from .reach import free_tilt_limit
        self.free_tilt = free_tilt_limit(arm, cfg)
        self.width_tolerance = float(cfg.get('robot.gripper.width_tolerance', 0.015))
        self.linear_descent = bool(cfg.get('grasp.linear_descent', True))
        # robot.motion.linear_moves false: short vertical moves (lift, release,
        # retreat, recovery) go as P2P, e.g. if move_l spins the wrist across the
        # +-180 deg roll wrap on the real controller.
        from ..robot.pyagxarm_arm import linear_moves_enabled
        self.linear_moves = linear_moves_enabled(cfg)
        self.verify = bool(cfg.get('grasp.verify_grasp', True))
        self.hover = float(cfg.get('runner.dry_run_hover', 0.05))
        self.feedback_max_age = float(cfg.get('robot.motion.feedback_max_age_s', 0.5))
        #: called with the stage name before every move (task.live watcher); None = off.
        self.on_stage = None
        # Explicit execution hooks may refuse motion; display observers above cannot.
        self.execution_guard = None
        self.before_release = None

    # ------------------------------------------------------------------ public
    def execute(self, grasp: GraspPlan, place: PlacePlan, dry_run: bool = False) -> PickResult:
        """Run the pick-and-place. Never raises for arm failures; see module docstring."""
        t0 = self.clock.now()
        r = PickResult(ok=False, stage='start', label=grasp.target.label, zone=place.zone,
                       detail={'fault': False, 'holding': False, 'dry_run': bool(dry_run),
                               'steps': []})
        if not grasp.feasible or not place.feasible:
            r.stage = 'plan'
            r.error = 'infeasible: %s' % (grasp.reason if not grasp.feasible else place.reason)
        else:
            try:
                if dry_run:
                    self._dry_run(grasp, place, r)
                else:
                    self._pick_place(grasp, place, r)
            except Exception as e:  # noqa: B902 - an arm backend bug must not crash the run
                _LOG.exception('pick-place raised')
                r.ok = False
                r.error = 'exception: %s' % e
                r.detail['fault'] = True
                try:
                    self.arm.stop()
                except Exception:  # noqa: B902
                    _LOG.exception('arm.stop() raised')
        r.duration = self.clock.now() - t0
        self._event('pick_result', ok=r.ok, stage=r.stage, error=r.error, label=r.label,
                    zone=r.zone, duration=round(r.duration, 3), grasp_width=r.grasp_width,
                    fault=r.detail.get('fault', False), holding=r.detail.get('holding', False))
        return r

    def place_held(self, grasp: GraspPlan, place: PlacePlan, grasp_width: float) -> PickResult:
        """Continue an explicitly requested placement without reopening a held grasp.

        This does not reset an abort or unhealthy arm. It first lifts vertically
        at the current XY when below the planned transit height; then the same
        checked transit, release and retreat primitives as a normal pick apply.
        A failed continuation holds position and keeps the gripper closed.
        """
        t0 = self.clock.now()
        r = PickResult(ok=False, stage='retry_place', label=grasp.target.label,
                       zone=place.zone, grasp_width=grasp_width,
                       detail={'fault': False, 'holding': True, 'steps': [],
                               'continuation': True})
        try:
            self._check_abort('retry_place')
            if not self._healthy():
                raise _StepFailed('retry_place', 'arm_error', 'arm unhealthy', fault=True)
            if not place.feasible or not grasp.feasible:
                raise _StepFailed('retry_place', 'infeasible', 'placement plan infeasible')
            if not math.isfinite(float(grasp_width)) or grasp_width < self.empty_width:
                raise _StepFailed('retry_place', 'no_gripper_feedback', 'invalid held width',
                                  fault=True)
            self._verify_held(r, 'dropped_before_retry')
            current = self.arm.get_tcp_pose()
            if (current is None or current.value is None or
                    self.clock.now() - float(current.stamp) > self.feedback_max_age):
                raise _StepFailed('retry_place', 'no_pose_feedback', 'missing current TCP',
                                  fault=True)
            lift = np.array(current.value, dtype=float, copy=True)
            if lift.shape != (4, 4) or not np.all(np.isfinite(lift)):
                raise _StepFailed('retry_place', 'no_pose_feedback', 'invalid current TCP',
                                  fault=True)
            height = float(place.T_pre[2, 3])
            if float(lift[2, 3]) < height - 1e-4:
                lift[2, 3] = height
                self._move(r, 'retry_lift', lift, True)
                self._verify_held(r, 'dropped')
            self._move(r, 'transit', place.T_pre, False)
            self._verify_held(r, 'dropped_in_transit')
            self._inspect_release(r, grasp, place)
            self._guard('release')
            self._move(r, 'release', place.T_release, self.linear_moves)
            self._guard('open_release')
            self._release_and_retreat(r, grasp, place)
        except _Lost as exc:
            r.detail['holding'] = False
            r.stage = r.error = exc.stage
            r.detail['lost_width'] = exc.width
        except _StepFailed as exc:
            r.stage, r.error = exc.stage, exc.error
            r.detail['reason'] = exc.detail
            r.detail['fault'] = bool(exc.fault)
        except Exception as exc:  # noqa: B902
            r.error = 'exception: %s' % exc
            r.detail['fault'] = True
            try:
                self.arm.stop()
            except Exception:  # noqa: B902
                _LOG.exception('arm.stop() raised during held placement')
        r.duration = self.clock.now() - t0
        self._event('pick_result', ok=r.ok, stage=r.stage, error=r.error, label=r.label,
                    zone=r.zone, duration=round(r.duration, 3), grasp_width=r.grasp_width,
                    fault=r.detail.get('fault', False), holding=r.detail.get('holding', False),
                    continuation=True)
        return r

    def pick_only(self, grasp: GraspPlan) -> PickResult:
        """Grasp, lift, and retain the object for a later explicit placement.

        A successful pick is a measured held grasp, not a completed placement.
        Recoverable failures after closing retain the grasp; a later controller
        step must choose what to do with it. No release destination is needed.
        """
        t0 = self.clock.now()
        r = PickResult(False, stage='start', label=grasp.target.label,
                       detail={'fault': False, 'holding': False, 'steps': [],
                               'pick_only': True})
        if not grasp.feasible:
            r.stage, r.error = 'plan', 'infeasible: %s' % grasp.reason
        else:
            try:
                self._pick_to_lift(grasp, r, require_feedback=True)
                if r.detail['holding']:
                    r.ok, r.stage = True, 'holding'
            except _Lost as exc:
                r.detail['holding'] = False
                r.stage = r.error = exc.stage
                r.detail['lost_width'] = exc.width
            except _StepFailed as exc:
                r.stage, r.error = exc.stage, exc.error
                r.detail.update(fault=bool(exc.fault), reason=exc.detail)
                if not exc.fault and not r.detail['holding'] and exc.stage in ('descend', 'close'):
                    self._retreat_empty(r, grasp, exc.error)
            except Exception as exc:  # noqa: B902
                _LOG.exception('pick-only raised')
                r.error = 'exception: %s' % exc
                r.detail['fault'] = True
                try:
                    self.arm.stop()
                except Exception:  # noqa: B902
                    _LOG.exception('arm.stop() raised during pick-only')
        r.duration = self.clock.now() - t0
        self._event('pick_result', ok=r.ok, stage=r.stage, error=r.error, label=r.label,
                    zone=r.zone, duration=round(r.duration, 3), grasp_width=r.grasp_width,
                    fault=r.detail.get('fault', False), holding=r.detail.get('holding', False),
                    pick_only=True)
        return r

    # -------------------------------------------------------------- primitives
    def _event(self, kind: str, **fields: Any) -> None:
        if self.logger is None:
            return
        try:
            self.logger.event(kind, **fields)
        except Exception:  # noqa: B902 - logging must never stop the arm logic
            _LOG.exception('logger.event failed')

    def _record(self, r: PickResult, stage: str, ok: bool, error: str, t_start: float,
                **extra: Any) -> None:
        dt = self.clock.now() - t_start
        step: Dict[str, Any] = {'stage': stage, 'ok': bool(ok), 'error': error,
                                'duration': round(dt, 3)}
        step.update(extra)
        r.detail['steps'].append(step)
        _LOG.info('step %-14s ok=%s %s %.2fs', stage, ok, error, dt)
        self._event('skill_step', label=r.label, zone=r.zone, **step)

    def _check_abort(self, stage: str) -> None:
        try:
            aborted = self.arm.abort_event.is_set()
        except Exception:  # noqa: B902
            aborted = False
        if aborted:
            raise _StepFailed(stage, 'aborted', 'abort event set', fault=True)

    def _healthy(self) -> bool:
        try:
            h = self.arm.health()
        except Exception:  # noqa: B902
            _LOG.exception('arm.health() raised')
            return False
        return bool(h is not None and h.ok)

    def _move(self, r: PickResult, stage: str, T: np.ndarray, linear: bool) -> MotionResult:
        self._check_abort(stage)
        if self.on_stage is not None:
            try:
                self.on_stage(stage)
            except Exception:  # noqa: B902 - an observer never changes the motion
                _LOG.exception('on_stage hook failed')
        t = self.clock.now()
        if self.free_tilt is not None and stage in FREE_STAGES:
            m = self.arm.move_tcp(T, linear=linear, max_tilt_deg=self.free_tilt)
        else:
            m = self.arm.move_tcp(T, linear=linear)
        self._record(r, stage, m.ok, m.error, t, target=_xyz(T), linear=bool(linear),
                     pos_err=m.pos_err, **({'detail': str(m.detail)[:500]} if not m.ok else {}))
        if not m.ok:
            r.detail['last_motion_detail'] = str(m.detail)[:500]
            err = m.error or 'unknown'
            raise _StepFailed(stage, err, m.detail, fault=is_fault_motion_error(err))
        return m

    def _read_width(self) -> Optional[float]:
        """Return a fresh measured gripper width (m) or None (missing / stale / NaN)."""
        try:
            s = self.arm.gripper_width()
        except Exception:  # noqa: B902
            _LOG.exception('gripper_width() raised')
            return None
        if s is None or s.value is None:
            return None
        if self.clock.now() - float(s.stamp) > self.feedback_max_age:
            return None
        w = float(s.value)
        return w if math.isfinite(w) else None

    def _gripper(self, r: PickResult, stage: str, width: float,
                 short_ok: Optional[float] = None) -> Optional[float]:
        """
        Command the gripper; return the measured width (m) or None if unknown.

        short_ok: an opening that stopped short of the command (gripper error
        'fault', detail 'opening stopped ...') is accepted when the measured
        width reaches short_ok (m); used at the release, where the fingers may
        touch the bin or an object already in it.
        """
        self._check_abort(stage)
        t = self.clock.now()
        g = self.arm.gripper(width, force=self.force, wait=True)
        w = g.width if (g.width is not None and math.isfinite(float(g.width))) else None
        if w is None:
            w = self._read_width()
        self._record(r, stage, g.ok, g.error, t, command=round(float(width), 4),
                     width=None if w is None else round(w, 4))
        if (short_ok is not None and g.error == 'fault' and OPENING_SHORT in (g.detail or '')
                and w is not None and w >= short_ok):
            _LOG.warning('%s: %s; open %.4f m >= %.4f m counts as released', stage, g.detail,
                         w, short_ok)
            r.detail['opening_short'] = g.detail
            return w
        if g.error and g.error != 'timeout':  # FAULT_GRIPPER_ERRORS and unknown ones
            raise _StepFailed(stage, 'gripper_' + g.error, g.detail, fault=True)
        return w

    def _open(self, r: PickResult, stage: str, width: float, need: Optional[float],
              short_ok: Optional[float] = None) -> None:
        w = self._gripper(r, stage, width, short_ok)
        if need is None:
            return
        if w is None:
            raise _StepFailed(stage, 'no_gripper_feedback', 'after opening', fault=True)
        if w < need:
            raise _StepFailed(stage, 'gripper_not_open',
                              'measured %.4f m < needed %.4f m' % (w, need))

    def _verify_held(self, r: PickResult, stage: str) -> Optional[float]:
        if not self.verify:
            return
        w = self._read_width()
        self._event('grasp_feedback', stage=stage,
                    ok=w is not None and w >= self.empty_width,
                    measured_width=w, empty_threshold=self.empty_width)
        if w is None:
            raise _StepFailed('verify_' + stage, 'no_gripper_feedback', 'while holding',
                              fault=True)
        if w < self.empty_width:
            raise _Lost(stage, w)
        return w

    # -------------------------------------------------------------------- flow
    def _pick_to_lift(self, g: GraspPlan, r: PickResult, require_feedback=False) -> None:
        """Shared checked opening, descent, closing, lift and held feedback."""
        self._open(r, 'open', g.open_width, need=g.expected_width + OPEN_CLEARANCE)
        self._move(r, 'pre', g.T_pre, False)
        self._guard('descend')
        self._move(r, 'descend', g.T_grasp, self.linear_descent)
        self._guard('close')
        w = self._gripper(r, 'close', g.close_width)
        r.grasp_width = w
        if w is None and (self.verify or require_feedback):
            if require_feedback:
                # Closure happened but its result is unknown. Treat the grip
                # conservatively as occupied; no later pick may reopen it.
                r.detail['holding'] = True
            raise _StepFailed('close', 'no_gripper_feedback', 'after closing', fault=True)
        if (self.verify or require_feedback) and w < self.empty_width:
            r.stage = r.error = 'grasp_miss'
            self._retreat_empty(r, g, 'grasp_miss')
            return
        if w is not None and abs(w - g.expected_width) > self.width_tolerance:
            _LOG.warning('held width %.4f m differs from expected %.4f m', w, g.expected_width)
        r.detail['holding'] = True
        self._move(r, 'lift', g.T_lift, self.linear_moves)
        held_width = self._verify_held(r, 'dropped')
        if require_feedback and not self.verify:
            held_width = self._read_width()
            if held_width is None:
                raise _StepFailed('verify_dropped', 'no_gripper_feedback', 'while holding', fault=True)
            if held_width < self.empty_width:
                raise _Lost('dropped', held_width)
        if held_width is not None:
            # A curved object can seat deeper between the pads while lifting.
            # Carry/release planning must use that measured grip, rather than
            # retaining the larger width that was read while still supported.
            r.grasp_width = held_width

    def _pick_place(self, g: GraspPlan, p: PlacePlan, r: PickResult) -> None:
        try:
            self._pick_to_lift(g, r)
            if not r.detail['holding']:
                return
            self._move(r, 'transit', p.T_pre, False)
            self._verify_held(r, 'dropped_in_transit')
            self._inspect_release(r, g, p)
            self._guard('release')
            self._move(r, 'release', p.T_release, self.linear_moves)
            self._guard('open_release')
            self._release_and_retreat(r, g, p)
        except _Lost as e:
            r.detail['holding'] = False
            r.stage = r.error = e.stage
            r.detail['lost_width'] = e.width
        except _StepFailed as e:
            r.stage = e.stage
            r.error = e.error
            if e.fault:
                r.detail['fault'] = True
                return
            if not r.detail['holding']:
                if e.stage in ('descend', 'close'):
                    self._retreat_empty(r, g, e.error)
                return
            if e.error == 'scene_changed':
                # Both the old drop site and the old pick site may have changed.
                # Retain the grasp until new camera evidence supplies a plan.
                return
            if e.stage == 'release':
                self._recover_release(r, g, p)
            elif e.stage in ('lift', 'transit'):
                self._put_back(r, g, from_transit=(e.stage == 'transit'))

    def _guard(self, stage):
        if self.execution_guard is not None:
            error = self.execution_guard(stage)
            if error:
                raise _StepFailed(stage, error, 'scene changed; camera confirmation required')

    def _inspect_release(self, r, grasp, place):
        if self.before_release is None:
            return
        updated = self.before_release(grasp, place, r.grasp_width, r)
        if updated is None or not updated.feasible:
            raise _StepFailed('inspect_place', 'placement_changed',
                              'fresh destination observation has no safe release')
        if not np.allclose(updated.T_pre, place.T_pre, atol=1e-5):
            self._move(r, 'transit', updated.T_pre, False)
            self._verify_held(r, 'dropped_in_transit')
        place.__dict__.update(updated.__dict__)

    def _release_and_retreat(self, r: PickResult, g: GraspPlan, p: PlacePlan) -> None:
        # A successful motion flag is insufficient: recovery and continuation
        # use this same gate, and no failed descent may turn into a high drop.
        self._check_abort('open_release')
        before, why = self._release_feedback(p)
        after = None
        if before is not None:
            start = self.clock.now()
            self.clock.sleep(RELEASE_SETTLE_S)
            self._check_abort('open_release')
            after, why = self._release_feedback(p)
            if after is not None:
                elapsed = self.clock.now() - start
                speed = (float(np.linalg.norm(after[:3, 3] - before[:3, 3])) / elapsed
                         if elapsed > 0 else math.inf)
                angular = (rotation_angle(before[:3, :3].T @ after[:3, :3]) / elapsed
                           if elapsed > 0 else math.inf)
                if speed > RELEASE_MAX_SPEED or angular > RELEASE_MAX_ANGULAR_SPEED:
                    why = 'release pose is still moving'
        if why or after is None:
            r.ok = False
            r.stage, r.error = 'release', 'release_pose_unconfirmed'
            r.detail.update(holding=True, reason=why or 'release pose is unavailable')
            return
        r.detail['release_feedback'] = dict(
            target=_xyz(p.T_release), actual=_xyz(after),
            height_error_m=float(after[2, 3] - p.T_release[2, 3]),
            linear_speed_m_s=speed, angular_speed_rad_s=angular,
            settled_s=elapsed)
        self._guard('open_release')
        held = r.grasp_width or 0.0
        width = p.open_width
        if r.grasp_width is not None:
            # The plan sized the opening from the detected width; an object that is
            # wider between the fingers would stay held by it.
            width = min(self.max_width, max(width, held + RELEASE_OPEN_EXTRA))
        need = min(width - RELEASE_FULL_OPEN_TOL, held + RELEASE_OPEN_MARGIN)
        try:
            # A finger stopped by the bin or an earlier object is not a fault once
            # the fingers are open past the held width: the object is released.
            self._open(r, 'open_release', width,
                       need=need if self.verify else None,
                       short_ok=need if r.grasp_width is not None else None)
        except _StepFailed as e:
            r.stage, r.error = e.stage, e.error
            if not e.fault:
                e = _StepFailed('open_release', 'holding_object', e.error, fault=True)
                r.error = e.error
            r.detail['fault'] = True
            return
        r.detail['holding'] = False
        r.ok = True
        r.stage = 'done'
        try:
            self._move(r, 'retreat', p.T_retreat, self.linear_moves)
        except _StepFailed as e:
            # The object is in the zone; report the failed retreat but keep ok=True.
            r.error = 'retreat_' + e.error
            r.detail['fault'] = bool(e.fault)
            r.detail['retreat_failed'] = True

    def _retreat_empty(self, r: PickResult, g: GraspPlan, why: str) -> None:
        """Nothing held: open the gripper and go straight up to T_pre (one try)."""
        if not self._healthy():
            r.detail['fault'] = True
            r.detail['recovery'] = 'skipped: arm unhealthy'
            return
        try:
            if why == 'grasp_miss':
                try:
                    self._open(r, 'recover_open', g.open_width, need=None)
                except _StepFailed as e:
                    if e.fault:
                        raise
            self._move(r, 'recover_retreat', g.T_pre, self.linear_moves)
            r.detail['recovery'] = 'retreated'
        except _StepFailed as e:
            # Stuck low next to the objects: further P2P motion is not safe.
            r.detail['recovery'] = 'failed: %s' % e.error
            r.detail['fault'] = True

    def _release_feedback(self, p: PlacePlan):
        """Fresh actual tool pose close enough to open over the measured support."""
        try:
            s = self.arm.get_tcp_pose()
            if s is None or s.value is None:
                return None, 'release TCP feedback is missing'
            age = self.clock.now() - float(s.stamp)
            if not math.isfinite(age) or not -.05 <= age <= self.feedback_max_age:
                return None, 'release TCP feedback is stale'
            actual = np.asarray(s.value, dtype=float)
            target = np.asarray(p.T_release, dtype=float)
            if (actual.shape != (4, 4) or not np.all(np.isfinite(actual))
                    or not is_rotation(actual[:3, :3], 1e-3)
                    or not np.allclose(actual[3], [0., 0., 0., 1.])):
                return None, 'release TCP feedback is invalid'
            delta = actual[:3, 3] - target[:3, 3]
            rotation = rotation_angle(target[:3, :3].T @ actual[:3, :3])
            if (np.linalg.norm(delta[:2]) > RELEASE_XY_TOL
                    or not -RELEASE_Z_BELOW_TOL <= delta[2] <= RELEASE_Z_ABOVE_TOL
                    or rotation > RELEASE_ROT_TOL):
                return None, ('release pose not reached: xyz error %s m, angle %.2f deg'
                              % (np.round(delta, 4).tolist(), math.degrees(rotation)))
            return actual.copy(), ''
        except Exception:  # noqa: B902 - unavailable feedback must never open
            return None, 'release TCP feedback is unavailable'

    def _near_release(self, p: PlacePlan) -> bool:
        actual, _why = self._release_feedback(p)
        return actual is not None

    def _recover_release(self, r: PickResult, g: GraspPlan, p: PlacePlan) -> None:
        """Descent into the zone stopped while holding the object."""
        if not self._healthy():
            r.detail['fault'] = True
            r.detail['recovery'] = 'skipped: arm unhealthy'
            return
        if self._near_release(p):
            r.detail['recovery'] = 'released near the release pose'
            self._release_and_retreat(r, g, p)
            return
        try:
            self._move(r, 'recover_up', p.T_pre, self.linear_moves)
        except _StepFailed as e:
            r.detail['fault'] = True
            r.error = 'holding_object'
            r.detail['recovery'] = 'failed: %s' % e.error
            return
        self._put_back(r, g, from_transit=True)

    def _put_back(self, r: PickResult, g: GraspPlan, from_transit: bool) -> None:
        """Put a held object back where it was picked; fault if that fails."""
        if not self._healthy():
            r.detail['fault'] = True
            r.detail['recovery'] = 'skipped: arm unhealthy'
            return
        try:
            if from_transit:
                self._move(r, 'putback_lift', g.T_lift, False)
            self._move(r, 'putback_down', g.T_grasp, self.linear_moves)
            self._open(r, 'putback_open', g.open_width, need=None)
        except _StepFailed as e:
            r.detail['fault'] = True
            r.error = 'holding_object'
            r.detail['recovery'] = 'put back failed: %s %s' % (e.stage, e.error)
            return
        r.detail['holding'] = False
        try:
            self._move(r, 'putback_retreat', g.T_pre, self.linear_moves)
            r.detail['recovery'] = 'put back'
        except _StepFailed as e:
            r.detail['fault'] = True
            r.detail['recovery'] = 'put back, retreat failed: %s' % e.error

    def _dry_run(self, g: GraspPlan, p: PlacePlan, r: PickResult) -> None:
        """Visit every pose of the plan without closing the gripper."""
        from .reach import reach_adjust
        hover_g = np.array(g.T_grasp, dtype=float)
        hover_g[2, 3] = min(hover_g[2, 3] + self.hover, g.T_pre[2, 3])
        hover_r = np.array(p.T_release, dtype=float)
        hover_r[2, 3] = min(hover_r[2, 3] + self.hover, p.T_pre[2, 3])
        # Raised poses can exceed joint5's vertical reach: lean them like planned poses.
        for name, T in (('hover_grasp', hover_g), ('hover_release', hover_r)):
            T2, _tilt, why = reach_adjust(self.arm, T)
            if T2 is None:
                r.stage, r.error = name, 'unreachable'
                r.detail['reason'] = why
                return
            T[:, :] = T2
        try:
            self._open(r, 'open', g.open_width, need=None)
            self._move(r, 'pre', g.T_pre, False)
            try:
                self._move(r, 'hover_grasp', hover_g, self.linear_descent)
            except _StepFailed as e:
                if not e.fault:
                    self._retreat_empty(r, g, e.error)
                raise
            self._move(r, 'up', g.T_pre, self.linear_moves)
            self._move(r, 'transit', p.T_pre, False)
            self._move(r, 'hover_release', hover_r, self.linear_moves)
            self._move(r, 'retreat', p.T_pre, self.linear_moves)
        except _StepFailed as e:
            r.stage, r.error = e.stage, e.error
            r.detail['fault'] = r.detail['fault'] or e.fault
            return
        r.ok = False
        r.stage = 'dry_run_done'
        r.detail['hover_grasp'] = _xyz(hover_g)
        r.detail['hover_release'] = _xyz(hover_r)
