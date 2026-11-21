"""
Competition runner for the L1 / L2 sorting rounds.

Flow: self check (a failure refuses a real run) -> operator keys active ->
refuse when the gripper holds an object -> straight up
when the TCP starts below the transit height -> observe pose + open gripper ->
eye_in_hand real runs: pick area visible from the observe pose -> operator start
(Enter typed before the prompt does not count) -> timed loop of observe / select
/ plan / pick-place. The wrist camera (eye_in_hand, the only camera of the
competition) captures at an observe pose after every pick. Returning there at the end
is optional.
Several wrist observe poses (robot.poses.observe_views, task/views.py): every view
is scanned once, the views' detections are merged, the target's view is captured
again before each pick when the arm moved since, and all views again when no
target is left.
A fault stops the run at once with
no further motion; the runner never restarts or resumes by itself (one
referee-approved restart per round is a human decision). Camera or perception
failures are retried until the deadline and do not count as 'nothing to pick'.
Poses are base-frame 4x4 TCP poses (m, rad); time is the injected Clock (s).
"""
from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.transforms import pose6_to_T
from ..core.types import Detection, PerceptionResult, PickResult, PlacePlan, TaskReport
from ..core.zones import parse_zones
from ..perception.grasp import (CAPSULE_RATIO, expected_table_z, GraspPlanner, LONG_ZONE_RATIO,
                                PlacePlanner, detection_table_z, held_offset_of,
                                max_gripper_width, table_floor_z, table_z_warning,
                                transit_height)
from .operator import Operator
from .selection import is_blacklisted, select_target
from .selfcheck import observe_tcp_pose, self_check
from .skills import PickPlaceSkill, RELEASE_OPEN_QUANTIZATION_TOL, is_fault_motion_error
from .views import (drop_near, MERGE_DIST, merge_view_detections, nearest_view, observe_views,
                    tag_view, view_of, views_near)
from .live import drop_live_seen

_LOG = logging.getLogger('cube_task.task.runner')

#: grasp-plan reasons that may come from one bad frame: blacklist only after this
#: many separate observations gave an infeasible plan.
REMEASURE_REASONS = ('too_wide', 'missing_geometry', 'bad_geometry')
INFEASIBLE_OBSERVATIONS = 2
#: a failed grasp may push the object: an unmatched detection with the same label
#: within this distance (m) of a failed attempt record continues that record.
PUSHED_MATCH_DIST = 0.10
#: after this many observation failures in a row, wait FAIL_BACKOFF_S between tries.
FAIL_BACKOFF_AFTER = 5
FAIL_BACKOFF_S = 2.0
#: at the start a gripper wider than empty_width and narrower than max_width minus
#: this (m) may hold an object: close it by PROBE_CLOSE and see whether it stops.
HOLD_OPEN_MARGIN = 0.01
PROBE_CLOSE = 0.008
PROBE_TOL = 0.003
#: eye_in_hand real runs: fresh frames tried for the pick-area check at the observe pose.
VIEW_CHECK_TRIES = 2
#: default grasp.min_approach_clearance (m): lowest pre-grasp TCP height above the object
#: top, and of the retreat above the released object, when higher poses are out of reach.
MIN_APPROACH_CLEARANCE = 0.02
#: default grasp.transit_object_clearance (m): the held object's bottom stays this far
#: above every other object near the straight line from the pick to the release.
TRANSIT_OBJECT_CLEARANCE = 0.03
#: release positions tried after an unreachable one (task/runner._attempt).
RELEASE_RETRIES = 6
# A small measured support has nearby centers and useful alternate finger
# angles. Reject individual poses there, not a 12 mm disk of valid support.
STACK_RELEASE_RETRIES = 30
#: an object counts as near that line within its and the held object's half footprints
#: plus this (m).
LINE_MARGIN = 0.03


class _NullLogger:
    dir = ''

    def event(self, kind: str, **fields: Any) -> None:
        pass

    def save_image(self, name: str, bgr: Any) -> None:
        return None

    def save_json(self, name: str, obj: Any) -> None:
        return None


def _xy(det: Detection) -> Tuple[float, float]:
    return float(det.center_base[0]), float(det.center_base[1])


def _has_xy(det: Detection) -> bool:
    c = det.center_base
    return (c is not None and len(c) >= 2 and math.isfinite(float(c[0]))
            and math.isfinite(float(c[1])))


def _is_stack_base(det: Detection, other: Detection) -> bool:
    """Match only the measured support named by this intentional contact."""
    support = (det.extra or {}).get('_stack_support')
    if not isinstance(support, dict) or other.label != support.get('label') or not _has_xy(other):
        return False
    try:
        xy = np.asarray(support['xy'], float)
        top = float(support['top_z'])
        return bool(xy.shape == (2,) and np.all(np.isfinite(xy)) and math.isfinite(top)
                    and np.linalg.norm(np.asarray(other.center_base[:2]) - xy) <= .012
                    and other.top_z is not None and abs(float(other.top_z) - top) <= .012)
    except (KeyError, TypeError, ValueError):
        return False


def _footprint(det: Detection) -> Optional[float]:
    """Largest horizontal extent of a detection (m), None when unknown."""
    vals = [float(v) for v in (det.length, det.width) if v is not None and math.isfinite(float(v))]
    return max(vals) if vals and max(vals) > 0.0 else None


def footprint_dims(det: Detection, nominal: Optional[Sequence[float]] = None
                   ) -> Optional[Tuple[float, float]]:
    """
    Return (length, width) of a detection's footprint (m), None when unknown.

    nominal: (length, width) of the class from place.nominal_footprint; the
    measured sides never count smaller (a curved or round object measures short).
    """
    vals = [float(v) for v in (det.length, det.width) if v is not None and math.isfinite(
        float(v))]
    if len(vals) < 2 or min(vals) <= 0.0:
        return None
    L, W = max(vals), min(vals)
    if nominal is not None:
        L, W = max(L, float(max(nominal))), max(W, float(min(nominal)))
    return L, W


def held_shape(det: Detection, gp, nominal: Optional[Sequence[float]] = None
               ) -> Optional[Dict[str, Any]]:
    """
    Return the held object's shape for PlacePlanner.plan(held=...), None if unknown.

    The long axis is perpendicular to the closing axis for 'elongated' grasps
    (GraspPlanner kind, det.extra['grasp']); grip_z = grasp TCP above the bottom.
    nominal: see footprint_dims.
    """
    dims = footprint_dims(det, nominal)
    if dims is None:
        return None
    tz = detection_table_z(det)
    kind = (det.extra.get('grasp') or {}).get('kind', '') if det.extra else ''
    shape = {'length': dims[0], 'width': dims[1], 'elongated': kind == 'elongated',
             'axis_offset': float(det.yaw or 0.) - float(gp.closing_angle),
             'grip_width': float(gp.expected_width),
             'grip_z': 0.0 if tz is None else max(0.0, float(gp.T_grasp[2, 3]) - tz)}
    desired_yaw = (det.extra or {}).get('_restore_yaw')
    if desired_yaw is not None:
        shape['_required_closing'] = gp.closing_angle + float(desired_yaw) - float(det.yaw or 0)
        # Match the restoration verifier's catalog-backed square symmetry.
        # A nearly square measured silhouette alone is not such evidence.
        from ..assistant.catalog import catalog_entries
        entry = next((e for e in catalog_entries() if e['label'] == det.label), {})
        known_dims = entry.get('dimensions', [])
        if (entry.get('shape') in ('box', 'cube') and len(known_dims) >= 2
                and abs(known_dims[0] - known_dims[1]) < .001):
            shape['_restore_quarter_turns'] = True
    return shape


def _placement_anchor_center(det, grasp):
    """Return the validated original source centre of a carried visual anchor."""
    for target in (grasp.target, det):
        anchor = (target.extra or {}).get('_held_translation_anchor')
        if not isinstance(anchor, dict) or anchor.get('translation_only') is not True:
            continue
        pair = anchor.get('pair')
        if not isinstance(pair, dict) or pair.get('ok') is not True or pair.get('translation_only') is not True:
            continue
        try:
            center = np.asarray(anchor['source_center'], float)
        except (KeyError, TypeError, ValueError):
            continue
        if center.shape == (3,) and np.all(np.isfinite(center)):
            return center.copy()
    return None


def _seg_point_dist(p, a, b) -> float:
    a, b, p = (np.asarray(v, dtype=float)[:2] for v in (a, b, p))
    d = b - a
    L2 = float(d @ d)
    t = 0.0 if L2 < 1e-18 else min(1.0, max(0.0, float((p - a) @ d) / L2))
    return float(np.linalg.norm(p - (a + t * d)))


def line_obstacle_top(det: Detection, others: Sequence[Detection], xy_to: Sequence[float],
                      margin: float = LINE_MARGIN) -> Optional[float]:
    """
    Return the highest top z (m) of the other objects near the pick -> release line.

    Near: center within margin + both half footprints of the straight base-XY
    segment from det to xy_to. None when no other object is near it.
    """
    if not _has_xy(det):
        return None
    a = _xy(det)
    half = (_footprint(det) or 0.0) / 2.0
    best = None
    for o in others:
        if o is det or not _has_xy(o) or o.top_z is None or not math.isfinite(float(o.top_z)):
            continue
        r = half + (_footprint(o) or 0.0) / 2.0 + margin
        if _seg_point_dist(_xy(o), a, xy_to) <= r:
            best = float(o.top_z) if best is None else max(best, float(o.top_z))
    return best


def pick_summary(r: PickResult) -> Dict[str, Any]:
    """Return a JSON-friendly summary of a PickResult (steps omitted)."""
    return {'ok': r.ok, 'stage': r.stage, 'error': r.error, 'label': r.label, 'zone': r.zone,
            'duration': round(float(r.duration), 3), 'grasp_width': r.grasp_width,
            'fault': bool(r.detail.get('fault', False)),
            'recovery': r.detail.get('recovery', '')}


class TaskRunner:
    """
    Run one timed round. See the module docstring and docs/CONTRACTS.md.

    targets: label list overriding tasks.<T>.order (only these labels are
    picked, first available label first). mode: 'real' | 'sim' (default from
    robot.backend).
    """

    def __init__(self, cfg, task: str, arm, camera, perception, clock, logger,
                 operator: Optional[Operator] = None, dry_run: bool = False,
                 extrinsics=None, targets: Optional[Sequence[str]] = None,
                 mode: Optional[str] = None):
        self.cfg = cfg
        self.task = str(task)
        self.arm = arm
        self.camera = camera
        self.perception = perception
        self.clock = clock
        self.logger = logger if logger is not None else _NullLogger()
        self.operator = operator if operator is not None else Operator(enabled=False)
        self.dry_run = bool(dry_run)
        self._ext_arg = extrinsics
        self.ext = None
        self.mode = mode or ('sim' if cfg.get('robot.backend', '') == 'sim' else 'real')
        t = 'tasks.%s.' % self.task
        self.labels = [str(x) for x in (cfg.get(t + 'labels', []) or [])]
        self.class_to_zone = dict(cfg.get(t + 'class_to_zone', {}) or {})
        self.order = [str(x) for x in (targets if targets else (cfg.get(t + 'order', []) or []))]
        tl = cfg.get(t + 'time_limit_s', None)
        if tl is None or not float(tl) > 0:
            raise ValueError('%stime_limit_s must be > 0, got %r' % (t, tl))
        self.time_limit = float(tl)
        self.required = int(cfg.get(t + 'required', 0) or 0)
        r = 'runner.'
        self.strategy = str(cfg.get(r + 'selection', 'nearest'))
        self.max_attempts = int(cfg.get(r + 'max_attempts_per_object', 3))
        mf = cfg.get(r + 'max_failures_per_label', None)
        self.max_label_failures = 2 * self.max_attempts if mf is None else int(mf)
        self.max_empty = int(cfg.get(r + 'max_empty_observations', 4))
        self.min_time = float(cfg.get(r + 'min_time_for_pick_s', 12.0))
        self.retry_s = float(cfg.get(r + 'observe_retry_s', 0.5))
        rh = cfg.get(r + 'return_home_at_end', 'home')
        self.return_home = 'observe' if rh is True else str(rh or '').lower()
        if self.return_home not in ('', 'false', 'observe', 'home'):
            raise ValueError('runner.return_home_at_end must be home, observe or false, got %r'
                             % (rh,))
        self.match_dist = float(cfg.get('perception.tracking.match_dist_m', 0.03))
        self.n_avg = int(cfg.get('camera.frames_to_average', 3))
        self.settle_s = float(cfg.get('camera.settle_s', 0.4))
        self.frame_timeout = float(cfg.get('camera.frame_timeout_s', 2.0))
        self.max_frame_age = float(cfg.get('camera.max_frame_age_s', 1.0))
        self.mount = 'eye_in_hand'              # the competition's only camera
        self.views = observe_views(cfg)
        self.multi_view = len(self.views) > 1
        self.view_merge = float(cfg.get('runner.view_merge_dist_m', MERGE_DIST))
        self.nominal = {str(k): (float(max(v)), float(min(v)))
                        for k, v in (cfg.get('place.nominal_footprint', {}) or {}).items()
                        if v is not None and len(v) == 2}
        # An elongated object seen shorter than this x its nominal length is cut by the
        # image border or the gripper: its center is not the object's (grasped at an end).
        self.partial_ratio = float(cfg.get('runner.partial_length_ratio', 0.6))
        # After a placement only the views with objects this close to the picked one are
        # captured again (the others were not touched); negative = every view.
        self.rescan_radius = float(cfg.get('runner.rescan_radius_m', 0.12))
        #: false: pick as soon as one view shows a target; true: scan every view first
        self.scan_all_first = bool(cfg.get('runner.scan_all_first', False))
        from .live import LiveSettings
        #: runner.live_watch: watch the pick area during each pick (real arm, several views)
        self.live = LiveSettings(cfg)
        self._live_dets: List[Detection] = []
        self._live_mark = 0
        self.save_images = bool(cfg.get('logging.save_images', True)) and bool(
            cfg.get('logging.image_every_observation', True))
        tcp = cfg.get('robot.tcp_offset', None)
        if tcp is None or len(tcp) != 6:
            raise ValueError('robot.tcp_offset must be [x y z roll pitch yaw], got %r' % (tcp,))
        self.T_flange_tcp = pose6_to_T(tcp)
        self.start_on_enter = bool(cfg.get('runner.start_on_enter', True))
        self.zones = parse_zones(cfg)
        self.planner = GraspPlanner(cfg, self.task)
        self.placer = PlacePlanner(cfg)
        self.max_width = max_gripper_width(cfg)
        self.skill = PickPlaceSkill(cfg, arm, clock, self.logger, task=self.task)
        self.skill.current_release_exit_check = self.current_release_exit_check
        from .reach import free_tilt_limit
        self.free_tilt = free_tilt_limit(arm, cfg)
        g = 'grasp.'
        self.min_approach = float(cfg.get(g + 'min_approach_clearance', MIN_APPROACH_CLEARANCE))
        self.transit_clear = float(cfg.get(g + 'transit_object_clearance',
                                           TRANSIT_OBJECT_CLEARANCE))
        #: label -> m the release goes below place.release_height(_flat) (place.release_lower)
        #: label -> m toward the robot of the point it is released as near to as the
        #: objects there allow (place.center_labels; a list = 0 for each)
        cl = cfg.get('place.center_labels', {}) or {}
        if isinstance(cl, dict):
            self.center_labels = {str(k): float(v or 0.0) for k, v in cl.items()}
        else:
            self.center_labels = {str(k): 0.0 for k in cl}
        #: zone -> labels in the order they must be placed (runner.zone_order); a later
        #: label waits while an earlier one may still come (_zone_order_wait)
        self.zone_order = {str(z): [str(x) for x in (v or [])]
                           for z, v in (cfg.get('runner.zone_order', {}) or {}).items()}
        self.release_lower = {str(k): float(v) for k, v in
                              (cfg.get('place.release_lower', {}) or {}).items()
                              if v is not None and math.isfinite(float(v))}
        self.expected_table_z = expected_table_z(cfg)
        self._tracks: List[Dict[str, Any]] = []
        self._label_failures: Dict[str, int] = {}
        #: zone name -> (x, y, footprint, height, length, width, long-axis angle or None) of
        #: the objects released into it
        self._placed: Dict[str, List[Tuple[Any, ...]]] = {}
        #: zone -> labels released there (runner.zone_order)
        self._placed_labels: Dict[str, List[str]] = {}
        #: every observe view scanned at least once (multi-view loop; True with one view)
        self._all_views_seen = True
        self._fail_streak = 0
        self._last_obs_failure = ''
        self._table_warned = False
        self._at_observe = False
        self._at_view = 0
        self._t_arrived: Optional[float] = None
        self._last_stamp: Optional[float] = None
        self._obs = 0
        self._observe_failures = 0
        self._deadline = math.inf
        self.fault_detail = ''
        self._held_recovery_context = None

    # ------------------------------------------------------------------ public
    def run(self) -> TaskReport:
        """Run the round and return the report (never raises for robot failures)."""
        rep = TaskReport(task=self.task, ok=False, required=self.required,
                         time_limit=self.time_limit, log_dir=str(self.logger.dir))
        t0 = None
        try:
            if self._prepare(rep):
                t0 = self.clock.now()
                self._deadline = t0 + self.time_limit
                self._event('start', t0=t0, deadline=self._deadline, dry_run=self.dry_run,
                            required=self.required, order=self.order)
                self._loop_sort(rep)
                rep.elapsed = self.clock.now() - t0
                if rep.stop_reason != 'done' and self._fail_streak > 0 and not self.fault_detail:
                    # The round ended while the camera / perception kept failing.
                    self.fault_detail = 'observation failures: %s (%d in a row)' % (
                        self._last_obs_failure, self._fail_streak)
                if rep.stop_reason not in ('fault', 'aborted') and self.return_home in (
                        'observe', 'home'):
                    m = self._go_home() if self.return_home == 'home' else self._go_observe()
                    if not m[0]:
                        _LOG.warning('could not return to the %s pose: %s', self.return_home,
                                     m[1])
        except Exception as e:  # noqa: B902 - a bug must stop the arm, not crash silently
            _LOG.exception('runner crashed')
            rep.stop_reason = 'fault'
            self.fault_detail = 'exception: %s' % e
            if t0 is not None:
                rep.elapsed = self.clock.now() - t0
            try:
                self.arm.stop()
            except Exception:  # noqa: B902
                _LOG.exception('arm.stop() raised')
        rep.ok = (rep.stop_reason == 'done' and rep.placed >= rep.required
                  and rep.elapsed <= self.time_limit and not self.dry_run)
        summary = {'task': rep.task, 'ok': rep.ok, 'placed': rep.placed,
                   'required': rep.required, 'attempts': rep.attempts,
                   'elapsed': round(rep.elapsed, 3), 'time_limit': rep.time_limit,
                   'stop_reason': rep.stop_reason, 'fault_detail': self.fault_detail,
                   'dry_run': self.dry_run, 'picks': [pick_summary(p) for p in rep.picks]}
        self._event('report', **summary)
        try:
            self.logger.save_json('report.json', summary)
        except Exception:  # noqa: B902
            _LOG.exception('could not save report.json')
        _LOG.info('%s finished: %s, placed %d/%d in %.1f s', rep.task, rep.stop_reason,
                  rep.placed, rep.required, rep.elapsed)
        return rep

    # --------------------------------------------------------------- prepare
    def _prepare(self, rep: TaskReport) -> bool:
        ok, msgs = self_check(self.cfg, self.task, self.arm, self.camera, self.mode,
                              extrinsics=self._ext_arg, clock=self.clock)
        for m in msgs:
            (_LOG.warning if not m.startswith('[ OK ]') else _LOG.info)('self check %s', m)
        self._event('self_check', ok=ok, mode=self.mode, messages=msgs)
        if not ok and self.mode == 'real':
            return self._refuse(rep, 'self_check', 'self check failed')
        try:
            if self._ext_arg is not None:
                self.ext = self._ext_arg
            else:
                from ..core.extrinsics import extrinsics_from_cfg
                self.ext = extrinsics_from_cfg(self.cfg)
        except (ImportError, ValueError) as e:
            return self._refuse(rep, 'config', 'extrinsics: %s' % e)
        if self.ext is None:
            return self._refuse(rep, 'config', 'no extrinsics')
        if not self.class_to_zone:
            return self._refuse(rep, 'config', 'class_to_zone is empty')
        if self.required <= 0:
            return self._refuse(rep, 'config', 'required must be > 0')
        self.arm.set_abort_event(self.operator.abort_event)
        if hasattr(self.operator, 'set_estop_callback'):
            self.operator.set_estop_callback(self.arm.emergency_stop)
        if hasattr(self.operator, 'start'):
            self.operator.start()               # p / space / e work from the first motion on
        if not self._healthy():
            return self._refuse(rep, 'fault', 'arm not healthy before start')
        held = self._holding_at_start()
        if held is not None:
            return self._refuse(rep, held[0], held[1])
        ok_up, err = self._lift_if_low()
        if not ok_up:
            return self._refuse(rep, 'aborted' if err == 'aborted' else 'fault',
                                'cannot go up to the transit height: %s' % err)
        ok_move, err = self._go_observe()
        if not ok_move:
            return self._refuse(rep, 'aborted' if err == 'aborted' else 'fault',
                                'cannot reach the observe pose: %s' % err)
        g = self.arm.gripper(self.max_width, wait=True)
        if g.error and g.error != 'timeout':
            if g.error == 'aborted' or self._aborted():
                return self._refuse(rep, 'aborted', 'aborted before start')
            return self._refuse(rep, 'fault', 'gripper open failed: %s %s' % (g.error, g.detail))
        if self.mode == 'real':
            # The self check could not test the wrist camera's view away from here.
            ok_view, msg = self._check_view_at_observe()
            if self._aborted():
                return self._refuse(rep, 'aborted', 'aborted before start')
            if not ok_view:
                return self._refuse(rep, 'fault',
                                    'pick area not visible from the observe pose: ' + msg)
        self._report_view_coverage()
        if self.start_on_enter:
            # Only an Enter typed after the prompt starts the clock.
            started = self.operator.wait_start(discard_earlier=True)
        else:
            started = not self._aborted()
        if not started:
            return self._refuse(rep, 'aborted', 'aborted before start')
        return True

    def _fresh_width(self) -> Optional[float]:
        """Return the measured gripper width (m) when fresh, else None."""
        try:
            s = self.arm.gripper_width()
        except Exception:  # noqa: B902
            _LOG.exception('gripper_width() raised')
            return None
        if s is None or s.value is None:
            return None
        if self.clock.now() - float(s.stamp) > self.skill.feedback_max_age:
            return None
        w = float(s.value)
        return w if math.isfinite(w) else None

    def _holding_at_start(self) -> Optional[Tuple[str, str]]:
        """
        Return (stop reason, detail) when the gripper holds an object, else None.

        After a fault or an abort while holding, the observe move would carry the
        object and the gripper would open there. A width between empty_width and
        max_width - HOLD_OPEN_MARGIN is either an object or an approach opening
        left by an abort, so the gripper closes by PROBE_CLOSE: an empty one
        follows, one on an object stops. Only the fingers move.
        """
        w = self._fresh_width()
        if w is None or not self.skill.empty_width < w < self.max_width - HOLD_OPEN_MARGIN:
            return None
        target = max(0.0, w - PROBE_CLOSE)
        g = self.arm.gripper(target, force=self.skill.force, wait=True)
        w2 = g.width if (g.width is not None and math.isfinite(float(g.width))) else None
        if w2 is None:
            w2 = self._fresh_width()
        self._event('gripper_probe', width=round(w, 4), target=round(target, 4),
                    result=None if w2 is None else round(w2, 4), error=g.error)
        if g.error == 'aborted' or self._aborted():
            return 'aborted', 'aborted before start'
        if g.error and g.error != 'timeout':
            return 'fault', 'gripper check failed: %s %s' % (g.error, g.detail)
        if w2 is None:
            return 'fault', 'no gripper feedback after the gripper check'
        if w2 > target + PROBE_TOL:
            return 'config', ('gripper is closed on something (%.3f m): remove the object, '
                              'then restart' % w2)
        return None

    def _lift_if_low(self) -> Tuple[bool, str]:
        """
        Go straight up to the transit height when the TCP starts below it.

        A restart after a fault can begin low beside objects or inside a bin,
        where the joint-space move to the observe pose could sweep through them.
        Returns (ok, error); a lift that SafeArm refuses (nothing moved) falls
        back to the observe move as before.
        """
        try:
            s = self.arm.get_tcp_pose()
        except Exception:  # noqa: B902
            _LOG.exception('get_tcp_pose() raised')
            return True, ''
        if s is None or s.value is None:
            return True, ''
        T = np.asarray(s.value, dtype=float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            return True, ''
        table = 0.0 if self.expected_table_z is None else self.expected_table_z
        try:
            tz = transit_height(self.cfg, list(self.zones.values()), table)
        except ValueError:
            return True, ''
        if float(T[2, 3]) >= tz - 0.005:
            return True, ''
        T_up = T.copy()
        T_up[2, 3] = tz
        from .reach import reach_adjust
        T2, tilt, why = reach_adjust(self.arm, T_up)
        if T2 is None:
            _LOG.warning('TCP z %.3f m is below the transit height %.3f m but the straight lift '
                         'is not possible (%s)', float(T[2, 3]), tz, why)
            self._event('start_lift', ok=False, error='rejected', reason=why)
            return True, ''
        m = self.arm.move_tcp(T2, linear=True)
        self._event('start_lift', ok=m.ok, error=m.error, from_z=round(float(T[2, 3]), 4),
                    to_z=round(tz, 4), tilt_deg=round(tilt, 1))
        if m.ok or m.error in ('rejected', 'unreachable'):
            return True, ''
        return False, m.error or 'unknown'

    def _check_view_at_observe(self) -> Tuple[bool, str]:
        """Eye-in-hand: check the pick area on a fresh frame at the observe pose."""
        from .selfcheck import check_pick_area
        settle_until = float(self._t_arrived) + self.settle_s
        wait = settle_until - self.clock.now()
        if wait > 0:
            self.clock.sleep(wait)
        ok, msg = False, ''
        for _try in range(VIEW_CHECK_TRIES):        # one bad frame does not refuse the run
            try:
                ok, msg = check_pick_area(self.cfg, self.camera, self.ext, self.arm,
                                          newer_than=settle_until)
            except Exception as e:  # noqa: B902 - a failed check refuses the run
                _LOG.exception('check_pick_area raised')
                ok, msg = False, 'check raised: %s' % e
            if ok or self._aborted():
                break
        (_LOG.info if ok else _LOG.error)('view at the observe pose: %s', msg)
        self._event('view_check', ok=ok, message=msg)
        return ok, msg

    def _refuse(self, rep: TaskReport, reason: str, detail: str) -> bool:
        rep.stop_reason = reason
        self.fault_detail = detail
        _LOG.error('run not started: %s (%s)', reason, detail)
        self._event('refused', reason=reason, detail=detail)
        return False

    # ---------------------------------------------------------------- helpers
    def _event(self, kind: str, **fields: Any) -> None:
        try:
            self.logger.event(kind, **fields)
        except Exception:  # noqa: B902 - logging never stops the run
            _LOG.exception('logger.event failed')

    def _healthy(self) -> bool:
        try:
            h = self.arm.health()
        except Exception:  # noqa: B902
            _LOG.exception('arm.health() raised')
            return False
        if h is None or not h.ok:
            self.fault_detail = 'arm unhealthy: %s' % (getattr(h, 'reason', 'no status'))
            return False
        return True

    def _aborted(self) -> bool:
        return self.operator.abort_event.is_set() or self.arm.abort_event.is_set()

    def _remaining(self) -> float:
        return self._deadline - self.clock.now()

    def _stop_check(self) -> Optional[str]:
        """Return a stop reason that applies before starting new work, else None."""
        if self._aborted():
            return 'aborted'
        if self._remaining() < self.min_time:
            return 'timeout'
        if not self._healthy():
            return 'fault'
        return None

    def _go_home(self) -> Tuple[bool, str]:
        """Go straight up when low, then to robot.poses.home_joints (default zero); (ok, error)."""
        q = self.cfg.get('robot.poses.home_joints', None) or [0.0] * 6
        ok, err = self._lift_if_low()
        if not ok:
            return False, 'going up first: %s' % err
        m = self.arm.move_joints([float(v) for v in q])
        self._event('home', ok=m.ok, error=m.error, joints=[round(float(v), 4) for v in q])
        return m.ok, m.detail or m.error

    def _go_observe(self, view: int = 0) -> Tuple[bool, str]:
        """Move to observe view `view` (taught joints, else observe_tcp); (ok, error)."""
        qj = self.views[view] if self.views else None
        if qj is not None:
            m = self.arm.move_joints([float(v) for v in qj])
        else:
            T = observe_tcp_pose(self.cfg)
            if T is None:
                return False, 'no observe pose configured'
            from .reach import reach_adjust
            T2, _tilt, why = reach_adjust(self.arm, T)
            if T2 is None:
                return False, 'rejected'
            m = self.arm.move_tcp(T2, linear=False)
        self._event('observe_move', ok=m.ok, error=m.error, duration=round(m.duration, 3),
                    view=view)
        if m.ok:
            self._at_observe = True
            self._at_view = view
            self._t_arrived = self.clock.now()
            return True, ''
        self._at_observe = False
        return False, m.error or 'unknown'

    # ------------------------------------------------------------ observation
    def _observe(self, view: int = 0) -> Tuple[str, Optional[PerceptionResult], Any]:
        """
        Capture at observe view `view` (moving there first when the arm is elsewhere).

        Returns (status, result, frame) with status 'ok', 'failed' (no usable
        frame, no flange pose, perception error or a first failed observe move;
        the reason is in _last_obs_failure), 'fault' or 'aborted'.
        """
        if not (self._at_observe and self._at_view == view):
            ok, err = self._go_observe(view)
            if not ok:
                if err == 'aborted' or self._aborted():
                    return 'aborted', None, None
                self._observe_failures += 1
                if is_fault_motion_error(err) or self._observe_failures >= 2:
                    self.fault_detail = 'observe move failed: %s' % err
                    return 'fault', None, None
                return self._obs_failed('observe move failed: %s' % err)
        self._observe_failures = 0
        settle_until = float(self._t_arrived) + self.settle_s
        wait = settle_until - self.clock.now()
        if wait > 0:
            self.clock.sleep(wait)
        newer = settle_until if self._last_stamp is None else max(settle_until, self._last_stamp)
        from ..camera.utils import capture_averaged
        try:
            frame = capture_averaged(self.camera, self.n_avg, newer_than=newer,
                                     timeout=self.frame_timeout)
        except Exception as e:  # noqa: B902 - camera trouble = failed observation
            _LOG.exception('capture failed')
            return self._obs_failed('capture error: %s' % e)
        if frame is None:
            return self._obs_failed('no frame')
        now = self.clock.now()
        if not float(frame.stamp) > newer or now - float(frame.stamp) > self.max_frame_age:
            return self._obs_failed('stale frame', stamp=frame.stamp, now=now, newer_than=newer)
        self._last_stamp = float(frame.stamp)
        extra = (frame.meta or {}).get('extra') or {}
        if frame.source == 'file' and extra.get('T_base_flange') is not None:
            # Replay: the recorded flange pose belongs to this frame, not the sim arm.
            T_bf = np.asarray(extra['T_base_flange'], dtype=float)
        else:
            s = self.arm.get_flange_pose()
            if s is None or float(s.stamp) < float(self._t_arrived):
                return self._obs_failed('no flange pose at capture')
            T_bf = np.asarray(s.value, dtype=float)
        try:
            from ..core.extrinsics import camera_pose_in_base
            T_bc = camera_pose_in_base(self.ext, T_bf, self.T_flange_tcp)
            result = self.perception.process(frame, T_bc, self.zones)
        except Exception as e:  # noqa: B902 - a perception bug = failed observation
            _LOG.exception('perception failed')
            return self._obs_failed('perception error: %s' % e)
        self._obs += 1
        self._fail_streak = 0
        warnings = list(result.warnings)
        table_warning = self._update_table_z(result)
        if table_warning:
            warnings.append(table_warning)
        self._event('observation', ok=True, n=self._obs, stamp=frame.stamp,
                    detections=[d.summary() for d in result.detections],
                    warnings=warnings)
        return 'ok', result, frame

    def _obs_failed(self, reason: str, **fields: Any) -> Tuple[str, None, None]:
        self._fail_streak += 1
        self._last_obs_failure = reason
        self._event('observation', ok=False, reason=reason, failures=self._fail_streak, **fields)
        return 'failed', None, None

    def _pause_after_failure(self) -> None:
        """Wait before the next try after a failed observation (longer after many)."""
        dt = self.retry_s
        if self._fail_streak >= FAIL_BACKOFF_AFTER:
            dt = max(dt, FAIL_BACKOFF_S)
        self._event('decision', action='observe_retry', failures=self._fail_streak,
                    reason=self._last_obs_failure, wait=round(dt, 3))
        self.clock.sleep(dt)

    def _update_table_z(self, result: PerceptionResult) -> str:
        """
        Give the perceived table z to SafeArm (lower-bounded by expected_z).

        Returns a warning when the perceived table differs from
        perception.table.expected_z by more than 10 mm, else ''.
        """
        z = None
        poly = self.cfg.get('perception.roi.pick_polygon', None) or []
        if result.table is not None and result.table.frame == 'base' and len(poly) >= 3:
            cx = float(np.mean([p[0] for p in poly]))
            cy = float(np.mean([p[1] for p in poly]))
            try:
                z = result.table.z_at(cx, cy)
            except ValueError:
                z = None
        if z is None:
            zs = [detection_table_z(d) for d in result.valid_detections()]
            zs = [v for v in zs if v is not None]
            z = float(np.median(zs)) if zs else None
        if z is None or not math.isfinite(z):
            return ''
        warning = table_z_warning(z, self.expected_table_z)
        if warning and not self._table_warned:
            self._table_warned = True
            _LOG.warning('%s', warning)
        if hasattr(self.arm, 'set_table_z'):
            self.arm.set_table_z(table_floor_z(z, self.expected_table_z))
        return warning

    def _candidates(self, result: PerceptionResult) -> List[Detection]:
        """Return valid, classified, mapped detections that pass the target filter."""
        return self._candidates_from(result.valid_detections())

    def _candidates_from(self, dets: Sequence[Detection]) -> List[Detection]:
        out: List[Detection] = []
        skipped = []
        for d in dets:
            zone = self.class_to_zone.get(d.label)
            if d.label in ('', 'unknown'):
                why = 'unknown_label'
            elif zone is None:
                why = 'unmapped_label'
            elif zone not in self.zones:
                why = 'zone_missing'
            elif self.order and d.label not in self.order:
                why = 'not_a_target'
            elif not _has_xy(d):
                why = 'no_geometry'
            elif self._partial(d):
                why = 'partial_view'
            else:
                out.append(d)
                continue
            skipped.append({'label': d.label, 'reason': why, 'track_id': d.track_id,
                            'center': d.summary()['center_base']})
        if skipped:
            self._event('decision', action='skip', skipped=skipped)
        return out

    def _partial(self, det: Detection) -> bool:
        """Return True for an elongated class seen much shorter than its nominal length."""
        nom = self.nominal.get(det.label)
        if nom is None or nom[0] < CAPSULE_RATIO * nom[1] or det.length is None:
            return False
        return float(det.length) < self.partial_ratio * nom[0]

    def _save_image(self, frame, result: Optional[PerceptionResult],
                    chosen: Optional[Detection]) -> None:
        if not self.save_images or frame is None:
            return
        base = result.debug_image if result is not None and result.debug_image is not None \
            else frame.color
        if base is None:
            return
        try:
            import cv2
            img = np.array(base, dtype=np.uint8, copy=True)
            if chosen is not None:
                x0, y0, x1, y1 = [int(v) for v in chosen.bbox]
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 3)
                cv2.putText(img, 'TARGET %s' % chosen.label, (x0, max(15, y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            t_run = self.time_limit - self._remaining()
            cv2.putText(img, '%s obs %d t=%.1fs' % (self.task, self._obs, t_run),
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.logger.save_image('obs_%03d.jpg' % self._obs, img)
        except Exception:  # noqa: B902 - images are diagnostics only
            _LOG.exception('could not save the observation image')

    # --------------------------------------------------------- tracks / picks
    def _track(self, det: Detection, others: Sequence[Detection] = ()) -> Dict[str, Any]:
        """
        Return the attempt record for the object at det's position (created on demand).

        Matches the record within match_dist; else (a failed grasp may have pushed
        the object) the nearest record with det's label within PUSHED_MATCH_DIST
        whose last attempt failed and whose old position is empty in this
        observation (others: its valid detections).
        """
        x, y = _xy(det)
        best, best_d = None, self.match_dist
        for t in self._tracks:
            d = math.hypot(t['xy'][0] - x, t['xy'][1] - y)
            if d <= best_d:
                best, best_d = t, d
        if best is None:
            best = self._pushed_track(det, others)
        if best is None:
            best = {'xy': (x, y), 'label': det.label, 'attempts': 0, 'blacklisted': False,
                    'reason': '', 'infeasible': 0, 'last_failed': False}
            self._tracks.append(best)
        return best

    def _pushed_track(self, det: Detection,
                      others: Sequence[Detection]) -> Optional[Dict[str, Any]]:
        x, y = _xy(det)
        seen = [_xy(o) for o in others if o is not det and o.label == det.label and _has_xy(o)]
        best, best_d = None, PUSHED_MATCH_DIST
        for t in self._tracks:
            if t['label'] != det.label or not t.get('last_failed'):
                continue
            tx, ty = t['xy']
            if any(math.hypot(tx - ox, ty - oy) <= self.match_dist for ox, oy in seen):
                continue                        # its object is still where it was
            d = math.hypot(tx - x, ty - y)
            if d <= best_d:
                best, best_d = t, d
        if best is not None:
            self._event('decision', action='track_moved', label=det.label,
                        from_xy=[round(v, 4) for v in best['xy']],
                        to_xy=[round(x, 4), round(y, 4)],
                        attempts=best['attempts'], blacklisted=best['blacklisted'])
        return best

    def _blacklist(self, track: Dict[str, Any], reason: str) -> None:
        track['blacklisted'] = True
        track['reason'] = reason
        self._event('decision', action='blacklist', label=track['label'],
                    xy=[round(v, 4) for v in track['xy']], reason=reason,
                    attempts=track['attempts'])

    def _blacklisted_xy(self) -> List[Tuple[float, float]]:
        return [t['xy'] for t in self._tracks if t['blacklisted']]

    def _attempt(self, det: Detection, zone_name: str, rep: TaskReport,
                 neighbors: Sequence[Detection], placement_zone=None,
                 placement_polygon=None, exact_xy=False) -> Tuple[Optional[str], Optional[PickResult]]:
        """Plan and execute one pick. Returns (stop_reason or None, PickResult or None)."""
        track = self._track(det, neighbors)
        if track['blacklisted']:
            # A blacklisted object pushed by its last failed grasp stays blacklisted.
            track['xy'] = _xy(det)
            return None, None
        track['xy'] = _xy(det)
        self._neighbors = list(neighbors)
        from ..perception.grasp import annotate_grasp_neighbors
        annotate_grasp_neighbors(det, neighbors)
        gp = self.planner.plan(det)
        from .grasp_contact import supported_grasp
        gp = supported_grasp(self.planner, self.arm, det, gp)
        if not gp.feasible:
            if gp.reason.startswith(REMEASURE_REASONS):
                # One bad frame can give these: measure again on the next observation.
                track['infeasible'] = track.get('infeasible', 0) + 1
                if track['infeasible'] < INFEASIBLE_OBSERVATIONS:
                    self._event('decision', action='remeasure', label=det.label,
                                reason=gp.reason, infeasible=track['infeasible'])
                    return None, None
            self._blacklist(track, 'grasp infeasible: %s' % gp.reason)
            return None, None
        tops = [float(d.top_z) for d in neighbors if d is not det and d.top_z is not None
                and math.isfinite(float(d.top_z))]
        held = held_offset_of(gp)
        tz = transit_height(self.cfg, list(self.zones.values()), detection_table_z(det), held,
                            max(tops) if tops else None)
        # Lift straight up to the transit height before moving sideways, so the held
        # object does not sweep low over the other objects on the way to the zone.
        gp.T_lift[2, 3] = max(float(gp.T_lift[2, 3]), float(tz))
        lower = self.release_lower.get(det.label, 0.0)
        zone = placement_zone if placement_zone is not None else self.zones.get(zone_name)
        footprint = self._footprint(det)
        occupied = list(self._placed.get(zone_name, []))
        shape = held_shape(det, gp, self.nominal.get(det.label))
        strict = placement_zone is not None
        avoided_poses = []
        place_options = {}
        if strict:
            from .reach import release_workspace_reason
            if shape is None:
                self._blacklist(track, 'place infeasible: missing footprint')
                return None, None
            shape['center_offset'] = np.asarray(det.center_base[:2]) - gp.T_grasp[:2, 3]
            shape['grasp_closing'] = gp.closing_angle
            place_options = dict(strict=True, fixed_xy=exact_xy,
                                 avoid_poses=avoided_poses,
                                 candidate_filter=lambda x, y: not release_workspace_reason(self.arm, (x, y)),
                                 pose_filter=self._strict_place_filter(det, gp, neighbors,
                                                                        placement_polygon))
        pending = [footprint_dims(d, self.nominal.get(d.label)) for d in neighbors
                   if d is not det and self.class_to_zone.get(d.label) == zone_name]
        pending = [p for p in pending if p is not None]
        pp = self.placer.plan(zone, held, tz, footprint=footprint, occupied=occupied,
                              center_first=self._center_first(zone_name), held=shape,
                              pending=pending, lower=lower,
                              near_center=self.center_labels.get(det.label), **place_options)
        place_info = dict(self.placer.last_info)
        if not pp.feasible:
            self._blacklist(track, 'place infeasible: %s' % pp.reason)
            return None, None
        from .reach import adjust_plan_poses
        # Checks every pose (SafeArm workspace + IK) and leans high poses outward
        # when joint5 needs it (free-space poses up to max_tilt_free_deg); free-space
        # poses still out of reach are lowered to their floors; a pose that stays
        # unreachable blacklists the object.
        report: Dict[str, Any] = {}
        ok, why, tilts = adjust_plan_poses(self.arm, gp, pp, self._floors(det, gp, pp, held),
                                           self.free_tilt, report)
        if ok and strict:
            ok, why = self._adjust_strict_release(det, gp, pp, neighbors,
                                                 placement_polygon, place_info['xy'], held)
        avoid: List[Tuple[float, float]] = []
        stacked = (det.extra or {}).get('_stack_support') is not None
        retry_limit = STACK_RELEASE_RETRIES if stacked else RELEASE_RETRIES
        while not ok and footprint is not None and len(avoid) + len(avoided_poses) < retry_limit:
            # That release position is out of reach: the next free one, not on top of
            # an object already in the zone unless nothing else is left.
            if strict and (exact_xy or stacked):
                avoided_poses.append((*place_info['xy'], place_info['closing_angle']))
            elif strict:
                # A region can use another center after the final leaned pose
                # fails clearance. Reserve yaw retries for a fixed point.
                avoid.append(tuple(place_info['xy']))
            else:
                avoid.append((float(pp.T_release[0, 3]), float(pp.T_release[1, 3])))
            pc = self.placer.plan(zone, held, tz, footprint=footprint, occupied=occupied,
                                  center_first=self._center_first(zone_name), held=shape,
                                  pending=pending, avoid=avoid, lower=lower,
                                  near_center=self.center_labels.get(det.label), **place_options)
            if not pc.feasible or (not strict and any(math.hypot(float(pc.T_release[0, 3]) - a,
                                                 float(pc.T_release[1, 3]) - b) < 1e-6
                                      for a, b in avoid)):
                break
            pp = pc
            place_info = dict(self.placer.last_info)
            ok, why, tilts = adjust_plan_poses(self.arm, gp, pp,
                                               self._floors(det, gp, pp, held),
                                               self.free_tilt, report)
            if ok and strict:
                ok, why = self._adjust_strict_release(det, gp, pp, neighbors,
                                                     placement_polygon, place_info['xy'], held)
        if not ok and footprint is not None and not strict:
            # Last resort: the zone center, on a flat card only when nothing lies there.
            pc = self.placer.plan(zone, held, tz, occupied=occupied, held=shape,
                                  center_only=True, lower=lower)
            fpc = self.placer.last_info.get('footprint_clearance')
            if pc.feasible and (zone.wall_height > 0.0 or fpc is None or fpc >= 0.0):
                ok_c, why_c, tilts_c = adjust_plan_poses(self.arm, gp, pc,
                                                         self._floors(det, gp, pc, held),
                                                         self.free_tilt, report)
                if ok_c:
                    ok, why, tilts, pp = ok_c, why_c, tilts_c, pc
                    place_info = dict(self.placer.last_info)
        if avoid:
            self._event('decision', action='release_moved', label=det.label, ok=ok,
                        unreachable=[[round(a, 4), round(b, 4)] for a, b in avoid],
                        tier=place_info.get('tier'))
        if not ok:
            self._blacklist(track, 'unreachable %s' % why)
            return None, None
        if report.get('lowered'):
            self._event('decision', action='lowered', label=det.label, lowered=report['lowered'])
        if any(v > 0 for v in tilts.values()):
            self._event('decision', action='tilt', label=det.label, tilt_deg=tilts)
        stop = self._stop_check()
        if stop is not None:
            return stop, None
        track['attempts'] += 1
        rep.attempts += 1
        self._event('decision', action='pick', label=det.label, zone=zone_name,
                    target=det.summary(), attempt=track['attempts'],
                    closing_angle=round(gp.closing_angle, 4),
                    expected_width=float(gp.expected_width), open_width=float(gp.open_width),
                    close_width=float(gp.close_width),
                    grasp_xyz=[round(float(v), 4) for v in gp.T_grasp[:3, 3]],
                    release_xyz=[round(float(v), 4) for v in pp.T_release[:3, 3]],
                    transit_z=round(float(pp.T_pre[2, 3]), 4), place=place_info,
                    grasp=det.extra.get('grasp', {}) if det.extra else {},
                    remaining=round(self._remaining(), 2))
        watcher = self._live_watcher(det)
        self._live_dets = []
        self._live_mark += 1                    # confirmations of earlier picks expire
        if watcher is not None:
            self.skill.on_stage = watcher.stage
        try:
            self._held_recovery_context = None
            res = self.skill.execute(gp, pp, dry_run=self.dry_run)
        finally:
            if watcher is not None:
                self.skill.on_stage = None
                self._live_dets = watcher.finish()
                self._event('decision', action='live_watch', label=det.label,
                            n=len(self._live_dets), **watcher.stats)
        self._save_held_recovery(det, gp, pp, res)
        if self._held_recovery_context is not None and strict:
            self._held_recovery_context.update(placement_zone=copy.deepcopy(zone),
                                              placement_polygon=copy.deepcopy(placement_polygon),
                                              exact_xy=bool(exact_xy),
                                              placement_avoid=[list(place_info['xy'])])
        self._at_observe = False
        if res.detail.get('fault') or res.detail.get('holding'):
            # A held object that could not be put down is a fault as well.
            self.fault_detail = 'pick %s at stage %s: %s' % (res.label, res.stage, res.error)
            return ('aborted' if self._aborted() else 'fault'), res
        if res.ok:
            self._tracks.remove(track)
            x_rel, y_rel = (float(v) for v in pp.T_release[:2, 3])
            h = float(det.height) if det.height is not None and math.isfinite(
                float(det.height)) else 0.0
            axis = place_info.get('axis') if shape is not None else None
            self._placed_labels.setdefault(zone_name, []).append(det.label)
            self._placed.setdefault(zone_name, []).append(
                (x_rel, y_rel, 0.0 if footprint is None else footprint, h,
                 shape['length'] if shape else None, shape['width'] if shape else None, axis))
        elif self.dry_run and res.stage == 'dry_run_done':
            self._blacklist(track, 'dry_run_visited')
        else:
            track['last_failed'] = True
            self._label_failures[det.label] = self._label_failures.get(det.label, 0) + 1
            if track['attempts'] >= self.max_attempts:
                self._blacklist(track, 'max_attempts (%s)' % res.error)
        return None, res

    @property
    def has_held_recovery(self) -> bool:
        """Whether a measured grasp from this runner can continue its placement."""
        return self._held_recovery_context is not None

    def _save_held_recovery(self, det, gp, pp, result) -> None:
        """Remember a measured pick-only grasp or a recoverable held failure."""
        errors = [s.get('error') for s in result.detail.get('steps', ()) if s.get('error')]
        allowed = ('rejected', 'unreachable', 'timeout', 'stalled', 'scene_changed',
                   'placement_changed', 'release_clearance_unconfirmed', 'held_anchor_unconfirmed')
        width = result.grasp_width
        picked = result.ok and result.detail.get('pick_only') and not result.error
        if (not result.detail.get('holding')
                or (not picked and result.error not in allowed + ('holding_object',))
                or any(error not in allowed for error in errors)
                or width is None or not math.isfinite(float(width))
                or width < self.skill.empty_width):
            self._held_recovery_context = None
            return
        self._held_recovery_context = {
            'det': copy.deepcopy(det), 'grasp': copy.deepcopy(gp),
            'place': copy.deepcopy(pp), 'width': float(width), 'zone': pp.zone,
            'avoid': [(float(pp.T_release[0, 3]), float(pp.T_release[1, 3]))]}

    def _held_transfer_feedback(self, det):
        context = self._held_recovery_context
        if context is None:
            return None, None, 'no recoverable held grasp from the last pick'
        if self._aborted() or not self._healthy():
            return None, None, 'arm stopped or unhealthy'
        original = context['det']
        if (det is None or det.label != original.label or not _has_xy(det)
                or np.linalg.norm(np.asarray(det.center_base[:2]) -
                                  np.asarray(original.center_base[:2])) > .02):
            return None, None, 'held transfer target changed'
        try:
            width = float(self.skill._read_width())
        except (TypeError, ValueError):
            return None, None, 'held grasp feedback missing or changed'
        if (not math.isfinite(width) or width < self.skill.empty_width or width > self.max_width
                or abs(width - context['width']) > self.skill.width_tolerance):
            return None, None, 'held grasp feedback missing or changed'
        return context, width, ''

    def plan_held_transfer(self, det: Detection, neighbors: Sequence[Detection],
                           placement_zone, placement_polygon, exact_xy=True,
                           cancelled=None, progress=None, max_attempts=None):
        """Check a temporary release without changing the original recovery goal.

        The caller owns the outer transaction and its restoration obligations.
        This local bundle retains the measured grip; no pick planner, motion or
        gripper command runs here. The caller must activate this destination's
        observation callback before executing it.
        """
        context, width, why = self._held_transfer_feedback(det)
        if context is None:
            return None, why
        if placement_zone is None:
            return None, 'temporary release destination is unavailable'
        original = copy.deepcopy(context['det'])
        grasp = copy.deepcopy(context['grasp'])
        placeholder = copy.deepcopy(context['place'])
        placeholder.zone = placement_zone.name
        place, why = self.replan_release(
            original, grasp, placeholder, width, neighbors,
            placement_zone=placement_zone, placement_polygon=placement_polygon,
            exact_xy=bool(exact_xy), cancelled=cancelled,
            progress=progress, max_attempts=max_attempts)
        if place is None:
            return None, why
        return {'det': original, 'grasp': grasp, 'place': place, 'width': width,
                'placement_zone': copy.deepcopy(placement_zone),
                'placement_polygon': copy.deepcopy(placement_polygon),
                'exact_xy': bool(exact_xy), '_context_id': id(context)}, ''

    def execute_held_transfer(self, bundle):
        """Release the same verified grasp into a locally checked temporary slot.

        A failed held continuation keeps its measured grip available. A release
        expires that grip; it does not count as completing the original goal.
        """
        if not isinstance(bundle, dict):
            return None, 'invalid held transfer bundle'
        context, width, why = self._held_transfer_feedback(bundle.get('det'))
        if context is None:
            return None, why
        if bundle.get('_context_id') != id(context):
            return None, 'held transfer context expired'
        try:
            grasp, place = bundle['grasp'], bundle['place']
            saved = context['grasp']
            same_grip = (np.allclose(grasp.T_grasp, saved.T_grasp, atol=1e-7, rtol=0)
                         and abs(float(grasp.closing_angle) - float(saved.closing_angle)) < 1e-7
                         and grasp.target.label == context['det'].label)
            same_width = abs(width - float(bundle['width'])) <= self.skill.width_tolerance
            same_zone = place.zone == bundle['placement_zone'].name
        except (KeyError, TypeError, ValueError, AttributeError):
            return None, 'invalid held transfer bundle'
        if not same_grip:
            return None, 'held transfer would change the held grasp transform'
        if not same_width:
            return None, 'held grasp feedback changed after planning'
        if not same_zone:
            return None, 'temporary release destination changed'
        # The skill's observation hook may refine the release plan. Keep the
        # offered bundle unchanged, and save the actually executed plan below.
        grasp, place = copy.deepcopy(grasp), copy.deepcopy(place)
        result = self.skill.place_held(grasp, place, width)
        self._save_held_recovery(context['det'], grasp, place, result)
        if self._held_recovery_context is not None:
            self._held_recovery_context.update(
                placement_zone=copy.deepcopy(bundle['placement_zone']),
                placement_polygon=copy.deepcopy(bundle['placement_polygon']),
                exact_xy=bool(bundle['exact_xy']))
        if result.ok:
            self.fault_detail = ''
        return result, ''

    def retry_held_placement(self, det: Detection, zone_name: str,
                             neighbors: Sequence[Detection] = (),
                             avoid: Sequence[Sequence[float]] = (),
                             placement_zone=None, placement_polygon=None, exact_xy=False
                             ) -> Tuple[Optional[PickResult], str]:
        """Explicitly replan the same held object's placement in the same zone.

        This opt-in seam does not change competition retry rules. It accepts
        measured geometry saved by the failed pick, never arbitrary model TCP
        coordinates. Abort/health/held-width checks and the full SafeArm path
        checks remain active; a refusal sends no motion or gripper commands.
        """
        from .reach import adjust_plan_poses, release_workspace_reason

        context = self._held_recovery_context
        if context is None:
            return None, 'no recoverable held grasp from the last pick'
        if self._aborted() or not self._healthy():
            return None, 'arm stopped or unhealthy'
        original = context['det']
        saved_zone = context.get('placement_zone')
        if (zone_name != context['zone'] or (saved_zone is None and zone_name not in self.zones)
                or det.label != original.label or not _has_xy(det)
                or np.linalg.norm(np.asarray(det.center_base[:2])
                                  - np.asarray(original.center_base[:2])) > 0.02):
            return None, 'held recovery target or destination changed'
        if saved_zone is not None:
            if (placement_zone is None or placement_zone.name != saved_zone.name
                    or not np.array_equal(placement_zone.center, saved_zone.center)
                    or tuple(placement_zone.size) != tuple(saved_zone.size)
                    or bool(exact_xy) != context.get('exact_xy')
                    or not np.array_equal(placement_polygon, context.get('placement_polygon'))):
                return None, 'held recovery target or destination changed'
        width = self.skill._read_width()
        if (width is None or width < self.skill.empty_width
                or abs(width - context['width']) > self.skill.width_tolerance):
            return None, 'held grasp feedback missing or changed'
        gp = copy.deepcopy(context['grasp'])
        self._neighbors = list(neighbors)
        held = held_offset_of(gp)
        tops = [float(d.top_z) for d in neighbors if d is not det and d.top_z is not None
                and math.isfinite(float(d.top_z))]
        transit = transit_height(self.cfg, list(self.zones.values()),
                                 detection_table_z(original), held, max(tops) if tops else None)
        occupied = list(self._placed.get(zone_name, ()))
        shape = held_shape(original, gp, self.nominal.get(det.label))
        if shape is not None:
            shape['grip_width'] = max(shape['grip_width'], width)
        strict_options = {}
        avoided_poses = []
        if saved_zone is not None:
            shape['center_offset'] = np.asarray(original.center_base[:2]) - gp.T_grasp[:2, 3]
            shape['grasp_closing'] = gp.closing_angle
            strict_options = dict(strict=True, fixed_xy=exact_xy,
                                  avoid_poses=avoided_poses,
                                  pose_filter=self._strict_place_filter(original, gp, neighbors,
                                                                         placement_polygon))
        stacked = (original.extra or {}).get('_stack_support') is not None
        banned = ([] if exact_xy or stacked else list(context.get('placement_avoid', context['avoid']))) if saved_zone is not None else list(context['avoid'])
        for point in avoid:
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError, IndexError):
                return None, 'invalid avoided release position'
            if not (math.isfinite(x) and math.isfinite(y)):
                return None, 'invalid avoided release position'
            banned.append((x, y))
        pp, why = None, 'no reachable alternative release position'
        for _ in range((STACK_RELEASE_RETRIES if stacked else RELEASE_RETRIES) + 1):
            candidate = self.placer.plan(
                saved_zone if saved_zone is not None else self.zones[zone_name], held, transit, footprint=self._footprint(original),
                occupied=occupied, held=shape, avoid=banned,
                candidate_filter=lambda x, y: not release_workspace_reason(self.arm, (x, y)),
                center_first=self._center_first(zone_name),
                lower=self.release_lower.get(det.label, 0.0), **strict_options)
            xy = tuple(float(v) for v in candidate.T_release[:2, 3])
            if (not candidate.feasible or (saved_zone is None and any(math.hypot(xy[0] - a, xy[1] - b) < 1e-6
                                              for a, b in banned))):
                break
            info = self.placer.last_info
            if any(info.get(k) is not None and info[k] < 0.0
                   for k in ('footprint_clearance', 'finger_clearance', 'edge_margin')):
                banned.append(xy)
                continue
            ok, why, _tilts = adjust_plan_poses(
                self.arm, gp, candidate, self._floors(original, gp, candidate, held),
                self.free_tilt)
            if ok and saved_zone is not None:
                ok, why = self._adjust_strict_release(original, gp, candidate, neighbors,
                                                     placement_polygon, info['xy'], held)
            if ok and not np.allclose(gp.T_grasp, context['grasp'].T_grasp, atol=1e-7, rtol=0):
                return None, 'release planning would change the held grasp transform'
            if ok:
                pp = candidate
                break
            if saved_zone is not None and (exact_xy or stacked):
                avoided_poses.append((*info['xy'], info['closing_angle']))
            elif saved_zone is not None:
                banned.append(tuple(info['xy']))
            else:
                banned.append(xy)
        if pp is None:
            return None, why
        self._event('decision', action='retry_held_placement', label=det.label, zone=zone_name,
                    release_xyz=[round(float(v), 4) for v in pp.T_release[:3, 3]],
                    place=dict(self.placer.last_info),
                    avoided=[[round(a, 4), round(b, 4)] for a, b in banned])
        result = self.skill.place_held(gp, pp, width)
        self._save_held_recovery(original, gp, pp, result)
        if self._held_recovery_context is not None and saved_zone is not None:
            self._held_recovery_context.update(placement_zone=copy.deepcopy(saved_zone),
                                              placement_polygon=copy.deepcopy(placement_polygon),
                                              exact_xy=bool(exact_xy),
                                              placement_avoid=banned + [list(self.placer.last_info['xy'])])
        if self._held_recovery_context is not None:
            self._held_recovery_context['avoid'] = banned + self._held_recovery_context['avoid']
        if result.ok:
            self.fault_detail = ''
            self._placed_labels.setdefault(zone_name, []).append(det.label)
            self._placed.setdefault(zone_name, []).append(
                (float(pp.T_release[0, 3]), float(pp.T_release[1, 3]),
                 self._footprint(original) or 0.0, original.height or 0.0,
                 shape['length'] if shape else None, shape['width'] if shape else None,
                 self.placer.last_info.get('axis') if shape else None))
        return result, ''

    def _release_search_geometry(self, det, gp, pp, width, neighbors,
                                 placement_zone=None, placement_polygon=None):
        """Build the same measured constraints for capacity and release searches."""
        try:
            width = float(width)
        except (TypeError, ValueError):
            return None, 'held grasp width is unavailable'
        if (not math.isfinite(width) or width < self.skill.empty_width
                or width > self.max_width):
            return None, 'held grasp width is invalid'
        zone = placement_zone if placement_zone is not None else self.zones.get(pp.zone)
        if zone is None or zone.name != pp.zone:
            return None, 'release destination changed'
        polygon = np.asarray(zone.corners_xy() if placement_polygon is None
                             else placement_polygon, float)
        if (polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3
                or not np.all(np.isfinite(polygon))):
            return None, 'release support polygon is unavailable'
        planning_grasp = copy.deepcopy(gp)
        planning_grasp.expected_width = max(float(gp.expected_width), width)
        shape = held_shape(det, planning_grasp, self.nominal.get(det.label))
        if shape is None:
            return None, 'held footprint is unavailable'
        shape['grip_width'] = max(shape['grip_width'], width)
        shape['center_offset'] = np.asarray(det.center_base[:2]) - gp.T_grasp[:2, 3]
        shape['grasp_closing'] = gp.closing_angle
        if (det.extra or {}).get('_stack_support') is not None:
            from ..perception.grasp import RELEASE_OPEN_EXTRA
            # Grip feedback measures the narrow contact band of a curved body.
            # A tilted exit needs clearance around its complete measured/nominal
            # horizontal envelope, including the driver's width quantization.
            required = (max(shape['length'], shape['width']) + .010 +
                        RELEASE_OPEN_QUANTIZATION_TOL)
            if required <= min(self.max_width, self.placer.open_width):
                shape['requested_release_opening'] = max(required,
                    min(self.placer.open_width, shape['grip_width'] + RELEASE_OPEN_EXTRA))
        neighbors = list(neighbors)
        self._neighbors = neighbors
        held = held_offset_of(planning_grasp)
        tops = [float(d.top_z) for d in neighbors if d is not det and d.top_z is not None
                and math.isfinite(float(d.top_z))]
        transit = transit_height(self.cfg, list(self.zones.values()), detection_table_z(det),
                                 held, max(tops) if tops else None)
        # Occupancy is rebuilt from current measurements, rather than retaining
        # old placement reservations after the user moved a table object.
        occupied = []
        for other in neighbors:
            if other is det or not _has_xy(other) or _is_stack_base(det, other):
                continue
            dims = footprint_dims(other, self.nominal.get(other.label))
            if dims is None or other.yaw is None:
                return None, 'neighbor footprint is unavailable before release'
            occupied.append((*_xy(other), dims[0], float(other.height or 0.),
                             *dims, float(other.yaw)))
        # Held skill transfers pass exact copies of configured named zones.
        # Preserve that goal contract without assigning it to arbitrary table
        # regions, points or object supports that also use explicit polygons.
        configured = self.zones.get(zone.name)
        completion_zone = (configured if configured is not None
            and np.array_equal(zone.center, configured.center)
            and np.array_equal(zone.size, configured.size)
            and float(zone.yaw) == float(configured.yaw)
            and np.array_equal(polygon, configured.corners_xy()) else None)
        return dict(width=width, zone=zone, polygon=polygon, grasp=planning_grasp,
                    shape=shape, held=held, transit=transit, neighbors=neighbors,
                    occupied=occupied, completion_zone=completion_zone), ''

    def has_release_capacity(self, det, gp, pp, width, neighbors, placement_zone=None,
                             placement_polygon=None, exact_xy=False, cancelled=None,
                             progress=None):
        """Return measured geometric capacity, never an executable placement.

        True permits looking for a better view, not releasing the object. False
        proves no strict geometric candidate was found. None means computation
        or measurements were insufficient; every actual release still goes
        through replan_release and its final pose/path/visibility checks.
        """
        from .reach import release_workspace_reason

        def interrupted():
            return self._aborted() or bool(cancelled is not None and cancelled())

        if interrupted():
            return None, 'aborted' if self._aborted() else 'placement_planning_budget_exceeded'
        if not self._healthy():
            return None, 'arm stopped or unhealthy before release'
        search, why = self._release_search_geometry(
            det, gp, pp, width, neighbors, placement_zone, placement_polygon)
        if search is None:
            return None, why
        if progress is not None:
            progress({'phase': 'release_capacity'})
        previous_info = self.placer.last_info
        try:
            candidate = self.placer.plan(
                search['zone'], search['held'], search['transit'], footprint=self._footprint(det),
                occupied=search['occupied'], held=search['shape'], strict=True,
                fixed_xy=bool(exact_xy),
                pose_filter=self._strict_place_filter(det, search['grasp'], search['neighbors'],
                                                      search['polygon'],
                                                      completion_zone=search['completion_zone'],
                                                      release_opening=search['shape'].get(
                                                          'requested_release_opening')),
                candidate_filter=lambda x, y: not release_workspace_reason(self.arm, (x, y)),
                center_first=self._center_first(search['zone'].name),
                lower=self.release_lower.get(det.label, 0.0), cancelled=interrupted)
        finally:
            # Capacity is observation evidence, never the executable release.
            self.placer.last_info = previous_info
        if interrupted():
            return None, 'aborted' if self._aborted() else 'placement_planning_budget_exceeded'
        if not candidate.feasible:
            from ..perception.grasp import FINGER_OUTER
            low, high = search['polygon'].min(axis=0), search['polygon'].max(axis=0)
            reach = .5 * max(self.placer.open_width,
                            math.hypot(search['shape']['length'], search['shape']['width']))
            for other in search['neighbors']:
                if other is det or other.valid or not _has_xy(other):
                    continue
                dims = footprint_dims(other, self.nominal.get(other.label))
                if dims is None:
                    continue
                radius = .5 * math.hypot(*dims) + reach + FINGER_OUTER
                xy = np.asarray(other.center_base[:2], float)
                if np.all(xy >= low - radius) and np.all(xy <= high + radius):
                    # A cropped silhouette's centroid is not its full body's
                    # measured centre. Keep its conservative obstacle, but let
                    # the caller seek a checked new view instead of treating
                    # this geometry failure as proof the support is full.
                    return None, 'release_occupancy_unconfirmed'
        return bool(candidate.feasible), '' if candidate.feasible else candidate.reason

    def replan_release(self, det, gp, pp, width, neighbors, placement_zone=None,
                       placement_polygon=None, exact_xy=False, observation_filter=None,
                       cancelled=None, progress=None, max_attempts=None, search_state=None):
        """Plan a held release with every geometric, reach and visibility check.

        max_attempts bounds one computation phase; its default retains the
        existing retry count. The physically held grasp is never replanned.
        An optional search_state retains only fully rejected candidates across
        computation slices of the SAME measurements and robot pose. Callers
        must discard it before recapturing, moving or applying an intervention.
        """
        from .reach import adjust_plan_poses, release_workspace_reason

        def interrupted():
            return self._aborted() or bool(cancelled is not None and cancelled())

        def interruption_reason():
            return 'aborted' if self._aborted() else 'placement_planning_budget_exceeded'

        if interrupted():
            return None, interruption_reason()
        if not self._healthy():
            return None, 'arm stopped or unhealthy before release'
        if max_attempts is not None and (type(max_attempts) is not int or max_attempts < 1):
            return None, 'invalid release attempt budget'
        stacked = (det.extra or {}).get('_stack_support') is not None
        attempt_limit = ((STACK_RELEASE_RETRIES if stacked else RELEASE_RETRIES) + 1
                         if max_attempts is None else max_attempts)
        search, why = self._release_search_geometry(
            det, gp, pp, width, neighbors, placement_zone, placement_polygon)
        if search is None:
            return None, why
        width, zone, polygon = search['width'], search['zone'], search['polygon']
        planning_grasp, shape = search['grasp'], search['shape']
        held, transit = search['held'], search['transit']
        neighbors, occupied = search['neighbors'], search['occupied']
        preferred_xy = ()
        if not exact_xy and not stacked:
            rotation = pp.T_release[:3, :3] @ gp.T_grasp[:3, :3].T
            anchor_center = _placement_anchor_center(det, gp)
            cloud = np.asarray((det.extra or {}).get('_placement_points_base'), float)
            if anchor_center is not None:
                center = pp.T_release[:3, 3] + rotation @ (anchor_center - gp.T_grasp[:3, 3])
            elif (cloud.ndim == 2 and cloud.shape[1:] == (3,) and len(cloud) >= 40
                    and np.all(np.isfinite(cloud))):
                carried = (cloud - gp.T_grasp[:3, 3]) @ rotation.T + pp.T_release[:3, 3]
                center = (carried.min(axis=0) + carried.max(axis=0)) / 2
            else:
                table = detection_table_z(det) or 0.0
                source = np.r_[det.center_base[:2], table + float(det.height or 0.0) / 2]
                center = pp.T_release[:3, 3] + rotation @ (source - gp.T_grasp[:3, 3])
            preferred_xy = (tuple(center[:2]),)
        allowed = self._strict_place_filter(det, planning_grasp, neighbors, polygon,
                                            completion_zone=search['completion_zone'],
                                            release_opening=shape.get('requested_release_opening'))
        def measured_allowed(x, y, closing):
            return (not interrupted() and
                    allowed(x, y, closing) and
                    (observation_filter is None or observation_filter(x, y, closing)))
        state = {} if search_state is None else search_state
        avoided = state.setdefault('avoided', [])
        avoided_poses = state.setdefault('avoided_poses', [])
        preferred_xy = tuple(state.get('preferred_xy', preferred_xy))
        why = 'no clear reachable position in the requested destination'
        for attempt in range(attempt_limit):
            if interrupted():
                return None, interruption_reason()
            if progress is not None:
                progress({'phase': 'release_candidates', 'attempt': attempt + 1,
                          'max_attempts': attempt_limit})
            candidate = self.placer.plan(
                zone, held, transit, footprint=self._footprint(det), occupied=occupied,
                held=shape, avoid=avoided, strict=True, fixed_xy=bool(exact_xy),
                avoid_poses=avoided_poses, pose_filter=measured_allowed,
                candidate_filter=lambda x, y: not release_workspace_reason(self.arm, (x, y)),
                center_first=self._center_first(zone.name),
                lower=self.release_lower.get(det.label, 0.0), cancelled=interrupted,
                preferred_xy=preferred_xy)
            if interrupted():
                return None, interruption_reason()
            if not candidate.feasible:
                return None, candidate.reason or why
            info = dict(self.placer.last_info)
            xy = tuple(info['xy'])
            ok, why, _tilts = adjust_plan_poses(
                self.arm, planning_grasp, candidate,
                self._floors(det, planning_grasp, candidate, held), self.free_tilt,
                held_only=True)
            if interrupted():
                return None, interruption_reason()
            if ok:
                ok, why = self._adjust_strict_release(
                    det, planning_grasp, candidate, neighbors, polygon, xy, held,
                    held_only=True, completion_zone=search['completion_zone'])
            if interrupted():
                return None, interruption_reason()
            if ok and observation_filter is not None:
                closing = math.atan2(float(candidate.T_release[1, 1]),
                                     float(candidate.T_release[0, 1]))
                if not observation_filter(*xy, closing, gripper_pose=candidate.T_release,
                                          opening=candidate.open_width):
                    ok, why = False, 'final release footprint is not visible in the fresh image'
            if ok and not np.allclose(planning_grasp.T_grasp, gp.T_grasp, atol=1e-7, rtol=0):
                ok, why = False, 'release planning would change the held grasp transform'
            if (ok and stacked and shape.get('requested_release_opening') is not None
                    and observation_filter is not None):
                ok, why = self._release_retreat_clear(
                    det, planning_grasp, candidate, neighbors, polygon, interrupted)
                if interrupted():
                    return None, interruption_reason()
            if ok:
                if interrupted():
                    return None, interruption_reason()
                if (stacked and shape.get('requested_release_opening') is not None
                        and observation_filter is not None):
                    # The skill accepts this only for these exact poses and
                    # aperture. Never certify a capacity-only or blind plan.
                    candidate.release_opening_checked = dict(width=float(candidate.open_width),
                        release_T=candidate.T_release.copy(), retreat_T=candidate.T_retreat.copy(),
                        support_polygon=polygon.copy(), measured_neighbors=[Detection(
                            other.label, other.score,
                            center_base=None if other.center_base is None else np.asarray(
                                other.center_base, float).copy(),
                            top_z=other.top_z, height=other.height, yaw=other.yaw,
                            length=other.length, width=other.width, valid=other.valid,
                            extra=copy.deepcopy({key: other.extra[key]
                                for key in ('resting_pose', 'table_z') if key in other.extra}))
                            for other in neighbors if other is not det])
                self._event('decision', action='replan_release', label=det.label, zone=zone.name,
                            release_xyz=[round(float(v), 4) for v in candidate.T_release[:3, 3]],
                            place=info, observed_neighbors=len(occupied),
                            changed=not np.allclose(candidate.T_release, pp.T_release, atol=.001),
                            avoided=[[round(float(a), 4), round(float(b), 4)] for a, b in avoided])
                return candidate, ''
            # The placer already established XY support/occupancy geometry.
            # Rejections here come from the leaned full pose, IK, retreat or
            # final camera footprint. Another wrist direction at this same
            # measured point can still work, including in an ordinary region.
            # Keep broad XY exclusions for caller-proven XY-only conflicts.
            avoided_poses.append((*xy, info['closing_angle']))
            # This point has already passed measured XY support/occupancy.
            # Try its remaining directions before rescanning the region grid;
            # the same-measurement continuation retains this bounded ordering.
            preferred_xy = (xy,) + tuple(p for p in preferred_xy
                if np.linalg.norm(np.asarray(p) - xy) > 1e-8)
            preferred_xy = preferred_xy[:8]
            state['preferred_xy'] = preferred_xy
            # Reach adjustment mutates both plans. Every new candidate must use
            # the same physically held grasp, not a failed candidate's anchor.
            planning_grasp = copy.deepcopy(gp)
            planning_grasp.expected_width = max(float(gp.expected_width), width)
        return None, why

    def ready_for_pick(self, go_observe: bool = False) -> Tuple[bool, str]:
        """
        Bring the arm to a state pick_detection()/observe_once() can start from.

        Arms the operator keys (p/space abort, e e-stop), refuses when the gripper holds
        something, lifts to the transit height, optionally goes to the observe pose and
        opens the gripper. Returns (True, '') or (False, why). Nothing here waits for Enter.
        """
        self.arm.set_abort_event(self.operator.abort_event)
        if hasattr(self.operator, 'set_estop_callback'):
            self.operator.set_estop_callback(self.arm.emergency_stop)
        if hasattr(self.operator, 'start'):
            self.operator.start()
        if not self._healthy():
            return False, 'arm not healthy: %s' % (self.fault_detail or 'see log')
        held = self._holding_at_start()
        if held is not None:
            return False, '%s: %s' % held
        ok_up, err = self._lift_if_low()
        if not ok_up:
            return False, 'cannot go up to the transit height: %s' % err
        if go_observe:
            ok_move, err = self._go_observe()
            if not ok_move:
                return False, 'cannot reach the observe pose: %s' % err
        g = self.arm.gripper(self.max_width, wait=True)
        if g.error and g.error != 'timeout':
            return False, 'gripper open failed: %s %s' % (g.error, g.detail)
        return True, ''

    def observe_once(self) -> Tuple[Optional[PerceptionResult], str]:
        """
        Move to the observe pose, capture and run perception once (tools/motion_ctl vision).

        Vision half of the vision -> motion seam: it returns the PerceptionResult whose
        detections pick_detection() takes. Returns (result, '') or (None, why).
        """
        if self.ext is None:
            try:
                from ..core.extrinsics import extrinsics_from_cfg
                self.ext = self._ext_arg if self._ext_arg is not None \
                    else extrinsics_from_cfg(self.cfg)
            except (ImportError, ValueError) as e:
                return None, 'extrinsics: %s' % e
            if self.ext is None:
                return None, 'no extrinsics (camera.extrinsics_file / extrinsics_inline)'
        if not math.isfinite(self._deadline):
            self._deadline = self.clock.now() + 3600.0
        status, result, _frame = self._observe()
        if status != 'ok' or result is None:
            return None, '%s: %s' % (status, self._last_obs_failure or self.fault_detail)
        return result, ''

    def pick_detection(self, det: Detection, zone_name: Optional[str] = None,
                       neighbors: Sequence[Detection] = (), budget_s: float = 3600.0,
                       placement_zone=None, placement_polygon=None, exact_xy=False
                       ) -> Tuple[Optional[PickResult], str]:
        """
        Plan and execute one pick of a detection that did not come from the camera loop.

        This is the seam between vision and motion: the same plan -> reach adjust ->
        PickPlaceSkill path a competition round takes in _attempt(), for one
        Detection built by hand (tools/motion_ctl pick). The caller owns the arm
        (enable, speed, abort event). zone_name defaults to class_to_zone[det.label];
        neighbors are the other objects on the table (transit clearance).

        Returns (PickResult, '') when the skill ran (check .ok), or (None, why) when
        nothing moved: no zone, grasp or place infeasible, unreachable pose, or a stop
        condition (aborted / unhealthy arm).
        """
        zone = placement_zone.name if placement_zone is not None else (zone_name or self.class_to_zone.get(det.label))
        if zone is None or (placement_zone is None and zone not in self.zones):
            return None, ('no zone for label %r (zone=%r, class_to_zone=%s, zones=%s)'
                          % (det.label, zone, self.class_to_zone, sorted(self.zones)))
        if not math.isfinite(self._deadline):
            self._deadline = self.clock.now() + float(budget_s)
        rep = TaskReport(task=self.task, ok=False, required=1, time_limit=self.time_limit,
                         log_dir=str(getattr(self.logger, 'dir', '')))
        stop, res = self._attempt(det, zone, rep, list(neighbors) or [det],
                                  placement_zone, placement_polygon, exact_xy)
        if res is not None:
            return res, ''
        if stop is not None:
            return None, 'stopped before moving: %s' % stop
        reason = str(self._tracks[-1].get('reason', '')) if self._tracks else ''
        return None, reason or 'not attempted (measure again)'

    def pick_only_detection(self, det: Detection, neighbors: Sequence[Detection] = (),
                            budget_s: float = 3600.0) -> Tuple[Optional[PickResult], str]:
        """Plan one measured grasp and lift, retaining it for a later place step.

        This deliberately requires no destination capacity. The complete supplied
        neighborhood constrains the grasp and lift; the ordinary reach and
        SafeArm checks remain active. Success neither reserves a place nor marks
        any sorting goal complete.
        """
        from .reach import reach_adjust, reach_lowered

        if self.dry_run:
            return None, 'pick-only requires measured gripper feedback, not dry-run'
        if self.has_held_recovery:
            return None, 'an object is already held; place it before another pick'
        if not math.isfinite(self._deadline):
            self._deadline = self.clock.now() + float(budget_s)
        stop = self._stop_check()
        if stop is not None:
            return None, 'stopped before moving: %s' % stop
        if not det.valid or not _has_xy(det):
            return None, 'target has no valid measured geometry'
        target = copy.deepcopy(det)
        others = [target] + [copy.deepcopy(d) for d in neighbors
                            if d is not det]
        self._neighbors = others
        from ..perception.grasp import annotate_grasp_neighbors
        annotate_grasp_neighbors(target, others)
        gp = self.planner.plan(target)
        from .grasp_contact import supported_grasp
        gp = supported_grasp(self.planner, self.arm, target, gp)
        if not gp.feasible:
            return None, 'grasp infeasible: %s' % gp.reason
        held = held_offset_of(gp)
        tops = [float(d.top_z) for d in others if d is not target and d.top_z is not None
                and math.isfinite(float(d.top_z))]
        transit = transit_height(self.cfg, list(self.zones.values()), detection_table_z(target),
                                 held, max(tops) if tops else None)
        gp.T_lift[2, 3] = max(float(gp.T_lift[2, 3]), float(transit))
        top = float(target.top_z if target.top_z is not None else target.center_base[2])
        floors = {'pre': top + self.min_approach,
                  'lift': max(float(gp.T_grasp[2, 3]) + self.min_approach, float(transit))}
        grasp, _tilt, why = reach_adjust(self.arm, gp.T_grasp)
        if grasp is None:
            return None, 'grasp: %s' % why
        adjusted = {'T_grasp': grasp}
        for stage, attribute in (('pre', 'T_pre'), ('lift', 'T_lift')):
            pose, _tilt, _lowered, why = reach_lowered(
                self.arm, getattr(gp, attribute), floors[stage], self.free_tilt,
                keep_R=grasp[:3, :3], vertical_first=stage == 'lift')
            if pose is None:
                return None, '%s: %s' % (stage, why)
            adjusted[attribute] = pose
        for attribute, pose in adjusted.items():
            setattr(gp, attribute, pose)
        stop = self._stop_check()
        if stop is not None:
            return None, 'stopped before moving: %s' % stop
        self._event('decision', action='pick_only', label=target.label,
                    target=target.summary(), observed_neighbors=len(others) - 1,
                    expected_width=float(gp.expected_width))
        watcher = self._live_watcher(target)
        self._live_dets = []
        self._live_mark += 1
        if watcher is not None:
            self.skill.on_stage = watcher.stage
        try:
            result = self.skill.pick_only(gp)
        finally:
            if watcher is not None:
                self.skill.on_stage = None
                self._live_dets = watcher.finish()
        # The saved anchor describes the measured grasp, not the object's later
        # wrist-camera appearance. A placeholder supplies no release authority.
        placeholder = PlacePlan('held', gp.T_lift.copy(), gp.T_grasp.copy(),
                                gp.T_lift.copy(), open_width=gp.open_width)
        self._save_held_recovery(target, gp, placeholder, result)
        self._at_observe = False
        if result.detail.get('fault'):
            self.fault_detail = 'pick %s at stage %s: %s' % (
                result.label, result.stage, result.error)
        elif result.ok:
            self.fault_detail = ''
        return result, ''

    def _live_watcher(self, det: Detection):
        """Return a task.live.LiveWatcher for this pick, None when live watching is off."""
        if (not self.live.enabled or self.mode != 'real' or self.dry_run or not self.multi_view
                or self.camera is None or self.ext is None):
            return None
        from .live import LiveWatcher
        fp = self._footprint(det)
        held_r = (0.10 if fp is None else 0.5 * float(fp)) + self.live.held_margin
        return LiveWatcher(self.live, self.arm, self.camera, self.perception, self.ext,
                           self.T_flange_tcp, self.zones, self.clock, held_r, self.view_merge)

    def _floors(self, det: Detection, gp, pp, held: float) -> Dict[str, float]:
        """
        Return the lowest z (m) adjust_plan_poses may lower each free-space pose to.

        pre_grasp: object top + min_approach_clearance; lift / pre_place: the
        held object's bottom transit_object_clearance above every other object
        near the pick -> release line and every zone wall top; retreat: the
        fingertips min_approach_clearance above the released object.
        """
        tz = detection_table_z(det) or 0.0
        top = float(det.top_z) if det.top_z is not None else float(det.center_base[2])
        line = line_obstacle_top(det, getattr(self, '_neighbors', ()), pp.T_release[:2, 3])
        high = max([tz] + [float(z.center[2]) + max(0.0, float(z.wall_height))
                           for z in self.zones.values()] + ([line] if line is not None else []))
        clear = high + self.transit_clear + held
        z_g, z_r = float(gp.T_grasp[2, 3]), float(pp.T_release[2, 3])
        h = float(det.height) if det.height is not None else top - tz
        return {'pre_grasp': top + self.min_approach,
                'lift': max(z_g + self.min_approach, clear),
                'pre_place': max(z_r, clear),
                'retreat': max(z_r + self.min_approach, z_r - held + h + self.min_approach)}

    def _center_first(self, zone_name: str) -> bool:
        """Release the first object at the zone center unless several labels share the zone."""
        return sum(1 for z in self.class_to_zone.values() if z == zone_name) <= 1

    def _strict_place_filter(self, det, gp, neighbors, support_polygon, completion_zone=None,
                             release_opening=None, gripper_only=False):
        """Complete body envelopes bound the destination and the open fingers."""
        from ..perception.geometry import points_in_polygon, object_footprint, object_prism
        from ..perception.grasp import FINGER_OUTER, FINGER_HALF_BAND, RELEASE_OPEN_EXTRA
        from ..perception.polygon_collision import prepare_polygon, polygons_overlap as overlaps

        polygon = np.asarray(support_polygon, dtype=float)
        dims = footprint_dims(det, self.nominal.get(det.label))
        if polygon.ndim != 2 or polygon.shape[1:] != (2,) or dims is None:
            return lambda *args, **kwargs: False
        if completion_zone is not None:
            from ..assistant.destinations import ZONE_COMPLETION_MARGIN
            goal_center = np.asarray(completion_zone.center, float)[:2]
            goal_half = np.asarray(completion_zone.size, float) / 2 - ZONE_COMPLETION_MARGIN
            c, s = math.cos(completion_zone.yaw), math.sin(completion_zone.yaw)
            goal_rotation = np.array([[c, -s], [s, c]])

        def corners(xy, size, yaw):
            local = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * (np.asarray(size) / 2)
            c, s = math.cos(yaw), math.sin(yaw)
            return local @ np.array([[c, -s], [s, c]]).T + xy

        stack_support = (det.extra or {}).get('_stack_support')
        if stack_support is not None and not any(_is_stack_base(det, other) for other in neighbors):
            return lambda *args, **kwargs: False
        obstacles = []
        for other in neighbors:
            if other is det or not _has_xy(other):
                continue
            odims = footprint_dims(other, self.nominal.get(other.label))
            if odims is None or other.yaw is None:
                # Missing footprint geometry cannot establish a free destination.
                return lambda *args, **kwargs: False
            height = float(other.height or 0.0)
            bottom = detection_table_z(other)
            bottom = 0.0 if bottom is None else float(bottom)
            top = float(other.top_z) if other.top_z is not None else bottom + height
            obstacles.append((prepare_polygon(object_footprint(other.center_base[:2], *odims, float(other.yaw),
                              label=other.label, resting_pose=(other.extra or {}).get('resting_pose'))),
                              height, bottom, top, _is_stack_base(det, other)))
        offset = np.asarray(det.center_base[:2]) - gp.T_grasp[:2, 3]
        margin_offsets = np.array([[.002, .002], [.002, -.002], [-.002, .002], [-.002, -.002]])
        opening = min(self.placer.open_width, gp.expected_width + RELEASE_OPEN_EXTRA)
        if release_opening is not None:
            opening = float(release_opening)
        finger_floor = self.placer.release_height_flat + max(0.0, float(gp.T_grasp[2, 3]) -
                                                            (detection_table_z(det) or 0.0)) - .015

        def gripper_clear(pose, width, padding=0.0):
            import cv2
            if pose is None:
                return False
            # Match the actual finger and palm boxes used by the final release
            # check. Padding covers the tool's movement between sweep samples.
            boxes = [(np.array([0., sign * (width / 2 + .006), -.025]),
                      np.array([.010, .006, .025])) for sign in (-1, 1)]
            boxes.append((np.array([0., 0., -.095]), np.array([.0225, .045, .045])))
            signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
            T = np.asarray(pose, float)
            for center, half in boxes:
                points = (center + signs * (half + padding)) @ T[:3, :3].T + T[:3, 3]
                footprint = prepare_polygon(cv2.convexHull(points[:, :2].astype(np.float32)).reshape(-1, 2))
                low_z, high_z = float(points[:, 2].min()), float(points[:, 2].max())
                for other, _height, bottom, top, _designated in obstacles:
                    if top + .002 > low_z and bottom - .002 < high_z and overlaps(footprint, other):
                        return False
            return True

        def allowed(x, y, closing, final_yaw=None, tcp_xy=None, held_projection=None,
                    gripper_pose=None, gripper_open_width=None, gripper_padding=0.0):
            width = opening if gripper_open_width is None else float(gripper_open_width)
            if not math.isfinite(width) or width <= 0.0 or width > self.placer.open_width:
                return False
            if gripper_only:
                return gripper_clear(gripper_pose, width, gripper_padding)
            if stack_support is not None:
                from ..assistant.stacking import center_supported
                if not center_supported([x, y], stack_support):
                    return False
            delta = closing - gp.closing_angle
            rotated = np.array([[math.cos(delta), -math.sin(delta)],
                                [math.sin(delta), math.cos(delta)]]) @ offset
            obj = object_footprint([x, y], *dims,
                          float(det.yaw or 0.0) + delta if final_yaw is None else final_yaw,
                          label=det.label, resting_pose=(det.extra or {}).get('resting_pose'))
            if held_projection is not None:
                import cv2
                obj = cv2.convexHull(np.vstack((obj, held_projection)).astype(np.float32)).reshape(-1, 2)
            # Named zones use the same full-outline margin as final camera
            # completion, in the zone's own axes. Explicit tabletop/stack
            # supports retain their independent placement contracts below.
            if (completion_zone is not None and
                    not np.all(np.abs((obj - goal_center) @ goal_rotation) <= goal_half)):
                return False
            expanded = (obj[:, None, :] + margin_offsets).reshape(-1, 2)
            if not np.all(points_in_polygon(expanded, polygon)):
                return False
            guard = getattr(self.arm, 'guard', None)
            if guard is not None and (np.any(expanded[:, 0] < guard.x[0])
                    or np.any(expanded[:, 0] > guard.x[1])
                    or np.any(expanded[:, 1] < guard.y[0])
                    or np.any(expanded[:, 1] > guard.y[1])
                    or np.any(np.linalg.norm(expanded, axis=1) > guard.r_max)):
                return False
            obj = prepare_polygon(obj)
            fingers = prepare_polygon(corners(np.array([x, y]) - rotated if tcp_xy is None else tcp_xy,
                              (width + 2 * FINGER_OUTER, 2 * FINGER_HALF_BAND), closing))
            if any(not designated and (overlaps(obj, other)
                   or (height > finger_floor and overlaps(fingers, other)))
                   for other, height, _bottom, _top, designated in obstacles):
                return False
            if gripper_pose is not None:
                T = np.asarray(gripper_pose, float)
                if stack_support is not None:
                    source_floor = detection_table_z(det)
                    if source_floor is None or det.height is None:
                        return False
                    source_box = object_prism(det.center_base[:2], *dims, float(det.yaw or 0.),
                        source_floor, float(det.height), label=det.label,
                        resting_pose=(det.extra or {}).get('resting_pose'))
                    rotation = T[:3, :3] @ gp.T_grasp[:3, :3].T
                    released = (source_box - gp.T_grasp[:3, 3]) @ rotation.T + T[:3, 3]
                    if float(released[:, 2].min()) < float(stack_support['top_z']) - .002:
                        return False
                if not gripper_clear(T, width):
                    return False
            return True
        return allowed

    def _release_retreat_clear(self, det, gp, pp, neighbors, polygon, cancelled):
        """Check only the open tool after release, including between samples."""
        from ..core.transforms import rotation_angle
        from ..robot.safety import interpolate_pose

        allowed = self._strict_place_filter(det, gp, neighbors, polygon,
            release_opening=pp.open_width, gripper_only=True)
        travel = float(np.linalg.norm(pp.T_retreat[:3, 3] - pp.T_release[:3, 3]))
        angle = rotation_angle(pp.T_release[:3, :3].T @ pp.T_retreat[:3, :3])
        # Every finger/palm corner lies within this tool-frame radius. For
        # linear translation+slerp, each corner moves at most travel+r*angle.
        radius = max(math.sqrt(.010**2 + (pp.open_width / 2 + .012)**2 + .050**2),
                     math.sqrt(.0225**2 + .045**2 + .140**2))
        extent = travel + radius * angle
        segments = max(1, int(math.ceil(extent / .004)))
        padding = extent / (2 * segments)
        for fraction in np.linspace(0., 1., segments + 1):
            if cancelled():
                return False, 'release exit check interrupted'
            pose = interpolate_pose(pp.T_release, pp.T_retreat, float(fraction))
            if not allowed(0., 0., 0., gripper_pose=pose,
                           gripper_open_width=pp.open_width, gripper_padding=padding):
                return False, 'open gripper retreat intersects measured occupancy'
        return True, ''

    def current_release_exit_check(self, grasp, place, actual, width):
        """Recheck the measured tool exit from the actual settled release TCP."""
        proof = getattr(place, 'release_opening_checked', None)
        if not isinstance(proof, dict) or self._aborted():
            return False
        try:
            if (abs(float(width) - float(proof['width'])) > 1e-9
                    or not np.array_equal(proof['release_T'], place.T_release)
                    or not np.array_equal(proof['retreat_T'], place.T_retreat)):
                return False
            neighbors = proof['measured_neighbors']
            polygon = np.asarray(proof['support_polygon'], float)
            actual = np.asarray(actual, float)
            if (not isinstance(neighbors, list) or actual.shape != (4, 4)
                    or not np.all(np.isfinite(actual))):
                return False
            current = copy.copy(place)
            current.T_release = actual.copy()
            current.open_width = float(width)
            ok, _why = self._release_retreat_clear(
                grasp.target, grasp, current, neighbors, polygon, self._aborted)
            return bool(ok and not self._aborted())
        except (KeyError, TypeError, ValueError, AttributeError):
            return False

    def _adjust_strict_release(self, det, gp, pp, neighbors, polygon, desired_xy, held,
                               held_only=False, completion_zone=None):
        """Recheck final leaned poses and compensate their measured center shift."""
        from .reach import adjust_plan_poses, _planning_cancelled
        from ..perception.geometry import object_prism

        desired = np.asarray(desired_xy, float)
        table = detection_table_z(det) or 0.0
        source_center = np.r_[det.center_base[:2], table + float(det.height or 0.0) / 2]
        anchor_center = _placement_anchor_center(det, gp)
        cloud = np.asarray((det.extra or {}).get('_placement_points_base'), float)
        if cloud.ndim != 2 or cloud.shape[1:] != (3,) or len(cloud) < 40 or not np.all(np.isfinite(cloud)):
            cloud = None
        source_axis = np.array([math.cos(float(det.yaw or 0)), math.sin(float(det.yaw or 0)), 0])
        allowed = self._strict_place_filter(det, gp, neighbors, polygon,
                                            completion_zone=completion_zone,
                                            release_opening=pp.open_width)
        for _ in range(4):
            if _planning_cancelled(self.arm):
                return False, 'aborted'
            rotation = pp.T_release[:3, :3] @ gp.T_grasp[:3, :3].T
            if anchor_center is not None:
                carried_center = pp.T_release[:3, 3] + rotation @ (anchor_center - gp.T_grasp[:3, 3])
            elif cloud is not None:
                carried_cloud = (cloud - gp.T_grasp[:3, 3]) @ rotation.T + pp.T_release[:3, 3]
                carried_center = (carried_cloud.min(axis=0) + carried_cloud.max(axis=0)) / 2
            else:
                carried_center = pp.T_release[:3, 3] + rotation @ (source_center - gp.T_grasp[:3, 3])
            delta = desired - carried_center[:2]
            if np.linalg.norm(delta) > .0002:
                for T in (pp.T_pre, pp.T_release, pp.T_retreat):
                    T[:2, 3] += delta
                ok, why, _tilts = adjust_plan_poses(self.arm, gp, pp,
                    self._floors(det, gp, pp, held), self.free_tilt, held_only=held_only)
                if not ok:
                    return False, why
                continue
            axis = rotation @ source_axis
            yaw = math.atan2(float(axis[1]), float(axis[0]))
            closing = math.atan2(float(pp.T_release[1, 1]), float(pp.T_release[0, 1]))
            length, width = footprint_dims(det, self.nominal.get(det.label))
            box = object_prism(det.center_base[:2], length, width, float(det.yaw or 0),
                table, float(det.height or 0), label=det.label,
                resting_pose=(det.extra or {}).get('resting_pose'))
            projection = ((box - gp.T_grasp[:3, 3]) @ rotation.T + pp.T_release[:3, 3])[:, :2]
            if cloud is not None:
                projection = np.vstack((projection,
                    ((cloud - gp.T_grasp[:3, 3]) @ rotation.T + pp.T_release[:3, 3])[:, :2]))
            if not allowed(*desired, closing, final_yaw=yaw, tcp_xy=pp.T_release[:2, 3],
                           held_projection=projection, gripper_pose=pp.T_release,
                           gripper_open_width=pp.open_width):
                return False, 'leaned release has no full footprint or finger clearance'
            return True, ''
        return False, 'release center does not converge at requested point'

    def _footprint(self, det: Detection) -> Optional[float]:
        """Largest horizontal extent (m), at least the class's place.nominal_footprint."""
        dims = footprint_dims(det, self.nominal.get(det.label))
        return dims[0] if dims is not None else _footprint(det)

    def _long_in(self, det: Detection, zone_name: Optional[str]) -> bool:
        """Return True when det is longer than LONG_ZONE_RATIO x the flat zone's smaller side."""
        zone = self.zones.get(zone_name) if zone_name else None
        fp = self._footprint(det)
        return (zone is not None and fp is not None and float(zone.wall_height) <= 0.0
                and fp > LONG_ZONE_RATIO * min(float(zone.size[0]), float(zone.size[1])))

    def _long_first(self, candidates: List[Detection]) -> List[Detection]:
        """
        Drop the other objects of flat zones that still wait for a long object.

        A long object (cucumber on a 200 mm card) lies along a zone axis on the
        side nearer the robot (PlacePlanner.long_strip); released first, it
        leaves the other side to the rest. Zones that
        already hold a long object, and long objects blacklisted at their
        position, do not wait.
        """
        black = self._blacklisted_xy()
        wait = set()
        for d in candidates:
            zn = self.class_to_zone.get(d.label)
            if not self._long_in(d, zn) or is_blacklisted(d, black, self.match_dist):
                continue
            side = min(float(v) for v in self.zones[zn].size[:2])
            if any(len(o) >= 5 and o[4] is not None and float(o[4]) > LONG_ZONE_RATIO * side
                   for o in self._placed.get(zn, [])):
                continue
            wait.add(zn)
        if not wait:
            return candidates
        return [d for d in candidates if self.class_to_zone.get(d.label) not in wait
                or self._long_in(d, self.class_to_zone.get(d.label))]

    def _zone_order_ok(self, candidates: List[Detection], capped: Set[str]) -> List[Detection]:
        """
        Drop the candidates whose zone (runner.zone_order) still waits for an earlier label.

        An earlier label waits only while it is not placed there yet and may still come:
        a pickable candidate of it (not blacklisted, label not at max failures) is in
        view, or an observe view has not been scanned yet.
        """
        black = self._blacklisted_xy()
        live = {d.label for d in candidates
                if d.label not in capped and not is_blacklisted(d, black, self.match_dist)}
        out = []
        for d in candidates:
            zn = self.class_to_zone.get(d.label)
            order = self.zone_order.get(zn or '', [])
            if d.label in order:
                done = self._placed_labels.get(zn, [])
                before = order[:order.index(d.label)]
                if any(e not in done and e not in capped
                       and (e in live or not self._all_views_seen) for e in before):
                    continue
            out.append(d)
        if len(out) < len(candidates):
            self._event('decision', action='zone_order_wait',
                        waiting=sorted({d.label for d in candidates} - {d.label for d in out}))
        return out

    def _select(self, candidates: List[Detection], valid: List[Detection]
                ) -> Optional[Detection]:
        """
        Return select_target(...) over the first pool that gives a target.

        Labels at max_failures_per_label only when nothing else; without a fixed
        order (tasks.<T>.order / targets) long objects of flat zones first
        (_long_first).
        """
        capped = {lbl for lbl, n in self._label_failures.items() if n >= self.max_label_failures}
        if self.zone_order and not self.order:     # a referee order (targets) wins
            candidates = self._zone_order_ok(candidates, capped)
            if not candidates:
                return None
        rest = [d for d in candidates if d.label not in capped] if capped else candidates
        pools = []
        if not self.order and rest:
            first = self._long_first(rest)
            if 0 < len(first) < len(rest):
                pools.append(first)
        if rest and len(rest) < len(candidates):
            pools.append(rest)
        pools.append(candidates)
        for pool in pools:
            t = select_target(pool, self.strategy, self.order, (0.0, 0.0),
                              self._blacklisted_xy(), self.match_dist, neighbors=valid)
            if t is not None:
                return t
        return None

    # ------------------------------------------------------------- L1 / L2
    def _loop_sort(self, rep: TaskReport) -> None:
        if self.multi_view:
            self._loop_sort_views(rep)
            return
        empty = 0
        while True:
            if rep.placed >= self.required:
                rep.stop_reason = 'done'
                return
            stop = self._stop_check()
            if stop is not None:
                rep.stop_reason = stop
                return
            status, result, frame = self._observe()
            if status in ('fault', 'aborted'):
                rep.stop_reason = status
                return
            if status == 'failed':
                self._pause_after_failure()     # not 'nothing to pick': retry until the deadline
                continue
            valid: List[Detection] = result.valid_detections()
            target = self._select(self._candidates(result), valid)
            self._save_image(frame, result, target)
            if target is None:
                empty += 1
                self._event('decision', action='nothing_to_pick', empty=empty)
                if empty >= self.max_empty:
                    rep.stop_reason = 'no_targets'
                    return
                self.clock.sleep(self.retry_s)
                continue
            empty = 0
            stop, res = self._attempt(target, self.class_to_zone[target.label], rep, valid)
            if res is not None:
                rep.picks.append(res)
                if res.ok:
                    rep.placed += 1
            if stop is not None:
                rep.stop_reason = stop
                return

    def _joints_now(self) -> Optional[List[float]]:
        try:
            s = self.arm.get_joints()
        except Exception:  # noqa: B902 - unknown joints: views in taught order
            return None
        return None if s is None or s.value is None else [float(v) for v in s.value]

    def _scan_view(self, view: int, scene: Dict[int, List[Detection]], stale: Set[int],
                   seen: Set[int]) -> str:
        """Capture observe view `view` into scene; return the _observe status."""
        status, result, frame = self._observe(view)
        if status != 'ok':
            return status
        scene[view] = tag_view(result.valid_detections(), view)
        drop_live_seen(scene, len(self.views), view, self.view_merge)
        stale.discard(view)
        seen.add(view)
        self._event('decision', action='view_scanned', view=view, n=len(scene[view]),
                    labels=[d.label for d in scene[view]])
        self._save_image(frame, result, None)
        return 'ok'

    def _loop_sort_views(self, rep: TaskReport) -> None:
        """
        L1 / L2 over several wrist observe views (robot.poses.observe_views).

        Every view is scanned once before the first pick (a long object in a
        later view must be known before its zone-mates). The target is chosen
        over the merged detections of all views; when its view was captured
        before the arm last moved, that view is captured again first. A pick
        makes every view stale and drops the picked object; with no target
        left the stale views are scanned, and max_empty fresh full scans
        without a target end the round.
        """
        n = len(self.views)
        scene: Dict[int, List[Detection]] = {}
        stale: Set[int] = set(range(n))
        seen: Set[int] = set()
        empty = 0
        while True:
            if rep.placed >= self.required:
                rep.stop_reason = 'done'
                return
            stop = self._stop_check()
            if stop is not None:
                rep.stop_reason = stop
                return
            self._all_views_seen = len(seen) >= n
            merged = merge_view_detections(scene, stale, self.view_merge, partial=self._partial)
            target = None
            if len(seen) == n or (seen and not self.scan_all_first):
                target = self._select(self._candidates_from(merged), merged)
            if target is None or (view_of(target) in stale
                                  and (target.extra or {}).get('live_ok') != self._live_mark):
                if target is not None:
                    view = view_of(target)
                elif stale:
                    todo = (set(range(n)) - seen) or stale
                    view = nearest_view(self.views, todo, self._joints_now())
                else:
                    empty += 1
                    self._event('decision', action='nothing_to_pick', empty=empty, views=n)
                    if empty >= self.max_empty:
                        rep.stop_reason = 'no_targets'
                        return
                    stale = set(range(n))
                    self.clock.sleep(self.retry_s)
                    continue
                status = self._scan_view(view, scene, stale, seen)
                if status in ('fault', 'aborted'):
                    rep.stop_reason = status
                    return
                if status == 'failed':
                    self._pause_after_failure()
                continue
            empty = 0
            stop, res = self._attempt(target, self.class_to_zone[target.label], rep, merged)
            if res is not None:
                rep.picks.append(res)
                if res.ok:
                    rep.placed += 1
                    xy = _xy(target)
                    if xy is not None:
                        drop_near(scene, xy, self.view_merge)
                    if xy is None or self.rescan_radius < 0:
                        stale = set(range(n))
                    else:
                        # objects next to the picked one may have been touched
                        stale |= views_near(scene, xy, self.rescan_radius) - {n}
                    self._apply_live(scene, xy)
                else:
                    stale = set(range(n))       # a failed pick may have moved things
                    scene.pop(n, None)          # live positions may predate a drop
            elif view_of(target) == n:
                # a live position that could not be picked: measure it in the views again
                scene[n] = [d for d in scene.get(n, []) if d is not target]
                xy = _xy(target)
                stale |= (views_near(scene, xy, self.rescan_radius) - {n}) if xy else set(range(n))
                if not (stale & set(range(n))):
                    stale = set(range(n))
            else:
                stale.add(view_of(target))      # not picked: measure it again
            if stop is not None:
                rep.stop_reason = stop
                return

    def _apply_live(self, scene: Dict[int, List[Detection]], xy) -> None:
        """
        Update scene with the detections seen during the last pick (task.live).

        A fruit that moved more than live_watch.move_m, or that was not known,
        gets the live position; one seen where it was is confirmed and is picked
        without capturing its view again. Live positions of earlier picks near
        this one that this pick did not see again are dropped (maybe touched).
        """
        if not self._live_dets:
            return
        from .live import apply_live, drop_unconfirmed_live
        n = len(self.views)
        ch = apply_live(scene, self._live_dets, n, self._live_mark, self.live.move,
                        self.live.match, self.view_merge)
        dropped = 0
        if xy is not None:
            dropped = drop_unconfirmed_live(scene, n, self._live_mark, xy, self.rescan_radius)
        self._live_dets = []
        self._event('decision', action='live_update', dropped=dropped, **ch)

    def _report_view_coverage(self) -> None:
        """Log how much of the pick area the observe views cover (eye_in_hand, never refuses)."""
        from .views import coverage_report
        try:
            intr = self.camera.intrinsics() if self.camera is not None else None
            if intr is None or self.ext is None or self.ext.parent_frame != 'flange':
                return
            frac, msg = coverage_report(self.cfg, self.views, self.ext.T, intr)
        except Exception:  # noqa: B902 - diagnostics only
            _LOG.exception('view coverage check failed')
            return
        if frac is None:
            return
        (_LOG.info if frac >= 0.95 else _LOG.warning)('%s', msg)
        self._event('view_coverage', fraction=round(frac, 3), message=msg,
                    views=len(self.views))
