"""Owner-thread RGB-D sampling and persistent, individually invalidated evidence.

Camera poses transform measurements into the base frame; moving the camera is
not an object change. Occlusion is not disappearance. Simulator change notices
only invalidate evidence: they never supply a replacement position or a verdict.
"""
from __future__ import annotations

import copy
import time
from collections import deque

import numpy as np

from ..core.types import Detection
from .observation_policy import measurement_absent, merge_local_detections, unique_target


class MotionObservation:
    def __init__(self, session):
        from .catalog import catalog_entries
        self.session = session
        self.height_priors = {e['label']: float(e['dimensions'][2]) for e in catalog_entries()}
        self.evidence = {}
        self.invalidated = set()
        self.last_wall = float('-inf')
        self.last_sim = float('-inf')
        self.capturing = False
        self.frames = deque(maxlen=12)
        self.stats = dict(frames=0, source_reuses=0, verification_reuses=0)

    def invalidate(self, labels, *, clear_views=True):
        labels = tuple(label for label in labels if label in self.height_priors)
        if labels and clear_views:
            self.frames.clear()
        for label in labels:
            self.invalidated.add(label)
            self.evidence.pop(label, None)
            getattr(self.session, '_released_observation_hints', {}).pop(label, None)

    def remember_frame(self, frame, pose, revision):
        """Retain actual ray/depth evidence, never fill self-occluded pixels."""
        if frame is not None and frame.depth is not None and revision == self.session._environment_revision:
            self.frames.append((copy.deepcopy(frame), np.asarray(pose).copy(), revision))

    def support_views(self):
        revision = self.session._environment_revision
        return [(frame, pose) for frame, pose, observed in self.frames if observed == revision]

    def remember(self, fresh, revision):
        """Only complete unique RGB-D instances establish usable evidence."""
        session = self.session
        if revision != session._environment_revision:
            return
        for label in {d.label for d in fresh} & self.height_priors.keys():
            candidates = [d for d in fresh if d.label == label]
            det = unique_target(candidates, label)
            if det is None:
                # A crop need not erase an earlier complete view, but two
                # valid same-class objects cannot share one cached identity.
                if sum(d.valid for d in candidates) > 1:
                    self.invalidate([label])
                continue
            if session._target_partial(det, det.summary()):
                continue
            old = self.evidence.get(label)
            observed = copy.deepcopy(det)
            stable = (old is not None and label not in self.invalidated
                and np.linalg.norm(old.center_base[:2] - det.center_base[:2]) <= .003
                and all(a is not None and b is not None and abs(a-b) <= .1*max(a,b)
                    for a,b in ((old.length,det.length),(old.width,det.width),(old.height,det.height))))
            if stable and label in ('banana', 'pear'):
                import math
                from .catalog import restoration_yaw_period
                period = restoration_yaw_period(label, det.extra.get('resting_pose'))
                stable = period is None or abs(math.remainder(
                    float(old.yaw or 0)-float(det.yaw or 0), period)) <= math.radians(4)
            points = observed.extra.get('_placement_points_base', observed.extra.get('_points_base'))
            old_points = old.extra.get('_placement_points_base') if stable else None
            clouds = [np.asarray(p, np.float32) for p in (old_points, points) if p is not None]
            clouds = [p for p in clouds if p.ndim == 2 and p.shape[1:] == (3,) and len(p) >= 40
                      and np.all(np.isfinite(p))]
            if clouds:
                combined = np.vstack(clouds)
                _, keep = np.unique(np.floor(combined/.002).astype(np.int64), axis=0, return_index=True)
                observed.extra['_placement_points_base'] = combined[np.sort(keep)]
                observed.extra['_cloud_views'] = max(det.extra.get('_cloud_views', 1),
                    old.extra.get('_cloud_views', 1) + 1 if old_points is not None else 1)
            observed.extra['_observed_revision'] = revision
            self.evidence[label] = observed
            self.invalidated.discard(label)

    def reconcile(self, fresh, revision):
        """Associate complete new RGB-D with an unchanged tracked identity.

        This supplies no replacement pixels or new completion evidence. The
        Crops cannot establish a new complete measurement. A release hint can
        associate identity only; geometry and completion still need new pixels.
        """
        if revision != self.session._environment_revision:
            return fresh
        from .catalog_perception import completeness
        tracked = {label:old for label,old in self.evidence.items()
                   if label not in self.invalidated}
        for label, hint in getattr(self.session, '_released_observation_hints', {}).items():
            if hint.extra.get('_identity_revision') == revision:
                tracked.setdefault(label, hint)
        for det in fresh:
            if det.center_base is None:
                continue
            matches = [old for label, old in tracked.items()
                if old.center_base is not None
                and np.linalg.norm(old.center_base[:2]-det.center_base[:2]) <= .018
                and all(a is not None and b is not None and abs(float(a)-float(b)) <= .015
                        for a,b in ((old.top_z,det.top_z),(old.height,det.height)))]
            if len(matches) == 1 and matches[0].label != det.label:
                candidate = copy.copy(det)
                candidate.label = matches[0].label
                if det.valid and completeness(candidate):
                    det.extra['_classifier_label'] = det.label
                    det.label, det.score_kind = candidate.label, 'tracked_rgbd'
                    det.extra['catalog_complete'] = True
                    if matches[0].extra.get('catalog_shape') is not None:
                        det.extra['catalog_shape'] = matches[0].extra['catalog_shape']
                    if det.label == 'pear':
                        from .catalog_perception import pear_pose
                        det.extra['resting_pose'] = pear_pose(det)
                    elif det.label == 'banana':
                        det.extra['resting_pose'] = 'flat'
                else:
                    det.valid = False
                    det.reject_reason = 'tracked_identity_conflict'
                    det.extra['catalog_complete'] = False
            elif (not matches and det.label in tracked
                    and np.linalg.norm(tracked[det.label].center_base[:2]-det.center_base[:2]) > .065):
                # One-per-class rebinding requires an actual change notice;
                # a distant classification cannot teleport a stable identity.
                det.valid = False
                det.reject_reason = 'tracked_identity_conflict'
                det.extra['catalog_complete'] = False
        return fresh

    def current(self, label=None):
        """Retain unchanged objects; invalidated ones require new camera evidence."""
        session = self.session
        if session.cancel_event.is_set() or getattr(session, '_interventions', ()):
            return []
        result = []
        for name, det in self.evidence.items():
            if name in self.invalidated or (label is not None and name != label):
                continue
            measured = copy.deepcopy(det)
            measured.extra['_observed_revision'] = session._environment_revision
            result.append(measured)
        return result

    def capture(self, *, force=False):
        """One synchronized frame, no robot command and no clock advancement."""
        session = self.session
        now, wall = session.clock.now(), time.monotonic()
        if (self.capturing or not session._prepared or session.cancel_event.is_set()
                or (not force and (now-self.last_sim < 1.25 or wall-self.last_wall < .45))):
            return False
        self.capturing = True
        self.last_sim, self.last_wall = now, wall
        try:
            from ..core.extrinsics import camera_pose_in_base
            previous_frame = getattr(session.camera, '_last', None)
            frame = session.camera.get_frame(timeout=0.)
            flange = session.arm.get_flange_pose()
            if (frame is None or frame is previous_frame or frame.depth is None or flange is None
                    or not np.isfinite(frame.stamp)
                    or abs(float(frame.stamp)-now) > 1e-6):
                return False
            pose = camera_pose_in_base(session.runner.ext, flange.value,
                                       session.runner.T_flange_tcp)
            revision = session._environment_revision
            context = getattr(session.runner, '_held_recovery_context', None)
            grasp = (context.get('grasp') if context else
                     getattr(session.runner.skill, 'active_grasp', None))
            target = (context.get('det') if context else
                      getattr(session, '_recovery_target', None))
            # Stationary observation can clear the display stage between a
            # successful pick-only and place. The measured held context still
            # establishes ownership of the body throughout that handoff.
            # Retreat follows release: the body must be observable there.
            carrying = session._stage != 'retreat' and (bool(context) or session._stage in (
                'close', 'lift', 'transit', 'release', 'placement_planning',
                'inspect_place', 'inspect_lift', 'retry_lift'))
            if carrying and grasp is not None:
                from .carried_observation import mask_carried_depth
                width = session.runner.skill._read_width()
                tcp = session.arm.get_tcp_pose()
                if width is not None and tcp is not None:
                    frame = mask_carried_depth(frame, pose, grasp, tcp.value, width,
                        nominal=session.runner.nominal.get(grasp.target.label))
            result = session.runner.perception.process(frame, pose, session.runner.zones)
            fresh = [copy.deepcopy(d) for d in result.detections
                     if not (carrying and target is not None and d.label == target.label)]
            for det in fresh:
                det.extra.update(_capture_stamp=float(frame.stamp),
                                 _capture_flange=np.asarray(flange.value).copy(),
                                 _observed_revision=revision, _motion_capture=True)
            if revision != session._environment_revision:
                return False
            self.reconcile(fresh, revision)
            singletons = ({d.label for d in fresh if d.valid and d.extra.get('catalog_complete')}
                          if session.editor.metadata().get('one_per_class') else ())
            previous = session._observed_detections
            merged = merge_local_detections(previous, fresh,
                lambda d: measurement_absent(d, frame, pose),
                singleton_labels=singletons, preserve_incomplete=True)
            removed = [d.label for d in previous if d.valid and
                       not any(n.label == d.label for n in merged)]
            self.invalidate(removed)
            self.remember(fresh, revision)
            self.remember_frame(frame, pose, revision)
            session._last_fresh = fresh
            session._observation_revision = revision
            session._last_observation_purpose = 'motion_observation'
            session._scan_stats['frames'] += 1
            self.stats['frames'] += 1
            session._cache_observation(6, result, frame, label='执行中实时观测')
            session._publish_scene(merged, result.warnings, [], frame.stamp, preserve_status=True)
            session._emit('motion_observation', dict(message='执行中更新可见物品，复用未变化状态',
                labels=sorted(d.label for d in fresh if d.valid),
                stats=copy.deepcopy(self.stats)))
            return True
        finally:
            # Rendering is synchronous owner-thread work. Its own duration
            # must not consume the entire spacing before the next capture.
            self.last_wall = time.monotonic()
            self.capturing = False

    def uncertainty_obstacles(self, exclude=()):
        """Unknown moved bodies occupy all tabletop XY until actually observed.

        This is an exclusion volume, not an inferred object position. It permits
        only a sufficiently high checked transit; a release cannot pass it.
        """
        session = self.session
        bounds = session._placement['table_bounds']
        low = np.array([bounds[a][0] for a in ('x', 'y')])
        high = np.array([bounds[a][1] for a in ('x', 'y')])
        result = []
        for label in self.invalidated - set(exclude):
            old = [d for d in session._observed_detections if d.label == label]
            height = max([float(d.top_z or 0.) for d in old] +
                         [self.height_priors.get(label, .22)]) + .02
            result.append(Detection('unobserved_' + label, 0., valid=False,
                center_base=np.r_[(low+high)/2, height], top_z=height, height=height,
                length=float(high[0]-low[0]), width=float(high[1]-low[1]), yaw=0.))
        return result

    def source(self, reference):
        """Return a measured source in base coordinates, independent of wrist pose."""
        session = self.session
        label = reference.get('label')
        det = unique_target(self.current(label), label)
        if det is None or session._target_partial(det, reference):
            return None
        if label in ('banana', 'pear') and det.extra.get('_cloud_views', 0) < 2:
            return None
        # Do not resolve an ambiguous public scene using a private cache.
        matches = [d for d in session._observed_detections if d.valid and d.label == label]
        if len(matches) != 1 or np.linalg.norm(matches[0].center_base[:2]-det.center_base[:2]) > .01:
            return None
        from .obstacle_recovery import _grasp
        peers = [d for d in session._observed_detections if d.label != label]
        _target, planned = _grasp(session, det, peers)
        if not planned.feasible:
            return None
        from .held_anchor import spherical_model, source_anchor
        if spherical_model(det) is not None and not source_anchor(planned).get('ok'):
            return None
        if reference.get('required_support') and not session._support_observed(
                reference, self.current(), session._environment_revision):
            return None
        return det
