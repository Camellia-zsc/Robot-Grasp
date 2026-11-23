"""Prove candidate support is visible, and propose guarded held-camera views.

Only camera depth, calibrated transforms, measured object geometry and the
robot's public reachability check are inputs. No world state is read here.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import numpy as np

from ..core.transforms import inv_T
from ..perception.geometry import points_in_polygon, project_points, object_prism, object_footprint
from ..perception.grasp import (FINGER_HALF_BAND, FINGER_OUTER, RELEASE_OPEN_EXTRA,
                                detection_table_z,measured_tapered_contact)
from ..task.reach import accepts_kwarg


class JointTransitRoute(list):
    """A joint-space segment with the exact start/end checked by FK sweep."""

    def __init__(self, pose, start, end):
        super().__init__([np.array(pose, copy=True)])
        self.start_joints = np.array(start, copy=True)
        self.end_joints = np.array(end, copy=True)


def _grip_bend(reference, pose):
    return math.acos(float(np.clip(reference[:3, 2] @ pose[:3, 2], -1., 1.)))


def _tapered_contact(grasp):
    return measured_tapered_contact(grasp.target)


def _dimensions(det, nominal):
    measured = np.array([det.length, det.width, det.height], float)
    if not np.all(np.isfinite(measured)) or np.any(measured <= 0):
        raise ValueError('candidate visibility requires complete measured geometry')
    length, width = sorted(measured[:2], reverse=True)
    if nominal is not None:
        length, width = max(length, max(nominal)), max(width, min(nominal))
    return float(length), float(width), float(measured[2])


def _rectangle(center, length, width, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return (np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * [length / 2, width / 2]
            @ np.array([[c, -s], [s, c]]).T + center)


def _samples(polygon, step):
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    x = np.linspace(low[0], high[0], max(2, int(math.ceil((high[0] - low[0]) / step)) + 1))
    y = np.linspace(low[1], high[1], max(2, int(math.ceil((high[1] - low[1]) / step)) + 1))
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return np.vstack([polygon, grid[points_in_polygon(grid, polygon, margin=1e-8)]])


def candidate_visibility_filter(det, grasp, frame, T_base_camera, nominal=None,
                                max_open_width=.1, sample_step=.008,
                                margin_px=8, depth_tolerance=.008, *, observations=(), landing_z=None):
    """Return a pose filter proving full support/finger footprints are observed.

    Object-support samples must see the landing plane in a fresh view; finger
    samples must prove free space below the conservative finger envelope. A
    low neighbor may lie safely below suspended fingers. Foreground entering
    the required space, depth holes and image crops cannot establish clearance.
    This complements measured collision checks and never treats unseen space
    as empty. ``landing_z`` is
    the actual destination support height. Source geometry keeps its original
    support frame when carrying an object from a pedestal to the table.
    """
    length, width, height = _dimensions(det, nominal)
    table = detection_table_z(det)
    if table is None:
        raise ValueError('candidate visibility requires a measured table height')
    stack_support = (det.extra or {}).get('_stack_support')
    if landing_z is None:
        landing_z = table if stack_support is None else stack_support['top_z']
    landing_z = float(landing_z)
    if not math.isfinite(landing_z):
        raise ValueError('candidate visibility requires a measured landing height')
    views = []
    for observed_frame, camera_pose in ((frame, T_base_camera), *observations):
        intr = observed_frame.intrinsics if observed_frame is not None else None
        depth = (None if observed_frame is None or observed_frame.depth is None
                 else np.asarray(observed_frame.depth))
        if intr is None or depth is None or depth.shape != (intr.height, intr.width):
            continue
        T_bc = np.asarray(camera_pose, float)
        views.append((T_bc, inv_T(T_bc), intr, depth))
    source_center = np.r_[det.center_base[:2], table + height / 2]
    original_yaw = float(det.yaw or 0.)
    resting_pose = det.extra.get('resting_pose')
    source_box = object_prism(source_center[:2], length, width, original_yaw, table, height,
                              label=det.label, resting_pose=resting_pose)
    offset = np.asarray(det.center_base[:2]) - grasp.T_grasp[:2, 3]
    opening_default = min(float(max_open_width), float(grasp.expected_width) + RELEASE_OPEN_EXTRA)
    # Bound depth tolerance along every calibrated image ray, leaving an
    # additional 4 mm below the real fingers. The object's support still
    # requires the landing plane; suspended fingers only require clear air.
    ray_bound = max((math.sqrt(1. + (max(intr.cx, intr.width-1-intr.cx)/intr.fx)**2
                                  + (max(intr.cy, intr.height-1-intr.cy)/intr.fy)**2)
                     for _bc, _cb, intr, _depth in views), default=1.)
    finger_margin = .004 + max(0., float(depth_tolerance))*ray_bound

    def allowed(x, y, closing, gripper_pose=None, opening=None):
        if not views:
            return False
        delta = float(closing) - float(grasp.closing_angle)
        c, s = math.cos(delta), math.sin(delta)
        rotated_offset = np.array([[c, -s], [s, c]]) @ offset
        tcp_xy = np.array([x, y]) - rotated_offset
        if gripper_pose is None:
            object_polygon = object_footprint([x, y], length + .004, width + .004,
                original_yaw + delta, label=det.label, resting_pose=resting_pose)
        else:
            import cv2
            pose = np.asarray(gripper_pose, float)
            rotation = pose[:3, :3] @ grasp.T_grasp[:3, :3].T
            carried_box = ((source_box - grasp.T_grasp[:3, 3]) @ rotation.T + pose[:3, 3])
            object_polygon = cv2.convexHull(carried_box[:, :2].astype(np.float32)).reshape(-1, 2)
            tcp_xy = pose[:2, 3]
        actual_opening = opening_default if opening is None else float(opening)
        if gripper_pose is None:
            low = landing_z + max(0., float(grasp.T_grasp[2, 3])-table) - .015
            fingers = [(_rectangle(tcp_xy, actual_opening + 2 * FINGER_OUTER,
                                  2 * FINGER_HALF_BAND, float(closing)), low)]
        else:
            import cv2
            signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
            fingers = []
            for sign in (-1, 1):
                center = np.array([0., sign * (actual_opening / 2 + .006), -.025])
                box = ((center + signs * [.010, .006, .025]) @ pose[:3, :3].T + pose[:3, 3])
                fingers.append((cv2.convexHull(box[:, :2].astype(np.float32)).reshape(-1, 2),
                                float(box[:, 2].min())))
        groups = [(object_polygon, landing_z + .001)] + [
            (polygon, max(landing_z + .001, low-finger_margin)) for polygon, low in fingers]
        points = []
        for polygon, z in groups:
            xy = _samples(polygon, sample_step)
            points.append(np.column_stack([xy, np.full(len(xy), z)]))
        points = np.vstack(points)
        covered = np.zeros(len(points), bool)
        for T_bc, T_cb, intr, depth in views:
            remaining = np.flatnonzero(~covered)
            samples = points[remaining]
            uv = project_points(samples, T_bc, intr)
            expected = samples @ T_cb[2, :3] + T_cb[2, 3]
            inside = np.flatnonzero(np.all(np.isfinite(uv), axis=1) & (expected > 0)
                & (uv[:, 0] >= margin_px) & (uv[:, 0] <= intr.width - 1 - margin_px)
                & (uv[:, 1] >= margin_px) & (uv[:, 1] <= intr.height - 1 - margin_px))
            pixels = np.rint(uv[inside]).astype(int)
            observed = depth[pixels[:, 1], pixels[:, 0]]
            covered[remaining[inside]] = (np.isfinite(observed) & (observed > 0)
                                         & (observed >= expected[inside] - depth_tolerance))
            if np.all(covered):
                return True
        return False
    return allowed


def _planning_cancelled(arm, cancelled=None):
    if cancelled is not None and cancelled():
        return True
    scoped = getattr(arm, '_planning_cancelled', None)
    if scoped is not None and scoped():
        return True
    event = getattr(arm, 'abort_event', None)
    return event is not None and event.is_set()


def _planning_call(arm, cancelled, function, *args, **kwargs):
    scope = getattr(arm, 'planning_scope', None)
    with scope(cancelled) if scope is not None and cancelled is not None else nullcontext():
        return function(*args, **kwargs)


def inspection_history(session, grasp):
    """Measured views are reusable only for the same held grasp and scene."""
    key = (session._environment_revision, grasp.target.label,
           tuple(np.round(grasp.T_grasp, 6).ravel()))
    if getattr(session, '_held_view_key', None) != key:
        session._held_view_key, session._held_view_history = key, []
    return session._held_view_history


def forget_inspection(session):
    """Object motion can make an old viewpoint informative again."""
    session._held_view_key = None
    session._held_view_history = []
    session._skill_coverage_revision = None
    session._skill_observation_coverage = {}


def remember_inspection(session, grasp, pose):
    history = inspection_history(session, grasp)
    history.append(np.asarray(pose, float).copy())
    del history[:-64]


def _inspection_geometry(grasp, nominal, highest_obstacle, table_z, clearance):
    length, width, height = _dimensions(grasp.target, nominal)
    det = grasp.target
    table = detection_table_z(det)
    if table is None:
        raise ValueError('inspection requires a measured table height')
    xy = _rectangle(np.asarray(det.center_base[:2]), length, width, float(det.yaw or 0.))
    corners = np.vstack([np.column_stack([xy, np.full(4, z)]) for z in (table, table + height)])
    radius = float(np.max(np.linalg.norm(corners - grasp.T_grasp[:3, 3], axis=1)))
    minimum = max(float(table_z), float(highest_obstacle)) + float(clearance) + radius
    return length, width, corners, minimum


def _inspection_visibility_score(T_bc, points, sizes, starts, held_polygon, intrinsics):
    """Project all 32 support groups once, preserving their original ranking."""
    T_cb = inv_T(T_bc)
    camera = points @ T_cb[:3, :3].T + T_cb[:3, 3]
    uv = camera[:, :2] / np.maximum(camera[:, 2:3], 1e-9)
    inside = camera[:, 2] > 1e-6
    if intrinsics is not None:
        inside &= ((uv[:, 0] * intrinsics.fx + intrinsics.cx >= 8)
                   & (uv[:, 0] * intrinsics.fx + intrinsics.cx <= intrinsics.width - 9)
                   & (uv[:, 1] * intrinsics.fy + intrinsics.cy >= 8)
                   & (uv[:, 1] * intrinsics.fy + intrinsics.cy <= intrinsics.height - 9))
    if held_polygon is not None:
        inside &= ~points_in_polygon(uv, held_polygon, margin=.015)
    counts = np.add.reduceat(inside, starts, dtype=np.int64)
    # Python sum retains the old group-by-group floating-point addition order.
    return int(np.count_nonzero(counts == sizes)), sum((counts / sizes).tolist())


def iter_safe_inspection_poses(arm, grasp, T_flange_camera, T_flange_tcp, target_xy,
                              highest_obstacle, table_z=0., clearance=.04,
                              free_tilt_deg=40., nominal=None, intrinsics=None, *,
                              cancelled=None, progress=None, max_poses=12, max_checks=None,
                              exclude_poses=(), focus_z=None, orientation_clearance=False):
    """Yield reachable held-camera poses, spending IK only as consumed.

    By default a bounding sphere clears arbitrary rotations. Callers which
    check every swept pose can request orientation_clearance: endpoint heights
    then use the full carried body and both fingers at that endpoint's actual
    attitude. A long flat body need not lift by its half length. This function
    sends no motion commands and never certifies the route between endpoints.
    Cancellation or an exhausted reachability-check budget ends the iterator.
    ``progress`` receives small dictionaries; no motion commands are sent.
    """
    if (_planning_cancelled(arm, cancelled) or (max_poses is not None and max_poses <= 0)
            or (max_checks is not None and max_checks <= 0)):
        return
    check = getattr(arm, 'check_reachable', None)
    length, width, corners, minimum_tcp_z = _inspection_geometry(
        grasp, nominal, highest_obstacle, table_z, clearance)
    det = grasp.target
    if check is None:
        return
    gaze_z = float(table_z) + .015 if focus_z is None else float(focus_z)
    if not math.isfinite(gaze_z) or gaze_z < float(table_z):
        return
    target = np.r_[np.asarray(target_xy, float), gaze_z]
    radial = target[:2] / max(float(np.linalg.norm(target[:2])), 1e-6)
    lateral = np.array([-radial[1], radial[0]])
    T_camera_tcp = inv_T(np.asarray(T_flange_camera, float)) @ np.asarray(T_flange_tcp, float)
    # The wrist camera and carried object move together. Merely changing the
    # eye position while always looking at the landing centre therefore leaves
    # a long carried object over the same image pixels. Offset the gaze as well
    # and rank views by projected full-footprint clearance, before spending IK
    # calls or physical motions on them. This is only an ordering prediction;
    # the new RGB-D frame must still pass candidate_visibility_filter.
    import cv2
    held_camera = (corners - grasp.T_grasp[:3, 3]) @ grasp.T_grasp[:3, :3]
    held_camera = held_camera @ T_camera_tcp[:3, :3].T + T_camera_tcp[:3, 3]
    positive = held_camera[:, 2] > 1e-6
    held_polygon = (cv2.convexHull((held_camera[positive, :2] / held_camera[positive, 2:3])
                                 .astype(np.float32)).reshape(-1, 2)
                    if np.count_nonzero(positive) >= 3 else None)
    closing_angles = np.linspace(-math.pi, math.pi, 32, endpoint=False)
    offset = np.asarray(det.center_base[:2]) - grasp.T_grasp[:2, 3]
    support = []
    opening = float(grasp.expected_width) + RELEASE_OPEN_EXTRA
    source_local = (corners-grasp.T_grasp[:3, 3]) @ grasp.T_grasp[:3, :3]
    signs = np.array([[i,j,k] for i in (-1,1) for j in (-1,1) for k in (-1,1)])
    held_local = np.vstack([source_local] + [
        np.array([0., sign*(opening/2+.006), -.025]) + signs*[.010,.006,.025]
        for sign in (-1,1)])
    def endpoint_floor(rotation):
        if not orientation_clearance:
            return minimum_tcp_z
        return (max(float(table_z), float(highest_obstacle)) + float(clearance)
                - float(np.min(held_local @ np.asarray(rotation)[2, :])))

    for closing in closing_angles:
        if _planning_cancelled(arm, cancelled):
            return
        delta = closing - float(grasp.closing_angle)
        c, s = math.cos(delta), math.sin(delta)
        tcp_xy = target[:2] - np.array([[c, -s], [s, c]]) @ offset
        polygons = [_rectangle(target[:2], length + .004, width + .004,
                               float(det.yaw or 0.) + delta),
                    _rectangle(tcp_xy, opening + 2 * FINGER_OUTER,
                               2 * FINGER_HALF_BAND, closing)]
        samples = np.vstack([_samples(p, .012) for p in polygons])
        support.append(np.column_stack([samples, np.full(len(samples), float(table_z) + .001)]))
    support_sizes = np.array([len(points) for points in support])
    support_starts = np.r_[0, np.cumsum(support_sizes)[:-1]]
    support_points = np.vstack(support)

    def visibility_score(T_bc):
        if _planning_cancelled(arm, cancelled):
            return 0, 0.
        return _inspection_visibility_score(T_bc, support_points, support_sizes,
                                            support_starts, held_polygon, intrinsics)

    proposals = []
    sample = arm.get_tcp_pose() if hasattr(arm, 'get_tcp_pose') else None
    attitude = grasp.T_grasp if sample is None else np.asarray(sample.value)
    # Moving the camera sideways can reveal a landing without bending the
    # contact axis. This is especially useful for a measured tapered grip:
    # repeatedly leaning it to the workspace tilt limit can slide the neck
    # through the pads, even when the complete swept geometry is collision-free.
    for height in (.23, .28, .33, .38):
        for distance in (-.12, -.06, 0., .06, .12):
            for side in (-.12, -.06, 0., .06, .12):
                pose = attitude.copy()
                pose[:2, 3] = target[:2] + distance*radial + side*lateral
                pose[2, 3] = max(endpoint_floor(pose[:3, :3]), float(table_z)+height+
                    (0. if orientation_clearance else max(0., gaze_z-float(table_z)-.015)))
                proposals.append((visibility_score(pose @ inv_T(T_camera_tcp)), pose))
    if progress is not None:
        progress({'phase': 'inspection_candidates', 'generated': 0, 'checked': 0, 'reachable': 0})
    for height in (.27, .32, .38, .44):
        # Short baselines also cover staging points near the robot. The longer
        # original baselines can otherwise put every proposed TCP inside the
        # arm's inner workspace even though the landing point is reachable.
        for distance, side in ((.14, 0.), (.18, .08), (.18, -.08), (.22, 0.),
                               (.06, 0.), (.08, .08), (.08, -.08), (.10, 0.)):
            eye = np.r_[target[:2] - distance * radial + side * lateral, target[2] + height]
            for forward, sideways in ((0., 0.), (-.10, 0.), (.10, 0.), (0., .10),
                                       (0., -.10), (-.08, .08), (-.08, -.08),
                                       (.08, .08), (.08, -.08), (0., .14), (0., -.14),
                                       (-.04, .14), (-.04, -.14), (.04, .14), (.04, -.14)):
                if _planning_cancelled(arm, cancelled):
                    return
                gaze = target + np.r_[forward * radial + sideways * lateral, 0.]
                z = gaze - eye
                z /= np.linalg.norm(z)
                x = np.cross(z, [0., 0., 1.])
                if np.linalg.norm(x) < 1e-9:
                    continue
                x /= np.linalg.norm(x)
                y = np.cross(z, x)
                for roll in ((0., -.75, .75) if distance <= .10 else (0.,)):
                    c, s = math.cos(roll), math.sin(roll)
                    T_bc = np.eye(4)
                    T_bc[:3, :3], T_bc[:3, 3] = np.column_stack([c*x+s*y, -s*x+c*y, z]), eye
                    T_tcp = T_bc @ T_camera_tcp
                    tilt = math.degrees(math.acos(float(np.clip(-T_tcp[2, 2], -1., 1.))))
                    # A target on the hard limit can land outside it after the
                    # solver/servo's finite orientation error. Keep proposals
                    # inside the limit; all actual path checks retain 40 deg.
                    if (float(T_tcp[2, 3]) >= endpoint_floor(T_tcp[:3, :3])
                            and tilt <= max(0., free_tilt_deg-.25)
                            and (not _tapered_contact(grasp)
                                 or _grip_bend(attitude, T_tcp) <= math.radians(15.))):
                        proposals.append((visibility_score(T_bc), T_tcp))
    if progress is not None:
        progress({'phase': 'inspection_reachability', 'generated': len(proposals),
                  'checked': 0, 'reachable': 0})
    checked, reachable = 0, 0
    excluded = np.asarray(list(exclude_poses), float).reshape(-1, 4, 4)
    def ranking(item):
        score, pose = item
        # Once a full landing is visible, preserve a tapered contact before
        # optimizing fractional visibility of other, unnecessary directions.
        if _tapered_contact(grasp):
            return (bool(score[0]), -_grip_bend(attitude, pose), score[0], score[1])
        return (bool(score[0]), score[0], score[1], -_grip_bend(attitude, pose))
    for _score, T_tcp in sorted(proposals, key=ranking, reverse=True):
        if (_planning_cancelled(arm, cancelled)
                or (max_checks is not None and checked >= max_checks)):
            return
        if len(excluded):
            near = np.linalg.norm(excluded[:, :3, 3] - T_tcp[:3, 3], axis=1) < .03
            # trace(R_old.T @ R_new) is the sum of elementwise products.
            traces = np.sum(excluded[:, :3, :3] * T_tcp[:3, :3], axis=(1, 2))
            if np.any(near & (traces > 1. + 2. * math.cos(math.radians(12.)))):
                continue
        kwargs = {'max_tilt_deg': free_tilt_deg} if accepts_kwarg(check, 'max_tilt_deg') else {}
        ok, _reason = _planning_call(arm, cancelled, check, T_tcp, **kwargs)
        checked += 1
        if _planning_cancelled(arm, cancelled):
            return
        reachable += int(ok)
        if progress is not None and (ok or checked == 1 or checked % 16 == 0):
            progress({'phase': 'inspection_reachability', 'generated': len(proposals),
                      'checked': checked, 'reachable': reachable})
        if _planning_cancelled(arm, cancelled):
            return
        if ok:
            yield T_tcp
            if max_poses is not None and reachable >= max_poses:
                return


def safe_inspection_poses(arm, grasp, T_flange_camera, T_flange_tcp, target_xy,
                          highest_obstacle, table_z=0., clearance=.04,
                          free_tilt_deg=40., nominal=None, intrinsics=None, *,
                          cancelled=None, progress=None, max_poses=12, max_checks=None):
    """Compatible list wrapper; cancelled searches never offer motion poses."""
    _length, _width, _corners, minimum = _inspection_geometry(
        grasp, nominal, highest_obstacle, table_z, clearance)
    poses = list(iter_safe_inspection_poses(
        arm, grasp, T_flange_camera, T_flange_tcp, target_xy, highest_obstacle,
        table_z, clearance, free_tilt_deg, nominal, intrinsics, cancelled=cancelled,
        progress=progress, max_poses=max_poses, max_checks=max_checks))
    return ([] if _planning_cancelled(arm, cancelled) else poses), minimum


def safe_inspection_route(arm, grasp, current, inspection, neighbors, opening,
                          table_z=0., nominal=None, neighbor_nominal=None,
                          free_tilt_deg=40., *, cancelled=None, progress=None, max_checks=None,
                          opening_uncertainty=0., allow_joint_route=False, bend_reference=None,
                          allow_table_departure=False, endpoint_clearance=0.):
    """Return a fully prechecked linear held-camera route, or hold position.

    Endpoint reachability alone does not establish a reachable lift or a safe
    swept volume. Each proposed route keeps the measured grasp fixed, checks
    the carried box and both finger boxes through the actual interpolated TCP
    path, and invokes the arm's normal linear-path preflight before any motion.
    The small interval padding bounds translation plus rotation between samples.
    Each finger also covers half of the permitted total opening variation.
    """
    if _planning_cancelled(arm, cancelled):
        return [], 'aborted'
    try:
        opening_uncertainty = float(opening_uncertainty)
        endpoint_clearance = float(endpoint_clearance)
    except (ValueError, TypeError):
        return [], 'inspection route has invalid opening uncertainty'
    if (not math.isfinite(opening_uncertainty) or opening_uncertainty < 0.
            or not math.isfinite(endpoint_clearance) or endpoint_clearance < 0.):
        return [], 'inspection route has invalid opening uncertainty'
    import cv2
    from ..robot.kinematics import rot_log
    from ..robot.safety import interpolate_pose

    check = getattr(arm, 'check_reachable', None)
    if check is None:
        return [], 'inspection route has no reachability preflight'
    det = grasp.target
    length, width, height = _dimensions(det, nominal)
    table = detection_table_z(det)
    if table is None:
        return [], 'inspection route has no measured support height'
    box = object_prism(det.center_base[:2], length, width, float(det.yaw or 0.),
        table, height, label=det.label, resting_pose=det.extra.get('resting_pose'))
    local_boxes = [(box - grasp.T_grasp[:3, 3]) @ grasp.T_grasp[:3, :3]]
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    for sign in (-1, 1):
        center = np.array([0., sign * (float(opening) / 2 + .006), -.025])
        local_boxes.append(center + signs * [.010, .006 + opening_uncertainty / 2, .025])
    radius = max(float(np.max(np.linalg.norm(b, axis=1))) for b in local_boxes)
    source_support = (det.extra.get('support_label')
                      if det.extra.get('support_observed') is True else None)
    try:
        support_z = float(det.extra.get('support_z', math.nan))
    except (TypeError, ValueError):
        support_z = math.nan
    obstacles = []
    for other in neighbors:
        if _planning_cancelled(arm, cancelled):
            return [], 'aborted'
        if other is det or other.center_base is None:
            continue
        try:
            a, b, h = _dimensions(other, (neighbor_nominal or {}).get(other.label))
            top = float(other.top_z)
            yaw = float(other.yaw)
        except (ValueError, TypeError):
            return [], 'inspection route has incomplete neighbor geometry'
        was_support = (other.valid and source_support == other.label
                       and abs(top - support_z) < .005)
        obstacles.append((object_footprint(other.center_base[:2], a, b, yaw,
                          label=other.label, resting_pose=other.extra.get('resting_pose')), top - h, top,
                          str(other.label), np.asarray(other.center_base[:2]), was_support))
    blocked_by = ['table or unconfirmed geometry']

    def overlap(a, b, gap):
        for polygon in (a, b):
            for edge in np.roll(polygon, -1, axis=0) - polygon:
                axis = np.array([-edge[1], edge[0]])
                size = float(np.linalg.norm(axis))
                if size < 1e-10:
                    continue
                axis /= size
                u, v = a @ axis, b @ axis
                if u.max() + gap < v.min() or v.max() + gap < u.min():
                    return False
        return True

    def clear(T, pad, departing_support=False):
        if _planning_cancelled(arm, cancelled):
            return False
        if (bend_reference is not None and _tapered_contact(grasp)
                and _grip_bend(bend_reference, T) > math.radians(15.25)):
            blocked_by[0] = 'measured tapered contact would bend beyond 15 degrees'
            return False
        for local in local_boxes:
            carried = local @ T[:3, :3].T + T[:3, 3]
            low, high = float(carried[:, 2].min()), float(carried[:, 2].max())
            if low < float(table_z) + pad:
                # A freshly remeasured release can already touch the table.
                # Permit only an unchanged-attitude vertical raise out of that
                # contact; sideways drag, rotation and penetration still fail.
                departing = (allow_table_departure and low >= float(table_z)-.002
                    and T[2,3] >= current[2,3]-1e-6
                    and np.linalg.norm(T[:2,3]-current[:2,3]) < 1e-6
                    and np.allclose(T[:3,:3],current[:3,:3],atol=1e-6,rtol=0.))
                if not departing:
                    blocked_by[0] = 'table'
                    return False
            polygon = cv2.convexHull(carried[:, :2].astype(np.float32)).reshape(-1, 2)
            for footprint, bottom, top, label, xy, was_support in obstacles:
                if low > top + pad or high < bottom - pad:
                    continue
                if overlap(polygon, footprint, pad):
                    # A lower grasp can leave the carried body just above its
                    # measured former support, inside the EXTRA sweep margin.
                    # Only a fixed-attitude vertical raise may restore that
                    # margin. Actual overlap, sideways motion and other bodies
                    # still fail; every segment endpoint gets the full check.
                    if (departing_support and was_support and low >= top + .001
                            and T[2, 3] >= current[2, 3] - 1e-6
                            and np.linalg.norm(T[:2, 3] - current[:2, 3]) < 1e-6
                            and np.allclose(T[:3, :3], current[:3, :3], atol=1e-6, rtol=0.)):
                        continue
                    blocked_by[0] = '%s at (%.3f, %.3f)' % (label, *xy)
                    return False
        return True

    current, inspection = np.asarray(current, float), np.asarray(inspection, float)
    choices = [[inspection]]
    if allow_table_departure or source_support is not None:
        # Full-height fixed-attitude raises can be unreachable from a low
        # release. Clear the contact vertically first, then turn above it.
        for height in (.016, .02, .04):
            departure = current.copy()
            departure[2, 3] += height
            choices.append([departure, inspection])
    # Retract without rotating the held object before a high/leaned view. This
    # avoids asking a near-extended arm for an impossible fixed-XY vertical lift.
    radial = current[:2, 3] / max(float(np.linalg.norm(current[:2, 3])), 1e-6)
    for distance in (.06, .12):
        inward = current.copy()
        inward[:2, 3] -= distance * radial
        choices.append([inward, inspection])
    raised = current.copy()
    raised[2, 3] = max(float(current[2, 3]), float(inspection[2, 3]))
    choices.append([raised, inspection])
    last_reason = 'no collision-free reachable inspection route'
    checks = 0
    for route_index, route in enumerate(choices):
        if _planning_cancelled(arm, cancelled):
            return [], 'aborted'
        if progress is not None:
            progress({'phase': 'inspection_route', 'route': route_index + 1, 'checked': checks})
        start = current
        joints = arm.get_joints() if hasattr(arm, 'get_joints') else None
        seed = None if joints is None else joints.value
        accepted = True
        for end in route:
            if _planning_cancelled(arm, cancelled):
                return [], 'aborted'
            if max_checks is not None and checks >= max_checks:
                return [], 'inspection route check budget exhausted'
            kwargs = {'max_tilt_deg': free_tilt_deg} if accepts_kwarg(check, 'max_tilt_deg') else {}
            if accepts_kwarg(check, 'q_seed'):
                kwargs['q_seed'] = seed
            ok, last_reason = _planning_call(arm, cancelled, check, end, **kwargs)
            checks += 1
            if _planning_cancelled(arm, cancelled):
                return [], 'aborted'
            if not ok:
                accepted = False
                break
            distance = float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
            angle = float(np.linalg.norm(rot_log(start[:3, :3].T @ end[:3, :3])))
            count = max(1, int(math.ceil(max(distance / .008, angle / math.radians(4)))))
            pad = .008 + (distance + radius * angle) / (2 * count)
            samples = [interpolate_pose(start, end, k / count) for k in range(count + 1)]
            if (not all(clear(pose, pad, departing_support=True) for pose in samples)
                    or not clear(end, max(pad, endpoint_clearance))):
                if _planning_cancelled(arm, cancelled):
                    return [], 'aborted'
                last_reason, accepted = ('inspection swept held object or fingers overlaps a measured obstacle: '
                                         + blocked_by[0]), False
                break
            native = getattr(arm, '_check_linear_path', None)
            if native is not None:
                ok, last_reason = _planning_call(arm, cancelled, native, start, end, seed, free_tilt_deg)
                if _planning_cancelled(arm, cancelled):
                    return [], 'aborted'
                if ok and getattr(arm, '_kin', None) is not None:
                    seed, _ep, _eo = _planning_call(arm, cancelled, arm._ik_tcp, end, seed)
                    if _planning_cancelled(arm, cancelled):
                        return [], 'aborted'
            else:
                ok = True
                for pose in samples:
                    if _planning_cancelled(arm, cancelled):
                        return [], 'aborted'
                    if max_checks is not None and checks >= max_checks:
                        return [], 'inspection route check budget exhausted'
                    ok, last_reason = _planning_call(arm, cancelled, check, pose, **kwargs)
                    checks += 1
                    if _planning_cancelled(arm, cancelled):
                        return [], 'aborted'
                    if not ok:
                        break
            if not ok:
                accepted = False
                break
            start = end
        if accepted:
            return route, ''
    # A Cartesian line may cross a wrist singularity although a joint-space
    # transfer is reachable. Check its actual FK sweep, not the Cartesian
    # chord, and execute precisely these joint endpoints when accepted.
    kin = getattr(arm, '_kin', None)
    joints = arm.get_joints() if hasattr(arm, 'get_joints') else None
    if (allow_joint_route and kin is not None and joints is not None
            and hasattr(arm, '_ik_tcp') and hasattr(arm, '_T_ft')):
        start_q = np.asarray(joints.value, float)
        end_q, _ep, _eo = _planning_call(arm, cancelled, arm._ik_tcp, inspection, start_q)
        if end_q is not None:
            count = max(2, int(math.ceil(float(np.max(np.abs(end_q-start_q))) / math.radians(1.))))
            poses = []
            for fraction in np.linspace(0., 1., count+1):
                if _planning_cancelled(arm, cancelled):
                    return [], 'aborted'
                pose = kin.fk(start_q+fraction*(end_q-start_q)) @ arm._T_ft
                ok, last_reason = arm._guard.check_tcp(pose, table_z, max_tilt_deg=free_tilt_deg)
                if not ok:
                    break
                poses.append(pose)
            if len(poses) == count+1:
                # Bound the between-sample travel of every carried corner.
                corners = np.vstack(local_boxes)
                swept = np.asarray([corners @ pose[:3, :3].T+pose[:3, 3] for pose in poses])
                pad = .008 + float(np.max(np.linalg.norm(np.diff(swept, axis=0), axis=2)))/2
                if (all(clear(pose, pad) for pose in poses)
                        and clear(poses[-1], max(pad, endpoint_clearance))):
                    return JointTransitRoute(inspection, start_q, end_q), ''
                last_reason = 'joint sweep held object or fingers overlaps a measured obstacle: '+blocked_by[0]
    return [], last_reason
