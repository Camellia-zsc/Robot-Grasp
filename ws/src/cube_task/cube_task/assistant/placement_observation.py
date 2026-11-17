"""Prove candidate support is visible, and propose guarded held-camera views.

Only camera depth, calibrated transforms, measured object geometry and the
robot's public reachability check are inputs. No world state is read here.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.transforms import inv_T
from ..perception.geometry import points_in_polygon, project_points
from ..perception.grasp import (FINGER_HALF_BAND, FINGER_OUTER, RELEASE_OPEN_EXTRA,
                                detection_table_z)
from ..task.reach import accepts_kwarg


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
                                margin_px=8, depth_tolerance=.008):
    """Return a pose filter proving full support/finger footprints are observed.

    Every sample must project inside the new image and see table depth. A
    foreground object, the carried target itself, a depth hole or image crop
    cannot establish a clear landing area. This complements measured collision
    checks; it never interprets an unobserved area as empty.
    """
    length, width, height = _dimensions(det, nominal)
    table = detection_table_z(det)
    if table is None:
        raise ValueError('candidate visibility requires a measured table height')
    T_bc = np.asarray(T_base_camera, float)
    T_cb = inv_T(T_bc)
    intr = frame.intrinsics if frame is not None else None
    depth = None if frame is None or frame.depth is None else np.asarray(frame.depth)
    source_center = np.r_[det.center_base[:2], table + height / 2]
    original_yaw = float(det.yaw or 0.)
    source_xy = _rectangle(source_center[:2], length, width, original_yaw)
    source_box = np.vstack([np.column_stack([source_xy, np.full(4, z)])
                            for z in (table, table + height)])
    offset = np.asarray(det.center_base[:2]) - grasp.T_grasp[:2, 3]
    opening_default = min(float(max_open_width), float(grasp.expected_width) + RELEASE_OPEN_EXTRA)

    def allowed(x, y, closing, gripper_pose=None, opening=None):
        if depth is None or intr is None or depth.shape != (intr.height, intr.width):
            return False
        delta = float(closing) - float(grasp.closing_angle)
        c, s = math.cos(delta), math.sin(delta)
        rotated_offset = np.array([[c, -s], [s, c]]) @ offset
        tcp_xy = np.array([x, y]) - rotated_offset
        if gripper_pose is None:
            object_polygon = _rectangle(np.array([x, y]), length + .004, width + .004,
                                        original_yaw + delta)
        else:
            import cv2
            pose = np.asarray(gripper_pose, float)
            rotation = pose[:3, :3] @ grasp.T_grasp[:3, :3].T
            carried_box = ((source_box - grasp.T_grasp[:3, 3]) @ rotation.T + pose[:3, 3])
            object_polygon = cv2.convexHull(carried_box[:, :2].astype(np.float32)).reshape(-1, 2)
            tcp_xy = pose[:2, 3]
        actual_opening = opening_default if opening is None else float(opening)
        if gripper_pose is None:
            fingers = [_rectangle(tcp_xy, actual_opening + 2 * FINGER_OUTER,
                                  2 * FINGER_HALF_BAND, float(closing))]
        else:
            import cv2
            signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
            fingers = []
            for sign in (-1, 1):
                center = np.array([0., sign * (actual_opening / 2 + .006), -.025])
                box = ((center + signs * [.010, .006, .025]) @ pose[:3, :3].T + pose[:3, 3])
                fingers.append(cv2.convexHull(box[:, :2].astype(np.float32)).reshape(-1, 2))
        xy = np.vstack([_samples(p, sample_step) for p in [object_polygon] + fingers])
        points = np.column_stack([xy, np.full(len(xy), table + .001)])
        uv = project_points(points, T_bc, intr)
        if (not np.all(np.isfinite(uv)) or np.any(uv[:, 0] < margin_px)
                or np.any(uv[:, 0] > intr.width - 1 - margin_px)
                or np.any(uv[:, 1] < margin_px)
                or np.any(uv[:, 1] > intr.height - 1 - margin_px)):
            return False
        pixels = np.rint(uv).astype(int)
        observed = depth[pixels[:, 1], pixels[:, 0]]
        expected = points @ T_cb[2, :3] + T_cb[2, 3]
        return bool(np.all(np.isfinite(observed)) and np.all(observed > 0)
                    and np.all(observed >= expected - depth_tolerance))
    return allowed


def safe_inspection_poses(arm, grasp, T_flange_camera, T_flange_tcp, target_xy,
                          highest_obstacle, table_z=0., clearance=.04,
                          free_tilt_deg=40., nominal=None, intrinsics=None):
    """Propose reachable TCP poses keeping the held object above all obstacles.

    Callers first lift vertically to at least ``minimum_tcp_z`` before moving
    sideways, then use their existing guarded motion path. The bound uses the
    farthest measured/nominal object corner from the *actual grasp*, so it also
    covers intermediate rotations. This function sends no motion commands.
    Return (poses, minimum_tcp_z); an empty list means hold and report inability
    to inspect, never release blindly.
    """
    check = getattr(arm, 'check_reachable', None)
    length, width, height = _dimensions(grasp.target, nominal)
    det = grasp.target
    table = detection_table_z(det)
    if table is None:
        raise ValueError('inspection requires a measured table height')
    xy = _rectangle(np.asarray(det.center_base[:2]), length, width, float(det.yaw or 0.))
    corners = np.vstack([np.column_stack([xy, np.full(4, z)]) for z in (table, table + height)])
    radius = float(np.max(np.linalg.norm(corners - grasp.T_grasp[:3, 3], axis=1)))
    minimum_tcp_z = max(float(table_z), float(highest_obstacle)) + float(clearance) + radius
    if check is None:
        return [], minimum_tcp_z
    target = np.r_[np.asarray(target_xy, float), float(table_z) + .015]
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
    for closing in closing_angles:
        delta = closing - float(grasp.closing_angle)
        c, s = math.cos(delta), math.sin(delta)
        tcp_xy = target[:2] - np.array([[c, -s], [s, c]]) @ offset
        polygons = [_rectangle(target[:2], length + .004, width + .004,
                               float(det.yaw or 0.) + delta),
                    _rectangle(tcp_xy, opening + 2 * FINGER_OUTER,
                               2 * FINGER_HALF_BAND, closing)]
        samples = np.vstack([_samples(p, .012) for p in polygons])
        support.append(np.column_stack([samples, np.full(len(samples), float(table_z) + .001)]))

    def visibility_score(T_bc):
        T_cb = inv_T(T_bc)
        clear, quality = 0, 0.
        for points in support:
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
            clear += int(np.all(inside))
            quality += float(np.mean(inside))
        return clear, quality

    proposals = []
    for height in (.27, .32, .38, .44):
        for distance, side in ((.14, 0.), (.18, .08), (.18, -.08), (.22, 0.)):
            eye = np.r_[target[:2] - distance * radial + side * lateral, target[2] + height]
            for forward, sideways in ((0., 0.), (-.10, 0.), (.10, 0.), (0., .10),
                                       (0., -.10), (-.08, .08), (-.08, -.08),
                                       (.08, .08), (.08, -.08)):
                gaze = target + np.r_[forward * radial + sideways * lateral, 0.]
                z = gaze - eye
                z /= np.linalg.norm(z)
                x = np.cross(z, [0., 0., 1.])
                if np.linalg.norm(x) < 1e-9:
                    continue
                x /= np.linalg.norm(x)
                y = np.cross(z, x)
                T_bc = np.eye(4)
                T_bc[:3, :3], T_bc[:3, 3] = np.column_stack([x, y, z]), eye
                T_tcp = T_bc @ T_camera_tcp
                if float(T_tcp[2, 3]) >= minimum_tcp_z:
                    proposals.append((visibility_score(T_bc), T_tcp))
    poses = []
    for _score, T_tcp in sorted(proposals, key=lambda item: item[0], reverse=True):
        kwargs = {'max_tilt_deg': free_tilt_deg} if accepts_kwarg(check, 'max_tilt_deg') else {}
        ok, _reason = check(T_tcp, **kwargs)
        if ok:
            poses.append(T_tcp)
            if len(poses) >= 12:
                break
    return poses, minimum_tcp_z
