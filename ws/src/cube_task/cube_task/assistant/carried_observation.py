"""Exclude measured hand/held geometry from an environment-only RGB-D copy.

The original capture stays available for grip calibration and visibility checks.
No current segmentation label or simulated object pose is self-mask evidence.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from ..core.transforms import inv_T, is_rotation
from ..perception.geometry import backproject
from ..perception.grasp import detection_table_z


class CarriedDetections(list):
    """Environment detections with the matching unfiltered calibration frame."""

    def __init__(self, environment, anchor_detections):
        super().__init__(environment)
        self.anchor_detections = anchor_detections


def _pose(value):
    pose = np.asarray(value, float)
    if (pose.shape != (4, 4) or not np.all(np.isfinite(pose))
            or not np.allclose(pose[3], [0., 0., 0., 1.])
            or not is_rotation(pose[:3, :3], 1e-3)):
        raise ValueError('self filtering requires a calibrated rigid pose')
    return pose


def _source_surface_supported(points, target):
    """A curved object's empty box corners cannot establish self geometry."""
    import cv2

    cloud = None
    for key in ('_placement_points_base', '_points_base'):
        try:
            value = np.asarray((target.extra or {}).get(key), dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if value.ndim == 2 and value.shape[1] == 3:
            value = value[np.all(np.isfinite(value), axis=1)]
            if len(value) >= 40:
                cloud = value
                break
    if cloud is None:
        return np.zeros(len(points), bool)
    # Uniform source cells retain sparse parts without weighting dense caps.
    _, indices = np.unique(np.floor(cloud / .002).astype(np.int64), axis=0,
                           return_index=True)
    cloud = cloud[np.sort(indices)]
    if len(cloud) > 4000:
        cloud = cloud[np.linspace(0, len(cloud) - 1, 4000).astype(int)]
    surface = np.ascontiguousarray(cloud, dtype=np.float32)
    result = np.zeros(len(points), bool)
    for begin in range(0, len(points), 4096):
        query = np.ascontiguousarray(points[begin:begin + 4096], dtype=np.float32)
        distance, _ = cv2.batchDistance(query, surface, cv2.CV_32F,
                                      normType=cv2.NORM_L2SQR, K=1)
        result[begin:begin + len(query)] = distance[:, 0] <= .006 ** 2
    return result


def _solid_box_source(target):
    """Static source shape prior; never inspect a current segmentation class."""
    from .catalog import catalog_entries

    return any(entry['id'] == target.label and entry['shape'] == 'box'
               for entry in catalog_entries())


def mask_carried_depth(frame, T_base_camera, grasp, T_base_tcp, opening, nominal=None):
    """Mask only actual depth samples inside the proven held/finger volumes.

    A complete source detection defines its box in the original base frame;
    the measured grasp and current TCP carry that box to the current frame.
    Only a known solid box fills that envelope. Curved/unknown shapes require
    nearby measured source surfaces, preserving objects in concavities and
    empty box corners. Missing source clouds mask only the known fingers.
    Finger boxes match the guarded observation route. A 4 mm envelope covers
    depth noise and small physical seating changes. Missing source geometry,
    invalid transforms, or invalid opening conservatively leave depth intact.
    Same-pixel surfaces outside those three-dimensional volumes stay intact.
    """
    if frame is None or frame.depth is None:
        return frame
    try:
        target = grasp.target
        if not target.valid or not getattr(grasp, 'feasible', True):
            return frame
        center = np.asarray(target.center_base, float)
        size = np.asarray([target.length, target.width, target.height], float)
        floor = detection_table_z(target)
        yaw, opening = float(target.yaw), float(opening)
        if (center.shape != (3,) or floor is None
                or not np.all(np.isfinite(np.r_[center, size, floor, yaw, opening]))
                or np.any(size <= 0.) or not 0. < opening <= .2):
            return frame
        length, width = sorted(size[:2], reverse=True)
        if nominal is not None:
            prior = np.asarray(nominal, float)[:2]
            if prior.shape != (2,) or np.any(prior <= 0.) or not np.all(np.isfinite(prior)):
                return frame
            length, width = max(length, max(prior)), max(width, min(prior))
        source, tcp, camera = (_pose(v) for v in (grasp.T_grasp, T_base_tcp, T_base_camera))
        depth = np.asarray(frame.depth)
        camera_points = backproject(depth, frame.intrinsics, zmin=1e-5, zmax=float('inf'))
        # Convert current camera samples into the fixed measured grasp frame.
        T_tcp_camera = inv_T(tcp) @ camera
        local = camera_points @ T_tcp_camera[:3, :3].T + T_tcp_camera[:3, 3]
        original = local @ source[:3, :3].T + source[:3, 3]
        c, s = math.cos(yaw), math.sin(yaw)
        xy = (original[..., :2] - center[:2]) @ np.array([[c, -s], [s, c]])
        pad = .004
        masked = ((np.abs(xy[..., 0]) <= length / 2 + pad)
                  & (np.abs(xy[..., 1]) <= width / 2 + pad)
                  & (original[..., 2] >= float(floor) - pad)
                  & (original[..., 2] <= float(floor) + size[2] + pad))
        if not _solid_box_source(target):
            positions = np.flatnonzero(masked)
            supported = _source_surface_supported(original.reshape(-1, 3)[positions], target)
            masked.flat[positions[~supported]] = False
        for sign in (-1., 1.):
            finger = np.array([0., sign * (opening / 2 + .006), -.025])
            masked |= np.all(np.abs(local - finger) <= np.array([.010, .006, .025]) + pad,
                             axis=-1)
        masked &= np.isfinite(depth) & (depth > 0.)
    except (AttributeError, TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        return frame
    count = int(np.count_nonzero(masked))
    if not count:
        return frame
    filtered = depth.copy()
    filtered[masked] = 0.
    return replace(frame, depth=filtered,
                   meta=dict(frame.meta or {}, carried_self_masked_pixels=count))
