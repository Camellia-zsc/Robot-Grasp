"""Camera-only support planes for observing an object above another object.

Only measured RGB-D surfaces enter this module. The object catalogue supplies
class priors through the existing classifier; there is no world/editor input.
An occluded lower object remains an incomplete observation, never a completed
50 mm box invented beneath an upper silhouette.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..core.types import Detection, TablePlane
from ..perception.classify_color import ColorClassifier
from ..perception.geometry import backproject, transform_cloud
from ..perception.pipeline import Perception, annotate_neighbors
from ..perception.segment import rect_geometry


class _LayerPerception(Perception):
    def __init__(self, cfg, classifier, plane):
        super().__init__(cfg, classifier=classifier)
        self._measured_support = plane
        self.draw = False

    def _table_for(self, frame, pts, zones, T_base_camera, res):
        return self._measured_support

    def _merged_blocks(self, frame, seg, dets, clf, res):
        return dets


def _observed_levels(points, table_z):
    """Find local horizontal-surface height peaks, not nominal stack heights."""
    z = points[..., 2]
    valid = np.isfinite(z)
    smooth = np.stack([cv2.GaussianBlur(np.where(valid, points[..., axis], 0.).astype(
        np.float32), (7, 7), 0) for axis in range(3)], axis=-1)
    # Evaluate normals in base space rather than a fixed pixel-depth jump:
    # curved pear shoulders must not become invented horizontal supports.
    du = np.roll(smooth, -3, 1) - np.roll(smooth, 3, 1)
    dv = np.roll(smooth, -3, 0) - np.roll(smooth, 3, 0)
    normals = np.cross(du, dv)
    magnitude = np.linalg.norm(normals, axis=-1)
    flat = (valid & (magnitude > 1e-10)
            & (np.abs(normals[..., 2]) >= .978 * magnitude))
    flat[:4] = flat[-4:] = False
    flat[:, :4] = flat[:, -4:] = False
    edges = np.arange(table_z + .025, table_z + .232, .003)
    hist, _ = np.histogram(z[flat], bins=edges)
    peaks = [i for i, count in enumerate(hist)
             if count >= 32 and count >= hist[max(0, i - 1)]
             and count >= hist[min(len(hist) - 1, i + 1)]]
    levels = []
    for i in sorted(peaks, key=lambda k: -hist[k]):
        band = flat & (z >= edges[i]) & (z < edges[i + 1])
        level = float(np.median(z[band]))
        if not any(abs(level - old[0]) < .012 for old in levels):
            levels.append((level, flat & (np.abs(z - level) < .003)))
    return sorted(levels, key=lambda item: item[0])[:6]


def _support_evidence(frame, points, flat_mask, det, color):
    """Keep measured support pixels close to the upper footprint only."""
    if det.center_base is None or det.length is None or det.width is None:
        return None
    center = np.asarray(det.center_base[:2])
    yaw = float(det.yaw or 0.)
    u = np.array([math.cos(yaw), math.sin(yaw)])
    v = np.array([-u[1], u[0]])
    delta = points[..., :2] - center
    near = (np.abs(delta @ u) <= float(det.length) / 2 + .007)
    near &= np.abs(delta @ v) <= float(det.width) / 2 + .007
    mask = flat_mask & near
    if np.count_nonzero(mask) < 24:
        return None
    hsv = cv2.cvtColor(frame.color, cv2.COLOR_BGR2HSV)
    classes = color.pixel_classes(hsv[mask])
    known = classes[classes >= 0]
    if len(known) < 24:
        return None
    counts = np.bincount(known, minlength=len(color.labels))
    index = int(np.argmax(counts))
    if counts[index] < 24 or counts[index] / len(classes) < .7:
        return None
    selected = np.zeros(mask.shape, dtype=bool)
    selected[mask] = classes == index
    p = points[selected]
    if len(p) < 24:
        return None
    xy = p[:, :2]
    cxy, axis, length, width = rect_geometry(xy)
    # A narrow color edge or a few isolated pixels do not establish a support
    # surface. This intentionally leaves a completely hidden base unconfirmed.
    if length < .015 or width < .004:
        return None
    polygon = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2)
    if len(polygon) < 3 or cv2.contourArea(polygon) < .00007:
        return None
    ys, xs = np.nonzero(selected)
    return dict(label=color.labels[index], level=float(np.median(p[:, 2])),
                polygon=polygon.astype(float).tolist(), points=len(p), mask=selected,
                center=cxy, yaw=axis, length=length, width=width,
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def refine_supported_objects(frame, T_base_camera, zones, result, pipeline, complete):
    """Replace measured merged stacks with complete upper-layer observations.

    Upper detections carry ``support_z``, ``support_label``,
    ``support_polygon`` and ``support_pixel_count`` in ``extra``. The polygon
    is the convex hull of visible support pixels, not a reconstructed full box.
    The lower patch is retained as ``valid=False / occluded_by_stack``.
    """
    if (result.table is None or frame.depth is None or frame.color is None
            or T_base_camera is None or pipeline.classifier is None):
        return False
    T = np.asarray(T_base_camera, float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    points = transform_cloud(backproject(frame.depth, frame.intrinsics,
                                         pipeline.zmin, pipeline.zmax), T)
    table_z = result.table.z_at(0., 0.)
    levels = _observed_levels(points, table_z)
    color = ColorClassifier(pipeline.cfg.get('perception.classifier.color.classes'),
                            allowed_labels=['red', 'green', 'yellow'])
    accepted, patches, ambiguous = [], [], []
    for level, flat_mask in levels:
        # Do not spend a complete perception pass on the highest visible cap.
        if not any(d.top_z is not None and d.top_z > level + .025
                   for d in result.detections):
            continue
        plane = TablePlane(np.array([0., 0., 1.]), -level, 'base', rms=result.table.rms)
        layer = _LayerPerception(pipeline.cfg, pipeline.classifier, plane)
        observed = layer.process(frame, T, zones)
        for det in observed.detections:
            if not det.valid or not complete(det):
                continue
            evidence = _support_evidence(frame, points, flat_mask, det, color)
            if evidence is None:
                continue
            # One measured top instance cannot be accepted at competing support
            # heights; retain ambiguity rather than selecting a guessed level.
            conflicts = [index for index, old in enumerate(accepted)
                         if old.mask is not None and det.mask is not None
                         and np.count_nonzero(old.mask & det.mask) > .5 * min(
                             np.count_nonzero(old.mask), np.count_nonzero(det.mask))]
            if conflicts:
                ambiguous.append(det.mask)
                for index in reversed(conflicts):
                    ambiguous.append(accepted[index].mask)
                    del accepted[index]
                    del patches[index]
                continue
            if any(np.count_nonzero(mask & det.mask) > .5 * np.count_nonzero(det.mask)
                   for mask in ambiguous):
                continue
            det.extra.update(support_z=evidence['level'], support_label=evidence['label'],
                support_polygon=evidence['polygon'], support_pixel_count=evidence['points'],
                support_observed=True, layered_observation=True)
            accepted.append(det)
            patches.append(evidence)
    if ambiguous:
        for old in result.detections:
            if old.mask is not None and any(np.count_nonzero(old.mask & mask)
                    > .3 * max(1, np.count_nonzero(mask)) for mask in ambiguous):
                old.valid = False
                old.reject_reason = 'support_height_ambiguous'
                old.extra['support_ambiguous'] = True
        result.warnings.append('support_height_ambiguous')
    if not accepted:
        return bool(ambiguous)
    kept = []
    for old in result.detections:
        consumed = any(old.mask is not None and new.mask is not None
                       and np.count_nonzero(old.mask & new.mask) > .3 * max(
                           1, np.count_nonzero(new.mask)) for new in accepted)
        if not consumed:
            kept.append(old)
    for patch in patches:
        z = patch['level']
        kept.append(Detection(patch['label'], 1., score_kind='color',
            bbox=patch['bbox'], center_base=np.r_[patch['center'], z], top_z=z,
            height=z - table_z, yaw=patch['yaw'], length=patch['length'], width=patch['width'],
            mask=patch['mask'], n_points=patch['points'], valid=False,
            reject_reason='occluded_by_stack', extra=dict(table_z=table_z,
                support_surface_only=True, support_polygon=patch['polygon'],
                reject_reasons=['occluded_by_stack'])))
    result.detections = kept + accepted
    annotate_neighbors(result.detections)
    result.warnings.append('layered_objects')
    return True
