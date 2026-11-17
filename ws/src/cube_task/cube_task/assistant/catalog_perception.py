"""RGB-D classification of the fixed object library, separate from world state.

Training is supervised by the known class of isolated synthetic objects.
Inference receives only a Frame, camera calibration and configured zones. It
cannot consult the edited scene, world labels or true object coordinates.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import tempfile
import threading

import numpy as np

from .catalog import catalog_entries, make_catalog_object
from ..core.config import load_config
from ..core.types import Frame
from ..perception.classify_knn import KnnClassifier
from ..perception.classify_color import ColorClassifier
from ..perception.features import feature_vector
from ..perception.pipeline import Perception
from ..perception.visualize import draw_result
from ..sim.render import render_rgbd
from ..sim.scenarios import layout_intrinsics, official_wrist_views
from ..sim.world import SimWorld
from ..core.types import CameraIntrinsics

_LOG = logging.getLogger(__name__)
_MODEL_LOCK = threading.Lock()
_MODEL = None
# Measured lower bounds for a sufficiently complete view, from fixed library
# dimensions. They reject cropped pears and thin slivers without filling in
# the hidden surface or substituting the true object center.
_MIN_GEOMETRY = {
    'red': (0.036, 0.032, 0.037),
    'green': (0.036, 0.032, 0.037),
    'yellow': (0.036, 0.032, 0.037),
    'apple': (0.043, 0.034, 0.046),
    'banana': (0.095, 0.023, 0.026),
    'pear': (0.055, 0.032, 0.095),
}
# Large merged instances must not become an executable class merely because
# their color resembles it. Bounds include room for RGB-D noise and stems.
_MAX_GEOMETRY = {
    'red': (0.075, 0.075, 0.065),
    'green': (0.075, 0.075, 0.065),
    'yellow': (0.075, 0.075, 0.065),
    'apple': (0.085, 0.085, 0.080),
    'banana': (0.215, 0.100, 0.060),
    'pear': (0.110, 0.110, 0.145),
}


def configure_sandbox(cfg) -> None:
    """Configure a general-shape task while keeping L1 block semantics intact."""
    entries = catalog_entries()
    cfg.set('tasks.L1.labels', ['red', 'green', 'yellow'])
    cfg.set('tasks.Sandbox', {'labels': [entry['label'] for entry in entries],
                            'class_to_zone': {entry['label']: entry['default_zone']
                                              for entry in entries},
                            'classifier': 'knn', 'time_limit_s': 300.0,
                            'required': len(entries), 'order': []})
    full = list(cfg.get('grasp.open_full_tasks', []) or [])
    if 'Sandbox' not in full:
        full.append('Sandbox')
    cfg.set('grasp.open_full_tasks', full)
    cfg.set('grasp.section_profile', True)
    depths = dict(cfg.get('grasp.max_grasp_depth_by_label', {}) or {})
    # The long stem must remain below the palm. Measured top minus 25 mm
    # keeps the pads around the neck, above the steep shoulder of this type.
    depths['pear'] = 0.025
    cfg.set('grasp.max_grasp_depth_by_label', depths)
    # Even separated objects can touch in the camera projection. Keep the
    # base-XY watershed so it separates these instances, but avoid 8 mm
    # height barriers which cut one pear's neck away from its own bulb.
    # Border, occlusion and completeness checks still apply to every part.
    cfg.set('perception.segmentation.split_touching', True)
    cfg.set('perception.segmentation.split_height_jump_m', 0.080)
    # Preserve the established geometry planners. These are library type
    # dimensions used to reject incomplete measurements, not live truth.
    nominal = dict(cfg.get('place.nominal_footprint', {}) or {})
    for entry in entries:
        nominal[entry['label']] = entry['dimensions'][:2]
    cfg.set('place.nominal_footprint', nominal)


def completeness(det) -> bool:
    """Whether a measured library detection has enough geometry to grasp.

    Unknown classes stay non-executable through the normal decision validator.
    This function is also suitable for the multi-view merge's partial filter.
    """
    minimum = _MIN_GEOMETRY.get(det.label)
    if minimum is None:
        return False
    try:
        measured = np.asarray([det.length, det.width, det.height], dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(measured)) and np.all(measured >= minimum)
                and np.all(measured <= _MAX_GEOMETRY[det.label]))


def _cache_path() -> Path:
    digest = hashlib.sha256(b'robot-grasp-catalog-rgbd-v1')
    root = Path(__file__).resolve().parents[1]
    for name in ('assistant/catalog.py', 'assistant/catalog_perception.py',
                 'perception/features.py', 'perception/segment.py',
                 'perception/classify_knn.py', 'sim/render.py', 'sim/world.py',
                 'sim/scenarios.py', '../config/default.yaml'):
        digest.update((root / name).read_bytes())
    return Path(tempfile.gettempdir()) / 'robot-grasp' / ('catalog-knn-' + digest.hexdigest()[:16] + '.npz')


def _training_samples():
    """Render isolated known examples; never accept an edited runtime world."""
    cfg = load_config(use_overlays=False)
    configure_sandbox(cfg)
    cfg.set('perception.roi.pick_polygon', [[0.08, -0.35], [0.50, -0.35],
                                           [0.50, 0.35], [0.08, 0.35]])
    cfg.set('perception.roi.exclude_zones', False)
    cfg.set('perception.debug_image', False)
    cfg.set('perception.table.reuse_s', 0.0)
    raw = Perception(cfg)
    intr = CameraIntrinsics.from_dict(layout_intrinsics('A'))
    poses = [np.asarray(T) for _q, T in official_wrist_views('A')]
    # Deliberately vary pose/texture/noise instead of training only on the
    # displayed initial scene. Six yaw/position examples per class, 3 views.
    placements = [(0.27, -0.08, 0.0), (0.28, 0.0, 0.5), (0.30, 0.08, 1.2),
                  (0.24, -0.03, 2.2), (0.32, 0.04, -0.5), (0.27, 0.0, -1.1)]
    rows, labels, names = [], [], None
    for entry in catalog_entries():
        label = entry['label']
        count_before = len(labels)
        for index, (x, y, yaw) in enumerate(placements):
            obj = make_catalog_object(label, 'teach-%s-%d' % (label, index), [x, y], yaw)
            # Small, deterministic illumination/color variation includes both
            # sides of the circular red hue boundary (0 and 179 in OpenCV).
            # The L1 blocks already vary by a few BGR levels; teaching only one
            # exact swatch made a red block nearer to apple samples after voting.
            color_offsets = ((0, 0, 0), (7, 0, 0), (0, 7, 0),
                             (-7, 0, 0), (0, -7, 0), (4, -4, 3))
            obj.color_bgr = tuple(int(v) for v in np.clip(
                np.asarray(obj.color_bgr, dtype=int) + color_offsets[index], 0, 255))
            world = SimWorld(objects=[obj], seed=index)
            for view, pose in enumerate(poses):
                color, depth = render_rgbd(world, intr, pose, gripper=False,
                                           rng=200 + 11 * index + view,
                                           noise_std=0.0008, dropout=0.002)
                frame = Frame(color, depth, intr, 1.0 + index * 3 + view, source='sim')
                result = raw.process(frame, pose, {})
                # The world contains one labelled teaching object. Only the
                # segmented visible object supplies features; coordinates and
                # surface points are never read from that world for a sample.
                candidates = [d for d in result.detections if d.valid]
                if not candidates:
                    continue
                det = max(candidates, key=lambda d: d.n_points)
                det.label = label
                if not completeness(det):
                    continue
                row, names = feature_vector(frame, det)
                if np.all(np.isfinite(row)):
                    rows.append(row)
                    labels.append(label)
        if len(labels) - count_before < 3:
            raise RuntimeError('物品 %s 的合成训练视角不足，不能启动识别。' % entry['name'])
    return np.asarray(rows), labels, names


def _classifier() -> KnnClassifier:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        path = _cache_path()
        expected = {entry['label'] for entry in catalog_entries()}
        try:
            model = KnnClassifier.load(path, k=3, max_distance=3.0)
            if set(model.labels) != expected:
                raise ValueError('cached catalog labels do not match')
        except (OSError, ValueError, KeyError, EOFError):
            _LOG.info('Training the fixed library from isolated synthetic RGB-D views')
            rows, labels, names = _training_samples()
            model = KnnClassifier(k=3, max_distance=3.0).fit(rows, labels, names)
            try:
                # Isolate writers from separate app/test processes too; the
                # classifier's own temporary filename is shared per target.
                staging = path.with_name(path.stem + '-%d.npz' % os.getpid())
                model.save(staging)
                os.replace(staging, path)
            except OSError:
                _LOG.warning('Cannot save catalog cache; using the in-memory model')
        _MODEL = model
        return model


class _ColorSeededKnn(ColorClassifier):
    """Expose conservative red/green split seeds, then classify shape with KNN.

    A color seed is only an instance boundary cue. It never becomes the final
    label: an apple remains eligible for apple, and a pear for pear.
    Brown stems and yellow banana bodies do not provide a second seed color.
    """

    def __init__(self, cfg, model):
        super().__init__(cfg.get('perception.classifier.color.classes'),
                         allowed_labels=['red', 'green'])
        self.model = model

    def classify(self, frame, detections):
        self.model.classify(frame, detections)


class SandboxPerception(Perception):
    """The existing image pipeline plus a conservative completeness gate."""

    def __init__(self, cfg, classifier=None, detector=None):
        seeded = _ColorSeededKnn(cfg, classifier) if classifier is not None else None
        super().__init__(cfg, classifier=seeded, detector=detector)

    def _merged_blocks(self, frame, seg, dets, clf, res):
        # Only use the shared color-boundary splitter. The fixed library also
        # includes round fruit and stems, so block-size cuts are inappropriate.
        return dets

    def process(self, frame, T_base_camera, zones=None):
        result = super().process(frame, T_base_camera, zones)
        changed = False
        for det in result.detections:
            if det.valid and det.label != 'unknown' and not completeness(det):
                det.valid = False
                det.reject_reason = 'partial_object'
                det.extra['catalog_partial'] = True
                changed = True
        if changed:
            result.warnings.append('partial_object')
            if self.draw:
                result.debug_image = draw_result(frame, result, T_base_camera,
                                                self.default_zones if zones is None else zones,
                                                self.pick_polygon)
        return result


def make_sandbox_perception(cfg) -> Perception:
    """Build RGB-D perception. No world, editor or live labels are accepted."""
    return SandboxPerception(cfg, classifier=_classifier())
