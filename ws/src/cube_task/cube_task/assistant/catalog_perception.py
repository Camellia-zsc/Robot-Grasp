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
from ..core.transforms import make_T, rpy_to_rot
from ..perception.classify_knn import KnnClassifier
from ..perception.classify_color import ColorClassifier
from ..perception.features import feature_vector
from ..perception.pipeline import Perception
from ..perception.visualize import draw_result
from ..sim.render import render_rgbd
from ..sim.scenarios import layout_intrinsics, look_at, official_wrist_views
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
    # The interactive tabletop extends past the competition's +/-40 cm strip.
    # Keep the radial envelope, x limits, joint IK and collision checks; the
    # old y rectangle must not veto an otherwise reachable tabletop grasp.
    radius = float(cfg.get('robot.workspace.r_max', .62))
    cfg.set('robot.workspace.y', [-radius, radius])
    # The interactive physical task lowers to the measured support, rather
    # than using the competition fruit's 10 mm drop and per-label lowering.
    # RGB-D/support and seated-grip error need a few mm of allowance: a 2 mm
    # command compressed the pear against the table before opening and slid it.
    # Applying the old 5 mm pear offset here would penetrate the support.
    cfg.set('place.release_height_flat', 0.004)
    cfg.set('place.release_lower', {})
    # Keep a small reserve for the difference between two RGB-D silhouettes.
    # Verification still requires its original five-millimetre whole outline.
    cfg.set('assistant.named_zone_measurement_margin', .015)
    cfg.set('assistant.dynamic_scene_clearance', True)
    # The tabletop is fixed in base coordinates in this editor. Reuse its
    # measured plane briefly across wrist poses; object depth/segmentation is
    # still recomputed for every image. Teaching examples explicitly refit.
    cfg.set('perception.table.reuse_s', 3.)
    # A curved/long body tilts at a free-space wrist pose. Keep its whole
    # envelope above tabletop objects, including a newly moved tall fruit.
    cfg.set('grasp.transit_z_min', max(.22, float(cfg.get('grasp.transit_z_min', .15))))
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
    # This dashboard's TCP is the fingertip: the distal 30 mm pad lies ABOVE
    # it, not +/-15 mm around it. A fictitious lower half rejected the wider
    # neck and made the actual fingers pinch the small tip instead.
    cfg.set('grasp.pad_center_offset', 0.015)
    # Search the existing mechanical depth limit (normally 40 mm), keeping
    # aperture, palm clearance, measured taper and neighboring-object checks.
    depths['pear'] = float(cfg.get('grasp.max_grasp_depth', 0.030))
    # On a 50 mm solid cube the 30 mm distal pad fits from z=20 to 50 mm.
    # The pear's 40 mm search depth put cube tips at z=10 mm; a leaned full
    # opening then scraped the table and changed the supposedly rigid grip.
    for label in ('red', 'green', 'yellow'):
        depths[label] = min(.030, float(cfg.get('grasp.max_grasp_depth', .030)))
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
    if not np.all(np.isfinite(measured)):
        return False
    if (np.all(measured >= minimum)
            and np.all(measured <= _MAX_GEOMETRY[det.label])):
        return True
    return det.label == 'pear' and pear_pose(det) == 'side'


def pear_pose(det) -> str:
    """Conservative resting pose from measured RGB-D, never the body transform.

    A low bulb alone cannot prove a sideways pear. Require the full long neck
    and broad body in the measured 3-D cloud as well as plausible extents.
    This preserves the existing cropped/occluded-instance rejection upstream.
    """
    try:
        size = np.asarray([det.length, det.width, det.height], dtype=float)
        if not np.all(np.isfinite(size)):
            return 'uncertain'
        if (np.all(size >= _MIN_GEOMETRY['pear'])
                and np.all(size <= _MAX_GEOMETRY['pear'])):
            return 'upright'
        if not (np.all(size >= [.085, .050, .060])
                and np.all(size <= [.135, .110, .095])):
            return 'uncertain'
        cloud = np.asarray(det.extra.get('_points_base', []), dtype=float)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            return 'uncertain'
        cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
        if len(cloud) < 40:
            return 'uncertain'
        centered = cloud - cloud.mean(axis=0)
        _values, axes = np.linalg.eigh(centered.T @ centered)
        axes = axes[:, ::-1]
        projected = centered @ axes
        low, high = np.percentile(projected, [.5, 99.5], axis=0)
        extents = high - low
        if (extents[0] >= .105 and extents[1] >= .060
                and abs(axes[2, 0]) <= .65):
            return 'side'
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        pass
    return 'uncertain'


def _center_round_fruit(det) -> bool:
    """Use a measured top cap for a complete upright round fruit's axis.

    The rectangle of an obliquely visible curved surface is biased toward the
    camera. Its top cap is a better axis estimate, already used by grasping.
    Do not apply this to a fallen pear, a crop, or a distant/noisy cap.
    """
    if (not det.valid or det.label not in ('pear', 'apple') or not completeness(det)
            or (det.label == 'pear' and pear_pose(det) != 'upright')):
        return False
    try:
        center = np.asarray(det.center_base, dtype=float)
        cap = np.asarray(det.extra.get('cap_center_xy'), dtype=float)
        if (center.shape != (3,) or cap.shape != (2,)
                or not np.all(np.isfinite(np.r_[center, cap]))
                or np.linalg.norm(cap - center[:2]) > .5 * float(det.width)):
            return False
        det.extra['visible_footprint_center_xy'] = center[:2].tolist()
        det.center_base = center.copy()
        det.center_base[:2] = cap
        return True
    except (TypeError, ValueError):
        return False


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
    # displayed initial scene. Pears also need side poses after a real fall;
    # teaching only upright examples makes their new height out of distribution.
    placements = [(0.27, -0.08, 0.0), (0.28, 0.0, 0.5), (0.30, 0.08, 1.2),
                  (0.24, -0.03, 2.2), (0.32, 0.04, -0.5), (0.27, 0.0, -1.1)]
    rows, labels, names = [], [], None
    for entry in catalog_entries():
        label = entry['label']
        count_before = len(labels)
        examples = [(x, y, yaw, 0., 0.) for x, y, yaw in placements]
        if label == 'pear':
            examples += [(x, y, yaw, roll, pitch)
                         for (x, y, yaw), (roll, pitch) in zip(placements,
                             [(1.3, 0.), (1.57, 0.), (1.8, 0.),
                              (0., 1.3), (0., 1.57), (0., 1.8)])]
        for index, (x, y, yaw, roll, pitch) in enumerate(examples):
            obj = make_catalog_object(label, 'teach-%s-%d' % (label, index), [x, y], yaw)
            sample_poses = poses
            if roll or pitch:
                pose = make_T(rpy_to_rot(roll, pitch, yaw), [x, y, 0.])
                obj.T_base_obj = pose
                floor, _top = obj.aabb()
                pose[2, 3] -= floor[2]
                obj.T_base_obj = pose
                sample_poses = poses + [look_at([x - .10, y, .5], [x, y, .03]),
                                        look_at([x + .10, y, .4], [x, y, .03])]
            # Small, deterministic illumination/color variation includes both
            # sides of the circular red hue boundary (0 and 179 in OpenCV).
            # The L1 blocks already vary by a few BGR levels; teaching only one
            # exact swatch made a red block nearer to apple samples after voting.
            color_offsets = ((0, 0, 0), (7, 0, 0), (0, 7, 0),
                             (-7, 0, 0), (0, -7, 0), (4, -4, 3))
            obj.color_bgr = tuple(int(v) for v in np.clip(
                np.asarray(obj.color_bgr, dtype=int) + color_offsets[index % len(color_offsets)], 0, 255))
            world = SimWorld(objects=[obj], seed=index)
            for view, pose in enumerate(sample_poses):
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
        self._catalog_shapes = {entry['label']:entry['shape'] for entry in catalog_entries()}

    def _merged_blocks(self, frame, seg, dets, clf, res):
        # Only use the shared color-boundary splitter. The fixed library also
        # includes round fruit and stems, so block-size cuts are inappropriate.
        return dets

    def process(self, frame, T_base_camera, zones=None):
        result = super().process(frame, T_base_camera, zones)
        from .stack_perception import refine_supported_objects
        changed = refine_supported_objects(frame, T_base_camera, zones, result, self, completeness)
        centered = False
        for det in result.detections:
            det.extra['catalog_shape'] = self._catalog_shapes.get(det.label)
            det.extra['catalog_complete'] = bool(det.valid and completeness(det))
            if det.label == 'pear':
                det.extra['resting_pose'] = pear_pose(det)
            elif det.label == 'banana':
                det.extra['resting_pose'] = 'flat' if completeness(det) else 'uncertain'
            if _center_round_fruit(det):
                centered = True
            if det.valid and det.label != 'unknown' and not completeness(det):
                det.valid = False
                det.reject_reason = 'partial_object'
                det.extra['catalog_partial'] = True
                changed = True
        if changed:
            result.warnings.append('partial_object')
        if (changed or centered) and self.draw:
            result.debug_image = draw_result(frame, result, T_base_camera,
                                            self.default_zones if zones is None else zones,
                                            self.pick_polygon)
        return result


def make_sandbox_perception(cfg) -> Perception:
    """Build RGB-D perception. No world, editor or live labels are accepted."""
    return SandboxPerception(cfg, classifier=_classifier())
