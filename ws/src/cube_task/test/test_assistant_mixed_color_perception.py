"""Recorded RGB-D after the real occupied-slot relocation, without world labels."""
import json
from pathlib import Path

import numpy as np
import pytest

from cube_task.assistant.catalog import make_catalog_object
from cube_task.assistant.catalog_perception import configure_sandbox, make_sandbox_perception
from cube_task.core.config import load_config
from cube_task.core.types import CameraIntrinsics, Frame
from cube_task.sim.render import render_rgbd
from cube_task.sim.world import SimWorld


FIXTURE = Path(__file__).with_name('fixtures') / 'assistant_mixed_color_neighbors.npz'


@pytest.fixture(scope='module')
def recorded():
    # The fixture was rendered from the independent final world saved by the
    # unchanged real intervention test. Object labels were deliberately replaced
    # before rendering. Only images, intrinsics and calibrated poses are stored.
    arrays = np.load(FIXTURE)
    intr = CameraIntrinsics.from_dict(json.loads(str(arrays['intrinsics'])))
    return arrays, intr


def perception(*, split_touching=True):
    cfg = load_config(use_overlays=False)
    configure_sandbox(cfg)
    cfg.set('perception.roi.pick_polygon', [[.07, -.45], [.6, -.45], [.6, .45], [.07, .45]])
    cfg.set('perception.roi.exclude_zones', False)
    cfg.set('perception.segmentation.split_touching', split_touching)
    cfg.set('perception.debug_image', False)
    return make_sandbox_perception(cfg)


def test_actual_red_block_at_hue_wrap_stays_red_and_its_neighbor_stays_green(recorded):
    arrays, intr = recorded
    pipeline = perception()
    frame = Frame(arrays['color0'], arrays['depth0'], intr, 100., source='recorded-rgbd')
    result = pipeline.process(frame, arrays['pose0'], {})
    valid = [d for d in result.detections if d.valid]
    assert sorted(d.label for d in valid) == ['green', 'red'], [d.summary() for d in result.detections]
    for label, key in (('red', 'final_xy'), ('green', 'blocker_xy')):
        det = next(d for d in valid if d.label == label)
        assert det.score_kind == 'knn'
        assert np.linalg.norm(det.center_base[:2] - arrays[key]) < .015


def test_large_red_and_green_seeds_separate_the_same_image_when_depth_split_is_unavailable(recorded):
    arrays, intr = recorded
    pipeline = perception(split_touching=False)
    frame = Frame(arrays['color0'], arrays['depth0'], intr, 101., source='recorded-rgbd')
    result = pipeline.process(frame, arrays['pose0'], {})
    valid = [d for d in result.detections if d.valid]
    assert sorted(d.label for d in valid) == ['green', 'red'], [d.summary() for d in result.detections]
    assert all(d.extra.get('split') == 'color' for d in valid)
    assert all(d.score_kind == 'knn' for d in valid)
    assert not np.any(valid[0].mask & valid[1].mask)


def test_an_actual_red_apple_of_similar_size_is_still_classified_by_shape(recorded):
    arrays, intr = recorded
    obj = make_catalog_object('apple', 'unseen-neighbor-apple', arrays['final_xy'], .4)
    obj.label = 'not-available-to-classifier'
    world = SimWorld(objects=[obj])
    color, depth = render_rgbd(world, intr, arrays['pose0'], gripper=False,
                               rng=781, noise_std=.0008, dropout=.002)
    result = perception().process(Frame(color, depth, intr, 102., source='sim'), arrays['pose0'], {})
    valid = [d for d in result.detections if d.valid]
    assert len(valid) == 1 and valid[0].label == 'apple', [d.summary() for d in result.detections]
    assert valid[0].score_kind == 'knn'
