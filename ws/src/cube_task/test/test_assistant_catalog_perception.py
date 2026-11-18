"""Library classification uses rendered RGB-D, with no edited scene access."""
import math

import numpy as np
import pytest

from cube_task.assistant.catalog import make_catalog_object
from cube_task.assistant.catalog_perception import (completeness, configure_sandbox,
                                                    make_sandbox_perception,
                                                    _center_round_fruit)
from cube_task.core.config import load_config
from cube_task.core.transforms import pose6_to_T
from cube_task.core.types import CameraIntrinsics, Detection, Frame
from cube_task.perception.grasp import GraspPlanner
from cube_task.sim.render import render_rgbd
from cube_task.sim.scenarios import layout_intrinsics, look_at, official_wrist_views, piper_fk
from cube_task.sim.world import SimWorld
from cube_task.task.views import merge_view_detections, tag_view


@pytest.fixture(scope='module')
def sandbox():
    cfg = load_config(use_overlays=False)
    configure_sandbox(cfg)
    cfg.set('perception.roi.pick_polygon', [[0.08, -0.30], [0.50, -0.30],
                                           [0.50, 0.30], [0.08, 0.30]])
    cfg.set('perception.roi.exclude_zones', False)
    cfg.set('perception.debug_image', False)
    return cfg, make_sandbox_perception(cfg)


def observe(cfg, pipeline, label, yaw=0.33):
    obj = make_catalog_object(label, 'unseen-texture', [0.28, 0.0], yaw)
    # Changing world semantic metadata must not alter camera classification.
    obj.label = 'incorrect-world-label'
    world = SimWorld(objects=[obj])
    intr = CameraIntrinsics.from_dict(layout_intrinsics('A'))
    out = []
    for view, (_q, pose) in enumerate(official_wrist_views('A')):
        color, depth = render_rgbd(world, intr, pose, gripper=False,
                                   rng=781 + view, noise_std=0.0008, dropout=0.002)
        out.extend(pipeline.process(Frame(color, depth, intr, 100 + view, source='sim'),
                                    pose, {}).detections)
    return out


def test_sandbox_does_not_turn_fruit_into_l1_blocks(sandbox):
    cfg, _pipeline = sandbox
    assert cfg.get('tasks.L1.labels') == ['red', 'green', 'yellow']
    assert cfg.get('tasks.Sandbox.class_to_zone')['apple'] == 'zone_red'
    assert cfg.get('tasks.Sandbox.class_to_zone')['pear'] == 'zone_green'
    assert cfg.get('tasks.Sandbox.class_to_zone')['banana'] == 'zone_yellow'
    assert cfg.get('perception.segmentation.split_touching') is True
    assert cfg.get('perception.segmentation.split_height_jump_m') == 0.080
    # The default CLI competition profile remains unchanged.
    assert load_config(use_overlays=False).get('perception.segmentation.split_touching') is True
    assert load_config(use_overlays=False).get('perception.segmentation.split_height_jump_m') == 0.008


@pytest.mark.parametrize('label', ['red', 'green', 'yellow', 'apple', 'banana', 'pear'])
def test_actual_noisy_rgbd_classifies_each_unseen_catalog_object(sandbox, label):
    cfg, pipeline = sandbox
    detections = observe(cfg, pipeline, label)
    valid = [d for d in detections if d.valid and d.label == label]
    assert valid, [(d.label, d.score, d.reject_reason, d.length, d.width, d.height)
                   for d in detections]
    best = max(valid, key=lambda d: d.n_points)
    assert best.score_kind == 'knn'
    assert best.center_base is not None
    assert np.linalg.norm(best.center_base[:2] - [0.28, 0.0]) < 0.025
    plan = GraspPlanner(cfg, 'Sandbox').plan(best)
    assert plan.feasible, plan.reason
    assert best.extra['grasp']['mode'] == 'profile'
    if label in ('apple', 'banana', 'pear'):
        assert best.extra['grasp']['kind'] != 'square'
    if label == 'banana':
        assert best.summary()['resting_pose'] == 'flat'
    elif label == 'pear':
        assert best.summary()['resting_pose'] == 'upright'


def test_cropped_pear_and_unknown_shapes_are_not_called_complete():
    pear = Detection('pear', 0.9, length=0.05, width=0.019, height=0.072)
    assert not completeness(pear)
    pear.length, pear.width, pear.height = 0.075, 0.044, 0.114
    assert completeness(pear)
    pear.label = 'unknown'
    assert not completeness(pear)


def test_a_merged_pear_and_apple_cannot_pass_as_one_executable_pear():
    merged = Detection('pear', 0.99, length=0.145, width=0.069, height=0.111)
    assert not completeness(merged)


def fallen_pear():
    """Replay the incident rotation; world semantics deliberately disagree."""
    obj = make_catalog_object('pear', 'fallen-camera-replay', [.35, -.20])
    pose = np.array([
        [.5535739493, -.7386684130, .3846100081, .3346393778],
        [-.1508933966, -.5431475409, -.8259672704, -.1509646834],
        [.8190159129, .3991988534, -.4121325151, .0598858815],
        [0., 0., 0., 1.]])
    obj.T_base_obj = pose
    obj.label = 'not-a-pear-in-world-metadata'
    return obj


def test_complete_sideways_pear_is_recognized_from_rgbd_but_hidden_neck_is_not(sandbox):
    _cfg, pipeline = sandbox
    intr = CameraIntrinsics.from_dict(layout_intrinsics('A'))
    world = SimWorld(objects=[fallen_pear()])
    # The old wrist view sees only the bulb. A new calibrated view reveals
    # the whole body and neck, even though its height decreased after a fall.
    hidden = np.asarray(official_wrist_views('A')[0][1])
    complete = look_at([.25, -.20, .5], [.35, -.20, .03])
    results = []
    for index, pose in enumerate((hidden, complete)):
        color, depth = render_rgbd(world, intr, pose, gripper=False,
            rng=901 + index, noise_std=.0008, dropout=.002)
        frame = Frame(color, depth, intr, 1000. + index, source='sim')
        results.append(pipeline.process(frame, pose, {}).detections)
    assert not any(d.valid and d.label == 'pear' for d in results[0])
    valid = [d for d in results[1] if d.valid and d.label == 'pear']
    assert len(valid) == 1, [(d.label, d.reject_reason, d.length, d.width, d.height)
                             for d in results[1]]
    assert valid[0].height < .095
    assert valid[0].extra['resting_pose'] == 'side'
    assert valid[0].extra['catalog_complete'] is True
    assert valid[0].summary()['resting_pose'] == 'side'


def test_short_pear_geometry_without_complete_camera_cloud_stays_incomplete():
    bulb_only = Detection('pear', .99, length=.076, width=.055, height=.082)
    assert not completeness(bulb_only)
    # A plausibly large rectangle alone does not prove a complete lying pear.
    bulb_only.length, bulb_only.width = .098, .075
    assert not completeness(bulb_only)


@pytest.mark.parametrize('label', ['pear', 'apple'])
def test_upright_round_fruit_center_uses_visible_top_cap_without_world_metadata(sandbox, label):
    _cfg, pipeline = sandbox
    obj = make_catalog_object(label, 'cap-center-camera-test', [.31, -.127])
    obj.label = 'deliberately-wrong-world-label'
    intr = CameraIntrinsics.from_dict(layout_intrinsics('A'))
    _q, pose = official_wrist_views('A')[1]
    color, depth = render_rgbd(SimWorld(objects=[obj]), intr, pose, gripper=False,
        rng=413, noise_std=.0008, dropout=.002)
    result = pipeline.process(Frame(color, depth, intr, 1300., source='sim'), pose, {})
    valid = [d for d in result.detections if d.valid and d.label == label]
    assert len(valid) == 1
    target = valid[0]
    assert np.allclose(target.center_base[:2], target.extra['cap_center_xy'])
    assert np.linalg.norm(target.center_base[:2] - [.31, -.127]) < .010
    assert 'visible_footprint_center_xy' in target.extra
    assert np.linalg.norm(np.asarray(target.extra['visible_footprint_center_xy'])
                          - [.31, -.127]) > np.linalg.norm(target.center_base[:2] - [.31, -.127])


@pytest.mark.parametrize('case', ['crop', 'side', 'distant_cap', 'nonfinite_cap'])
def test_round_axis_center_does_not_correct_unreliable_or_sideways_views(case):
    target = Detection('pear', .99, center_base=np.array([.3, -.1, .113]),
        length=.075, width=.045, height=.113, extra={'cap_center_xy': [.31, -.1]})
    if case == 'crop':
        target.valid = False
    elif case == 'side':
        target.height = .083
    elif case == 'distant_cap':
        target.extra['cap_center_xy'] = [.34, -.1]
    elif case == 'nonfinite_cap':
        target.extra['cap_center_xy'] = [np.nan, -.1]
    before = target.center_base.copy()
    assert not _center_round_fruit(target)
    np.testing.assert_array_equal(target.center_base, before)


def test_a_single_pear_neck_and_bulb_stay_one_instance_in_the_actual_wrist_view(sandbox):
    cfg, pipeline = sandbox
    obj = make_catalog_object('pear', 'wrist-pear', [0.28, 0], 0.6)
    world = SimWorld(objects=[obj])
    intr = CameraIntrinsics.from_dict(layout_intrinsics('A'))
    q, pose = official_wrist_views('A')[1]
    tcp = piper_fk(q) @ pose6_to_T(cfg.get('robot.tcp_offset'))
    color, depth = render_rgbd(world, intr, pose, gripper=(tcp, 0.1),
                               rng=543, noise_std=0.0008, dropout=0.002)
    result = pipeline.process(Frame(color, depth, intr, 200.0, source='sim'), pose, {})
    valid = [d for d in result.detections if d.valid and d.label == 'pear']
    assert len(valid) == 1, [(d.label, d.reject_reason, d.length, d.width, d.height)
                             for d in result.detections]
    det = valid[0]
    assert det.height > 0.095 and det.length > 0.055
    plan = GraspPlanner(cfg, 'Sandbox').plan(det)
    assert plan.feasible, plan.reason
    assert det.extra['grasp']['mode'] == 'profile'


def test_mixed_fruit_stays_separate_when_their_camera_projections_connect(sandbox):
    cfg, pipeline = sandbox
    placements = [('banana', [0.26, -0.12], -0.4), ('pear', [0.26, 0.07], 0.6),
                  ('apple', [0.35, 0.05], 0.3)]
    world = SimWorld(objects=[make_catalog_object(label, 'mixed-' + label, xy, yaw)
                              for label, xy, yaw in placements])
    # No semantic labels survive into the scene rendered for inference.
    for obj in world.objects:
        obj.label = 'not-an-inference-source'
    intr = CameraIntrinsics.from_dict(layout_intrinsics('A'))
    views = {}
    for index, (q, pose) in enumerate(official_wrist_views('A')):
        tcp = piper_fk(q) @ pose6_to_T(cfg.get('robot.tcp_offset'))
        color, depth = render_rgbd(world, intr, pose, gripper=(tcp, 0.1),
                                   rng=771 + index, noise_std=0.0008, dropout=0.002)
        frame = Frame(color, depth, intr, 300.0 + index, source='sim')
        result = pipeline.process(frame, pose, {})
        views[index] = tag_view(result.detections, index)
    merged = merge_view_detections(views, partial=lambda d: not d.valid)
    known = [d for d in merged if d.valid and d.label in {'apple', 'banana', 'pear'}]
    assert sorted(d.label for d in known) == ['apple', 'banana', 'pear'], [
        (d.label, d.valid, d.reject_reason, d.length, d.width, d.height) for d in merged]
    for label, xy, _yaw in placements:
        det = next(d for d in known if d.label == label)
        assert np.linalg.norm(det.center_base[:2] - xy) < 0.025


@pytest.mark.parametrize('yaw', [0.0, math.pi / 2, -0.7])
def test_banana_different_yaws_keep_shape_aware_profile_grasps(sandbox, yaw):
    cfg, pipeline = sandbox
    valid = [d for d in observe(cfg, pipeline, 'banana', yaw) if d.valid and d.label == 'banana']
    assert valid
    best = max(valid, key=lambda d: d.n_points)
    plan = GraspPlanner(cfg, 'Sandbox').plan(best)
    assert plan.feasible, plan.reason
    assert best.extra['grasp']['kind'] == 'elongated'
