"""Layered observations are reconstructed from RGB-D, never scene metadata."""
import socket

import numpy as np
import pytest

from cube_task.assistant.catalog import make_catalog_object
from cube_task.assistant.catalog_perception import configure_sandbox, make_sandbox_perception
from cube_task.core.config import load_config
from cube_task.core.types import CameraIntrinsics, Frame
from cube_task.core.types import Detection, PerceptionResult, TablePlane
from cube_task.perception.grasp import GraspPlanner, detection_table_z
from cube_task.sim.render import render_rgbd
from cube_task.sim.scenarios import layout_intrinsics, official_wrist_views
from cube_task.sim.world import SimWorld


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('layered camera tests must not access online APIs')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


@pytest.fixture(scope='module')
def camera_pipeline():
    cfg = load_config(use_overlays=False)
    configure_sandbox(cfg)
    cfg.set('perception.roi.pick_polygon', [[.08, -.30], [.50, -.30], [.50, .30], [.08, .30]])
    cfg.set('perception.roi.exclude_zones', False)
    cfg.set('perception.debug_image', False)
    return cfg, make_sandbox_perception(cfg)


def observe(pipeline, objects, view=1, seed=301):
    # Deliberately corrupt every semantic field accepted by the world renderer.
    # The classifier receives only colors, depth and calibration.
    for obj in objects:
        obj.label = 'wrong-world-class'
        obj.name = 'opaque-object-' + str(id(obj))
    intr = CameraIntrinsics.from_dict(layout_intrinsics('A'))
    _q, pose = official_wrist_views('A')[view]
    color, depth = render_rgbd(SimWorld(objects=objects), intr, pose, gripper=False,
                               rng=seed, noise_std=.0008, dropout=.002)
    return pipeline.process(Frame(color, depth, intr, 100. + seed, source='sim'), pose, {})


def stacked_objects(label='apple', offset=0.):
    return [make_catalog_object('red', 'base', [.28, 0.]),
            make_catalog_object(label, 'top', [.28, offset], .2, table_z=.05)]


def test_apple_on_red_box_has_measured_own_height_and_support_identity(camera_pipeline):
    cfg, pipeline = camera_pipeline
    result = observe(pipeline, stacked_objects())
    apples = [d for d in result.detections if d.valid and d.label == 'apple']
    assert len(apples) == 1, [(d.label, d.valid, d.reject_reason, d.height) for d in result.detections]
    target = apples[0]
    assert target.height == pytest.approx(.060, abs=.007)
    assert detection_table_z(target) == pytest.approx(.05, abs=.003)
    assert target.extra['support_label'] == 'red'
    assert target.extra['support_observed'] is True
    assert target.extra['support_pixel_count'] >= 24
    assert len(target.extra['support_polygon']) >= 3
    assert np.linalg.norm(target.center_base[:2] - [.28, 0.]) < .015
    plan = GraspPlanner(cfg, 'Sandbox').plan(target)
    assert plan.feasible, plan.reason
    assert plan.T_grasp[2, 3] > .05


def test_occluded_base_stays_an_incomplete_surface_not_an_invented_full_box(camera_pipeline):
    _cfg, pipeline = camera_pipeline
    result = observe(pipeline, stacked_objects())
    bases = [d for d in result.detections if d.label == 'red']
    assert len(bases) == 1
    assert not bases[0].valid and bases[0].reject_reason == 'occluded_by_stack'
    assert bases[0].extra['support_surface_only']
    # Visible patch width remains measured; never fill to the nominal 50 mm.
    assert bases[0].width < .03
    assert not any(d.valid and d.label == 'red' for d in result.detections)


def test_offset_stacked_block_is_observed_in_a_view_with_visible_support(camera_pipeline):
    _cfg, pipeline = camera_pipeline
    result = observe(pipeline, stacked_objects('green', offset=.015), view=2, seed=302)
    targets = [d for d in result.detections if d.valid and d.label == 'green']
    assert len(targets) == 1, [(d.label, d.valid, d.reject_reason, d.height) for d in result.detections]
    assert targets[0].height == pytest.approx(.05, abs=.003)
    assert targets[0].extra['support_z'] == pytest.approx(.05, abs=.003)
    assert targets[0].extra['support_label'] == 'red'


def test_covered_top_can_use_a_fresh_planar_side_without_inventing_a_graspable_base(camera_pipeline):
    _cfg, pipeline = camera_pipeline
    result = observe(pipeline, stacked_objects('green'), view=1)
    upper = next(d for d in result.detections if d.valid and d.label=='green')
    assert upper.extra['support_surface_kind']=='vertical_face'
    assert upper.extra['support_label']=='red'
    assert not any(d.valid and d.label == 'red' for d in result.detections)


@pytest.mark.parametrize('spurious_peak',[None,.039])
def test_centered_pear_with_covered_base_top_is_confirmed_from_visible_side_only(camera_pipeline,monkeypatch,spurious_peak):
    from cube_task.sim.scenarios import look_at
    from cube_task.perception.geometry import backproject,transform_cloud
    _cfg,pipeline = camera_pipeline
    objects = [make_catalog_object('yellow','base',[.25,0]),
               make_catalog_object('pear','upper',[.25,0],.3,table_z=.05)]
    intr = CameraIntrinsics.from_dict(layout_intrinsics('A'))
    pose = look_at([-.03,0.,.18],[.25,0.,.08])
    color,depth = render_rgbd(SimWorld(objects=objects),intr,pose,gripper=False,
                              rng=301,noise_std=.0008,dropout=.002)
    frame = Frame(color,depth,intr,100.,source='sim')
    if spurious_peak is not None:
        from cube_task.assistant import stack_perception
        monkeypatch.setattr(stack_perception,'_observed_levels',lambda points,table:
                            [(spurious_peak,np.zeros(depth.shape,dtype=bool))])
    result = pipeline.process(frame,pose,{})
    upper = next(d for d in result.detections if d.valid and d.label=='pear')
    assert upper.extra['support_surface_kind']=='vertical_face'
    assert upper.extra['support_label']=='yellow'
    assert detection_table_z(upper)==pytest.approx(.05,abs=.003)
    # Removing every measured base pixel removes the support evidence too.
    points = transform_cloud(backproject(depth,intr,.15,3.),pose)
    hidden = (points[...,2]<.052) & (np.linalg.norm(points[...,:2]-[.25,0.],axis=-1)<.04)
    depth = depth.copy();depth[hidden]=0.
    result = pipeline.process(Frame(color,depth,intr,101.,source='sim'),pose,{})
    assert not any(d.extra.get('support_observed') for d in result.detections)


def test_single_tall_pear_cannot_supply_its_own_fictional_support_plane(camera_pipeline):
    _cfg, pipeline = camera_pipeline
    result = observe(pipeline, [make_catalog_object('pear', 'pear', [.28, 0.], .3)])
    pears = [d for d in result.detections if d.valid and d.label == 'pear']
    assert len(pears) == 1, [(d.label, d.valid, d.reject_reason, d.height) for d in result.detections]
    assert detection_table_z(pears[0]) == pytest.approx(0., abs=.003)
    assert not any(d.extra.get('layered_observation') for d in result.detections)


def test_cropped_upper_object_retains_existing_image_border_rejection(camera_pipeline):
    _cfg, pipeline = camera_pipeline
    result = observe(pipeline, stacked_objects(), view=0, seed=300)
    assert not any(d.valid and d.label == 'apple' for d in result.detections)
    assert any(d.reject_reason == 'touches_image_border' for d in result.detections)


def test_empty_or_missing_depth_frame_does_not_create_layers(camera_pipeline):
    _cfg, pipeline = camera_pipeline
    intr = CameraIntrinsics.from_dict(layout_intrinsics('A'))
    _q, pose = official_wrist_views('A')[1]
    result = pipeline.process(Frame(np.zeros((intr.height, intr.width, 3), np.uint8),
                                    None, intr, 1.), pose, {})
    assert not result.detections and 'no_depth' in result.warnings


def test_competing_measured_support_heights_remain_ambiguous(camera_pipeline, monkeypatch):
    from cube_task.assistant import stack_perception
    _cfg, pipeline = camera_pipeline
    intr = CameraIntrinsics(width=20, height=20, fx=30., fy=30., cx=10., cy=10.)
    mask = np.ones((20, 20), dtype=bool)
    frame = Frame(np.zeros((20, 20, 3), np.uint8), np.ones((20, 20), np.float32), intr, 1.)
    old = Detection('apple', .99, mask=mask, center_base=np.array([.28, 0., .15]),
                    top_z=.15, height=.15, valid=True)
    result = PerceptionResult([old], table=TablePlane(np.array([0., 0., 1.]), 0., 'base'))
    monkeypatch.setattr(stack_perception, '_observed_levels',
                        lambda *args: [(.05, mask), (.08, mask)])
    monkeypatch.setattr(stack_perception._LayerPerception, 'process',
        lambda self, *args: PerceptionResult([Detection('apple', .99, mask=mask,
            center_base=np.array([.28, 0., .15]), top_z=.15, height=.06,
            length=.055, width=.044, valid=True)]))
    monkeypatch.setattr(stack_perception, '_support_evidence', lambda *args: dict(
        label='red', level=.05, polygon=[[.25, -.02], [.30, -.02], [.30, .02]],
        points=100, mask=mask, center=np.array([.28, 0.]), yaw=0., length=.05,
        width=.02, bbox=(0, 0, 20, 20)))
    assert stack_perception.refine_supported_objects(frame, np.eye(4), {}, result,
                                                     pipeline, lambda det: True)
    assert not any(d.valid for d in result.detections)
    assert old.reject_reason == 'support_height_ambiguous'
