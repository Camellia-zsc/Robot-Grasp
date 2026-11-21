"""A carried silhouette must not become an environment obstacle of any class."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.carried_observation import CarriedDetections, mask_carried_depth
from cube_task.core.types import CameraIntrinsics, Detection, Frame


@pytest.fixture
def capture():
    intr = CameraIntrinsics(200., 200., 40., 40., 81, 81)
    depth = np.full((81, 81), 1.5, dtype=np.float32)
    color = np.full((81, 81, 3), [160, 190, 60], dtype=np.uint8)
    frame = Frame(color, depth, intr, 12., source='test', meta={'sequence': 7})
    camera = np.eye(4)
    camera[2, 3] = -.5
    target = Detection('red', 1., center_base=np.array([0., 0., .5]),
                       top_z=.5, height=.05, length=.06, width=.05, yaw=0.)
    grasp = SimpleNamespace(target=target, T_grasp=np.eye(4), feasible=True)
    return frame, camera, grasp, np.eye(4)


def test_self_and_external_points_in_one_color_blob_remain_separate(capture):
    frame, camera, grasp, tcp = capture
    # One connected color/depth strip can include both a close carried item and
    # a nearby obstacle. Its left half is self; its right half is external.
    frame.depth[40, 40:52] = .975
    result = mask_carried_depth(frame, camera, grasp, tcp, .08)
    np.testing.assert_array_equal(result.depth[40, 40:47], 0.)
    np.testing.assert_array_equal(result.depth[40, 48:52], frame.depth[40, 48:52])
    assert result.depth[40, 47] == frame.depth[40, 47]
    assert result.meta['carried_self_masked_pixels'] == 7


def test_same_pixel_nearer_or_farther_surface_is_not_a_silhouette_mask(capture):
    frame, camera, grasp, tcp = capture
    # All three samples are in the projected held-object rectangle. Only the
    # actual depth inside its three-dimensional volume belongs to self.
    frame.depth[40, 39:42] = [.8, .975, 1.2]
    result = mask_carried_depth(frame, camera, grasp, tcp, .08)
    assert result.depth[40, 40] == 0.
    assert result.depth[40, 39] == frame.depth[40, 39]
    assert result.depth[40, 41] == frame.depth[40, 41]


def test_finger_volume_uses_current_opening_and_does_not_mask_gap(capture):
    frame, camera, grasp, tcp = capture
    frame.depth[59, 40] = .475  # y=.0451 lies in the positive finger box.
    frame.depth[40, 40] = .475  # Same height in the opening is free space.
    result = mask_carried_depth(frame, camera, grasp, tcp, .08)
    assert result.depth[59, 40] == 0.
    assert result.depth[40, 40] == frame.depth[40, 40]
    wider = mask_carried_depth(frame, camera, grasp, tcp, .12)
    assert wider.depth[59, 40] == frame.depth[59, 40]


def test_filter_keeps_original_capture_for_anchor_and_visibility(capture):
    frame, camera, grasp, tcp = capture
    frame.depth[40, 40] = .975
    before = copy.deepcopy(frame)
    result = mask_carried_depth(frame, camera, grasp, tcp, .08)
    assert result is not frame and result.depth is not frame.depth
    np.testing.assert_array_equal(frame.depth, before.depth)
    np.testing.assert_array_equal(frame.color, before.color)
    assert frame.meta == before.meta
    assert result.color is frame.color and result.intrinsics is frame.intrinsics
    assert result.stamp == 12. and result.source == 'test'
    assert result.meta['sequence'] == 7


def test_current_rigid_pose_change_preserves_camera_relative_mask(capture):
    frame, camera, grasp, tcp = capture
    frame.depth[40, 40] = .975
    angle = .6
    rigid = np.array([[np.cos(angle), 0., np.sin(angle), .4],
                      [0., 1., 0., -.1],
                      [-np.sin(angle), 0., np.cos(angle), .2], [0., 0., 0., 1.]])
    baseline = mask_carried_depth(frame, camera, grasp, tcp, .08)
    changed = mask_carried_depth(frame, rigid @ camera, grasp, rigid @ tcp, .08)
    np.testing.assert_array_equal(baseline.depth, changed.depth)


@pytest.mark.parametrize('bad', ['invalid_source', 'missing_height', 'missing_yaw',
                                  'bad_camera', 'bad_grasp', 'bad_tcp', 'nan_opening',
                                  'no_opening', 'bad_nominal', 'no_depth'])
def test_unproven_geometry_never_erases_environment_depth(capture, bad):
    frame, camera, grasp, tcp = capture
    frame.depth[40, 40] = .975
    opening, nominal = .08, None
    if bad == 'invalid_source':
        grasp.target.valid = False
    elif bad == 'missing_height':
        grasp.target.height = None
    elif bad == 'missing_yaw':
        grasp.target.yaw = None
    elif bad == 'bad_camera':
        camera[0, 0] = 2.
    elif bad == 'bad_grasp':
        grasp.T_grasp[2, 3] = np.nan
    elif bad == 'bad_tcp':
        tcp[3, 0] = 1.
    elif bad == 'nan_opening':
        opening = float('nan')
    elif bad == 'no_opening':
        opening = None
    elif bad == 'bad_nominal':
        nominal = [float('nan'), .05]
    elif bad == 'no_depth':
        frame.depth = None
    assert mask_carried_depth(frame, camera, grasp, tcp, opening, nominal) is frame


def test_rgb_does_not_determine_self_mask(capture):
    frame, camera, grasp, tcp = capture
    frame.depth[40, 40] = .975
    red = mask_carried_depth(frame, camera, grasp, tcp, .08)
    frame.color[:] = [0, 0, 0]
    unknown = mask_carried_depth(frame, camera, grasp, tcp, .08)
    np.testing.assert_array_equal(red.depth, unknown.depth)


@pytest.mark.parametrize('label', ['banana', 'pear', 'apple', 'unknown'])
def test_curved_or_unknown_source_preserves_external_point_in_empty_box_gap(capture, label):
    frame, camera, grasp, tcp = capture
    grasp.target.label = label
    grasp.target.length, grasp.target.width = .14, .06
    # Measured separate curved ribs leave an empty region in their box. A
    # real external surface at the center must survive even at the same z.
    y, z = np.meshgrid(np.linspace(-.015, .015, 15), np.linspace(.46, .49, 15))
    grasp.target.extra['_placement_points_base'] = np.vstack([
        np.column_stack([np.full(y.size, side), y.ravel(), z.ravel()])
        for side in (-.045, .045)])
    frame.depth[40, 40] = .98
    frame.depth[40, 49] = .98
    result = mask_carried_depth(frame, camera, grasp, tcp, .08)
    assert result.depth[40, 40] == frame.depth[40, 40]
    assert result.depth[40, 49] == 0.


@pytest.mark.parametrize('cloud', [None, [], [[0., 0., .475]],
                                    [[float('nan'), 0., .475]] * 50])
def test_complex_source_without_measured_surface_masks_only_known_fingers(capture, cloud):
    frame, camera, grasp, tcp = capture
    grasp.target.label = 'pear'
    grasp.target.extra['_points_base'] = cloud
    frame.depth[40, 40] = .975
    frame.depth[59, 40] = .475
    result = mask_carried_depth(frame, camera, grasp, tcp, .08)
    assert result.depth[40, 40] == frame.depth[40, 40]
    assert result.depth[59, 40] == 0.


def test_frame_bound_raw_detections_survive_environment_filtering():
    raw = [Detection('pear', .8)]
    environment = [Detection('apple', 1.)]
    result = CarriedDetections(environment, raw)
    assert result == environment
    assert result.anchor_detections is raw
    assert copy.deepcopy(result).anchor_detections == raw
