"""Keep live viewer rendering separate from perception and mutable simulation."""
from __future__ import annotations

import copy
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cube_task.assistant import preview as preview_module
from cube_task.assistant.preview import LivePreview
from cube_task.assistant.simulation import SimulationSession
from cube_task.core.types import CameraIntrinsics


def _wait_for(predicate, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail('live preview did not update before the deadline')


def test_live_preview_updates_during_motion_without_capturing_or_mutating(
        tmp_path, monkeypatch):
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False) as session:
        session.prepare()
        clock_before = session.clock.now()
        frames_before = session.camera.frames_rendered
        rng_before = copy.deepcopy(session.camera._rng.bit_generator.state)
        poses_before = [obj.T_base_obj.copy() for obj in session.world.objects]
        tcp_before = session.world.tcp_pose.copy()
        scene_before = session.scene
        last_frame = session.camera._last
        renders = []
        render_rgbd = preview_module.render_rgbd

        def render(*args, **kwargs):
            renders.append(kwargs)
            return render_rgbd(*args, **kwargs)

        def forbidden_capture(*args, **kwargs):
            pytest.fail('a viewer must not acquire a perception frame')

        monkeypatch.setattr(preview_module, 'render_rgbd', render)
        monkeypatch.setattr(session.camera, 'get_frame', forbidden_capture)
        preview = LivePreview(session)
        assert preview.info() == {
            'available': False, 'sim_time': None, 'width': 320, 'height': 240,
            'updated_at': None, 'fps': 3.0, 'mode': 'live'}
        assert preview.frame_jpeg() is None
        try:
            assert preview.start() is preview
            assert preview.start() is preview
            _wait_for(lambda: preview.frame_jpeg() is not None)
            first_jpeg = preview.frame_jpeg()
            first_info = preview.info()
            image = cv2.imdecode(np.frombuffer(first_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert image.shape[:2] == (240, 320)
            assert first_info['available']
            assert first_info['sim_time'] == session.snapshot()['sim_time']
            # Polling an idle session neither renders the same snapshot again
            # nor pretends that the cached image is a newly captured frame.
            time.sleep(0.42)
            assert len(renders) == 1
            assert renders[0] == {'rng': None, 'noise_std': 0.0, 'dropout': 0.0}
            assert preview.info() == first_info
            assert session.clock.now() == clock_before
            assert session.camera.frames_rendered == frames_before
            assert session.camera._rng.bit_generator.state == rng_before
            assert session.camera._last is last_frame
            for obj, pose in zip(session.world.objects, poses_before):
                np.testing.assert_array_equal(obj.T_base_obj, pose)
            np.testing.assert_array_equal(session.world.tcp_pose, tcp_before)
            assert session.scene == scene_before

            joints = session.arm.get_joints().value.copy()
            joints[0] += 0.14
            assert session.arm.move_joints(joints).ok
            session._publish_snapshot()
            expected_time = session.snapshot()['sim_time']
            _wait_for(lambda: preview.info()['sim_time'] == expected_time)
            assert preview.frame_jpeg() != first_jpeg
            assert preview.info()['updated_at'] > first_info['updated_at']
            assert session.camera.frames_rendered == frames_before
            assert session.camera._rng.bit_generator.state == rng_before
            assert session.scene == scene_before
        finally:
            preview.close()
            preview.close()
        assert not preview._thread.is_alive()
        with pytest.raises(RuntimeError, match='closed'):
            preview.start()


def test_snapshot_reconstruction_keeps_held_pose_and_draws_the_gripper(monkeypatch):
    tcp = np.eye(4)
    tcp[:3, 3] = (0.25, 0.10, 0.20)
    flange = np.eye(4)
    flange[:3, 3] = (0.25, 0.10, 0.34)
    mount = np.eye(4)
    mount[:3, 3] = (0.03, 0.0, 0.04)
    held_pose = np.eye(4)
    held_pose[:3, 3] = (0.25, 0.10, 0.17)
    snapshot = {
        'objects': [{'id': 'red_held', 'label': 'red', 'shape': 'box',
                     'size': [0.05, 0.05, 0.05], 'color_rgb': [200, 40, 40],
                     'transform': held_pose.tolist(), 'held': True, 'placed': False}],
        'zones': [{'name': 'zone_red', 'label': 'red', 'center': [0.3, 0.2, 0.0],
                   'size': [0.2, 0.2], 'yaw': 0.2, 'color_rgb': [150, 70, 70]}],
        'table_z': 0.0, 'tcp_transform': tcp.tolist(), 'gripper_width': 0.05,
        'link_frames': [flange.tolist()], 'sim_time': 1.25}
    original = copy.deepcopy(snapshot)
    camera = SimpleNamespace(
        intrinsics=lambda: CameraIntrinsics(400, 400, 320, 240, 640, 480),
        T_flange_camera=mount)
    session = SimpleNamespace(camera=camera, snapshot=lambda: copy.deepcopy(snapshot))
    captured = {}

    def render(world, intrinsics, camera_pose, **kwargs):
        captured.update(world=world, intrinsics=intrinsics, camera_pose=camera_pose,
                        kwargs=kwargs)
        return np.zeros((intrinsics.height, intrinsics.width, 3), np.uint8), None

    monkeypatch.setattr(preview_module, 'render_rgbd', render)
    preview = LivePreview(session, fps=50, width=160)
    camera.T_flange_camera[:] = np.eye(4)  # the preview copied its mount at setup
    assert preview._render(snapshot)
    world = captured['world']
    assert world is not getattr(session, 'world', None)
    assert world.held is world.objects[0]
    np.testing.assert_array_equal(world.held.T_base_obj, held_pose)
    np.testing.assert_array_equal(world.tcp_pose, tcp)
    np.testing.assert_allclose(captured['camera_pose'][:3, 3], (0.28, 0.10, 0.38))
    assert world.gripper_width == 0.05
    assert world.objects[0].color_bgr == (40, 40, 200)
    assert world.zones[0].color_bgr == (70, 70, 150)
    assert (captured['intrinsics'].width, captured['intrinsics'].height) == (160, 120)
    assert captured['kwargs'] == {'rng': None, 'noise_std': 0.0, 'dropout': 0.0}
    assert snapshot == original
    assert preview.info()['fps'] == 3
    preview.close()


def test_render_failure_marks_preview_unavailable_and_retries_previous_snapshot(monkeypatch):
    current = {'sim_time': 1.0}
    camera = SimpleNamespace(
        intrinsics=lambda: CameraIntrinsics(400, 400, 320, 240, 640, 480),
        T_flange_camera=np.eye(4))
    session = SimpleNamespace(camera=camera, snapshot=lambda: copy.deepcopy(current))
    preview = LivePreview(session)
    rendered = []

    def render(snapshot):
        rendered.append(snapshot['sim_time'])
        if snapshot['sim_time'] == 2.0:
            raise RuntimeError('test render failure')
        return b'first-valid-frame' if rendered.count(1.0) == 1 else b'recovered-frame'

    monkeypatch.setattr(preview, '_render', render)
    try:
        preview.start()
        _wait_for(lambda: preview.info()['available'])
        assert preview.frame_jpeg() == b'first-valid-frame'
        first_info = preview.info()

        current['sim_time'] = 2.0
        _wait_for(lambda: not preview.info()['available'], timeout=2.0)
        # Keep the last good image and its timestamp, while accurately marking
        # that the live preview currently cannot render the simulation state.
        assert preview.frame_jpeg() == b'first-valid-frame'
        assert preview.info()['sim_time'] == first_info['sim_time']
        assert preview.info()['updated_at'] == first_info['updated_at']

        # The source can return to a previously rendered state during recovery.
        # Equality with that state must not suppress the retry after failure.
        current['sim_time'] = 1.0
        _wait_for(lambda: preview.info()['available'] and
                  preview.frame_jpeg() == b'recovered-frame', timeout=2.0)
        assert rendered.count(1.0) == 2
        assert preview.info()['updated_at'] > first_info['updated_at']
    finally:
        preview.close()


def test_nonvisual_progress_does_not_retrace_the_same_camera_scene(monkeypatch):
    current = {
        'objects': [{'id': 'apple', 'transform': np.eye(4).tolist(), 'held': False}],
        'zones': [], 'table_z': 0., 'tcp_transform': np.eye(4).tolist(),
        'gripper_width': .04, 'link_frames': [np.eye(4).tolist()],
        'sim_time': 1., 'stage': 'observing', 'scene_version': 1, 'feedback': {},
    }
    camera = SimpleNamespace(
        intrinsics=lambda: CameraIntrinsics(400, 400, 320, 240, 640, 480),
        T_flange_camera=np.eye(4))
    session = SimpleNamespace(camera=camera, snapshot=lambda: copy.deepcopy(current))
    preview = LivePreview(session)
    rendered = []

    def render(snapshot):
        rendered.append(copy.deepcopy(snapshot))
        return str(snapshot['sim_time']).encode()

    monkeypatch.setattr(preview, '_render', render)
    try:
        preview.start()
        _wait_for(lambda: preview.info()['available'])
        first_info = preview.info()
        # Camera averaging and planning update time/diagnostics without moving
        # anything visible. The cached image retains its actual capture time.
        current.update(sim_time=2., stage='planning', scene_version=2,
                       feedback={'message': 'new planning progress'})
        time.sleep(.42)
        assert len(rendered) == 1
        assert preview.info() == first_info
        assert preview.frame_jpeg() == b'1.0'

        # A manual object move must refresh even if the simulated clock did
        # not advance; display invalidation follows geometry, not timestamps.
        current['objects'][0]['transform'][0][3] += .04
        _wait_for(lambda: len(rendered) == 2)
        assert preview.info()['sim_time'] == 2.
        assert preview.frame_jpeg() == b'2.0'
    finally:
        preview.close()


@pytest.mark.parametrize('field', ['objects', 'zones', 'table_z', 'tcp_transform',
                                  'gripper_width', 'link_frames'])
def test_each_visual_input_invalidates_preview_without_a_clock_tick(monkeypatch, field):
    current = {
        'objects': [], 'zones': [], 'table_z': 0.,
        'tcp_transform': np.eye(4).tolist(), 'gripper_width': .04,
        'link_frames': [np.eye(4).tolist()], 'sim_time': 1.,
    }
    camera = SimpleNamespace(
        intrinsics=lambda: CameraIntrinsics(400, 400, 320, 240, 640, 480),
        T_flange_camera=np.eye(4))
    preview = LivePreview(SimpleNamespace(
        camera=camera, snapshot=lambda: copy.deepcopy(current)))
    rendered = []
    monkeypatch.setattr(preview, '_render',
                        lambda snapshot: rendered.append(copy.deepcopy(snapshot)) or b'frame')
    try:
        preview.start()
        _wait_for(lambda: len(rendered) == 1)
        if field in ('objects', 'zones'):
            current[field].append({'visible': 'new geometry'})
        elif field == 'table_z':
            current[field] = .01
        elif field == 'gripper_width':
            current[field] = .06
        elif field == 'tcp_transform':
            current[field][0][3] += .02
        else:
            current[field][-1][1][3] += .02
        _wait_for(lambda: len(rendered) == 2)
        assert rendered[-1][field] != rendered[0][field]
    finally:
        preview.close()


@pytest.mark.parametrize('kwargs', [{'fps': 0}, {'fps': float('nan')},
                                    {'width': 0}, {'width': True}])
def test_invalid_preview_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LivePreview(None, **kwargs)
