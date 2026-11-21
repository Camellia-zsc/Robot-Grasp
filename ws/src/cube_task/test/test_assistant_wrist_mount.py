"""The wrist camera moves physically, with matching calibration after rebuilds."""
import copy
import socket

import numpy as np
import pytest

pytest.importorskip('mujoco')

from cube_task.assistant.physics import DashboardPhysics
from cube_task.assistant.simulation import SimulationSession
from cube_task.core.clock import SimClock
from cube_task.core.config import Config, deep_merge, load_config
from cube_task.core.extrinsics import camera_pose_in_base
from cube_task.sim.scenarios import make_scenario, scenario_config_overlay


OFFSET_KEY = 'assistant.physics.wrist_camera_flange_z_offset'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError('camera mounting verification must not use an online API')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def configuration():
    world, info = make_scenario('L1', 0, mount='eye_in_hand', difficulty='normal', layout='A')
    base = load_config(use_overlays=False)
    cfg = Config(deep_merge(base.data, scenario_config_overlay(info)), base.root, base.sources)
    return world, info, cfg


@pytest.mark.parametrize('offset', [None, 0., -.015])
def test_mount_offset_is_applied_once_to_a_copy_before_any_build(monkeypatch, offset):
    world, info, cfg = configuration()
    if offset is not None:
        cfg.set(OFFSET_KEY, offset)
    before = copy.deepcopy(info)
    builds = []
    monkeypatch.setattr(DashboardPhysics, '_build',
                        lambda self: builds.append(np.array(self.info['T_flange_camera'])))
    monkeypatch.setattr(DashboardPhysics, 'sync', lambda *_args, **_kwargs: None)

    physics = DashboardPhysics(cfg, world, info, SimClock())
    physics._build()
    expected = np.asarray(before['T_flange_camera'], float).copy()
    expected[2, 3] += -.025 if offset is None else offset

    assert len(builds) == 2
    for mounted in builds:
        np.testing.assert_array_equal(mounted, expected)
    np.testing.assert_array_equal(info['T_flange_camera'], before['T_flange_camera'])
    assert not np.shares_memory(physics.info['T_flange_camera'], info['T_flange_camera'])
    assert physics.cfg.get(OFFSET_KEY) == (-.025 if offset is None else offset)
    assert cfg.get(OFFSET_KEY, None) == offset


@pytest.mark.parametrize('offset', [float('nan'), float('inf'), -float('inf'), 'bad', None])
def test_invalid_mount_offset_is_rejected_before_building_a_model(monkeypatch, offset):
    world, info, cfg = configuration()
    cfg.set(OFFSET_KEY, offset)
    builds = []
    monkeypatch.setattr(DashboardPhysics, '_build', lambda *_args: builds.append(1))
    with pytest.raises(ValueError, match='finite distance'):
        DashboardPhysics(cfg, world, info, SimClock())
    assert not builds


def test_physical_camera_and_runner_calibration_stay_equal_after_actual_rebuild(tmp_path):
    with SimulationSession(backend='mujoco', realtime=False, log_dir=tmp_path) as session:
        original = np.asarray(session.info['T_flange_camera'], float).copy()
        expected = original.copy()
        expected[2, 3] -= .025

        def check_camera():
            physics = session._physics
            np.testing.assert_allclose(physics.mujoco_world.T_flange_camera, expected, atol=1e-12)
            np.testing.assert_allclose(session.camera.T_flange_camera, expected, atol=1e-12)
            np.testing.assert_allclose(session.runner.ext.T, expected, atol=1e-12)
            np.testing.assert_allclose(physics.cfg.get('camera.extrinsics_inline')['matrix'],
                                       expected, atol=1e-12)
            estimated = camera_pose_in_base(session.runner.ext,
                session.arm.get_flange_pose().value, session.runner.T_flange_tcp)
            np.testing.assert_allclose(session.camera.current_T_base_camera(), estimated, atol=1e-12)

        check_camera()
        camera, arm = session.camera, session.arm
        joints = arm.get_joints().value.copy()
        session.world.objects = session.world.objects[:-1]
        session._physics.apply_scene_changes()
        check_camera()
        assert session.camera is camera and session.arm is arm
        np.testing.assert_allclose(arm.get_joints().value, joints, atol=1e-12)
        np.testing.assert_array_equal(session.info['T_flange_camera'], original)
