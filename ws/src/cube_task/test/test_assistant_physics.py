"""Actual MuJoCo free-body/contact tests; no online model calls."""
import socket

import numpy as np
import pytest

pytest.importorskip('mujoco')

from cube_task.assistant.catalog import make_catalog_object
from cube_task.assistant.physics import DashboardPhysics
from cube_task.core.clock import SimClock
from cube_task.core.config import Config, deep_merge, load_config
from cube_task.sim.scenarios import make_scenario, scenario_config_overlay


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('physics verification must not access an online API')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


@pytest.fixture
def build():
    backends = []
    def factory(offset=0., upper='apple'):
        world, info = make_scenario('L1', 0, mount='eye_in_hand', difficulty='normal', layout='A')
        base = load_config(use_overlays=False)
        cfg = Config(deep_merge(base.data, scenario_config_overlay(info)), base.root, base.sources)
        red = make_catalog_object('red', 'base-red', [.28, 0.])
        top = make_catalog_object(upper, 'upper-'+upper, [.28+offset, 0.])
        transform = top.T_base_obj
        transform[2, 3] += .05
        top.T_base_obj = transform
        world.objects = [red, top]
        events = []
        physics = DashboardPhysics(cfg, world, info, SimClock(),
            on_event=lambda kind, event: events.append((kind, event)))
        physics.arm.connect()
        assert physics.arm.enable()
        physics.camera.start()
        backends.append(physics)
        return physics, red, top, events
    yield factory
    for physics in backends:
        physics.camera.stop()
        physics.arm.close()
        physics.close()


def test_centered_apple_has_real_support_contact_without_rigid_attachment(build):
    physics, base, apple, _events = build()
    physics.settle(1.2)
    result = physics.support_status(apple.name, 'red')
    assert result['success'] and result['contact'] and result['stable'], result
    assert apple.bottom_z() == pytest.approx(base.top_z(), abs=.003)
    assert physics.world.held is None and physics.world._T_tcp_obj is None
    assert physics.state()['gravity'] == [0., 0., -9.81]
    assert physics.state()['physics_steps'] >= 600
    assert not any(int(t) == int(physics.mujoco_world._mj.mjtEq.mjEQ_WELD)
                   for t in physics.mujoco_world.model.eq_type)


def test_unsupported_off_center_apple_physically_falls_and_rolls(build):
    physics, _base, apple, events = build(offset=.043)
    before = apple.T_base_obj.copy()
    for _ in range(45):
        physics.settle(.05)
    assert apple.bottom_z() < .004
    assert np.linalg.norm(apple.T_base_obj[:2, 3]-before[:2, 3]) > .008
    assert not physics.support_status(apple.name, 'red')['success']
    assert any(event['kind'] == 'rolled' and event['object'] == apple.name
               for _kind, event in events)


def test_moving_base_away_releases_upper_object_to_gravity(build):
    physics, base, apple, events = build(upper='green')
    physics.settle(1.2)
    assert physics.support_status(apple.name, 'red')['success']
    base.place_upright(.43, .15, 0., physics.world.table_z)
    physics.apply_scene_changes()
    assert any(event['kind'] == 'support_lost' and event['object'] == apple.name
               for _kind, event in events)
    for _ in range(30):
        physics.settle(.05)
    assert apple.bottom_z() == pytest.approx(physics.world.table_z, abs=.003)
    assert not physics.support_status(apple.name, 'red')['success']


def test_editor_removal_rebuilds_physics_without_replacing_dashboard_components(build):
    physics, base, apple, _events = build()
    physics.settle(1.2)
    arm, camera, world = physics.arm, physics.camera, physics.world
    q = arm.get_joints().value.copy()
    physics.world.objects = [apple]
    physics.apply_scene_changes()
    assert physics.arm is arm and physics.camera is camera and physics.world is world
    np.testing.assert_allclose(physics.arm.get_joints().value, q, atol=1e-10)
    assert len(physics.clock._listeners) == 2
    assert len(physics.mujoco_world.objects) == 1
    physics.settle(1.)
    assert apple.bottom_z() < .003
    assert physics.arm.health().ok


def test_actual_model_fk_and_sensor_use_same_physical_robot(build):
    physics, _base, _apple, _events = build()
    physics.settle(.2)
    q = physics.arm.get_joints().value
    np.testing.assert_allclose(physics.arm.kin.fk(q), physics.arm.get_flange_pose().value,
                               atol=1e-12)
    frames = physics.arm.kin.fk_all(q)
    assert len(frames) == 7
    np.testing.assert_allclose(frames[-1], physics.arm.get_flange_pose().value, atol=1e-12)
    frame = physics.camera.get_frame(timeout=1.)
    assert frame is not None and frame.depth is not None
    assert np.count_nonzero(frame.depth > 0) > 1000


def test_physical_finger_stroke_matches_the_explicit_project_configuration(build):
    physics, _base, _apple, _events = build()
    configured = float(physics.cfg.get('robot.gripper.max_width'))
    assert physics.mujoco_world.max_width == pytest.approx(configured)
    assert physics.state()['gripper_max_width'] == pytest.approx(configured)
    model = physics.mujoco_world.model
    mj = physics.mujoco_world._mj
    # Both real sliding-joint limits must permit the configured total opening.
    for joint in ('joint7', 'joint8'):
        ident = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint)
        assert max(abs(model.jnt_range[ident])) >= configured / 2 - 1e-9
