"""The mixed catalog task must retain physical fruit after its measured lift."""
import socket

import numpy as np
import pytest

pytest.importorskip('mujoco')

from cube_task.assistant.catalog import make_catalog_object
from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.core.config import load_config
from cube_task.task.skills import gripper_force_for


def test_mixed_catalog_uses_model_grasp_force_and_respects_explicit_override():
    cfg = load_config(use_overlays=False)
    assert gripper_force_for(cfg, 'Sandbox') == gripper_force_for(cfg, 'L2')
    assert gripper_force_for(cfg, 'L1') < gripper_force_for(cfg, 'Sandbox')
    cfg.set('tasks.Sandbox.gripper_force', 1.7)
    assert gripper_force_for(cfg, 'Sandbox') == 1.7


@pytest.mark.parametrize('destination', ['point', 'zone_green'])
def test_camera_measured_pear_lift_and_placement_use_real_contacts(tmp_path, monkeypatch, destination):
    def denied(*args, **kwargs):
        raise AssertionError('contact-grasp verification must not access an online API')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    with SimulationSession(backend='mujoco', realtime=False, log_dir=tmp_path) as session:
        session.world.objects = [make_catalog_object('pear', 'physical-pear', [.31, -.127])]
        session._physics.apply_scene_changes()
        scene = session.prepare()
        executor = SkillExecutor(session, originals=scene['objects'])
        goals = ([dict(label='pear', destination=dict(type='point', xy=[.30, .08]))]
                 if destination == 'point' else [dict(label='pear', zone_name='zone_green')])
        picked = executor.execute(dict(id='pick-pear', skill='pick', label='pear'), goals)
        assert picked['ok'] and picked['holding'], picked
        assert session._physics.mujoco_world.held_objects() == ['physical-pear']
        assert session.world.objects[0].bottom_z() > session.world.table_z + .04
        assert picked['grasp_width'] == pytest.approx(session.arm.gripper_width().value, abs=.001)
        assert session.world._T_tcp_obj is None

        placed = executor.execute(dict(id='place-pear', skill='place', label='pear',
            destination=dict(type='goal', index=0)), goals)
        assert placed['ok'] and placed['camera_verified'], placed
        session._physics.settle(2.)
        pear = session.world.objects[0]
        assert not session._physics.mujoco_world.held_objects()
        assert pear.bottom_z() == pytest.approx(session.world.table_z, abs=.002)
        assert pear.T_base_obj[2, 2] > np.cos(np.deg2rad(10))
        if destination == 'point':
            assert np.linalg.norm(pear.center_xy() - [.30, .08]) < .02
        else:
            assert session.runner.zones['zone_green'].contains_xy(*pear.center_xy())
            verified = executor.execute(dict(id='verify-pear', skill='verify'), goals)
            assert verified['ok'] and verified['truth']['success'], verified
