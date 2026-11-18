"""Dashboard-to-contact-gripper integration; never uses an online model."""
import socket

import pytest

pytest.importorskip('mujoco')

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.assistant.catalog import make_catalog_object


def test_measured_pick_and_physical_transit_drop_are_not_reported_as_success(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('physical session test must not call an online API')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    events = []
    with SimulationSession(backend='mujoco', realtime=False, log_dir=tmp_path,
            on_event=lambda kind, data: events.append((kind, data))) as session:
        scene = session.prepare()
        assert sorted(o['label'] for o in scene['objects'] if o['valid']) == ['green', 'red', 'yellow']
        executor = SkillExecutor(session, originals=scene['objects'])
        goals = [dict(label='red', zone_name='zone_red')]
        picked = executor.execute(dict(id='pick-red', skill='pick', label='red'), goals)
        assert picked['ok'] and picked['holding'], picked
        assert session.world.held is not None
        assert session.world._T_tcp_obj is None
        assert session._physics.state()['physics_steps'] > 1000
        assert any(kind == 'physics_event' and data['kind'] == 'grasped'
                   for kind, data in events)

        # The driver opens the actual physical fingers during the next motion.
        # The target must fall under gravity and the normal execution feedback
        # must fail, rather than retaining the geometry viewer's old attachment.
        session.arm.inner.inject('drop')
        outcome = executor.execute(dict(id='place-red', skill='place', label='red',
            destination=dict(type='goal', index=0)), goals)
        assert not outcome['ok'], outcome
        session._physics.settle(.5)
        assert session.world.held is None
        assert not session.runner.has_held_recovery
        assert any(kind == 'physics_event' and data['kind'] in ('released', 'dropped')
                   for kind, data in events)
        assert outcome.get('error') in ('dropped_in_transit', 'dropped', 'scene_changed'), outcome


def test_camera_grounded_contact_grasp_stacks_apple_and_detects_removed_support(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('physical session test must not call an online API')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    events = []
    with SimulationSession(backend='mujoco', realtime=False, log_dir=tmp_path,
            on_event=lambda kind, data: events.append((kind, data))) as session:
        # Explicit initial scene editing is allowed; the result is produced
        # entirely by the camera, robot servo, contact fingers and gravity.
        session.world.objects = [make_catalog_object('red', 'base-red', [.28, .01]),
                                 make_catalog_object('apple', 'upper-apple', [.28, -.15])]
        session._physics.apply_scene_changes()
        scene = session.prepare()
        executor = SkillExecutor(session, originals=scene['objects'])
        destination = dict(type='on_object', label='red')
        goals = [dict(label='apple', destination=destination)]
        result = executor.execute(dict(id='stack-apple', skill='move', label='apple',
            destination=dict(type='goal', index=0)), goals)
        assert result['ok'] and result['camera_verified'], result
        assert result['truth']['support']['contact'] and result['truth']['support']['stable'], result
        assert session._physics.support_status('upper-apple', 'red')['success']
        assert session.world.held is None and session.world._T_tcp_obj is None
        assert any(kind == 'physics_event' and data['kind'] == 'grasped'
                   and data['label'] == 'apple' for kind, data in events)

        snap = session.snapshot()
        session.intervene_scene(dict(action='move', object_id='base-red', xy=[.43, .15],
                                     scene_version=snap['scene_version']))
        session._physics.settle(1.)
        assert not session._physics.support_status('upper-apple', 'red')['success']
        apple = next(o for o in session.world.objects if o.label == 'apple')
        assert apple.bottom_z() < .004
        verification = session.verify_goals(goals)
        assert not verification['ok'], verification
        assert any(kind == 'physics_event' and data['kind'] == 'support_lost'
                   and data['label'] == 'apple' for kind, data in events)
