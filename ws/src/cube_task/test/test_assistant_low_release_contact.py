"""Measure the release gap with real contact bodies; no truth is a planning input."""
import socket

import pytest

pytest.importorskip('mujoco')

from cube_task.assistant.catalog import make_catalog_object
from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor


@pytest.mark.parametrize('label', ['banana', 'pear'])
def test_camera_measured_stack_respects_low_release_and_confirmation(
        tmp_path, monkeypatch, label):
    def denied(*args, **kwargs):
        raise AssertionError('release contact verification must not use an online API')

    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    events = []
    with SimulationSession(backend='mujoco', realtime=False, log_dir=tmp_path,
            on_event=lambda kind, data: events.append((kind, data))) as session:
        target = make_catalog_object(label, 'physical-' + label, [.28, -.15])
        support = make_catalog_object('red', 'physical-base', [.28, .01])
        session.world.objects = [target, support]
        session._physics.apply_scene_changes()
        scene = session.prepare()
        executor = SkillExecutor(session, originals=scene['objects'])
        assert any(obj['label'] == label and obj['valid'] for obj in scene['objects'])
        assert any(obj['label'] == 'red' and obj['valid'] for obj in scene['objects'])
        destination = dict(type='on_object', label='red')
        goals = [dict(label=label, destination=destination)]
        # Independent evaluation only. These values never alter a pose, a
        # detector, the planner, the servo or the physical contact model.
        release_samples = []
        raw_gripper = session.arm.inner.gripper

        def measured_gripper(width, *args, **kwargs):
            physical = session._physics
            physical.sync()
            measured_width = session.arm.inner.gripper_width().value
            if (target.name in physical.mujoco_world.held_objects()
                    and width > measured_width + .005):
                linear_speed, angular_speed = physical.mujoco_world.object_speed(target.name)
                release_samples.append(dict(
                    gap=target.bottom_z() - support.top_z(),
                    linear_speed=linear_speed, angular_speed=angular_speed))
            return raw_gripper(width, *args, **kwargs)

        monkeypatch.setattr(session.arm.inner, 'gripper', measured_gripper)
        placed = executor.execute(dict(id='move-' + label, skill='move', label=label,
            destination=dict(type='goal', index=0)), goals)
        print('placement outcome:', {key: placed.get(key) for key in (
            'ok', 'camera_verified', 'controller_ok', 'error', 'message', 'truth')}, flush=True)
        if label == 'pear':
            # For this measured source/destination, the reachable release
            # leans enough that full fruit/finger clearance is unconfirmed.
            # Holding it higher and opening anyway must never be the fallback.
            assert not placed['ok'] and not placed['controller_ok'], placed
            assert 'leaned release has no full footprint or finger clearance' in placed['error']
            assert not release_samples
            assert not session._physics.mujoco_world.held_objects()
            assert not session.runner.has_held_recovery
            assert target.bottom_z() == pytest.approx(session.world.table_z, abs=.003)
            assert session._physics.resting_status(target.name)['success']
            assert not session._physics.support_status(target.name, support.label)['success']
            return
        assert release_samples, placed
        print(label, '50 mm support release evaluation:', release_samples, flush=True)
        assert len(release_samples) == 1, release_samples
        sample = release_samples[0]
        assert -.003 <= sample['gap'] <= .006, sample
        assert sample['linear_speed'] <= .015, sample
        assert any(kind == 'physics_event' and data['kind'] == 'grasped'
                   and data['object'] == target.name for kind, data in events)
        assert session.world._T_tcp_obj is None
        session._physics.settle(2.)
        print('settled evaluation:', dict(
            resting=session._physics.resting_status(target.name),
            support=session._physics.support_status(target.name, support.label),
            center=target.center_xy().tolist()), flush=True)
        assert not session._physics.mujoco_world.held_objects()
        assert session._physics.resting_status(target.name)['success']
        assert session._physics.support_status(target.name, support.label)['success']
        assert placed['controller_ok'] and placed['truth']['success'], placed
        # A banana may roll to a stable slanted rest on this narrow support.
        # Real support does not replace the separate camera confirmation: an
        # unrecognised final view must remain a pending/failed task.
        assert bool(placed['ok']) is bool(placed['camera_verified']), placed
        if not placed['camera_verified']:
            assert placed['error'] == 'placement_not_observed', placed
