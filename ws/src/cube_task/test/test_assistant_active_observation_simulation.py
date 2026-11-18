"""Real RGB-D, guarded camera motion and held release, entirely offline."""
import copy
import json
import socket
import time
from pathlib import Path

import numpy as np

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.core.transforms import make_T, rot_z
from cube_task.sim.world import SimObject


def test_measured_grip_survives_active_local_view_then_verified_release(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('this simulation must not access an API/network')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    goals = [{'label': 'red', 'zone_name': 'zone_red'}]
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False) as session:
        scene = session.prepare()
        executor = SkillExecutor(session, originals=scene['objects'])
        picked = executor.execute({'id': 'pick', 'skill': 'pick', 'label': 'red'}, goals)
        assert picked['ok'] and picked['holding'], picked
        grasp = session.runner._held_recovery_context['grasp'].T_grasp.copy()
        untouched = {o.name: o.T_base_obj.copy() for o in session.world.objects
                     if o is not session.world.held}
        current = session.arm.get_tcp_pose().value.copy()
        frames = session.camera.frames_rendered
        events = []
        session.on_event = lambda kind, fields: events.append((kind, copy.deepcopy(fields)))
        observed = executor.execute({'id': 'look', 'skill': 'observe', 'mode': 'active',
            'destination': {'type': 'goal', 'index': 0}}, goals)
        assert observed['ok'] and observed['holding'], observed
        assert observed['observation_mode'] == 'active_local' and observed['views'] > 0
        assert session.camera.frames_rendered > frames
        assert not np.allclose(current, session.arm.get_tcp_pose().value, atol=1e-4)
        assert any(k == 'active_observation' for k, _ in events)
        assert not any(k == 'stage' and f.get('stage') in ('open_release', 'release')
                       for k, f in events)
        np.testing.assert_array_equal(session.runner._held_recovery_context['grasp'].T_grasp, grasp)
        for obj in session.world.objects:
            if obj.name in untouched:
                np.testing.assert_array_equal(obj.T_base_obj, untouched[obj.name])
        placed = executor.execute({'id': 'place', 'skill': 'place', 'label': 'red',
            'destination': {'type': 'goal', 'index': 0}}, goals)
        assert placed['ok'] and placed['camera_verified'] and placed['truth']['success'], placed
        assert not session.snapshot()['held_object']
        verified = executor.execute({'id': 'verify', 'skill': 'verify'}, goals)
        assert verified['ok'] and verified['camera_verified'], verified


def test_crowded_incident_geometry_uses_new_measured_grasp_and_local_feedback(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('crowded scene replay must not access an API/network')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    fixture = json.loads((Path(__file__).parent / 'fixtures' /
                          'assistant_crowded_six_objects_scene.json').read_text(encoding='utf-8'))
    goals = [{'label': 'green', 'zone_name': None,
              'destination': {'type': 'region', 'name': 'pick_area'}}]
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False) as session:
        session.world.objects = [SimObject(o['id'], o['label'], o['shape'], o['size'],
            list(reversed(o['color_rgb'])), o['transform']) for o in fixture['objects']]
        green = next(o for o in session.world.objects if o.label == 'green')
        # Test setup only: the old log cannot restore a precise original grasp.
        green.T_base_obj = make_T(rot_z(-.4694), [.214, .314, 0.])
        scene = session.prepare()
        executor = SkillExecutor(session, originals=scene['objects'])
        untouched = {o.name: o.T_base_obj.copy() for o in session.world.objects if o.label != 'green'}
        history, events, timing = [], [], []
        for owner, name in ((session.runner.placer, 'plan'),
                            (session.runner, '_adjust_strict_release')):
            original = getattr(owner, name)
            def timed(*args, _name=name, _original=original, **kwargs):
                started = time.monotonic()
                try:
                    return _original(*args, **kwargs)
                finally:
                    timing.append({'function': _name, 'seconds': time.monotonic()-started})
            monkeypatch.setattr(owner, name, timed)
        session.on_event = lambda kind, fields: events.append({'kind': kind, **copy.deepcopy(fields)})
        placed = executor.execute({'id': 'move', 'skill': 'move', 'label': 'green',
            'destination': {'type': 'goal', 'index': 0}}, goals)
        history.append({k: placed.get(k) for k in ('ok', 'error', 'diagnostics', 'holding')})
        for attempt in range(2):
            if placed['ok']:
                break
            assert placed['holding'], history
            grasp = session.runner._held_recovery_context['grasp'].T_grasp.copy()
            observed = executor.execute({'id': 'view'+str(attempt), 'skill': 'observe',
                'mode': 'active', 'destination': {'type': 'goal', 'index': 0}}, goals)
            history.append({k: observed.get(k) for k in ('ok', 'error', 'changed', 'views', 'reason')})
            assert observed['ok'] and observed['holding'], history
            placed = executor.execute({'id': 'place'+str(attempt), 'skill': 'place', 'label': 'green',
                'destination': {'type': 'goal', 'index': 0}}, goals)
            history.append({k: placed.get(k) for k in ('ok', 'error', 'diagnostics', 'holding')})
            if placed['ok']:
                break
            np.testing.assert_array_equal(session.runner._held_recovery_context['grasp'].T_grasp, grasp)
        (tmp_path / 'crowded-feedback.json').write_text(json.dumps(
            {'history': history, 'events': events, 'scene': session.scene, 'timing': timing},
            ensure_ascii=False, indent=2), encoding='utf-8')
        assert placed['ok'] and placed['camera_verified'] and placed['truth']['success'], history
        assert not session.snapshot()['held_object']
        for obj in session.world.objects:
            if obj.name in untouched:
                np.testing.assert_array_equal(obj.T_base_obj, untouched[obj.name])
