"""Measured apple placement must use the open parts of a crowded region."""
import json
from pathlib import Path
import socket

import numpy as np

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.sim.world import SimObject


def test_measured_apple_reaches_open_region_without_moving_nearby_objects(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('this measured simulation must not access an API/network')

    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    fixture = json.loads((Path(__file__).parent / 'fixtures' /
        'assistant_apple_open_region_scene.json').read_text(encoding='utf-8'))
    goals = [{'label': 'apple', 'zone_name': None,
              'destination': {'type': 'region', 'name': 'pick_area'}}]
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False) as session:
        # Geometry sets up a test scene only. Detection, grasp anchors and all
        # placement plans below come from the actual RGB-D pipeline.
        session.world.objects = [SimObject(o['id'], o['label'], o['shape'], o['size'],
            list(reversed(o['color_rgb'])), o['transform']) for o in fixture['objects']]
        untouched = {o.name: o.T_base_obj.copy() for o in session.world.objects
                     if o.label != 'apple'}
        scene = session.prepare()
        executor = SkillExecutor(session, originals=scene['objects'])
        placed = executor.execute({'id': 'move', 'skill': 'move', 'label': 'apple',
            'destination': {'type': 'goal', 'index': 0}}, goals)
        assert placed['ok'] and placed['camera_verified'] and placed['truth']['success'], placed
        assert not session.snapshot()['held_object']
        for obj in session.world.objects:
            if obj.name in untouched:
                np.testing.assert_array_equal(obj.T_base_obj, untouched[obj.name])
