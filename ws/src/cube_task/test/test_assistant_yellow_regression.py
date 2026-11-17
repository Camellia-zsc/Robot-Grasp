"""Replay the exact rendered incident scene through measured RGB-D and motion.

The scene retains object IDs because SimObject derives surface texture from its
name. Simplifying names changes the measured cloud and can hide this failure.
"""
import copy
import json
from pathlib import Path

import numpy as np

from cube_task.assistant.simulation import SimulationSession
from cube_task.sim.world import SimObject, SimZone


SCENE_FIXTURE = Path(__file__).with_name('fixtures') / 'assistant_six_object_scene.json'


def restore_captured_scene(session, layout='yellow_failure'):
    """Restore geometry only; observations and grasp plans must be measured anew."""
    scene = json.loads(SCENE_FIXTURE.read_text(encoding='utf-8'))
    assert session.seed == scene['seed']
    poses = scene['layouts'][layout]
    session.world.objects = [
        SimObject(obj['id'], obj['label'], obj['shape'], copy.deepcopy(obj['size']),
                  obj['color_rgb'][::-1], poses.get(obj['id'], obj['transform']))
        for obj in scene['objects']]
    session.world.table_z = scene['table_z']
    session.world.zones = [
        SimZone(zone['name'], zone['center'], zone['size'], yaw=zone['yaw'],
                color_bgr=zone['color_rgb'][::-1], wall_height=zone['wall_height'],
                label=zone['label']) for zone in scene['zones']]
    assert session.runner.class_to_zone == scene['class_to_zone']
    session._publish_snapshot()


def test_exact_yellow_is_placed_with_feedback_without_moving_five_neighbors(tmp_path):
    with SimulationSession(seed=0, realtime=False, log_dir=tmp_path) as session:
        restore_captured_scene(session)
        before = {obj.name: obj.T_base_obj.copy() for obj in session.world.objects}
        print('Exact yellow regression: preparing six-view RGB-D scene', flush=True)
        scene = session.prepare()
        targets = [obj for obj in scene['objects'] if obj['valid'] and obj['label'] == 'yellow']
        assert len(targets) == 1, scene['objects']
        target = targets[0]
        result = session.execute({'action': 'pick', 'scene_version': scene['scene_version'],
                                  'object_id': target['object_id'], 'zone_name': 'zone_yellow'})
        failure = {key: result.get(key) for key in ('stage', 'error', 'grasp_width', 'diagnostics')}
        failure.update(world_last_grasp=session.world.last_grasp, observed_target=target)
        assert result['ok'], failure
        assert result['controller_ok'] and not result['holding'] and not result['fault']
        assert not result['aborted'] and session.world.held is None
        assert result['truth']['success']
        assert result['truth']['object_name'] == 'yellow_cube'
        assert result['truth']['expected_zone'] == result['truth']['actual_zone'] == 'zone_yellow'

        checks = {check['stage']: check for check in session.snapshot()['feedback']['checks']}
        for stage in ('close', 'dropped', 'dropped_in_transit', 'placement'):
            assert stage in checks and checks[stage]['ok'], checks
        assert checks['close']['measured_width'] > 0
        assert result['grasp_width'] > 0
        for obj in session.world.objects:
            if obj.name != 'yellow_cube':
                np.testing.assert_allclose(obj.T_base_obj, before[obj.name], atol=1e-8, rtol=0,
                                           err_msg='yellow grasp moved neighbor ' + obj.name)
