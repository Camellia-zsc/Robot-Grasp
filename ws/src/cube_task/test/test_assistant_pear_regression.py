"""The captured six-object scene must yield a stable, verified pear grasp.

World geometry reconstructs the incident only. Target coordinates and the grasp
plan still come from the production six-view RGB-D pipeline.
"""
import numpy as np

from cube_task.assistant.simulation import SimulationSession
from test_assistant_observations import captured_layout


def pear_failure_layout():
    objects = captured_layout()
    for obj in objects:
        obj.label = obj.name
        if obj.label == 'banana':
            center = obj.center_xy()
            obj.place_upright(float(center[0]), float(center[1]), 1.6580627893946132, 0.0)
    return objects


def test_captured_pear_is_grasped_with_feedback_without_moving_neighbors(tmp_path):
    with SimulationSession(realtime=False, log_dir=tmp_path) as session:
        session.world.objects = pear_failure_layout()
        before = {obj.name: obj.T_base_obj.copy() for obj in session.world.objects}
        print('Pear regression: production modules loaded; preparing captured scene', flush=True)
        scene = session.prepare()
        targets = [obj for obj in scene['objects'] if obj['valid'] and obj['label'] == 'pear']
        assert len(targets) == 1, scene['objects']
        target = targets[0]
        result = session.execute({'action': 'pick', 'scene_version': scene['scene_version'],
                                  'object_id': target['object_id'], 'zone_name': 'zone_green'})
        failure = {key: result.get(key) for key in ('stage', 'error', 'grasp_width', 'diagnostics')}
        failure.update(world_last_grasp=session.world.last_grasp, observed_target=target)
        assert result['ok'], failure
        assert result['controller_ok'] and not result['holding'] and not result['fault']
        assert not result['aborted'] and session.world.held is None
        assert result['truth']['success']
        assert result['truth']['expected_zone'] == result['truth']['actual_zone'] == 'zone_green'

        checks = {check['stage']: check for check in session.snapshot()['feedback']['checks']}
        for stage in ('close', 'dropped', 'dropped_in_transit', 'placement'):
            assert stage in checks and checks[stage]['ok'], checks
        assert checks['close']['measured_width'] > 0
        assert result['grasp_width'] > 0
        for obj in session.world.objects:
            if obj.label != 'pear':
                np.testing.assert_allclose(obj.T_base_obj, before[obj.name], atol=1e-8,
                                           err_msg='pear grasp moved neighbor ' + obj.name)
