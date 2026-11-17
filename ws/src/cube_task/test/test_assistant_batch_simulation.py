"""One intent request drives two real RGB-D/IK/feedback grasp executions.

Only the online intent response is replaced. Observations, identity rebinding,
motion planning, simulation and independent placement checks use production code.
"""
import copy
import json

import numpy as np
import pytest

from cube_task.assistant.catalog import make_catalog_object
from cube_task.assistant.server import DashboardController
from cube_task.assistant.simulation import SimulationSession


class LocalSettings:
    def public_info(self):
        return {'configured': True, 'provider': 'offline-intent-test',
                'model': 'fixed-batch', 'api_style': 'responses'}


class BatchClient:
    """Return initial measured IDs once; never supply a later target or pose."""

    def __init__(self, pear_zone):
        self.pear_zone = pear_zone
        self.calls = 0
        self.initial_scene = None

    def decide(self, instruction, scene, on_delta, cancel_event=None):
        self.calls += 1
        self.initial_scene = copy.deepcopy(scene)
        targets = {}
        for label in ('apple', 'pear'):
            candidates = [obj for obj in scene['objects']
                          if obj.get('valid') is True and obj.get('label') == label]
            assert len(candidates) == 1, scene['objects']
            targets[label] = candidates[0]['object_id']
        decision = {'scene_version': scene['scene_version'], 'action': 'pick_batch',
                    'items': [{'object_id': targets['apple'], 'zone_name': 'zone_red'},
                              {'object_id': targets['pear'], 'zone_name': self.pear_zone}],
                    'message': '先搬苹果，再搬梨子，逐件核验完成。'}
        on_delta(json.dumps(decision, ensure_ascii=False))
        return decision


class FruitBatchSession(SimulationSession):
    """Set the editable world, then record calls around the real adapter."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world.objects[:] = [
            make_catalog_object('apple', 'batch-apple', [.35, .05], .3, self.world.table_z),
            make_catalog_object('pear', 'batch-pear', [.26, .07], .6, self.world.table_z),
        ]
        self.executions = []

    def execute(self, decision, cancel_event=None):
        call = {'decision': copy.deepcopy(decision), 'scene_before': self.scene,
                'poses_before': {obj.label: obj.T_base_obj.copy() for obj in self.world.objects}}
        self.executions.append(call)
        result = super().execute(decision, cancel_event=cancel_event)
        call.update(result=copy.deepcopy(result), scene_after=self.scene,
                    poses_after={obj.label: obj.T_base_obj.copy() for obj in self.world.objects})
        return result


@pytest.mark.parametrize('pear_zone', ['zone_green', 'zone_red'])
def test_one_batch_rebinds_second_fruit_and_verifies_both_placements(tmp_path, pear_zone):
    client = BatchClient(pear_zone)
    ctl = DashboardController(realtime=False, session_factory=FruitBatchSession,
                              settings=LocalSettings(), client=client, log_dir=tmp_path)
    try:
        assert ctl.wait_idle(180), 'initial camera scan did not finish'
        assert ctl.snapshot()['status'] == 'ready', ctl.snapshot()
        ctl.submit('先把苹果放到红色区域，再把梨子放到%s区域' %
                   ('绿色' if pear_zone == 'zone_green' else '红色'))
        assert ctl.wait_idle(240), 'batch simulation did not finish'
        state = ctl.snapshot()
        assert state['status'] == 'complete', state
        assert not state['needs_reset'] and not state['error']
        assert client.calls == 1
        assert state['metrics'] == {'requests': 1, 'successes': 1, 'model_requests': 1}
        assert state['batch']['status'] == 'complete'
        assert state['batch']['completed'] == state['batch']['total'] == 2
        assert state['last_result']['ok'] and state['last_result']['batch']
        assert state['last_result']['completed'] == 2

        for item, label, zone in zip(state['batch']['items'], ('apple', 'pear'),
                                     ('zone_red', pear_zone)):
            assert item['label'] == label and item['zone_name'] == zone
            assert item['status'] == 'succeeded', item
            result = item['result']
            assert result['ok'] and result['controller_ok'], result
            assert not result['holding'] and not result['fault'] and not result['aborted']
            assert result['truth']['success']
            assert result['truth']['expected_zone'] == result['truth']['actual_zone'] == zone

        first, second = ctl._session.executions
        initial = client.initial_scene
        initial_pear = next(obj for obj in initial['objects'] if obj['valid'] and obj['label'] == 'pear')
        rescanned = first['scene_after']
        fresh_pear = next(obj for obj in rescanned['objects'] if obj['valid'] and obj['label'] == 'pear')
        assert first['decision']['scene_version'] == initial['scene_version']
        assert rescanned['scene_version'] > initial['scene_version']
        assert second['decision']['scene_version'] == rescanned['scene_version']
        assert second['decision']['object_id'] == fresh_pear['object_id']
        assert second['decision']['object_id'] != initial_pear['object_id']
        assert second['scene_before'] == rescanned
        np.testing.assert_allclose(first['poses_after']['apple'],
                                   second['poses_before']['apple'], atol=1e-8)
        np.testing.assert_allclose(first['poses_after']['apple'],
                                   second['poses_after']['apple'], atol=1e-8)
        assert ctl._session.world.held is None
        for obj in ctl._session.world.objects:
            assert ctl._session.world.zone_of(obj).name == ('zone_red' if obj.label == 'apple' else pear_zone)
    finally:
        ctl.close()
