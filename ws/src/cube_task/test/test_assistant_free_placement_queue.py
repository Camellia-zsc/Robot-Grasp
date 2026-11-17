"""Extended goals survive fresh IDs, measured skip checks and bounded recovery."""
from copy import deepcopy

import pytest

from cube_task.assistant.batch import (bind_item, build_items, matching_objects,
                                       whole_footprint_at_goal)
from cube_task.assistant.recovery import model_recovery_context, validate_recovery_decision
from test_assistant_batch import BatchSession, create
from test_assistant_free_placement_protocol import POINT, REGION, observed


def scene():
    result = observed(1)
    first = result['objects'][0]
    first.update(center_base=[.17, -.35, .06], length=.07, width=.07, yaw=0.)
    second = deepcopy(first)
    second.update(object_id='v1-second-apple', center_base=[.36, .05, .06])
    result['objects'] = [first, second]
    return result


def plan(current):
    return {'scene_version': current['scene_version'], 'action': 'pick_batch',
            'message': '先放回抓取区，再搬另一苹果到桌面点。',
            'items': [{'object_id': obj['object_id'], 'zone_name': None,
                       'destination': deepcopy(REGION if i == 0 else POINT)}
                      for i, obj in enumerate(current['objects'])]}


class PlacementClient:
    def decide(self, instruction, current, on_delta, cancel_event=None):
        return plan(current)


class PlacementSession(BatchSession):
    missing_second = False
    no_release = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scene = scene()

    def observe(self):
        self.observe_calls += 1
        self.scene = deepcopy(self.scene)
        version = self.scene['scene_version'] = self.scene['scene_version'] + 1
        for index, obj in enumerate(self.scene['objects']):
            obj['object_id'] = f'v{version}-apple-{index}'
        return self.scene

    def execute(self, decision, cancel_event=None):
        assert decision['scene_version'] == self.scene['scene_version']
        self.executions.append(deepcopy(decision))
        target = next(o for o in self.scene['objects'] if o['object_id'] == decision['object_id'])
        destination = decision['destination']
        xy = [.25, -.10] if destination['type'] == 'region' else destination['xy']
        target['center_base'] = list(xy) + [.06]
        if self.missing_second and len(self.executions) == 1:
            self.scene['objects'] = self.scene['objects'][:1]
        self.observe()
        result = {'ok': True, 'message': '落位核验成功', 'destination': deepcopy(destination)}
        if not self.no_release:
            result['placement_center'] = list(xy) + [.06]
        return result


def test_queue_rebinds_fresh_ids_preserving_detached_destinations():
    initial = scene()
    decision = plan(initial)
    items = build_items(decision, initial)
    later = deepcopy(initial)
    later['scene_version'] = 8
    later['objects'][0]['object_id'] = 'fresh-apple'
    single, _ = bind_item(items[0], later)
    assert single['object_id'] == 'fresh-apple' and single['scene_version'] == 8
    assert single['zone_name'] is None and single['destination'] == REGION
    items[1]['destination']['xy'][0] = .4
    assert decision['items'][1]['destination'] == POINT


def test_completed_pick_area_goal_excludes_only_the_completed_local_identity():
    ctl = create(PlacementSession, PlacementClient())
    try:
        ctl.submit('依次搬两个苹果')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'complete' and state['batch']['completed'] == 2
        first, second = ctl._session.executions
        assert second['scene_version'] > first['scene_version']
        assert first['destination'] == REGION and second['destination'] == POINT
        assert [i['status'] for i in state['batch']['items']] == ['succeeded', 'succeeded']
        assert state['batch']['items'][0]['completion_xy'] == [.25, -.10]
    finally:
        ctl.close()


@pytest.mark.parametrize('missing_release', [False, True])
def test_missing_or_unverified_completed_identity_stops_later_same_class(missing_release):
    class Changed(PlacementSession):
        no_release = missing_release
        missing_second = not missing_release

    ctl = create(Changed, PlacementClient())
    try:
        ctl.submit('依次搬两个苹果')
        assert ctl.wait_idle(3)
        assert ctl.snapshot()['batch']['completed'] == 1
        assert len(ctl._session.executions) == 1
        assert ctl.snapshot()['batch']['items'][1]['status'] == 'failed'
    finally:
        ctl.close()


def test_completed_point_cannot_substitute_missing_pending_same_class():
    current = scene()
    current['objects'] = current['objects'][:1]
    current['objects'][0]['center_base'] = [.30, .08, .06]
    assert matching_objects(current, 'apple', [.33, .08])
    assert matching_objects(current, 'apple', [.33, .08],
                            completed_positions=([.30, .08],)) == []
    with pytest.raises(ValueError, match='观测位置无效'):
        matching_objects(current, 'apple', [.33, .08], completed_positions=([float('nan'), 0.],))


def test_point_skip_needs_measured_tolerance_and_supported_complete_footprint():
    current = scene()
    target = current['objects'][0]
    target['center_base'] = [.30, .08, .06]
    assert whole_footprint_at_goal(target, current, {'zone_name': None, 'destination': POINT})
    target['center_base'][0] += .016
    assert not whole_footprint_at_goal(target, current, {'zone_name': None, 'destination': POINT})
    target['center_base'] = [.11, .08, .06]
    edge = {'type': 'point', 'xy': [.11, .08]}
    assert not whole_footprint_at_goal(target, current, {'zone_name': None, 'destination': edge})


@pytest.mark.parametrize('destination', [POINT, REGION])
def test_recovery_visible_goal_preserves_destination_and_omits_private_data(destination):
    context = {'scene_version': 3, 'attempt': 1, 'max_attempts': 3,
               'goal': {'label': 'apple', 'zone_name': None, 'destination': deepcopy(destination)},
               'placement': observed()['placement'], 'failure': {'holding': True},
               'observations': [], 'history': [],
               'options': [{'id': 'continue', 'description': '继续原目标'}],
               'private_world': {'actual': 'must not leave process'}}
    visible = model_recovery_context(context)
    assert visible['goal'] == context['goal']
    assert 'placement' not in visible and 'private_world' not in visible
    if destination['type'] == 'point':
        visible['goal']['destination']['xy'][0] = .4
        assert context['goal']['destination'] == POINT
    chosen = {'scene_version': 3, 'action': 'recover', 'option_id': 'continue', 'message': '继续原目标'}
    assert validate_recovery_decision(chosen, context) == chosen
    with pytest.raises(ValueError):
        validate_recovery_decision(dict(chosen, destination=destination), context)


def test_controller_passes_extended_goal_and_local_identities_to_recovery():
    class Recovering(PlacementSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.recovery_goals = []

        def execute(self, decision, cancel_event=None):
            if len(self.executions) == 1:
                self.executions.append(deepcopy(decision))
                return {'ok': False, 'error': 'grasp_miss'}
            return super().execute(decision, cancel_event)

        def prepare_recovery(self, failure, goal, attempt, history,
                             reference_xy=None, completed_zones=(), completed_positions=()):
            self.recovery_goals.append(deepcopy((goal, reference_xy, completed_positions)))
            return {'options': []}

    client = PlacementClient()
    client.recover = lambda *a, **k: pytest.fail('No local option must not invoke a model')
    ctl = create(Recovering, client)
    try:
        ctl.submit('依次搬两个苹果')
        assert ctl.wait_idle(3)
        goal, reference, positions = ctl._session.recovery_goals[0]
        assert goal == {'label': 'apple', 'zone_name': None, 'destination': POINT}
        assert reference == [.36, .05] and positions == ([.25, -.10],)
    finally:
        ctl.close()


@pytest.mark.parametrize('batch', [False, True])
def test_validation_refusal_keeps_scene_and_allows_reselection_without_recovery(batch):
    class Refusing(PlacementSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.recovery_calls = 0

        def execute(self, decision, cancel_event=None):
            if not self.executions:
                self.executions.append(deepcopy(decision))
                return {'ok': False, 'stage': 'validation', 'error': 'destination_not_observed',
                        'message': '当前无法确认目标点周围空间，未执行抓取',
                        'scene': deepcopy(self.scene)}
            return super().execute(decision, cancel_event)

        def prepare_recovery(self, *args, **kwargs):
            self.recovery_calls += 1
            raise AssertionError('A refused grasp must not enter task recovery')

    class Choosing(PlacementClient):
        batch = False
        destination = deepcopy(POINT)
        recovery_calls = 0

        def decide(self, instruction, current, on_delta, cancel_event=None):
            if self.batch:
                result = plan(current)
                result['items'][0]['destination'] = deepcopy(self.destination)
                return result
            return {'scene_version': current['scene_version'], 'action': 'pick',
                    'object_id': current['objects'][0]['object_id'], 'zone_name': None,
                    'destination': deepcopy(self.destination), 'message': '放到指定桌面点'}

        def recover(self, *args, **kwargs):
            self.recovery_calls += 1
            raise AssertionError('A validation refusal must not invoke a recovery model')

    client = Choosing()
    client.batch = batch
    ctl = create(Refusing, client)
    try:
        ctl.submit('将物品放到指定点')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'error' and not state['needs_reset']
        result = state['batch']['items'][0]['result'] if batch else state['last_result']
        assert result['error'] == 'destination_not_observed'
        assert result['message'] == '当前无法确认目标点周围空间，未执行抓取'
        assert not result.get('fault') and not result.get('recovery_history')
        assert state['scene']['scene_version'] == 2
        if batch:
            assert [i['status'] for i in state['batch']['items']] == ['failed', 'cancelled']
        assert len(ctl._session.executions) == 1
        assert client.recovery_calls == ctl._session.recovery_calls == 0
        client.batch = False
        client.destination = {'type': 'point', 'xy': [.32, .08]}
        ctl.submit('改放到重新选择的桌面点')
        assert ctl.wait_idle(3)
        assert ctl.snapshot()['status'] == 'complete' and not ctl.snapshot()['needs_reset']
        assert ctl._session.executions[-1]['destination'] == client.destination
        assert client.recovery_calls == ctl._session.recovery_calls == 0
    finally:
        ctl.close()
