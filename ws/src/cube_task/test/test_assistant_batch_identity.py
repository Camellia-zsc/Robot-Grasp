"""Completed placements cannot become identities for later batch targets."""
import copy
import math
from unittest.mock import Mock

import pytest

from cube_task.assistant.batch import bind_item, matching_objects
from test_assistant_batch import BatchClient, BatchSession, create, scene


@pytest.mark.parametrize('local_xy, excluded', [
    ((.139, .089), True),
    ((-.139, -.089), True),
    ((.141, 0.), False),
    ((0., .091), False),
])
def test_completed_region_uses_the_rotated_rectangle(local_xy, excluded):
    observed = scene()
    zone = observed['zones'][0]
    zone.update(center=[0., 0., 0.], size=[.20, .10], yaw=math.pi / 4)
    x, y = local_xy
    c, s = math.cos(zone['yaw']), math.sin(zone['yaw'])
    xy = [c * x - s * y, s * x + c * y]
    apple = observed['objects'][0]
    apple.update(center_base=xy + [.06], zone=None, in_zone=False)

    matches = matching_objects(observed, 'apple', xy, ('zone_red',))

    assert matches == ([] if excluded else [apple])


@pytest.mark.parametrize('axis, sign, outside', [
    (axis, sign, outside)
    for axis in (0, 1) for sign in (-1, 1) for outside in (False, True)
])
def test_completed_region_includes_exactly_40mm_but_not_beyond(axis, sign, outside):
    observed = scene()
    observed['zones'][0].update(center=[0., 0., 0.], size=[.20, .10], yaw=0.)
    boundary = (.10 if axis == 0 else .05) + .04
    xy = [0., 0.]
    xy[axis] = sign * (boundary + (1e-6 if outside else 0.))
    apple = observed['objects'][0]
    apple['center_base'] = xy + [.06]

    matches = matching_objects(observed, 'apple', xy, ('zone_red',))

    assert matches == ([apple] if outside else [])


@pytest.mark.parametrize('distance, found', [
    (-.040001, False), (-.04, True), (.04, True), (.040001, False),
])
def test_original_position_matching_has_a_40mm_limit(distance, found):
    observed = scene()
    apple = observed['objects'][0]
    apple['center_base'] = [distance, 0., .06]

    matches = matching_objects(observed, 'apple', [0., 0.])

    assert matches == ([apple] if found else [])


@pytest.mark.parametrize('change', [
    'missing', 'duplicate', 'missing_center', 'short_center', 'nan_center',
    'missing_size', 'short_size', 'zero_size', 'negative_size', 'nan_size',
    'nan_yaw', 'infinite_yaw', 'nonnumeric_yaw',
])
def test_unusable_completed_zone_geometry_stops_identity_matching(change):
    observed = scene()
    apple = observed['objects'][0]
    apple['center_base'] = [.17, -.35, .06]
    zone = observed['zones'][0]
    if change == 'missing':
        observed['zones'] = observed['zones'][1:]
    elif change == 'duplicate':
        observed['zones'].append(copy.deepcopy(zone))
    elif change.startswith('missing_'):
        del zone[change.removeprefix('missing_')]
    else:
        field, value = {
            'short_center': ('center', [.17, -.35]),
            'nan_center': ('center', [.17, float('nan'), 0.]),
            'short_size': ('size', [.2]),
            'zero_size': ('size', [.2, 0.]),
            'negative_size': ('size', [-.2, .2]),
            'nan_size': ('size', [.2, float('nan')]),
            'nan_yaw': ('yaw', float('nan')),
            'infinite_yaw': ('yaw', float('inf')),
            'nonnumeric_yaw': ('yaw', 'unknown'),
        }[change]
        zone[field] = value

    with pytest.raises(ValueError, match='区域信息无效'):
        matching_objects(observed, 'apple', [.17, -.35], ('zone_red',))


def test_same_class_target_outside_completed_region_can_be_bound():
    observed = scene(8)
    completed = observed['objects'][0]
    completed['center_base'] = [.17, -.35, .06]
    pending = copy.deepcopy(completed)
    pending.update(object_id='v8-other-apple', center_base=[.36, .05, .06])
    observed['objects'] = [completed, pending]
    item = dict(index=2, label='apple', reference_xy=[.36, .05], zone_name='zone_green')

    decision, target = bind_item(item, observed, ('zone_red',))

    assert target is pending
    assert decision['object_id'] == 'v8-other-apple'
    assert decision['scene_version'] == 8 and decision['zone_name'] == 'zone_green'


def test_completed_zone_protection_is_scoped_to_the_same_class():
    class PearInRedZone(BatchSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.scene['objects'][1]['center_base'] = [.17, -.35, .10]

    ctl = create(PearInRedZone)
    try:
        ctl.submit('先把苹果放到红区，再把红区里的梨放到绿区')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['batch']['completed'] == 2
        assert [item['status'] for item in state['batch']['items']] == ['succeeded', 'succeeded']
        assert [d['zone_name'] for d in ctl._session.executions] == ['zone_red', 'zone_green']
    finally:
        ctl.close()


def test_completed_regions_do_not_carry_over_to_a_new_user_instruction():
    client = BatchClient()
    ctl = create(client=client)
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl.wait_idle(3)
        assert ctl.snapshot()['batch']['completed'] == 2

        ctl.submit('重新检查苹果和梨，都放到各自的区域')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == state['batch']['status'] == 'complete'
        assert state['batch']['completed'] == state['batch']['total'] == 2
        assert [item['status'] for item in state['batch']['items']] == ['skipped', 'skipped']
        assert len(ctl._session.executions) == 2 and len(client.calls) == 2
        assert state['metrics'] == {'requests': 2, 'successes': 2, 'model_requests': 2}
    finally:
        ctl.close()


def test_second_same_class_failure_passes_completed_regions_to_recovery():
    class FailingSecondApple(BatchSession):
        fail_at = 2

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            first = self.scene['objects'][0]
            first['center_base'] = [.30, 0., .06]
            second = copy.deepcopy(first)
            second['center_base'] = [.36, .05, .06]
            self.scene['objects'] = [first, second]
            self.recovery_calls = []

        def observe(self):
            super().observe()
            for index, obj in enumerate(self.scene['objects']):
                obj['object_id'] += f'-{index}'
            return self.scene

        def prepare_recovery(self, failure, goal, attempt, history,
                             reference_xy=None, completed_zones=()):
            self.recovery_calls.append(copy.deepcopy(dict(
                goal=goal, reference_xy=reference_xy, completed_zones=completed_zones)))
            return {'options': []}

    client = BatchClient()
    client.recover = Mock(side_effect=AssertionError('No safe recovery option should reach the model'))
    ctl = create(FailingSecondApple, client)
    try:
        ctl.submit('依次把两个苹果放到红区')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['batch']['completed'] == 1
        assert [item['status'] for item in state['batch']['items']] == ['succeeded', 'failed']
        assert len(ctl._session.executions) == 2
        assert ctl._session.recovery_calls == [{
            'goal': {'label': 'apple', 'zone_name': 'zone_red'},
            'reference_xy': [.36, .05], 'completed_zones': ('zone_red',),
        }]
        client.recover.assert_not_called()
        assert state['metrics']['model_requests'] == 1
    finally:
        ctl.close()
