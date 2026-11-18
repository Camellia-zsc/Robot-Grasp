"""The queue uses fresh measurements and never runs two grasps concurrently."""
import copy
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cube_task.assistant.batch import bind_item, build_items, whole_footprint_in_zone
from cube_task.assistant.server import BusyError, DashboardController
from test_assistant_server import Settings


def scene(version=1):
    return {'scene_version': version,
            'objects': [dict(object_id=f'v{version}-{label}', label=label, valid=True,
                             center_base=xy + [.10 if label == 'pear' else .06],
                             length=.085 if label == 'pear' else .06,
                             width=.085 if label == 'pear' else .06, yaw=0., zone=None)
                        for label, xy in [('apple', [.24, -.08]), ('pear', [.30, .05])]],
            'class_to_zone': {'apple': 'zone_red', 'pear': 'zone_green'},
            'zones': [dict(name='zone_red', label='red', center=[.17, -.35, 0], size=[.2, .2], yaw=0),
                      dict(name='zone_green', label='green', center=[.17, .35, 0], size=[.2, .2], yaw=0)]}


def plan(observed):
    return {'scene_version': observed['scene_version'], 'action': 'pick_batch',
            'items': [{'object_id': o['object_id'], 'zone_name': None} for o in observed['objects']],
            'message': '先放苹果，再放梨'}


class BatchClient:
    def __init__(self):
        self.calls = []

    def decide(self, instruction, observed, on_delta, cancel_event=None):
        self.calls.append(copy.deepcopy(observed))
        on_delta('{"message":"先放苹果，再放梨"}')
        return plan(observed)


class BatchSession:
    fail_at = None
    missing_after_first = False
    drift_after_first = False
    ambiguous_after_first = False
    refresh_on_execute = True
    gate = None
    placed_with_fault = False

    def __init__(self, **kwargs):
        self.scene = scene()
        self.executions = []
        self.observe_calls = 0
        self.active = 0
        self.stopped = False
        self.reached = threading.Event()

    def set_speed(self, speed):
        pass

    def prepare(self):
        return self.scene

    def observe(self):
        self.observe_calls += 1
        self.scene = copy.deepcopy(self.scene)
        self.scene['scene_version'] += 1
        for o in self.scene['objects']:
            o['object_id'] = f"v{self.scene['scene_version']}-{o['label']}"
        return self.scene

    def execute(self, decision, cancel_event=None):
        assert decision['action'] == 'pick'
        assert decision['scene_version'] == self.scene['scene_version']
        assert self.active == 0
        self.active += 1
        self.executions.append(copy.deepcopy(decision))
        self.reached.set()
        try:
            if self.gate is not None and len(self.executions) == 1:
                assert self.gate.wait(3)
            if self.stopped or cancel_event.is_set():
                return {'ok': False, 'aborted': True, 'message': '已停止'}
            if self.fail_at == len(self.executions):
                return {'ok': False, 'message': '抓取失败', 'error': 'grasp_miss',
                        'truth': {'success': self.placed_with_fault}}
            target = next(o for o in self.scene['objects'] if o['object_id'] == decision['object_id'])
            zone = next(z for z in self.scene['zones'] if z['name'] == decision['zone_name'])
            target['center_base'] = list(zone['center'][:2]) + [target['center_base'][2]]
            target['zone'] = zone['name']
            if len(self.executions) == 1:
                if self.missing_after_first:
                    self.scene['objects'] = [o for o in self.scene['objects'] if o['label'] != 'pear']
                if self.drift_after_first:
                    self.scene['objects'][1]['center_base'][0] += .12
                if self.ambiguous_after_first:
                    extra = copy.deepcopy(self.scene['objects'][1])
                    extra.update(object_id='nearby-other-pear')
                    extra['center_base'][0] += .01
                    self.scene['objects'].append(extra)
            if self.refresh_on_execute:
                self.observe()
                if self.ambiguous_after_first and len(self.executions) == 1:
                    self.scene['objects'][-1]['object_id'] += '-decoy'
            return {'ok': True, 'message': '独立落位核验成功', 'truth': {'success': True}}
        finally:
            self.active -= 1

    def snapshot(self):
        return {'status': 'stopped' if self.stopped else 'ready', 'held_object': None}

    def stop(self):
        self.stopped = True

    def close(self):
        pass


def create(session_factory=BatchSession, client=None):
    ctl = DashboardController(session_factory=session_factory, client=client or BatchClient(),
                              settings=Settings(), realtime=False)
    assert ctl.wait_idle(3)
    return ctl


def test_each_item_uses_the_new_id_and_version_and_one_user_request():
    ctl = create()
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'complete' and state['last_result']['ok']
        assert state['batch']['completed'] == state['batch']['total'] == 2
        assert [i['status'] for i in state['batch']['items']] == ['succeeded', 'succeeded']
        a, b = ctl._session.executions
        assert a['object_id'] == 'v2-apple' and b['object_id'] == 'v3-pear'
        assert b['scene_version'] > a['scene_version']
        assert state['metrics'] == {'requests': 1, 'successes': 1, 'model_requests': 1}
    finally:
        ctl.close()


@pytest.mark.parametrize('cancel', [False, True])
def test_serial_queue_and_stop_never_start_the_next_grasp(cancel):
    gate = threading.Event()
    class Paused(BatchSession):
        pass
    Paused.gate = gate
    ctl = create(Paused)
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl._session.reached.wait(2)
        state = ctl.snapshot()
        assert state['batch']['current_index'] == 1
        assert [i['status'] for i in state['batch']['items']] == ['executing', 'pending']
        with pytest.raises(BusyError):
            ctl.submit('再抓一批')
        if cancel:
            ctl.stop()
        gate.set()
        assert ctl.wait_idle(3)
        assert len(ctl._session.executions) == (1 if cancel else 2)
        if cancel:
            state = ctl.snapshot()
            assert state['status'] == state['batch']['status'] == 'stopped'
            assert state['needs_reset'] and not state['last_result']['ok']
            assert all(i['status'] == 'cancelled' for i in state['batch']['items'])
    finally:
        gate.set()
        ctl.close()


@pytest.mark.parametrize('change', ['missing_after_first', 'drift_after_first', 'ambiguous_after_first'])
def test_next_target_must_still_be_uniquely_observed_near_its_original_position(change):
    class Changed(BatchSession):
        pass
    setattr(Changed, change, True)
    ctl = create(Changed)
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'error' and state['batch']['completed'] == 1
        assert len(ctl._session.executions) == 1
        assert state['batch']['items'][0]['status'] == 'succeeded'
        assert state['batch']['items'][1]['status'] == 'failed'
    finally:
        ctl.close()


def test_first_failure_cancels_remaining_items():
    class Failing(BatchSession):
        fail_at = 1
    ctl = create(Failing)
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert [i['status'] for i in state['batch']['items']] == ['failed', 'cancelled']
        assert len(ctl._session.executions) == 1 and state['batch']['completed'] == 0
    finally:
        ctl.close()


def test_adapter_without_automatic_rescan_is_observed_before_the_second_item():
    class NoRefresh(BatchSession):
        refresh_on_execute = False
    ctl = create(NoRefresh)
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl.wait_idle(3)
        assert ctl.snapshot()['last_result']['ok']
        assert ctl._session.executions[1]['scene_version'] == 3
    finally:
        ctl.close()


def test_center_inside_zone_does_not_skip_an_object_extending_outside_it():
    observed = scene()
    apple = observed['objects'][0]
    apple['center_base'] = [.26, -.35, .06]
    apple['zone'] = 'zone_red'
    assert not whole_footprint_in_zone(apple, observed, 'zone_red')
    apple['center_base'] = [.17, -.35, .06]
    assert whole_footprint_in_zone(apple, observed, 'zone_red')


def test_complete_footprint_already_in_destination_is_skipped_without_grasping():
    class AlreadyThere(BatchSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.scene['objects'][0]['center_base'] = [.17, -.35, .06]
    ctl = create(AlreadyThere)
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['last_result']['ok'] and state['batch']['completed'] == 2
        assert state['batch']['items'][0]['status'] == 'skipped'
        assert len(ctl._session.executions) == 1
    finally:
        ctl.close()


@pytest.mark.parametrize('audit', ['upright', 'fallen', 'moving', 'missing',
                                   'outside', 'outside_point', 'outside_region'])
def test_physical_batch_completion_skip_requires_released_supported_rest(audit):
    """Camera geometry alone cannot complete a physically unsettled pear."""
    from cube_task.core.types import Detection
    import numpy as np

    class PhysicalAlreadyThere(BatchSession):
        targeted_observation = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._environment_revision = self._observation_revision = 0
            self.audits = []
            self.geometry_audits = []
            self._physics = SimpleNamespace(resting_status=self.resting_status)
            self.scene['placement'] = {
                'bounds': {'x': [.08, .6], 'y': [-.4, .4], 'r_min': .1, 'r_max': .7},
                'table_bounds': {'x': [-.3, .9], 'y': [-.7, .7]},
                'pick_polygon': [[.05, -.5], [.55, -.5], [.55, .5], [.05, .5]],
                'point_tolerance': .015, 'margin': .002}
            for target in self.scene['objects']:
                zone = next(z for z in self.scene['zones']
                            if z['name'] == self.scene['class_to_zone'][target['label']])
                target['center_base'][:2] = zone['center'][:2]
                if target['label'] == 'pear':
                    target['resting_pose'] = 'upright'

        def observe(self, **kwargs):
            observed = super().observe()
            self._last_fresh = [Detection(target['label'], .99,
                center_base=np.asarray(target['center_base'], float))
                for target in observed['objects']]
            return observed

        def _evaluation_object(self, measured):
            return (None if measured.label == 'pear' and audit == 'missing' else
                    SimpleNamespace(name=measured.label))

        def resting_status(self, name):
            self.audits.append(name)
            success = name != 'pear' or audit == 'upright' or audit.startswith('outside')
            return dict(success=success, stable=audit != 'moving',
                        posture_ok=audit != 'fallen',
                        reason='pear_not_upright' if audit == 'fallen' else
                               'object_motion_or_support_unconfirmed' if not success else '')

        def _verify_zone(self, obj, zone_name):
            self.geometry_audits.append((obj.name, zone_name))
            return obj.name != 'pear' or audit != 'outside'

        def _verify_destination(self, obj, destination):
            self.geometry_audits.append((obj.name, destination['type']))
            return False

    class PhysicalClient(BatchClient):
        def decide(self, instruction, observed, on_delta, cancel_event=None):
            decision = super().decide(instruction, observed, on_delta, cancel_event)
            if audit == 'outside_point':
                decision['items'][1].update(zone_name=None,
                    destination=dict(type='point', xy=observed['objects'][1]['center_base'][:2]))
            elif audit == 'outside_region':
                decision['items'][1].update(zone_name=None,
                    destination=dict(type='region', name='pick_area'))
            return decision

    ctl = create(PhysicalAlreadyThere, client=PhysicalClient())
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert not ctl._session.executions
        assert state['batch']['items'][0]['status'] == 'skipped'
        if audit == 'upright':
            assert state['status'] == 'complete'
            assert state['last_result']['ok']
            assert ctl._session.audits == ['apple', 'pear']
            assert ctl._session.geometry_audits == [('apple', 'zone_red'), ('pear', 'zone_green')]
        else:
            assert state['status'] == 'error'
            assert not state['last_result']['ok']
            assert state['batch']['items'][1]['status'] == 'failed'
            assert state['batch']['completed'] == 1
            assert state['metrics']['successes'] == 0
            if audit.startswith('outside'):
                assert ctl._session.geometry_audits[-1] == (
                    'pear', audit.removeprefix('outside_') if '_' in audit else 'zone_green')
    finally:
        ctl.close()


def test_a_malformed_last_item_prevents_any_execution():
    class BadClient(BatchClient):
        def decide(self, instruction, observed, on_delta, cancel_event=None):
            decision = super().decide(instruction, observed, on_delta, cancel_event)
            decision['items'][-1]['object_id'] = 'invented'
            return decision
    ctl = create(client=BadClient())
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl.wait_idle(3)
        assert ctl.snapshot()['status'] == 'error'
        assert not ctl._session.executions
    finally:
        ctl.close()


def test_successful_placement_with_arm_failure_is_not_picked_again():
    class PlacedWithFault(BatchSession):
        fail_at = 1
        placed_with_fault = True
        def prepare_recovery(self, *args, **kwargs):
            raise AssertionError('An already placed item must not be regrasped')
    client = BatchClient()
    client.recover = lambda *a, **k: pytest.fail('Must not call recovery')
    ctl = create(PlacedWithFault, client)
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'error' and state['needs_reset']
        assert len(ctl._session.executions) == 1
        assert state['metrics']['model_requests'] == 1
    finally:
        ctl.close()


class RecoveringBatch(BatchSession):
    fail_at = 1
    repair_ok = True
    repair_placed_with_fault = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.repairs = []
        self.holding = False
        self.references = []

    def execute(self, decision, cancel_event=None):
        result = super().execute(decision, cancel_event)
        if not result['ok']:
            self.holding = True
            result.update(holding=True, fault=True)
        return result

    def prepare_recovery(self, failure, goal, attempt, history, reference_xy=None):
        self.references.append(reference_xy)
        return dict(scene_version=self.scene['scene_version'], attempt=attempt, max_attempts=3,
                    goal=goal, failure=failure, history=history, observations=[],
                    observation_mode='stationary_with_last_scan',
                    options=[{'id': 'place_held_alternate', 'description': '保持夹持续接放置'}])

    def execute_recovery(self, chosen, cancel_event=None):
        self.repairs.append(chosen)
        placed = self.repair_ok or self.repair_placed_with_fault
        self.holding = not placed
        if placed:
            self.scene['objects'][0]['center_base'] = [.17, -.35, .06]
            self.observe()
        return dict(ok=self.repair_ok, truth={'success': placed}, holding=self.holding,
                    fault=not self.repair_ok, message='已恢复' if self.repair_ok else '恢复失败')

    def snapshot(self):
        return dict(super().snapshot(), held_object='apple' if self.holding else None)


class RecoveryClient(BatchClient):
    def __init__(self, gate=None):
        super().__init__()
        self.gate = gate
        self.recovering = threading.Event()

    def recover(self, instruction, context, on_delta, cancel_event=None):
        on_delta('保持夹持，先完成当前物品')
        self.recovering.set()
        if self.gate:
            assert self.gate.wait(3)
        return dict(scene_version=context['scene_version'], action='recover',
                    option_id='place_held_alternate', message='先完成当前物品')


@pytest.mark.parametrize('success', [False, True])
def test_each_item_finishes_recovery_before_the_next_item_can_start(success):
    class Recovering(RecoveringBatch):
        repair_ok = success
    ctl = create(Recovering, RecoveryClient())
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['last_result']['ok'] is success
        assert len(ctl._session.executions) == (2 if success else 1)
        assert len(ctl._session.repairs) == (1 if success else 3)
        assert state['metrics']['model_requests'] == (2 if success else 4)
        assert all(xy == [.24, -.08] for xy in ctl._session.references)
        if not success:
            assert state['needs_reset'] and state['last_result']['holding']
            assert state['batch']['items'][1]['status'] == 'cancelled'
    finally:
        ctl.close()


def test_stop_during_model_recovery_preserves_the_held_item_and_cancels_the_rest():
    gate = threading.Event()
    client = RecoveryClient(gate)
    ctl = create(RecoveringBatch, client)
    try:
        ctl.submit('依次抓苹果和梨')
        assert client.recovering.wait(2)
        assert ctl.snapshot()['batch']['items'][0]['status'] == 'recovering'
        ctl.stop()
        gate.set()
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['batch']['status'] == 'stopped'
        assert state['last_result']['holding'] and state['last_result']['aborted']
        assert not ctl._session.repairs and len(ctl._session.executions) == 1
    finally:
        gate.set()
        ctl.close()


def test_a_repair_that_places_the_object_but_fails_retreat_is_not_repeated():
    class PlacedWithFault(RecoveringBatch):
        repair_ok = False
        repair_placed_with_fault = True
    ctl = create(PlacedWithFault, RecoveryClient())
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert not state['last_result']['ok'] and state['needs_reset']
        assert len(ctl._session.repairs) == 1 and len(ctl._session.executions) == 1
        assert state['metrics']['model_requests'] == 2
    finally:
        ctl.close()


@pytest.mark.parametrize('missing_second', [False, True])
def test_completed_object_cannot_replace_a_missing_next_object_of_the_same_class(missing_second):
    class TwoApples(BatchSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            first = self.scene['objects'][0]
            first['center_base'] = [.30, 0., .06]
            second = copy.deepcopy(first)
            second['center_base'] = [.20, -.35, .06] if missing_second else [.36, .05, .06]
            self.scene['objects'] = [first, second]

        def observe(self):
            super().observe()
            for index, obj in enumerate(self.scene['objects']):
                obj['object_id'] += f'-{index}'
            return self.scene

        def execute(self, decision, cancel_event=None):
            result = super().execute(decision, cancel_event)
            if missing_second and len(self.executions) == 1:
                self.scene['objects'] = self.scene['objects'][:1]
            return result

    ctl = create(TwoApples)
    try:
        ctl.submit('依次抓取这两个苹果放到红区')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['batch']['completed'] == (1 if missing_second else 2)
        assert len(ctl._session.executions) == (1 if missing_second else 2)
        assert state['batch']['items'][1]['status'] == ('failed' if missing_second else 'succeeded')
    finally:
        ctl.close()


def test_actual_recovery_does_not_substitute_a_far_away_object_of_the_same_class():
    from cube_task.assistant.simulation import SimulationSession
    session = object.__new__(SimulationSession)
    session._operation_lock = threading.RLock()
    session._check_open = Mock()
    session.cancel_event = threading.Event()
    session._recovery_goal = {'label': 'apple', 'zone_name': 'zone_red'}
    session._recovery_target = SimpleNamespace(summary=lambda: {'label': 'apple'}, extra={})
    session.editor = SimpleNamespace(metadata=lambda: {'one_per_class': False})
    session.arm = SimpleNamespace(health=lambda: SimpleNamespace(ok=True))
    session.world = SimpleNamespace(held=None)
    session._scene_version = 8
    session._publish_snapshot = Mock()
    observed = scene(8)
    observed['objects'] = [observed['objects'][0]]
    observed['objects'][0]['center_base'] = [.46, .05, .06]
    session.observe = lambda **_kwargs: observed
    context = session.prepare_recovery({'ok': False}, session._recovery_goal, 1, [],
                                       reference_xy=[.24, -.08])
    assert not context['options'] and session._recovery_retry is None


def test_actual_recovery_does_not_regrasp_a_completed_same_class_object():
    from cube_task.assistant.simulation import SimulationSession
    session = object.__new__(SimulationSession)
    session._operation_lock = threading.RLock()
    session._check_open = Mock()
    session.cancel_event = threading.Event()
    session._recovery_goal = {'label': 'apple', 'zone_name': 'zone_red'}
    session._recovery_target = SimpleNamespace(summary=lambda: {'label': 'apple'}, extra={})
    session.editor = SimpleNamespace(metadata=lambda: {'one_per_class': False})
    session.arm = SimpleNamespace(health=lambda: SimpleNamespace(ok=True))
    session.world = SimpleNamespace(held=None)
    session._scene_version = 8
    session._publish_snapshot = Mock()
    observed = scene(8)
    observed['objects'] = [observed['objects'][0]]
    observed['objects'][0]['center_base'] = [.17, -.35, .06]
    session.observe = lambda **_kwargs: observed
    context = session.prepare_recovery({'ok': False}, session._recovery_goal, 1, [],
                                       reference_xy=[.20, -.35], completed_zones=['zone_red'])
    assert not context['options'] and session._recovery_retry is None
