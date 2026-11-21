"""Resuming one held obligation must preserve the original queue's accounting."""
import copy
import threading
from types import SimpleNamespace

import pytest

from cube_task.assistant.batch import whole_footprint_at_goal
from test_assistant_batch import RecoveryClient, create
from test_assistant_recovery_approval import ConfirmedSession, pending


class Resumable(ConfirmedSession):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.runner = SimpleNamespace(has_held_recovery=True)
        self.cancel_event = threading.Event()
        self._recovery_goal = {'label': 'apple', 'zone_name': 'zone_red'}

    def execute(self, decision, cancel_event=None):
        result = super().execute(decision, cancel_event)
        if not result['ok'] and not result.get('aborted'):
            # This fixture represents a held scene refusal. A driver fault
            # requires intervention and must not make resume available.
            result.update(error='scene_changed', fault=False)
        return result


def answer(controller, approval, approved):
    controller.approve_recovery(dict(request_id=approval['request_id'],
        scene_version=approval['scene_version'], approved=approved))


def decline_initial_recovery(controller):
    controller.submit('依次抓苹果和梨')
    answer(controller, pending(controller), False)
    assert controller.wait_idle(3)
    assert controller.snapshot()['can_resume_recovery']


def test_legacy_held_driver_fault_cannot_resume_through_the_request_boundary():
    ctl = create(Resumable, RecoveryClient())
    try:
        decline_initial_recovery(ctl)
        ctl._last_result.update(fault=True, error='comm', aborted=False)
        assert not ctl.snapshot()['can_resume_recovery']
        with pytest.raises(ValueError, match='故障'):
            ctl.resume_recovery()
        assert not ctl._session.repairs
        assert not ctl.snapshot()['busy']
    finally:
        ctl.close()


def test_partial_batch_resume_does_not_count_the_whole_request_as_success():
    ctl = create(Resumable, RecoveryClient())
    try:
        decline_initial_recovery(ctl)
        ctl.resume_recovery()
        answer(ctl, pending(ctl), True)
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert [item['status'] for item in state['batch']['items']] == ['succeeded', 'cancelled']
        assert state['batch']['completed'] == 1
        assert not state['last_result']['ok']
        assert state['status'] == 'error'
        assert state['metrics']['successes'] == 0
    finally:
        ctl.close()


def test_stop_while_resumed_recovery_waits_for_approval_ends_the_queue_item():
    ctl = create(Resumable, RecoveryClient())
    try:
        decline_initial_recovery(ctl)
        ctl.resume_recovery()
        pending(ctl)
        ctl.stop()
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == state['batch']['status'] == 'stopped'
        assert all(item['status'] != 'recovering' for item in state['batch']['items'])
        assert state['last_result']['aborted'] and state['last_result']['holding']
        assert not ctl._session.repairs
    finally:
        ctl.close()


def test_stop_after_recovery_result_does_not_publish_a_completed_resume():
    ctl = create(Resumable, RecoveryClient())
    reached = threading.Event()
    release = threading.Event()
    original_event = ctl._event

    def event(kind, payload=None):
        if kind == 'recovery_result':
            reached.set()
            assert release.wait(3)
        original_event(kind, payload)

    try:
        decline_initial_recovery(ctl)
        ctl._event = event
        ctl.resume_recovery()
        answer(ctl, pending(ctl), True)
        assert reached.wait(2)
        ctl.stop()
        release.set()
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == state['batch']['status'] == 'stopped'
        assert state['last_result']['aborted']
        assert state['metrics']['successes'] == 0
    finally:
        release.set()
        ctl.close()


class MeasuredFinalItem(Resumable):
    fail_at = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targeted_observation = False
        self._environment_revision = 0
        self.protected = []
        self.final_checks = []

    def execute(self, decision, cancel_event=None):
        target = next(obj for obj in self.scene['objects']
                      if obj['object_id'] == decision['object_id'])
        result = super().execute(decision, cancel_event)
        self.targeted_observation = True
        if not result['ok']:
            self._recovery_goal = {'label': target['label'], 'zone_name': decision['zone_name']}
        return result

    def prepare_recovery(self, *args, protected_goals=(), **kwargs):
        self.protected.append(copy.deepcopy(list(protected_goals)))
        return super().prepare_recovery(*args, **kwargs)

    def execute_recovery(self, chosen, cancel_event=None, confirmed=False):
        assert confirmed is True
        self.repairs.append(chosen)
        self.holding = False
        goal = self._recovery_goal
        target = next(obj for obj in self.scene['objects'] if obj['label'] == goal['label'])
        zone = next(zone for zone in self.scene['zones'] if zone['name'] == goal['zone_name'])
        target['center_base'] = list(zone['center'][:2]) + [target['center_base'][2]]
        self.observe()
        return dict(ok=True, holding=False, fault=False, truth={'success': True},
                    verified_revision=self._environment_revision, message='当前目标已完成')

    def verify_goals(self, goals):
        self.final_checks.append(copy.deepcopy(goals))
        items = []
        for goal in goals:
            target = next(obj for obj in self.scene['objects'] if obj['label'] == goal['label'])
            items.append(dict(goal=copy.deepcopy(goal),
                ok=whole_footprint_at_goal(target, self.scene, goal),
                verified_revision=self._environment_revision))
        return dict(ok=all(item['ok'] for item in items), items=items,
                    verified_revision=self._environment_revision, message='已重新核验整批目标')


@pytest.mark.parametrize('displace_prior', [False, True])
def test_resume_rechecks_prior_goal_and_counts_only_current_successes(displace_prior):
    ctl = create(MeasuredFinalItem, RecoveryClient())
    try:
        decline_initial_recovery(ctl)
        session = ctl._session
        assert ctl.snapshot()['batch']['completed'] == 1
        # A runtime edit may displace the earlier completed apple while the pear is held.
        if displace_prior:
            session.scene['objects'][0]['center_base'][:2] = [.4, 0.]
        session.scene['scene_version'] += 1
        session._environment_revision += 1
        ctl.resume_recovery()
        answer(ctl, pending(ctl), True)
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        expected = [{'label': 'apple', 'zone_name': 'zone_red'}]
        assert session.protected == [expected, expected]
        assert session.final_checks
        assert state['batch']['items'][0]['status'] == ('failed' if displace_prior else 'succeeded')
        assert state['batch']['completed'] == (1 if displace_prior else 2)
        assert state['last_result']['ok'] is not displace_prior
        assert state['status'] == ('error' if displace_prior else 'complete')
        assert state['metrics']['successes'] == int(not displace_prior)
    finally:
        ctl.close()
