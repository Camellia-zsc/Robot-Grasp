"""A current audit cannot override later camera/physical verification evidence."""
import copy

import pytest

from test_assistant_deferred_final_audit import create, AuditExecutor, GOALS, deny_network
from test_assistant_skill_protocol import context, step


@pytest.mark.parametrize('skill', ['skip', 'verify'])
def test_new_positive_evidence_replaces_an_incomplete_audit_before_finish(create, monkeypatch, skill):
    original = AuditExecutor.execute

    def execute(self, action, goals, *args, **kwargs):
        if skill == 'verify' and action['skill'] == 'verify' and 'goal_index' in action:
            self.session._scene['objects'][0]['zone'] = goals[0]['zone_name']
        result = original(self, action, goals, *args, **kwargs)
        if action['skill'] == 'move':
            result.update(skipped=True, changed=False, label=action['label'], controller_ok=True,
                          resolved_destination=dict(type='named_region', name=goals[0]['zone_name']))
        return result

    monkeypatch.setattr(AuditExecutor, 'execute', execute)

    def compose(_context, number):
        if number == 1:
            return [step('normal', 'observe', 'apple')]
        if number == 2:
            check = (step('found-complete', 'move', 'apple', dict(type='goal', index=0))
                     if skill == 'skip' else step('new-check', 'verify', goal_index=0))
            return [check, step('finish', 'finish')]
        return dict(action='stop', steps=[])

    ctl, session, client = create(compose, GOALS[:1])
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert len(client.contexts) == 2
    assert client.contexts[1]['final_audit']['items'][0]['ok'] is False
    assert state['last_result']['ok'] is True
    item = state['skill_task']['final_audit']['items'][0]
    assert item['ok'] is True and item['camera_verified'] is True
    if skill == 'skip':
        assert len(session.checked) == 1
    else:
        assert len(session.checked) == 2


@pytest.mark.parametrize('skill', ['skip', 'verify'])
def test_unchanged_positive_evidence_keeps_incomplete_audit_without_rescanning_other_goals(create, monkeypatch, skill):
    original = AuditExecutor.execute

    def execute(self, action, goals, *args, **kwargs):
        result = original(self, action, goals, *args, **kwargs)
        if action['skill'] == 'move':
            result.update(skipped=True, changed=False, label=action['label'], controller_ok=True,
                          resolved_destination=dict(type='named_region', name=goals[0]['zone_name']))
        return result

    monkeypatch.setattr(AuditExecutor, 'execute', execute)

    def compose(_context, number):
        if number == 1:
            return [step('normal', 'observe', 'apple')]
        if number == 3:
            check = (step('still-complete', 'move', 'apple', dict(type='goal', index=0))
                     if skill == 'skip' else step('unchanged-check', 'verify', goal_index=0))
            return [check, step('finish', 'finish')]
        return dict(action='stop', steps=[])

    ctl, session, client = create(compose)
    session._scene['objects'][0]['zone'] = 'zone_red'
    ctl.submit('依次把苹果、香蕉、梨子放到各自默认区域')
    assert ctl.wait_idle(3)
    assert len(client.contexts) == 3
    assert client.contexts[2]['final_audit']['items'][0]['ok'] is True
    assert not ctl.snapshot()['last_result']['ok']
    assert session.checked.count(GOALS) == 1


def test_new_negative_verify_invalidates_old_positive_item_without_environment_edit(create, monkeypatch):
    original = AuditExecutor.execute

    def execute(self, action, goals, *args, **kwargs):
        result = original(self, action, goals, *args, **kwargs)
        if action['skill'] == 'verify' and action.get('goal_index') == 0:
            result.update(ok=False, camera_verified=True, truth=dict(success=False),
                          error='support_not_stable')
            result['items'][0].update(ok=False, camera_verified=True, truth=dict(success=False),
                                      error='support_not_stable')
        return result

    monkeypatch.setattr(AuditExecutor, 'execute', execute)
    later = []

    def compose(value, number):
        if number == 1:
            return [step('normal', 'observe', 'apple')]
        if number == 3:
            return [step('new-failed-check', 'verify', goal_index=0)]
        if number == 4:
            later.append(copy.deepcopy(value))
        return dict(action='stop', steps=[])

    ctl, session, client = create(compose)
    session._scene['objects'][0]['zone'] = 'zone_red'
    ctl.submit('依次把苹果、香蕉、梨子放到各自默认区域')
    assert ctl.wait_idle(3)
    assert session._environment_revision == 0
    assert later
    audit = later[0].get('final_audit')
    assert audit is None or audit['items'][0]['ok'] is False
    assert not ctl.snapshot()['last_result']['ok']
