"""Keep placement capacity separate from a blocked camera inspection route."""
import copy
import json
import threading
from types import SimpleNamespace

import pytest

from cube_task.assistant.skill_orchestrator import SkillOrchestrator
from cube_task.assistant.skill_protocol import model_skill_context
from test_assistant_skill_protocol import context, step
from test_assistant_skill_orchestrator import create, Executor


def checks(capacity):
    return dict(release_rejection='release support not sufficiently visible',
                capacity=capacity, capacity_reason='measured geometric preflight',
                inspection_rejection='held-camera path blocked')


@pytest.mark.parametrize('capacity', [True, False, None])
def test_feedback_history_and_next_failure_preserve_three_state_capacity(context, capacity):
    events = []
    session = SimpleNamespace(snapshot=lambda: dict(held_object='apple'))
    controller = SimpleNamespace(_session=session, _lock=threading.RLock(),
        _skill_task=dict(history=[], executed=0),
        _event=lambda kind, value: events.append((kind, value)))
    orchestrator = SkillOrchestrator(controller)
    result = dict(ok=False, stage='inspect_place', error='placement_changed', holding=True,
                  placement_checks=checks(capacity), message='仍保持夹持')
    before = copy.deepcopy(result)
    orchestrator._feedback(dict(id='place-1', skill='place', label='apple',
                               destination=dict(type='goal', index=0)), result, 2)
    context.update(holding=True, held_label='apple', failure=result,
                   last_action_failure=result, history=controller._skill_task['history'])
    visible = model_skill_context(context)
    assert visible['failure']['placement_checks'] == checks(capacity)
    assert visible['last_action_failure']['placement_checks'] == checks(capacity)
    assert visible['history'][0]['placement_checks'] == checks(capacity)
    assert visible['failure']['placement_checks']['capacity'] is capacity
    assert events[0][1]['feedback']['placement_checks'] == checks(capacity)
    assert result == before
    result['placement_checks']['release_rejection'] = 'changed afterward'
    assert controller._skill_task['history'][0]['placement_checks'] == checks(capacity)


def test_placement_diagnostics_allowlist_drops_nontext_geometry_and_unknown_keys(context):
    value = dict(capacity=False, release_rejection='x' * 700,
                 capacity_reason={'truth': 'private-sentinel'},
                 inspection_rejection=['private-sentinel'],
                 object_transform='private-sentinel', truth='private-sentinel')
    context.update(failure=dict(placement_checks=value),
                   last_action_failure=dict(placement_checks=value),
                   history=[dict(skill='place', placement_checks=value)])
    visible = model_skill_context(context)
    expected = dict(capacity=False, release_rejection='x' * 500)
    for entry in (visible['failure'], visible['last_action_failure'], visible['history'][0]):
        assert entry['placement_checks'] == expected
    assert 'private-sentinel' not in json.dumps(visible)


@pytest.mark.parametrize('capacity', [0, 1, 'false', [], {}])
def test_invalid_capacity_is_never_coerced_into_a_geometric_conclusion(context, capacity):
    context['failure'] = dict(placement_checks=dict(capacity=capacity,
                              inspection_rejection='blocked'))
    value = model_skill_context(context)['failure']['placement_checks']
    assert value == dict(inspection_rejection='blocked')


@pytest.mark.parametrize('malformed', [None, [], 'private-sentinel', False])
def test_nonobject_placement_checks_are_not_forwarded(context, malformed):
    context.update(failure=dict(placement_checks=malformed),
                   history=[dict(placement_checks=malformed)])
    visible = model_skill_context(context)
    assert 'placement_checks' not in visible['failure']
    assert 'placement_checks' not in visible['history'][0]


def test_placement_capacity_is_retained_after_an_observe_only_replan(create, monkeypatch):
    original = Executor.execute

    def execute(self, action, *args, **kwargs):
        result = original(self, action, *args, **kwargs)
        if action['skill'] == 'place' and not result['ok']:
            result['placement_checks'] = dict(checks(True), simulator_pose='private-sentinel')
        if action['skill'] == 'observe':
            result.update(changed=False, reason='no_new_measured_coverage')
        return result

    monkeypatch.setattr(Executor, 'execute', execute)

    def compose(_context, number):
        if number == 1:
            return [step(1, 'pick', 'apple'),
                    step(2, 'place', 'apple', dict(type='goal', index=0))]
        if number == 2:
            return [step(3, 'observe')]
        return [step(4, 'place', 'apple', dict(type='goal', index=0)), step(5, 'finish')]

    controller, session, client = create(compose)
    session.fail_place = True
    controller.submit('把苹果放红区')
    assert controller.wait_idle(3)
    assert controller.snapshot()['last_result']['ok']
    visible = model_skill_context(client.contexts[2])
    assert visible['failure']['error'] == 'goal_not_finished'
    assert visible['last_action_failure']['placement_checks'] == checks(True)
    assert visible['history'][-2]['placement_checks'] == checks(True)
    assert 'placement_checks' not in visible['history'][-1]
    assert 'private-sentinel' not in json.dumps(visible)
