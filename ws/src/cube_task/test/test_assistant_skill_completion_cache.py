"""Offline regressions for completion skips and manipulation-proof freshness."""
import copy
import socket
import threading
from types import SimpleNamespace

import pytest

from cube_task.assistant.server import DashboardController
from cube_task.assistant.skill_orchestrator import SkillOrchestrator
from test_assistant_skill_orchestrator import (
    Client, Executor as BaseExecutor, GOALS, Session as BaseSession, Settings, approve,
)
from test_assistant_skill_protocol import context, step


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('completion-cache tests must not access the network')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


class Session(BaseSession):
    def __init__(self, observed):
        super().__init__(observed)
        self._operation_lock = threading.RLock()
        self._intervention_lock = threading.Lock()
        self._interventions = []
        self.completion_checks = []
        self.result_hook = None
        self.completion_hook = None
        self.world = SimpleNamespace(held=None, objects=[SimpleNamespace(name=o['label'], label=o['label'])
                                                       for o in self._scene['objects']])
        self.support_stable = True
        self._physics = SimpleNamespace(sync=lambda: None, support_status=self._support_status)

    def _support_status(self, object_id, support_label):
        target = next(o for o in self._scene['objects'] if o['label'] == object_id)
        contact = target.get('support_label') == support_label
        return dict(success=contact and self.support_stable, contact=contact, stable=self.support_stable)

    def snapshot(self):
        snap = super().snapshot()
        snap['editor']['intervention_pending'] = len(self._interventions)
        return snap


def resolved(action, goals):
    destination = action.get('destination', {})
    if destination.get('type') == 'goal':
        goal = goals[destination['index']]
        return copy.deepcopy(goal.get('destination') or
                             dict(type='named_region', name=goal['zone_name']))
    if destination.get('type') == 'zone':
        return dict(type='named_region', name=destination['name'])
    return copy.deepcopy(destination)


class Executor(BaseExecutor):
    """Keep physical calls distinct from read-only task-completion checks."""

    def execute(self, action, goals, *args, **kwargs):
        s = self.session
        if action['skill'] == 'observe' and s._interventions:
            s._interventions.clear()
            s._environment_revision += 1
        selected = ([goals[action['goal_index']]]
                    if action['skill'] == 'verify' and 'goal_index' in action else goals)
        if action['skill'] == 'verify' and any(
                (g.get('destination') or {}).get('type') == 'on_object' for g in selected):
            s.calls.append(copy.deepcopy(action))
            items = []
            for goal in selected:
                target = next(o for o in s._scene['objects'] if o['label'] == goal['label'])
                ok = target.get('support_label') == goal['destination']['label']
                items.append(dict(goal=copy.deepcopy(goal), ok=ok, truth={'success': ok}))
            ok = all(item['ok'] for item in items)
            result = dict(ok=ok, camera_verified=ok, truth={'success': ok}, items=items,
                          verified_revision=s._environment_revision, holding=False)
        else:
            result = super().execute(action, selected, *args, **kwargs)
        if action['skill'] in ('move', 'place') and result['ok']:
            result.update(label=action['label'], controller_ok=True,
                          resolved_destination=resolved(action, goals))
            if result['resolved_destination'].get('type') == 'on_object':
                target = next(o for o in s._scene['objects'] if o['label'] == action['label'])
                target['support_label'] = result['resolved_destination']['label']
        if s.result_hook:
            s.result_hook(action, result)
        return result

    def check_completed(self, action, goals, **kwargs):
        s = self.session
        s.completion_checks.append(copy.deepcopy(action))
        target = next(o for o in s._scene['objects'] if o['label'] == action['label'])
        destination = resolved(action, goals)
        complete = (not s.holding and destination.get('type') == 'named_region'
                    and target.get('zone') == destination.get('name'))
        result = (dict(ok=True, skipped=True, label=action['label'], controller_ok=True,
                       resolved_destination=destination, camera_verified=True,
                       truth={'success': True}, verified_revision=s._environment_revision,
                       holding=False, message='现场目标已完成，跳过重复动作')
                  if complete else None)
        if s.completion_hook:
            result = s.completion_hook(action, result)
        return result


@pytest.fixture
def create(context, monkeypatch):
    from cube_task.assistant import skill_execution
    monkeypatch.setattr(skill_execution, 'SkillExecutor', Executor)
    controllers = []

    def factory(fn, goals=GOALS):
        session, client = Session(context['scene']), Client(fn)
        original_plan = client.plan_skills

        def plan(*args, **kwargs):
            return dict(original_plan(*args, **kwargs), goals=copy.deepcopy(goals))

        client.plan_skills = plan
        ctl = DashboardController(session_factory=lambda **kwargs: session,
                                  client=client, settings=Settings())
        assert ctl.wait_idle(2)
        controllers.append(ctl)
        return ctl, session, client

    yield factory
    for ctl in controllers:
        ctl.close()


def run(ctl):
    ctl.submit('按原始目标逐个完成任务')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert state['status'] == 'complete', state['last_result']
    return state


def normal(_context, _number):
    return [step(1, 'move', 'apple', {'type': 'goal', 'index': 0}),
            step(2, 'verify', goal_index=0), step(3, 'finish')]


def test_verified_single_move_supplies_goal_and_finish_without_duplicate_camera_pass(create):
    ctl, session, _ = create(normal)
    state = run(ctl)
    assert [s['skill'] for s in session.calls] == ['move']
    assert state['last_result']['verification_reused']
    assert state['skill_task']['executed'] == 3


def test_verified_place_supplies_goal_and_finish_without_duplicate_camera_pass(create):
    ctl, session, _ = create(lambda c, n: [
        step(1, 'pick', 'apple'), step(2, 'place', 'apple', {'type': 'goal', 'index': 0}),
        step(3, 'verify', goal_index=0), step(4, 'finish')])
    run(ctl)
    assert [s['skill'] for s in session.calls] == ['pick', 'place']


@pytest.mark.parametrize('actions', [('move',), ('pick', 'place')])
def test_settled_stack_action_supplies_verify_and_finish_without_duplicate_scan(create, actions):
    goals = [dict(label='apple', zone_name=None,
                  destination=dict(type='on_object', label='pear'))]
    def compose(c, n):
        steps = [step(i, skill, 'apple', {'type': 'goal', 'index': 0}
                      if skill != 'pick' else None) for i, skill in enumerate(actions)]
        return steps + [step(8, 'verify'), step(9, 'finish')]
    ctl, session, _ = create(compose, goals)
    state = run(ctl)
    assert [s['skill'] for s in session.calls] == list(actions)
    assert state['last_result']['verification_reused']
    assert state['last_result']['items'][0]['goal'] == goals[0]


def test_stack_verify_proof_is_not_scanned_again_by_finish(create):
    goals = [dict(label='apple', zone_name=None,
                  destination=dict(type='on_object', label='pear'))]
    ctl, session, _ = create(lambda c, n: [step(1, 'verify'), step(2, 'finish')], goals)
    session._scene['objects'][0]['support_label'] = 'pear'
    run(ctl)
    assert [s['skill'] for s in session.calls] == ['verify']


@pytest.mark.parametrize('change', ['lost_contact', 'unstable'])
def test_stack_camera_proof_rechecks_physics_without_a_revision_change(create, change):
    goals = [dict(label='apple', zone_name=None,
                  destination=dict(type='on_object', label='pear'))]
    ctl, session, _ = create(normal, goals)
    orchestrator, executor = SkillOrchestrator(ctl, Executor), Executor(session)
    action = step(1, 'move', 'apple', {'type': 'goal', 'index': 0})
    proof = orchestrator._action_proof(action, executor.execute(action, goals), goals)
    assert orchestrator._cached_verification(proof, goals) is not None
    if change == 'lost_contact':
        session._scene['objects'][0]['support_label'] = None
    else:
        session.support_stable = False
    assert session._environment_revision == proof['result']['verified_revision']
    assert orchestrator._cached_verification(proof, goals) is None


def test_stack_support_is_checked_again_when_final_success_is_published(create):
    goals = [dict(label='apple', zone_name=None,
                  destination=dict(type='on_object', label='pear'))]
    ctl, session, _ = create(normal, goals)
    original_event = ctl._event
    def change_after_verification(kind, payload=None):
        original_event(kind, payload)
        if kind == 'skill_feedback' and payload['feedback']['skill'] == 'finish':
            session.support_stable = False
    ctl._event = change_after_verification
    ctl.submit('把苹果放在梨上')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert not state['last_result']['ok']
    assert state['last_result']['truth']['success'] is False
    assert state['metrics']['successes'] == 0
    assert [s['skill'] for s in session.calls] == ['move']


@pytest.mark.parametrize('change', ['physics_revision', 'queued_intervention', 'new_support'])
def test_stack_proof_is_not_reused_after_physical_change_or_for_another_support(create, change):
    goals = [dict(label='apple', zone_name=None,
                  destination=dict(type='on_object', label='pear'))]
    ctl, session, _ = create(normal, goals)
    orchestrator, executor = SkillOrchestrator(ctl, Executor), Executor(session)
    action = step(1, 'move', 'apple', {'type': 'goal', 'index': 0})
    proof = orchestrator._action_proof(action, executor.execute(action, goals), goals)
    assert proof is not None
    if change == 'physics_revision':
        session._environment_revision += 1
    elif change == 'queued_intervention':
        session._interventions.append({'label': 'pear'})
    else:
        goals[0]['destination']['label'] = 'banana'
    assert orchestrator._cached_verification(proof, goals) is None


@pytest.mark.parametrize('field,value', [
    ('camera_verified', None), ('camera_verified', False),
    ('truth', None), ('truth', {'success': False}),
    ('controller_ok', None), ('controller_ok', False),
    ('verified_revision', None), ('verified_revision', -1),
    ('label', 'banana'), ('resolved_destination', None),
    ('resolved_destination', {'type': 'named_region', 'name': 'zone_green'}),
])
def test_incomplete_or_mismatched_action_evidence_requires_fresh_verify(create, field, value):
    ctl, session, _ = create(normal)

    def corrupt(action, result):
        if action['skill'] == 'move':
            if value is None:
                result.pop(field, None)
            else:
                result[field] = copy.deepcopy(value)

    session.result_hook = corrupt
    run(ctl)
    assert [s['skill'] for s in session.calls] == ['move', 'verify']


@pytest.mark.parametrize('flag', ['fault', 'aborted'])
def test_faulted_or_aborted_actions_never_count_as_cached_success(create, flag):
    ctl, session, _ = create(normal)
    session.result_hook = lambda action, result: result.update(
        {flag: True, 'ok': False}) if action['skill'] == 'move' else None
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert state['metrics']['successes'] == 0
    assert not state['last_result']['ok']


def test_later_physical_move_invalidates_earlier_goal_proof_even_without_revision_change(create):
    goals = [dict(label='apple', zone_name='zone_red'),
             dict(label='banana', zone_name='zone_green')]
    ctl, session, _ = create(lambda c, n: [
        step(1, 'move', 'apple', {'type': 'goal', 'index': 0}),
        step(2, 'verify', goal_index=0),
        step(3, 'move', 'banana', {'type': 'goal', 'index': 1}),
        step(4, 'verify', goal_index=1), step(5, 'finish')], goals)
    run(ctl)
    assert session._environment_revision == 0
    assert [s['skill'] for s in session.calls] == ['move', 'move', 'verify']
    assert session.calls[-1]['id'] == 'final-check'


def test_later_move_cannot_hide_displacement_of_previously_verified_target(create):
    goals = [dict(label='apple', zone_name='zone_red'),
             dict(label='banana', zone_name='zone_green')]

    def compose(c, n):
        if n == 1:
            return [step(1, 'move', 'apple', {'type': 'goal', 'index': 0}),
                    step(2, 'verify', goal_index=0),
                    step(3, 'move', 'banana', {'type': 'goal', 'index': 1}),
                    step(4, 'finish')]
        return [step(5, 'move', 'apple', {'type': 'goal', 'index': 0}), step(6, 'finish')]

    ctl, session, client = create(compose, goals)

    def disturb(action, result):
        if action['skill'] == 'move' and action['label'] == 'banana':
            next(o for o in session._scene['objects'] if o['label'] == 'apple')['zone'] = None

    session.result_hook = disturb
    run(ctl)
    assert len(client.contexts) == 2
    assert client.contexts[1]['failure']['ok'] is False
    assert [s['label'] for s in session.calls if s['skill'] == 'move'] == ['apple', 'banana', 'apple']


@pytest.mark.parametrize('queued', [False, True])
def test_manual_interference_invalidates_cached_result_before_finish(create, queued):
    def compose(c, n):
        return normal(c, n) if n == 1 else [step(4, 'verify'), step(5, 'finish')]

    ctl, session, client = create(compose)

    def interfere(action, result):
        if action['skill'] == 'move':
            if queued:
                session._interventions.append({'label': 'banana'})
            else:
                session._environment_revision += 1

    session.result_hook = interfere
    run(ctl)
    assert len(client.contexts) == 2
    assert client.contexts[1]['failure']['error'] == 'scene_changed'
    assert [s['skill'] for s in session.calls] == ['move', 'observe', 'verify']


def test_already_complete_pick_observe_place_pair_skips_both_physical_actions(create):
    ctl, session, _ = create(lambda c, n: [
        step(1, 'pick', 'apple'), step(2, 'observe', 'apple'),
        step(3, 'place', 'apple', {'type': 'goal', 'index': 0}), step(4, 'finish')])
    session._scene['objects'][0]['zone'] = 'zone_red'
    state = run(ctl)
    assert [s['skill'] for s in session.calls] == ['observe']
    assert len(session.completion_checks) == 2
    assert [s['skipped'] for s in state['skill_task']['history']
            if s['skill'] in ('pick', 'place')] == [True, True]


def test_pair_rechecks_completion_after_intervening_observe_and_never_places_with_empty_gripper(create):
    def compose(c, n):
        if n == 1:
            return [step(1, 'pick', 'apple'), step(2, 'observe', 'apple'),
                    step(3, 'place', 'apple', {'type': 'goal', 'index': 0}), step(4, 'finish')]
        return normal(c, n)

    ctl, session, client = create(compose)
    session._scene['objects'][0]['zone'] = 'zone_red'

    def disturb(action, result):
        if action['skill'] == 'observe':
            session._scene['objects'][0]['zone'] = None

    session.result_hook = disturb
    run(ctl)
    assert len(session.completion_checks) == 2
    assert len(client.contexts) == 2
    assert client.contexts[1]['failure']['error'] == 'completion_changed'
    assert [s['skill'] for s in session.calls] == ['observe', 'move']


def test_independent_pick_and_hold_is_not_skipped_when_task_destination_is_already_satisfied(create):
    def compose(c, n):
        return ([step(1, 'pick', 'apple'), step(2, 'observe', 'apple')] if n == 1 else
                [step(3, 'place', 'apple', {'type': 'goal', 'index': 0}), step(4, 'finish')])

    ctl, session, _ = create(compose)
    session._scene['objects'][0]['zone'] = 'zone_red'
    run(ctl)
    assert [s['skill'] for s in session.calls] == ['pick', 'observe', 'place']
    assert not session.completion_checks


def test_ask_boundary_prevents_speculative_pairing(create):
    ctl, session, _ = create(lambda c, n: [
        step(1, 'pick', 'apple'),
        step(2, 'ask', question='确认继续放下苹果吗？', affected_goals=[0]),
        step(3, 'place', 'apple', {'type': 'goal', 'index': 0}), step(4, 'finish')])
    session._scene['objects'][0]['zone'] = 'zone_red'
    ctl.submit('把苹果放红区')
    approve(ctl)
    assert ctl.wait_idle(3)
    assert ctl.snapshot()['last_result']['ok']
    assert [s['skill'] for s in session.calls] == ['pick', 'place']
    assert not session.completion_checks


def test_temporary_auto_free_place_is_never_skipped_as_if_it_were_final_goal(create):
    ctl, session, _ = create(lambda c, n: [
        step(1, 'pick', 'apple'), step(2, 'place', 'apple', {'type': 'auto_free'}),
        step(3, 'move', 'apple', {'type': 'goal', 'index': 0}), step(4, 'finish')])
    session._scene['objects'][0]['zone'] = 'zone_red'
    ctl.submit('暂放后继续把苹果放红区')
    approve(ctl)
    assert ctl.wait_idle(3)
    assert ctl.snapshot()['last_result']['ok']
    assert [s['skill'] for s in session.calls] == ['pick', 'place', 'move']
    assert not session.completion_checks


@pytest.mark.parametrize('additional', ['protected', 'restoration'])
def test_protected_or_restored_objects_require_fresh_full_final_verification(create, additional):
    ctl, session, _ = create(normal)
    session._scene['objects'][0]['zone'] = 'zone_red'
    session._scene['objects'][1]['zone'] = 'zone_green'
    orchestrator = SkillOrchestrator(ctl, Executor)
    executor = Executor(session)
    evidence = executor.execute(step(1, 'verify'), GOALS)
    proof = orchestrator._proof(evidence, GOALS)
    assert proof is not None
    memory = dict(goals=copy.deepcopy(GOALS), originals=session.scene['objects'],
                  protected_goals=[], obligations={}, restorations=[])
    if additional == 'protected':
        memory['protected_goals'] = [dict(label='banana', zone_name='zone_green')]
    else:
        memory['restorations'] = ['banana']
    session.calls.clear()
    result = orchestrator._verify_all(executor, memory, proof)
    assert result['ok'], result
    assert [s['skill'] for s in session.calls] == ['verify']
    assert [item['goal']['label'] for item in result['items']] == ['apple', 'banana']


def test_multi_goal_verification_grouped_by_destination_finishes_without_replanning(create):
    goals = [dict(label='apple', zone_name='zone_red'),
             dict(label='banana', zone_name='zone_green'),
             dict(label='pear', zone_name='zone_red')]
    ctl, session, client = create(lambda c, n: [step(1, 'verify')], goals)
    for goal in goals:
        next(o for o in session._scene['objects']
             if o['label'] == goal['label'])['zone'] = goal['zone_name']

    def destination_order(action, result):
        if action['skill'] == 'verify':
            # The real verifier shares each destination view, yielding A,C,B.
            result['items'] = [result['items'][i] for i in (0, 2, 1)]

    session.result_hook = destination_order
    state = run(ctl)
    assert len(client.contexts) == 1
    assert [s['skill'] for s in session.calls] == ['verify']
    assert [item['goal'] for item in state['last_result']['items']] == goals
