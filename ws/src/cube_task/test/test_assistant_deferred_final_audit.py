"""Deferred final moves and a bounded final audit before a model stops."""
import copy
import socket
from types import SimpleNamespace

import pytest

from cube_task.assistant import skill_execution
from cube_task.assistant.server import DashboardController
from test_assistant_skill_orchestrator import Client, Executor, Session, Settings, approve
from test_assistant_skill_protocol import context, step


GOALS = [dict(label='apple', zone_name='zone_red'),
         dict(label='banana', zone_name='zone_green'),
         dict(label='pear', zone_name='zone_green')]


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('Deferred/audit tests must never use a network')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


class AuditExecutor(Executor):
    def defer_unobserved(self, action, goals):
        if action['skill'] != 'move' or self.session.holding:
            return None
        destination = action.get('destination') or {}
        if destination.get('type') != 'goal':
            return None
        if goals[destination['index']]['label'] != action['label']:
            return None
        found = [o for o in self.session._scene['objects']
                 if o['label'] == action['label'] and o.get('valid')]
        if len(found) != 1:
            return dict(ok=False, deferred=True, manipulation_started=False,
                        changed=False, holding=False, label=action['label'],
                        error='goal_deferred', reason='unreliable_measurement', message='暂缓未可靠测量的目标')
        return None

    def execute(self, action, goals, *args, **kwargs):
        s = self.session
        if action['skill'] == 'verify':
            selected = [goals[action['goal_index']]] if 'goal_index' in action else goals
            s.calls.append(copy.deepcopy(action))
            s.checked.append(copy.deepcopy(selected))
            items = []
            for goal in selected:
                found = [o for o in s._scene['objects'] if o['label'] == goal['label'] and o.get('valid')]
                ok = len(found) == 1 and found[0].get('zone') == goal.get('zone_name')
                items.append(dict(goal=copy.deepcopy(goal), label=goal['label'], ok=ok,
                                  camera_verified=ok, truth={'success': ok},
                                  error='' if ok else 'camera_unconfirmed',
                                  message='已核验' if ok else '尚未可靠确认',
                                  verified_revision=s._environment_revision))
            ok = all(item['ok'] for item in items)
            result = dict(ok=ok, camera_verified=ok, truth={'success': ok}, items=items,
                          verified_revision=s._environment_revision, holding=False,
                          error='' if ok else 'verification_incomplete', message='完整目标核验')
            if s.verify_hook:
                s.verify_hook(action, result)
            return result
        return super().execute(action, goals, *args, **kwargs)


@pytest.fixture
def create(context, monkeypatch):
    monkeypatch.setattr(skill_execution, 'SkillExecutor', AuditExecutor)
    controllers = []

    def factory(compose, goals=GOALS):
        session = Session(context['scene'])
        session._scene['zones'] = [dict(name='zone_red', center=[.25, -.2, 0.], size=[.25, .25], yaw=0.),
                                   dict(name='zone_green', center=[.4, .2, 0.], size=[.35, .3], yaw=0.)]
        session._scene['objects'][0].update(center_base=[.25, -.2, .06], top_z=.06)
        session._scene['objects'][1].update(center_base=[.4, .2, .04], top_z=.04)
        session._scene['objects'][2]['top_z'] = .04
        session.checked, session.verify_hook = [], None
        client = Client(lambda _context, _number: [])

        def plan(instruction, context, on_delta, cancel_event=None):
            client.contexts.append(copy.deepcopy(context))
            proposal = compose(context, len(client.contexts))
            if isinstance(proposal, list):
                proposal = dict(action='plan', steps=proposal)
            return dict(scene_version=context['scene']['scene_version'], message='离线技能方案',
                        goals=copy.deepcopy(context['goals'] or goals), **proposal)

        client.plan_skills = plan
        ctl = DashboardController(session_factory=lambda **kwargs: session, client=client, settings=Settings())
        assert ctl.wait_idle(2)
        controllers.append(ctl)
        return ctl, session, client

    yield factory
    for ctl in controllers:
        ctl.close()


def stop(_context, _number):
    return dict(action='stop', steps=[])


def run(ctl):
    ctl.submit('把所有物品放到各自默认区域')
    assert ctl.wait_idle(3)
    return ctl.snapshot()


def test_model_stop_audits_every_goal_and_returns_summary_for_one_decision(create):
    ctl, session, client = create(stop)
    session._scene['objects'][2]['valid'] = False
    session._scene['objects'][1]['zone'] = 'zone_green'
    state = run(ctl)

    assert len(session.checked) == 1 and session.checked[0] == GOALS
    assert len(client.contexts) == 2
    review = client.contexts[1]['final_audit']
    assert review['phase'] == 'reviewed'
    assert [item['label'] for item in review['items']] == [g['label'] for g in GOALS]
    assert all(set(item) == {'goal_index', 'label', 'ok', 'camera_verified', 'error', 'message'}
               for item in review['items'])
    assert not state['last_result']['ok'] and state['metrics']['successes'] == 0
    assert state['skill_task']['final_audit'] == review
    assert state['skill_task']['goal_worklist'][-1]['status'] == 'deferred'


@pytest.mark.parametrize('dynamic', [False, True])
def test_unreliable_move_defers_and_later_independent_goal_executes(create, monkeypatch, dynamic):
    def compose(_context, number):
        return ([step('pear', 'move', 'pear', dict(type='goal', index=2)),
                 step('apple', 'move', 'apple', dict(type='goal', index=0))]
                if number == 1 else stop(None, None))
    ctl, session, client = create(compose)
    session._scene['objects'][2]['valid'] = False
    if dynamic:
        original = AuditExecutor.execute
        monkeypatch.setattr(AuditExecutor, 'defer_unobserved', lambda *args: None)

        def execute(self, action, *args, **kwargs):
            if action.get('label') == 'pear' and action['skill'] == 'move':
                return dict(ok=False, deferred=True, manipulation_started=False, changed=False,
                            holding=False, label='pear', error='goal_deferred', message='动态测量无法确认')
            return original(self, action, *args, **kwargs)

        monkeypatch.setattr(AuditExecutor, 'execute', execute)
    state = run(ctl)

    assert [a.get('label') for a in session.calls if a['skill'] == 'move'] == ['apple']
    assert session.checked == [GOALS]
    assert len(client.contexts) == 2
    history = state['skill_task']['history']
    assert history[0]['deferred'] is True and history[0]['ok'] is False
    assert ctl._skill_memory['last_action_failure'] == {}
    assert state['skill_task']['steps'][0]['status'] == 'deferred'
    assert state['skill_task']['goals'] == GOALS and not state['last_result']['ok']


def test_failed_finish_is_the_audit_and_is_not_scanned_again_for_model_stop(create):
    ctl, session, client = create(lambda c, n: [step('done', 'finish')] if n == 1 else stop(c, n))
    state = run(ctl)
    assert session.checked == [GOALS]
    assert len(client.contexts) == 2 and 'final_audit' in client.contexts[1]
    assert not state['last_result']['ok']


def test_missing_audit_items_remain_explicitly_unconfirmed(create):
    ctl, session, client = create(stop)

    def incomplete(action, result):
        result['items'] = result['items'][:1]
        result['truth'] = {'success': True, 'private_world_pose': 'never-forward'}
        result['items'][0]['truth']['private_world_pose'] = 'never-forward'

    session.verify_hook = incomplete
    state = run(ctl)
    review = client.contexts[1]['final_audit']
    assert len(review['items']) == 3
    assert all(item['ok'] is False for item in review['items'][1:])
    assert 'never-forward' not in str(review) and not state['last_result']['ok']


def test_audit_success_can_finish_without_another_model_call(create):
    ctl, session, client = create(stop)
    for obj, goal in zip(session._scene['objects'], GOALS):
        obj['zone'] = goal['zone_name']
    state = run(ctl)
    assert state['status'] == 'complete' and state['last_result']['ok']
    assert len(client.contexts) == 1 and session.checked == [GOALS]


def test_real_fault_or_held_state_prevents_end_audit(create):
    ctl, session, client = create(stop)
    session.holding = 'apple'
    session.runner._held_recovery_context = dict(det=SimpleNamespace(label='apple'))
    session.runner.has_held_recovery = True
    state = run(ctl)
    assert not session.checked and not state['last_result']['ok']
    assert len(client.contexts) == 1


def test_declined_risk_plan_does_not_run_an_audit(create):
    ctl, session, client = create(lambda c, n: [step('temporary', 'move', 'apple', dict(type='auto_free'))])
    ctl.submit('只在明确同意后暂放苹果')
    approve(ctl, False)
    assert ctl.wait_idle(3)
    assert session.calls == [] and session.checked == [] and len(client.contexts) == 1


def test_newly_observed_pending_goal_returns_to_planner_before_final_audit(create):
    checked_when_replanning = []

    def compose(_context, number):
        if number == 1:
            return [step('look', 'observe', 'pear')]
        if number == 2:
            checked_when_replanning.append(copy.deepcopy(session.checked))
            return [step('move', 'move', 'pear', dict(type='goal', index=2))]
        return stop(None, None)

    ctl, session, client = create(compose)
    session._scene['objects'][2]['valid'] = False
    session._scene['objects'][0]['zone'] = 'zone_red'
    session._scene['objects'][1]['zone'] = 'zone_green'

    def reveal(action, result):
        if action['skill'] == 'observe':
            session._scene['objects'][2]['valid'] = True
            result['changed'] = True

    session.hook = reveal
    state = run(ctl)
    assert checked_when_replanning == [[]]
    assert client.contexts[1]['goal_worklist'][2]['status'] == 'observed_pending'
    assert [a['label'] for a in session.calls if a['skill'] == 'move'] == ['pear']
    assert state['last_result']['ok'] and session.checked == [GOALS]


@pytest.mark.parametrize('changed, audits, requests', [(False, 1, 2), (True, 2, 4)])
def test_changed_measurement_invalidates_review_but_unchanged_view_does_not(
        create, changed, audits, requests):
    def compose(_context, number):
        return [step('look', 'observe', 'pear')] if number == 2 else stop(None, None)

    ctl, session, client = create(compose)
    session._scene['objects'][2]['valid'] = False

    def capture(action, result):
        if action['skill'] == 'observe':
            result['changed'] = changed
            if changed:
                session._scene['objects'][2]['valid'] = True

    session.hook = capture
    state = run(ctl)
    assert len(session.checked) == audits and len(client.contexts) == requests
    assert not state['last_result']['ok']
    assert session._environment_revision == 0
    if changed:
        assert 'final_audit' not in client.contexts[2]


def test_last_model_request_is_reserved_for_audit_decision(create, monkeypatch):
    from cube_task.assistant import skill_orchestrator
    monkeypatch.setattr(skill_orchestrator, 'plan_limit', lambda *args: 3)
    ctl, session, client = create(lambda c, n: [step(str(n), 'observe', 'pear')]
                                   if n < 3 else stop(c, n))
    state = run(ctl)
    assert len(client.contexts) == 3
    assert 'final_audit' not in client.contexts[1]
    assert 'final_audit' in client.contexts[2]
    assert session.checked == [GOALS] and not state['last_result']['ok']


def test_missing_camera_flag_is_not_inferred_from_item_ok(create):
    ctl, session, client = create(stop)
    session._scene['objects'][0]['zone'] = 'zone_red'

    def no_camera_flag(action, result):
        result['items'][0].pop('camera_verified')

    session.verify_hook = no_camera_flag
    run(ctl)
    item = client.contexts[1]['final_audit']['items'][0]
    assert item['ok'] is False and item['camera_verified'] is False


def test_single_action_proof_does_not_audit_while_another_observed_goal_is_pending(create, monkeypatch):
    original = AuditExecutor.execute
    checked_before_next_work = []

    def verified_move(self, action, goals, *args, **kwargs):
        result = original(self, action, goals, *args, **kwargs)
        if action['skill'] == 'move':
            goal = goals[action['destination']['index']]
            result.update(label=action['label'], controller_ok=True,
                          resolved_destination=dict(type='named_region', name=goal['zone_name']))
        return result

    monkeypatch.setattr(AuditExecutor, 'execute', verified_move)

    def compose(_context, number):
        if number == 1:
            return [step('apple', 'move', 'apple', dict(type='goal', index=0))]
        checked_before_next_work.append(copy.deepcopy(session.checked))
        return stop(None, None)

    ctl, session, client = create(compose)
    state = run(ctl)
    assert checked_before_next_work[0] == []
    assert client.contexts[1]['goal_worklist'][2]['status'] == 'observed_pending'
    assert session.checked == [GOALS]
    assert not state['last_result']['ok']


def test_single_goal_cached_action_has_consistent_camera_audit_summary(create, monkeypatch):
    original = AuditExecutor.execute

    def verified_move(self, action, goals, *args, **kwargs):
        result = original(self, action, goals, *args, **kwargs)
        if action['skill'] == 'move':
            result.update(label=action['label'], controller_ok=True,
                          resolved_destination=dict(type='named_region', name=goals[0]['zone_name']))
        return result

    monkeypatch.setattr(AuditExecutor, 'execute', verified_move)
    ctl, session, client = create(lambda c, n: [
        step('move', 'move', 'apple', dict(type='goal', index=0)), step('done', 'finish')], GOALS[:1])
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert state['last_result']['ok'] and not session.checked
    item = state['skill_task']['final_audit']['items'][0]
    assert item['ok'] is True and item['camera_verified'] is True
    assert [a['skill'] for a in session.calls] == ['move']


def test_actual_move_invalidates_old_audit_without_environment_revision_change(create):
    def compose(_context, number):
        return [step('move', 'move', 'apple', dict(type='goal', index=0))] if number == 2 else stop(None, None)

    ctl, session, client = create(compose)
    state = run(ctl)
    assert session._environment_revision == 0
    assert len(session.checked) == 2 and len(client.contexts) == 4
    assert 'final_audit' not in client.contexts[2]
    assert [a['label'] for a in session.calls if a['skill'] == 'move'] == ['apple']
    assert not state['last_result']['ok']


@pytest.mark.parametrize('unsafe', ['unhealthy', 'queued_intervention', 'user_stop'])
def test_unsafe_end_conditions_never_start_audit_camera_commands(create, unsafe):
    ctl, session, client = create(stop)
    if unsafe == 'unhealthy':
        session.arm = SimpleNamespace(health=lambda: SimpleNamespace(ok=False))
    elif unsafe == 'queued_intervention':
        session._interventions = [('move', {})]
        # A pending edit invalidates the plan before its stop is handled.
        # The existing observation refresh is read-only; it must not verify.
    else:
        original = client.plan_skills

        def cancelled(*args, **kwargs):
            result = original(*args, **kwargs)
            ctl.stop()
            return result

        client.plan_skills = cancelled
    state = run(ctl)
    assert session.checked == [] and not state['last_result']['ok']
    assert not any(a['skill'] in ('pick', 'move', 'place') for a in session.calls)


def test_intervention_after_audit_publication_invalidates_its_model_summary(create):
    ctl, session, client = create(stop)
    original = ctl._event
    edited = []

    def intervene(kind, payload=None):
        original(kind, payload)
        if kind == 'final_audit' and not edited:
            edited.append(True)
            session._environment_revision += 1

    ctl._event = intervene
    state = run(ctl)
    assert len(session.checked) == 2 and len(client.contexts) == 3
    assert 'final_audit' not in client.contexts[1]
    assert 'final_audit' in client.contexts[2]
    assert not state['last_result']['ok']


def test_resumed_deferred_move_preserves_prior_action_diagnostics(create):
    ctl, session, client = create(lambda c, n: [
        step('pear', 'move', 'pear', dict(type='goal', index=2))] if n == 1 else stop(c, n))
    session._scene['objects'][2]['valid'] = False
    previous = dict(ok=False, stage='inspect_place', error='placement_changed',
                    diagnostics='retained physical-action failure', holding=False)
    ctl._skill_memory = dict(instruction='把所有物品放到各自默认区域', goals=copy.deepcopy(GOALS),
        protected_goals=[], originals=session.scene['objects'], obligations={}, restorations=[],
        last_action_failure=copy.deepcopy(previous))
    ctl._skill_task['status'] = 'failed'
    ctl._last_result = copy.deepcopy(previous)
    ctl.resume_recovery()
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert ctl._skill_memory['last_action_failure'] == previous
    assert client.contexts[1]['last_action_failure'] == previous
    assert state['skill_task']['goals'] == GOALS and not state['last_result']['ok']
    assert not any(a['skill'] in ('pick', 'place', 'move') for a in session.calls)


@pytest.mark.parametrize('error', ['original_not_measured', 'task_scope_unconfirmed'])
def test_inconclusive_audit_without_revision_gets_one_model_decision(create, monkeypatch, error):
    from cube_task.assistant.skill_orchestrator import SkillOrchestrator
    attempts = []

    def unconfirmed(*args, **kwargs):
        attempts.append(True)
        return dict(ok=False, error=error, message='缺少完整审核依据')

    monkeypatch.setattr(SkillOrchestrator, '_verify_all', unconfirmed)
    ctl, session, client = create(stop)
    state = run(ctl)
    assert len(attempts) == 1 and len(client.contexts) == 2
    assert len(client.contexts[1]['final_audit']['items']) == len(GOALS)
    assert all(item['ok'] is False and item['camera_verified'] is False
               for item in client.contexts[1]['final_audit']['items'])
    assert not state['last_result']['ok'] and not session.calls


def test_camera_unavailable_audit_without_proof_is_not_repeated(create):
    ctl, session, client = create(stop)

    def missing_evidence(action, result):
        result.pop('verified_revision')
        result['camera_verified'] = False
        result['items'] = []

    session.verify_hook = missing_evidence
    state = run(ctl)
    assert session.checked == [GOALS] and len(client.contexts) == 2
    assert all(item['ok'] is False for item in client.contexts[1]['final_audit']['items'])
    assert state['status'] != 'complete' and not state['last_result']['ok']
