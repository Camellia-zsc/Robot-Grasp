"""Offline regressions for final evidence freshness and the real default client."""
import copy
import json
import socket
import threading

import pytest

from cube_task.assistant import llm, skill_execution
from cube_task.assistant.server import DashboardController
from cube_task.assistant.skill_orchestrator import SkillOrchestrator
from test_assistant_llm import FakeStream, completed, install_stream, settings, sse
from test_assistant_skill_orchestrator import Executor, GOALS, Session, create
from test_assistant_skill_protocol import context, step


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('feedback boundary tests must never access the network')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def _enable_locks(session):
    session._operation_lock = threading.RLock()
    session._intervention_lock = threading.Lock()
    session._interventions = []


def _final_plan(terminal):
    return lambda _context, _number: [
        step(1, 'move', 'apple', {'type': 'goal', 'index': 0}), step(2, terminal)]


def _assert_never_succeeded(state):
    assert not state['last_result']['ok']
    assert state['metrics']['successes'] == 0
    assert state['status'] != 'complete'
    assert not any(e.get('result', {}).get('ok')
                   for e in state['events'] if e['kind'] == 'result')


@pytest.mark.parametrize('detail', [
    '叠放支撑物必须是当前唯一有效的三维相机测量',
    '底座未完整观测；' * 250,
], ids=['support_measurement_reason', 'bounded_long_reason'])
def test_placement_refusal_detail_survives_observation_and_model_context(create, monkeypatch, detail):
    from cube_task.assistant.skill_protocol import model_skill_context

    original = Executor.execute
    refused = []

    def execute(self, action, *args, **kwargs):
        if action['skill'] == 'move' and not refused:
            refused.append(True)
            return dict(ok=False, stage='validation', error='invalid_destination',
                        message='放置目标无效', detail=detail, holding=False,
                        truth={'private': 'private-feedback-sentinel'})
        return original(self, action, *args, **kwargs)

    monkeypatch.setattr(Executor, 'execute', execute)

    def compose(_context, number):
        if number == 1:
            return [step(1, 'move', 'apple', {'type': 'goal', 'index': 0})]
        if number == 2:
            return [step(2, 'observe', 'apple')]
        return [step(3, 'move', 'apple', {'type': 'goal', 'index': 0}), step(4, 'finish')]

    ctl, _session, client = create(compose)
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    assert ctl.snapshot()['last_result']['ok']
    assert len(client.contexts) == 3
    first_retry = model_skill_context(client.contexts[1])
    after_observe = model_skill_context(client.contexts[2])
    expected = detail[:1000]
    assert first_retry['failure']['detail'] == expected
    assert after_observe['failure']['error'] == 'goal_not_finished'
    assert after_observe['last_action_failure']['detail'] == expected
    assert after_observe['history'][0]['detail'] == expected
    assert ctl.snapshot()['skill_task']['history'][0]['detail'] == expected
    assert 'private-feedback-sentinel' not in str(first_retry) + str(after_observe)


def test_intervention_after_step_publication_cancels_old_plan_before_dispatch(create):
    ctl, session, client = create(_final_plan('finish'))
    event = ctl._event
    changed = []

    def publish(kind, payload=None):
        event(kind, payload)
        if kind == 'skill_step' and not changed:
            session._environment_revision += 1
            changed.append(True)

    ctl._event = publish
    ctl.submit('把苹果放到红区')
    assert ctl.wait_idle(3)
    assert changed
    assert len(client.contexts) >= 2
    assert client.contexts[1]['failure']['error'] == 'scene_changed'
    assert not client.contexts[1]['history'], 'expired plan must not dispatch its move'


@pytest.mark.parametrize('terminal', ['verify', 'finish'])
def test_final_verification_from_an_old_environment_never_counts_success(create, terminal):
    ctl, session, client = create(_final_plan(terminal))
    _enable_locks(session)

    def intervention_after_measurement(action, result):
        if action['skill'] == 'verify':
            assert result['verified_revision'] == session._environment_revision
            session._environment_revision += 1
    session.hook = intervention_after_measurement
    ctl.submit('把苹果放到红区')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    _assert_never_succeeded(state)
    assert len(client.contexts) == state['skill_task']['max_plans']
    assert any(c.get('failure', {}).get('error') == 'scene_changed' for c in client.contexts[1:])
    assert session._environment_revision > 0


@pytest.mark.parametrize('terminal', ['verify', 'finish'])
@pytest.mark.parametrize('pending_source', ['queue', 'snapshot'])
def test_queued_unapplied_intervention_blocks_success(create, monkeypatch, terminal, pending_source):
    ctl, session, client = create(_final_plan(terminal))
    _enable_locks(session)
    queued = {'pending': False}
    original_snapshot = session.snapshot

    def snapshot():
        result = original_snapshot()
        result['editor']['intervention_pending'] = int(queued['pending'])
        return result
    monkeypatch.setattr(session, 'snapshot', snapshot)

    def enqueue_after_measurement(action, result):
        if action['skill'] == 'verify':
            assert result['verified_revision'] == session._environment_revision
            if pending_source == 'queue':
                session._interventions[:] = [('pending-edit', {'action': 'move'})]
            else:
                queued['pending'] = True
    session.hook = enqueue_after_measurement
    ctl.submit('把苹果放到红区')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    _assert_never_succeeded(state)
    # A queued request has not changed the environment revision yet. Revision
    # equality alone therefore cannot justify publishing task success.
    assert session._environment_revision == 0
    assert session._interventions or queued['pending']
    assert any(c.get('failure', {}).get('error') == 'scene_changed' for c in client.contexts[1:])


@pytest.mark.parametrize('pending', [False, True])
def test_success_publication_rechecks_evidence_after_verification(create, pending):
    ctl, session, _client = create(_final_plan('verify'))
    _enable_locks(session)
    memory = dict(goals=copy.deepcopy(GOALS), protected_goals=[],
                  originals=session.scene['objects'], obligations={}, restorations=[])
    verified = dict(ok=True, camera_verified=True, truth={'success': True}, verified_revision=0)
    if pending:
        session._interventions.append(('pending-edit', {'action': 'move'}))
    else:
        session._environment_revision = 1
    result = SkillOrchestrator(ctl, Executor)._finish(True, '全部原始目标已通过核验', memory, verified)
    assert not result['ok'] and ctl.snapshot()['metrics']['successes'] == 0
    assert ctl.snapshot()['status'] != 'complete'


@pytest.mark.parametrize('style', ['responses', 'chat_completions'])
@pytest.mark.parametrize('action', ['answer', 'plan'])
def test_default_real_model_client_streams_new_skill_protocol(monkeypatch, context, style, action):
    session = Session(context['scene'])
    _enable_locks(session)
    monkeypatch.setattr(skill_execution, 'SkillExecutor', Executor)

    def forbidden_legacy_method(*args, **kwargs):
        raise AssertionError('the real default client must use plan_skills')
    monkeypatch.setattr(llm.ModelClient, 'decide', forbidden_legacy_method)
    monkeypatch.setattr(llm.ModelClient, 'recover', forbidden_legacy_method)
    decision = dict(scene_version=context['scene']['scene_version'], action=action,
                    message='按当前观测处理这条指令。',
                    goals=copy.deepcopy(GOALS) if action == 'plan' else [],
                    steps=_final_plan('verify')(context, 1) if action == 'plan' else [])
    text = json.dumps(decision, ensure_ascii=False)
    if style == 'responses':
        raw = completed(text)
    else:
        raw = (sse({'choices': [{'index': 0, 'delta': {'content': text}}]})
               + sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
               + b'data: [DONE]\n\n')
    stream = FakeStream([raw])
    requests = install_stream(monkeypatch, stream)
    # No injected client: DashboardController must construct the production
    # ModelClient, then choose its new streaming planning method itself.
    ctl = DashboardController(session_factory=lambda **kwargs: session,
                              settings=settings(style), realtime=False)
    try:
        assert ctl.wait_idle(3)
        assert type(ctl._client) is llm.ModelClient
        ctl.submit('把苹果放到红区' if action == 'plan' else '桌上有苹果吗？')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert len(requests) == 1 and stream.closed
        body = json.loads(requests[0][0].data)
        visible = json.loads(body['input'] if style == 'responses' else body['messages'][1]['content'])
        assert 'context' in visible and 'scene' not in visible and 'recovery' not in visible
        assert visible['context']['scene']['scene_version'] == context['scene']['scene_version']
        assert visible['context']['perception'] == 'structured_camera_observations'
        assert visible['context']['images_provided'] is False
        assert state['messages'][-1]['kind'] == 'skill_plan'
        assert state['messages'][-1]['status'] == 'complete'
        assert state['messages'][-1]['decision'] == decision
        if action == 'plan':
            assert state['status'] == 'complete' and state['last_result']['ok']
            assert [s['skill'] for s in session.calls] == ['move', 'verify']
        else:
            assert state['status'] == 'ready' and session.calls == []
    finally:
        ctl.close()
