"""Recovery HTTP boundaries reject ambiguous or stale requests without moving."""
import json
import threading
from http.client import HTTPConnection
from types import SimpleNamespace

import pytest

from cube_task.assistant.server import DashboardServer
from test_assistant_batch import RecoveryClient, create
from test_assistant_recovery_approval import ConfirmedSession, pending


@pytest.fixture
def recovery_http():
    ctl = create(ConfirmedSession, RecoveryClient())
    server = DashboardServer(('127.0.0.1', 0), ctl)
    worker = threading.Thread(target=server.serve_forever,
                              kwargs={'poll_interval': .01}, daemon=True)
    worker.start()

    def post(path, payload):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode('utf-8')
        connection = HTTPConnection('127.0.0.1', server.server_port, timeout=3)
        try:
            connection.request('POST', path, body=body,
                               headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        yield ctl, post
    finally:
        ctl.stop()
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
        ctl.close()


@pytest.fixture
def pending_http(recovery_http):
    ctl, post = recovery_http
    ctl.submit('依次抓苹果和梨')
    request = pending(ctl)
    answer = {key: request[key] for key in ('request_id', 'scene_version')}
    yield ctl, post, dict(answer, approved=True)


@pytest.mark.parametrize('change', [
    {'approved': 'true'},
    {'approved': 1},
    {'scene_version': True},
    {'scene_version': '2'},
    {'request_id': None},
    {'unexpected': True},
])
def test_http_rejects_malformed_approval_fields_without_consuming_the_gate(pending_http, change):
    ctl, post, answer = pending_http
    status, result = post('/api/recovery-approval', dict(answer, **change))
    assert status == 400 and result['error']
    assert ctl.snapshot()['recovery_approval']['status'] == 'pending'
    assert not ctl._session.repairs


def test_http_rejects_missing_approval_field(pending_http):
    ctl, post, answer = pending_http
    answer.pop('approved')
    status, result = post('/api/recovery-approval', answer)
    assert status == 400 and result['error']
    assert ctl.snapshot()['recovery_approval']['status'] == 'pending'
    assert not ctl._session.repairs


def test_http_rejects_duplicate_json_keys_in_an_otherwise_valid_approval(pending_http):
    ctl, post, answer = pending_http
    raw = (json.dumps(answer)[:-1] + ',"approved":false}').encode('utf-8')
    status, result = post('/api/recovery-approval', raw)
    assert status == 400 and result['error']
    assert ctl.snapshot()['recovery_approval']['status'] == 'pending'
    assert not ctl._session.repairs


@pytest.mark.parametrize('path', ['/api/recovery-approval', '/api/resume-recovery'])
@pytest.mark.parametrize('raw', [b'{', b'[]', b'null'])
def test_recovery_http_requires_a_valid_json_object(pending_http, path, raw):
    ctl, post, _answer = pending_http
    status, result = post(path, raw)
    assert status == 400 and result['error']
    assert ctl.snapshot()['recovery_approval']['status'] == 'pending'
    assert not ctl._session.repairs


def test_http_approval_cannot_reuse_an_answer_after_scene_change(pending_http):
    ctl, post, answer = pending_http
    ctl._session.scene['scene_version'] += 1
    status, result = post('/api/recovery-approval', answer)
    assert status == 400 and result['error']
    assert ctl.wait_idle(3)
    assert ctl.snapshot()['recovery_approval']['status'] == 'stale'
    assert not ctl._session.repairs
    assert len(ctl._session.executions) == 1


def test_http_approval_after_stop_cannot_restart_motion(pending_http):
    ctl, post, answer = pending_http
    status, result = post('/api/stop', {})
    assert status == 202 and result['ok']
    status, result = post('/api/recovery-approval', answer)
    assert status == 400 and result['error']
    assert ctl.wait_idle(3)
    assert ctl.snapshot()['status'] == 'stopped'
    assert not ctl._session.repairs
    assert len(ctl._session.executions) == 1


def test_http_resume_rejects_new_goal_even_with_a_recoverable_held_object(pending_http):
    ctl, post, answer = pending_http
    session = ctl._session
    session.runner = SimpleNamespace(has_held_recovery=True)
    session.cancel_event = threading.Event()
    session._recovery_goal = {'label': 'apple', 'zone_name': 'zone_red'}
    snapshot = session.snapshot
    session.snapshot = lambda: dict(snapshot(), held_object='held-apple')
    status, result = post('/api/recovery-approval', dict(answer, approved=False))
    assert status == 202 and result['ok']
    assert ctl.wait_idle(3)
    # This boundary concerns changing a healthy held task, not overriding the
    # controller-fault stop supplied by RecoveringBatch's default failure.
    ctl._last_result['fault'] = False
    assert ctl.snapshot()['can_resume_recovery']

    status, result = post('/api/resume-recovery', {'instruction': '改抓梨放到绿区'})
    assert status == 400 and result['error']
    assert ctl.snapshot()['can_resume_recovery']
    assert not session.repairs
    assert len(session.executions) == 1


def test_http_resume_rejects_an_empty_request_without_held_recovery(recovery_http):
    ctl, post = recovery_http
    status, result = post('/api/resume-recovery', {})
    assert status == 400 and result['error']
    assert not ctl.snapshot()['can_resume_recovery']
    assert not ctl._session.executions
    assert not ctl._session.repairs
