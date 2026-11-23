"""Intervention remains independent of the active model/robot worker."""
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from cube_task.assistant.server import DashboardController, DashboardServer
from test_assistant_scene_server import EditableSession
from test_assistant_server import Client, Settings


class InterventionSession(EditableSession):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interventions = []

    def snapshot(self):
        return dict(super().snapshot(), editor={
            'editable': True, 'intervention_available': not self.stopped,
            'intervention_pending': len(self.interventions)})

    def intervene_scene(self, request):
        if request.get('object_id') == self.held_object:
            raise ValueError('夹持中的物品不能移动')
        self.interventions.append(request)
        return {'queued': True, 'request_id': len(self.interventions)}


def create(client=None):
    ctl = DashboardController(session_factory=InterventionSession,
                              settings=Settings(), client=client or Client())
    assert ctl.wait_idle(3)
    return ctl


def move(object_id='other-object'):
    return {'action': 'move', 'object_id': object_id, 'xy': [.3, .1],
            'yaw': .3, 'scene_version': 1}


def test_intervention_does_not_wait_for_model_worker_or_change_task_intent():
    gate = threading.Event()
    client = Client(gate=gate)
    ctl = create(client)
    try:
        ctl.submit('抓红色方块')
        assert client.streaming.wait(2)
        before = ctl.snapshot()
        request = move()
        result = ctl.intervene_scene(request)
        request['xy'][0] = 99
        after = ctl.snapshot()
        assert result == {'queued': True, 'request_id': 1}
        assert after['busy'] and client.streaming.is_set() and not gate.is_set()
        assert after['scene'] == before['scene']
        assert after['messages'] == before['messages']
        assert after['batch'] == before['batch']
        assert after['metrics'] == before['metrics']
        assert after['simulation']['editor']['intervention_available']
        assert not after['simulation']['editor']['editable']
        assert ctl._session.interventions == [move()]
        event = after['events'][-1]
        assert event['kind'] == 'scene_intervention_queued'
        assert event['request_id'] == 1
    finally:
        gate.set()
        ctl.wait_idle(3)
        ctl.close()


def test_holding_one_object_only_prevents_moving_that_object():
    ctl = create()
    try:
        ctl._session.held_object = 'held-object'
        with pytest.raises(ValueError, match='夹持中的物品'):
            ctl.intervene_scene(move('held-object'))
        assert ctl.intervene_scene(move())['queued']
        assert ctl._session.interventions == [move()]
    finally:
        ctl._session.held_object = None
        ctl.close()


def test_environment_revision_reaches_session_through_controller():
    ctl = create()
    try:
        request = move()
        request.pop('scene_version')
        request['environment_revision'] = 4
        assert ctl.intervene_scene(request)['queued']
        assert ctl._session.interventions == [request]
    finally:
        ctl.close()


@pytest.mark.parametrize('payload', [None, [], 1, 'move'])
def test_intervention_requires_an_object(payload):
    ctl = create()
    try:
        with pytest.raises(ValueError, match='JSON 对象'):
            ctl.intervene_scene(payload)
        assert not ctl._session.interventions
    finally:
        ctl.close()


@pytest.mark.parametrize('payload', [
    {'action': 'clear'}, {'action': 'remove', 'object_id': 'other-object'},
    dict(move(), unexpected=True), {'action': 'add', 'catalog_id': 'banana'},
])
def test_intervention_cannot_add_or_delete_objects(payload):
    ctl = create()
    try:
        with pytest.raises(ValueError, match='只允许移动或旋转'):
            ctl.intervene_scene(payload)
        assert not ctl._session.interventions
    finally:
        ctl.close()


@pytest.mark.parametrize('condition,message', [
    ('reset', '先重置'), ('missing_session', '尚未准备好'),
    ('missing_scene', '尚未准备好'), ('closed', '先重置'),
])
def test_unavailable_scene_refuses_intervention(condition, message):
    ctl = create()
    session = ctl._session
    try:
        if condition == 'reset':
            ctl._needs_reset = True
        elif condition == 'missing_session':
            ctl._session = None
        elif condition == 'missing_scene':
            ctl._scene = {}
        else:
            ctl._closed = True
        with pytest.raises(ValueError, match=message):
            ctl.intervene_scene(move())
        assert not session.interventions
    finally:
        ctl._closed = False
        ctl._session = session
        ctl.close()


def test_http_intervention_returns_queue_id_and_keeps_existing_edit_lock():
    ctl = create()
    server = DashboardServer(('127.0.0.1', 0), ctl)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = 'http://127.0.0.1:%s' % server.server_port

    def post(path, payload):
        return urlopen(Request(base + path, data=json.dumps(payload).encode('utf-8'),
                               headers={'Content-Type': 'application/json'}), timeout=3)

    try:
        with ctl._lock:
            ctl._busy = True
        with post('/api/scene-intervene', move()) as response:
            assert response.status == 202
            assert json.loads(response.read()) == {'ok': True, 'queued': True, 'request_id': 1}
        with pytest.raises(HTTPError) as error:
            post('/api/scene-edit', {'action': 'clear', 'scene_version': 1})
        assert error.value.code == 409
        with pytest.raises(HTTPError) as error:
            post('/api/scene-intervene', {'action': 'clear'})
        assert error.value.code == 400
        assert ctl._session.interventions == [move()]
    finally:
        with ctl._lock:
            ctl._busy = False
        server.shutdown()
        server.server_close()
        ctl.close()
