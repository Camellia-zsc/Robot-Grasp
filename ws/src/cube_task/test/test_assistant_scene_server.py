"""Scene editing shares the robot worker and never requests a model response."""
import copy
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from cube_task.assistant.server import BusyError, DashboardController, DashboardServer
from test_assistant_server import Client, SCENE, Session, Settings


class EditableSession(Session):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scene = copy.deepcopy(SCENE)
        self.held_object = None
        self.editable = True
        self.edit_requests = []
        self.edit_entered = threading.Event()
        self.edit_gate = None

    def snapshot(self):
        return dict(super().snapshot(), held_object=self.held_object,
                    status='stopped' if self.stopped else 'ready',
                    editor={'editable': self.editable, 'catalog': ['red_cube']})

    def edit_scene(self, request):
        self.edit_requests.append(copy.deepcopy(request))
        self.edit_entered.set()
        if self.edit_gate:
            self.edit_gate.wait(3)
        if self.stopped:
            raise RuntimeError('布局调整已停止')
        if request.get('scene_version') != self.scene['scene_version']:
            raise ValueError('场景版本已过期，请刷新后重试')
        if request.get('action') != 'clear':
            raise ValueError('未知布局操作')
        self.scene = dict(self.scene, scene_version=self.scene['scene_version'] + 1,
                          objects=[])
        return copy.deepcopy(self.scene)


def create(client=None):
    ctl = DashboardController(session_factory=EditableSession, settings=Settings(),
                              client=client or Client())
    assert ctl.wait_idle(3)
    return ctl


def test_edit_clears_result_recovery_and_error_but_preserves_chat_and_model_counts():
    ctl = create()
    try:
        ctl._messages = [{'id': 10, 'role': 'user', 'content': '上次指令',
                          'status': 'complete'}]
        ctl._last_result = {'ok': False}
        ctl._recovery.update(attempt=2, status='failed', history=[{'ok': False}])
        ctl._error = '上次失败'
        before = ctl.snapshot()
        ctl.edit_scene({'action': 'clear', 'scene_version': 1})
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'ready'
        assert '下次指令将重新扫描' in state['stage']
        assert state['scene']['scene_version'] == 2
        assert state['scene']['objects'] == []
        assert state['last_result'] is None and state['error'] == ''
        assert state['recovery'] == {
            'attempt': 0, 'max_attempts': 3, 'status': 'idle',
            'reason': '', 'message': '', 'history': []}
        assert state['messages'] == before['messages']
        assert state['metrics'] == before['metrics']
        assert not ctl._client.streaming.is_set()
        assert state['simulation']['editor'] == {
            'editable': True, 'catalog': ['red_cube']}
        assert state['events'][-1]['kind'] == 'scene_edited'
    finally:
        ctl.close()


@pytest.mark.parametrize('payload', [None, [], 'clear', 3])
def test_edit_requires_an_object(payload):
    ctl = create()
    try:
        with pytest.raises(ValueError, match='JSON 对象'):
            ctl.edit_scene(payload)
        assert ctl._session.edit_requests == []
    finally:
        ctl.close()


def test_busy_model_worker_rejects_edit_and_disables_editor():
    gate = threading.Event()
    client = Client(gate=gate)
    ctl = create(client)
    try:
        ctl.submit('抓红色方块')
        assert client.streaming.wait(2)
        assert not ctl.snapshot()['simulation']['editor']['editable']
        with pytest.raises(BusyError):
            ctl.edit_scene({'action': 'clear', 'scene_version': 1})
        assert ctl._session.edit_requests == []
    finally:
        gate.set()
        ctl.wait_idle(3)
        ctl.close()


def test_edit_worker_excludes_commands_and_other_edits_and_copies_request():
    gate = threading.Event()
    ctl = create()
    ctl._session.edit_gate = gate
    request = {'action': 'clear', 'scene_version': 1}
    try:
        ctl.edit_scene(request)
        assert ctl._session.edit_entered.wait(2)
        request['action'] = 'invalid'
        assert not ctl.snapshot()['simulation']['editor']['editable']
        with pytest.raises(BusyError):
            ctl.submit('抓红色方块')
        with pytest.raises(BusyError):
            ctl.edit_scene({'action': 'clear', 'scene_version': 1})
        gate.set()
        assert ctl.wait_idle(3)
        assert ctl.snapshot()['scene']['scene_version'] == 2
        assert ctl._session.edit_requests == [{'action': 'clear', 'scene_version': 1}]
        assert ctl.snapshot()['metrics']['model_requests'] == 0
    finally:
        gate.set()
        ctl.wait_idle(3)
        ctl.close()


@pytest.mark.parametrize('condition,message', [
    ('held', '仍夹持'), ('needs_reset', '先重置'), ('stopped', '先重置'),
    ('missing_session', '尚未准备好'), ('missing_scene', '尚未准备好')])
def test_unavailable_scene_refuses_edit_and_disables_editor(condition, message):
    ctl = create()
    session = ctl._session
    try:
        if condition == 'held':
            session.held_object = 'red_cube_1'
        elif condition == 'needs_reset':
            ctl._needs_reset = True
        elif condition == 'stopped':
            session.stop()
        elif condition == 'missing_session':
            ctl._session = None
        else:
            ctl._scene = {}
        assert not ctl.snapshot()['simulation']['editor']['editable']
        with pytest.raises(ValueError, match=message):
            ctl.edit_scene({'action': 'clear', 'scene_version': 1})
        assert session.edit_requests == []
    finally:
        ctl._session = session
        ctl.close()


def test_session_editor_can_disable_editing_in_snapshot():
    ctl = create()
    try:
        ctl._session.editable = False
        assert not ctl.snapshot()['simulation']['editor']['editable']
        assert ctl._session.snapshot()['editor']['catalog'] == ['red_cube']
    finally:
        ctl.close()


def test_reset_hides_old_layout_and_editor_until_new_session_is_ready():
    ctl = create()
    constructing = threading.Event()
    release = threading.Event()

    def factory(**kwargs):
        constructing.set()
        release.wait(3)
        return EditableSession(**kwargs)

    try:
        ctl.edit_scene({'action': 'clear', 'scene_version': 1})
        assert ctl.wait_idle(3)
        previous = ctl._session
        ctl._session_factory = factory
        ctl.reset()
        assert constructing.wait(2)
        state = ctl.snapshot()
        assert state['busy'] and state['needs_reset']
        assert state['scene'] == {}
        assert state['simulation'] == {'editor': {'editable': False}}
        assert previous.closed
        release.set()
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['scene'] == SCENE
        assert state['simulation']['editor']['editable']
        assert ctl._session is not previous
    finally:
        release.set()
        ctl.wait_idle(3)
        ctl.close()


def test_reader_cannot_modify_controller_scene_or_editor_metadata():
    ctl = create()
    try:
        state = ctl.snapshot()
        state['scene']['objects'][0]['center_base'][0] = 999
        state['simulation']['editor']['catalog'].clear()
        state['simulation']['editor']['editable'] = False
        current = ctl.snapshot()
        assert current['scene'] == SCENE
        assert current['simulation']['editor'] == {
            'editable': True, 'catalog': ['red_cube']}
    finally:
        ctl.close()


def test_worker_rechecks_held_object_after_request_is_accepted(monkeypatch):
    ctl = create()
    launch = ctl._launch

    def hold_before_launch(*args):
        ctl._session.held_object = 'red_cube_1'
        launch(*args)

    try:
        monkeypatch.setattr(ctl, '_launch', hold_before_launch)
        ctl.edit_scene({'action': 'clear', 'scene_version': 1})
        assert ctl.wait_idle(3)
        assert ctl._session.edit_requests == []
        assert ctl.snapshot()['status'] == 'error'
        assert '仍夹持' in ctl.snapshot()['error']
    finally:
        ctl.close()


@pytest.mark.parametrize('payload,message', [
    ({'action': 'clear', 'scene_version': 0}, '版本已过期'),
    ({'action': 'invalid', 'scene_version': 1}, '未知布局操作')])
def test_session_validation_error_keeps_scene_and_chat(payload, message):
    ctl = create()
    try:
        ctl._messages = [{'id': 10, 'role': 'user', 'content': '保留对话'}]
        before = ctl.snapshot()
        ctl.edit_scene(payload)
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'error' and message in state['error']
        assert state['scene'] == before['scene']
        assert state['messages'] == before['messages']
        assert state['metrics']['model_requests'] == 0
        assert state['simulation']['editor']['editable']
    finally:
        ctl.close()


def test_stop_during_edit_preserves_stopped_state_and_refuses_followup():
    gate = threading.Event()
    ctl = create()
    ctl._session.edit_gate = gate
    try:
        ctl.edit_scene({'action': 'clear', 'scene_version': 1})
        assert ctl._session.edit_entered.wait(2)
        ctl.stop()
        gate.set()
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'stopped' and state['needs_reset']
        assert state['scene'] == SCENE
        assert not state['simulation']['editor']['editable']
        with pytest.raises(ValueError, match='先重置'):
            ctl.edit_scene({'action': 'clear', 'scene_version': 1})
        assert len(ctl._session.edit_requests) == 1
    finally:
        gate.set()
        ctl.wait_idle(3)
        ctl.close()


def test_http_scene_edit_uses_existing_json_and_acceptance_boundaries(tmp_path, monkeypatch):
    from cube_task.assistant import server as server_module

    (tmp_path / 'editor.js').write_text('window.editorLoaded = true;', encoding='utf-8')
    monkeypatch.setattr(server_module, 'STATIC', tmp_path)
    ctl = create()
    server = DashboardServer(('127.0.0.1', 0), ctl)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = 'http://127.0.0.1:%s' % server.server_port

    def post(payload):
        return urlopen(Request(base + '/api/scene-edit',
                               data=json.dumps(payload).encode('utf-8'),
                               headers={'Content-Type': 'application/json'}), timeout=3)

    try:
        with urlopen(base + '/static/editor.js', timeout=3) as response:
            assert response.status == 200
            assert response.headers.get_content_type() == 'text/javascript'
            assert b'editorLoaded' in response.read()
        with pytest.raises(HTTPError) as error:
            post([])
        assert error.value.code == 400
        ctl._session.held_object = 'red_cube_1'
        with pytest.raises(HTTPError) as error:
            post({'action': 'clear', 'scene_version': 1})
        assert error.value.code == 400
        ctl._session.held_object = None
        ctl._session.edit_gate = threading.Event()
        with post({'action': 'clear', 'scene_version': 1}) as response:
            assert response.status == 202
            assert json.loads(response.read()) == {'ok': True}
        assert ctl._session.edit_entered.wait(2)
        with pytest.raises(HTTPError) as error:
            post({'action': 'clear', 'scene_version': 1})
        assert error.value.code == 409
        ctl._session.edit_gate.set()
        assert ctl.wait_idle(3)
        with post({'action': 'clear', 'scene_version': 1}) as response:
            assert response.status == 202
        assert ctl.wait_idle(3)
        with urlopen(base + '/api/state', timeout=3) as response:
            state = json.loads(response.read())
        assert state['status'] == 'error' and '版本已过期' in state['error']
        assert state['scene']['scene_version'] == 2
    finally:
        if ctl._session.edit_gate:
            ctl._session.edit_gate.set()
        ctl.wait_idle(3)
        server.shutdown()
        server.server_close()
        ctl.close()
