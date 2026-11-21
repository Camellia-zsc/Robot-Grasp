"""Dashboard concurrency and request-boundary tests, without paid API calls."""
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from cube_task.assistant.server import BusyError, DashboardController, DashboardServer


SCENE = {
    'scene_version': 1,
    'objects': [{'object_id': 'obj_1', 'label': 'red', 'score': 0.99,
                 'valid': True, 'center_base': [0.25, 0, 0.05]}],
    'zones': [{'name': 'zone_red', 'label': 'red'}],
    'class_to_zone': {'red': 'zone_red'},
}


class Settings:
    def public_info(self):
        return {'configured': True, 'provider': 'test', 'model': 'fake', 'api_style': 'responses'}


class Session:
    def __init__(self, **kwargs):
        self.scene = dict(SCENE)
        self.executions = []
        self.stopped = False
        self.closed = False

    def set_speed(self, speed):
        self.speed = speed

    def prepare(self):
        return self.scene

    def observe(self):
        return self.scene

    def execute(self, decision, cancel_event=None):
        self.executions.append(decision)
        return {'ok': True, 'message': '测试抓取完成'}

    def snapshot(self):
        return {'sim_time': 0, 'objects': [], 'joint_positions': []}

    def frame_jpeg(self):
        return None

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class Client:
    def __init__(self, action='pick', gate=None, stale=False):
        self.action = action
        self.gate = gate
        self.stale = stale
        self.streaming = threading.Event()

    def decide(self, instruction, scene, on_delta, cancel_event=None):
        on_delta('{"message":"正在选择红色方块"')
        self.streaming.set()
        if self.gate:
            self.gate.wait(3)
        return {'scene_version': 0 if self.stale else 1, 'action': self.action,
                'object_id': 'obj_1' if self.action == 'pick' else None,
                'zone_name': 'zone_red' if self.action == 'pick' else None,
                'message': '抓取红色方块' if self.action == 'pick' else '未观察到目标'}


def controller(client=None):
    ctl = DashboardController(session_factory=Session, settings=Settings(), client=client or Client())
    assert ctl.wait_idle(3)
    return ctl


def test_stream_visible_while_task_pending_and_duplicate_is_refused():
    gate = threading.Event()
    client = Client(gate=gate)
    ctl = controller(client)
    try:
        ctl.submit('抓红色方块')
        assert client.streaming.wait(2)
        state = ctl.snapshot()
        assert state['status'] == 'thinking'
        assert state['messages'][-1]['content'].startswith('{"message"')
        assert state['messages'][-1]['status'] == 'streaming'
        with pytest.raises(BusyError):
            ctl.submit('再抓一个')
        assert ctl._session.executions == []
        gate.set()
        assert ctl.wait_idle(3)
        assert len(ctl._session.executions) == 1
        assert ctl.snapshot()['last_result']['ok']
    finally:
        gate.set()
        ctl.close()


@pytest.mark.parametrize('action,stale', [('target_not_observed', False), ('pick', True)])
def test_absent_or_stale_model_target_never_executes(action, stale):
    ctl = controller(Client(action=action, stale=stale))
    try:
        ctl.submit('选择目标')
        assert ctl.wait_idle(3)
        assert ctl._session.executions == []
        assert ctl.snapshot()['status'] == ('error' if stale else 'ready')
    finally:
        ctl.close()


def test_cancel_during_api_does_not_execute_and_can_reset():
    gate = threading.Event()
    client = Client(gate=gate)
    ctl = controller(client)
    try:
        ctl.submit('抓红色方块')
        assert client.streaming.wait(2)
        ctl.stop()
        gate.set()
        assert ctl.wait_idle(3)
        assert ctl.snapshot()['status'] == 'stopped'
        assert ctl._session.executions == []
        previous = ctl._session
        ctl.reset()
        assert ctl.wait_idle(3)
        assert ctl._session is not previous
        assert ctl.snapshot()['status'] == 'ready'
    finally:
        gate.set()
        ctl.close()


def test_http_rejects_cross_origin_and_no_key_in_state():
    ctl = controller()
    server = DashboardServer(('127.0.0.1', 0), ctl)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = 'http://127.0.0.1:%s' % server.server_port
    try:
        with urlopen(base + '/api/state', timeout=3) as response:
            state = json.loads(response.read())
        assert state['model']['configured']
        assert 'api_key' not in json.dumps(state)
        req = Request(base + '/api/command', data=b'{"instruction":"pick"}',
                      headers={'Content-Type': 'application/json', 'Origin': 'https://example.com'})
        with pytest.raises(HTTPError) as error:
            urlopen(req, timeout=3)
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(base + '/../llm.py', timeout=3)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        ctl.close()


def test_dashboard_polling_preserves_ui_events_without_copying_audit_payloads():
    ctl = controller()
    server = DashboardServer(('127.0.0.1', 0), ctl)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = 'http://127.0.0.1:%s' % server.server_port
    try:
        ctl._event('scene_intervention_applied', {
            'request_id': 42, 'message': '干预已生效', 'stage': 'inspect_place',
            'ok': False, 'feedback': {'ok': False, 'audit': ['detail'] * 1000},
            'result': {'trajectory': ['large audit'] * 1000}})
        with urlopen(base + '/api/state?view=dashboard', timeout=3) as response:
            dashboard = json.loads(response.read())
        with urlopen(base + '/api/state', timeout=3) as response:
            full = json.loads(response.read())
        event = dashboard['events'][-1]
        assert event == {key: full['events'][-1][key] for key in
                         ('id', 'kind', 'time', 'request_id', 'message', 'stage', 'ok')} | {
                             'feedback': {'ok': False}}
        assert full['events'][-1]['result']['trajectory'] == ['large audit'] * 1000
        assert full['events'][-1]['feedback']['audit'] == ['detail'] * 1000
        assert {key: value for key, value in dashboard.items() if key != 'events'} == {
            key: value for key, value in full.items() if key != 'events'}
        event['message'] = 'changed browser copy'
        assert ctl.snapshot()['events'][-1]['message'] == '干预已生效'
    finally:
        server.shutdown()
        server.server_close()
        ctl.close()


def test_stop_during_session_construction_stops_the_new_session():
    constructing = threading.Event()
    release = threading.Event()

    def factory(**kwargs):
        constructing.set()
        release.wait(3)
        return Session(**kwargs)

    ctl = DashboardController(session_factory=factory, settings=Settings(), client=Client())
    try:
        assert constructing.wait(2)
        ctl.stop()
        release.set()
        assert ctl.wait_idle(3)
        assert ctl._session.stopped
        assert ctl.snapshot()['status'] == 'stopped'
        assert ctl.snapshot()['needs_reset']
        assert ctl.snapshot()['scene'] == {}
    finally:
        release.set()
        ctl.close()


def test_late_model_worker_releases_resources_after_controller_close(monkeypatch):
    gate = threading.Event()
    client = Client(gate=gate)
    ctl = controller(client)
    session = ctl._session
    wait_idle = ctl.wait_idle
    try:
        ctl.submit('抓红色方块')
        assert client.streaming.wait(2)
        monkeypatch.setattr(ctl, 'wait_idle', lambda timeout=30: False)
        ctl.close()
        assert not session.closed
        gate.set()
        assert wait_idle(3)
        assert session.closed
        assert ctl._session is None
        assert session.executions == []
    finally:
        gate.set()
        wait_idle(3)
        ctl.close()


def test_failed_reset_clears_the_closed_scene_and_refuses_commands():
    ctl = controller()

    def broken(**kwargs):
        raise ValueError('test construction failed')

    try:
        ctl._session_factory = broken
        ctl.reset()
        assert ctl.wait_idle(3)
        assert ctl._session is None
        assert ctl.snapshot()['scene'] == {}
        assert ctl.snapshot()['needs_reset']
        with pytest.raises(ValueError):
            ctl.submit('抓红色方块')
    finally:
        ctl.close()


def test_live_preview_and_frozen_observations_are_separate_and_cleaned_up():
    previews = []

    class CameraSession(Session):
        def observation_views(self):
            return [{'id': 3, 'label': '黄色放置区', 'detected_labels': ['red']}]

        def observation_frame_jpeg(self, view):
            return b'processed-detection-image' if view == 3 else None

    class Preview:
        def __init__(self, session):
            self.closed = False
            previews.append(self)

        def start(self):
            pass

        def close(self):
            self.closed = True

        def info(self):
            return {'mode': 'live', 'available': True, 'sim_time': 12.3}

        def frame_jpeg(self):
            return b'live-wrist-image'

    ctl = DashboardController(session_factory=CameraSession, settings=Settings(),
                              client=Client(), preview_factory=Preview)
    try:
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['camera']['mode'] == 'live'
        assert state['camera_available']
        assert state['observation_views'][0]['detected_labels'] == ['red']
        assert ctl.camera_jpeg() == b'live-wrist-image'
        assert ctl.camera_jpeg(mode='observation', view=3) == b'processed-detection-image'
        assert ctl.camera_jpeg(mode='observation', view=99) is None
        with pytest.raises(ValueError):
            ctl.camera_jpeg(mode='invalid')
        ctl.reset()
        assert ctl.wait_idle(3)
        assert previews[0].closed and not previews[1].closed
    finally:
        ctl.close()
    assert all(preview.closed for preview in previews)


def test_stop_between_model_decision_and_terminal_state_is_preserved():
    ctl = controller(Client(action='answer'))
    deciding = threading.Event()
    release = threading.Event()
    original = ctl._event

    def event(kind, payload=None):
        if kind == 'decision':
            deciding.set()
            release.wait(3)
        original(kind, payload)

    ctl._event = event
    try:
        ctl.submit('桌上有什么')
        assert deciding.wait(2)
        ctl.stop()
        release.set()
        assert ctl.wait_idle(3)
        assert ctl.snapshot()['status'] == 'stopped'
        assert ctl.snapshot()['needs_reset']
        assert ctl._session.executions == []
    finally:
        release.set()
        ctl.close()
