"""Free placement stays explicit, bounded and detached from world truth."""
from copy import deepcopy
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from cube_task.assistant import llm
from cube_task.assistant.server import DashboardController, DashboardServer
from test_assistant_llm import FakeStream, completed, install_stream, settings, sse
from test_assistant_server import Session, Settings


POINT = {'type': 'point', 'xy': [.30, .08]}
REGION = {'type': 'region', 'name': 'pick_area'}


def observed(version=7):
    return {
        'scene_version': version,
        'objects': [
            {'object_id': f'v{version}-apple', 'label': 'apple', 'valid': True,
             'center_base': [.25, -.08, .06]},
            {'object_id': f'v{version}-pear', 'label': 'pear', 'valid': True,
             'center_base': [.35, -.05, .10]},
        ],
        'class_to_zone': {'apple': 'zone_red', 'pear': 'zone_green'},
        'zones': [{'name': 'zone_red'}, {'name': 'zone_green'}],
        'placement': {
            'bounds': {'x': [.08, .60], 'y': [-.40, .40], 'r_min': .10, 'r_max': .62},
            'table_bounds': {'x': [.08, .60], 'y': [-.40, .40]},
            'pick_polygon': [[.15, -.22], [.45, -.22], [.45, .22], [.15, .22]],
            'point_tolerance': .015, 'margin': .002,
        },
    }


def pick(scene, destination=None):
    result = {'scene_version': scene['scene_version'], 'action': 'pick',
              'object_id': scene['objects'][0]['object_id'], 'zone_name': None,
              'message': '将苹果放到指定位置。'}
    if destination is not None:
        result['destination'] = deepcopy(destination)
    return result


@pytest.mark.parametrize('destination', [POINT, REGION])
def test_new_destination_is_preserved_and_detached(destination):
    scene = observed()
    decision = pick(scene, destination)
    before_scene, before_decision = deepcopy(scene), deepcopy(decision)
    result = llm.validate_decision(decision, scene)
    assert result['zone_name'] is None and result['destination'] == destination
    if destination['type'] == 'point':
        result['destination']['xy'][0] = .5
    else:
        result['destination']['name'] = 'changed'
    assert scene == before_scene and decision == before_decision


def test_legacy_normalization_keeps_its_exact_shape():
    scene = observed()
    decision = pick(scene)
    assert llm.validate_decision(decision, scene) == dict(decision, zone_name='zone_red')


def test_explicit_null_destination_keeps_legacy_default_without_changing_input():
    scene = observed()
    decision = dict(pick(scene), destination=None)
    before = deepcopy(decision)
    assert llm.validate_decision(decision, scene) == dict(decision, zone_name='zone_red')
    assert decision == before


@pytest.mark.parametrize('destination', [
    {}, 'pick_area', [], {'type': 'region'},
    {'type': 'region', 'name': 'invented'},
    {'type': 'region', 'name': 'pick_area', 'xy': [.3, 0.]},
    {'type': 'point'}, {'type': 'point', 'xy': [.3]},
    {'type': 'point', 'xy': [.3, 0., .1]},
    {'type': 'point', 'xy': [True, 0.]},
    {'type': 'point', 'xy': ['.3', 0.]},
    {'type': 'point', 'xy': [float('nan'), 0.]},
    {'type': 'point', 'xy': [float('inf'), 0.]},
    {'type': 'point', 'xy': [10 ** 500, 0.]},
    {'type': 'point', 'xy': [.70, 0.]},
    {'type': 'point', 'xy': [.08, 0.]},
    {'type': 'point', 'xy': [.3, 0.], 'z': .2},
])
def test_malformed_unbounded_or_invented_destination_is_rejected(destination):
    scene = observed()
    with pytest.raises(ValueError):
        llm.validate_decision(pick(scene, destination), scene)


@pytest.mark.parametrize('destination', [POINT, REGION])
def test_destination_cannot_conflict_with_a_legacy_zone(destination):
    scene = observed()
    decision = pick(scene, destination)
    decision['zone_name'] = 'zone_red'
    with pytest.raises(ValueError):
        llm.validate_decision(decision, scene)


@pytest.mark.parametrize('action', ['answer', 'clarify', 'target_not_observed'])
def test_nonmotion_decision_cannot_carry_an_executable_destination(action):
    scene = observed()
    decision = pick(scene, POINT)
    decision.update(action=action, object_id=None)
    with pytest.raises(ValueError):
        llm.validate_decision(decision, scene)


def test_mixed_batch_keeps_each_destination_and_validates_the_last_before_execution():
    scene = observed()
    decision = {'scene_version': 7, 'action': 'pick_batch', 'message': '按顺序搬运。',
                'items': [{'object_id': 'v7-apple', 'zone_name': None,
                           'destination': deepcopy(POINT)},
                          {'object_id': 'v7-pear', 'zone_name': None,
                           'destination': deepcopy(REGION)}]}
    before = deepcopy(decision)
    result = llm.validate_decision(decision, scene)
    assert result['items'] == before['items'] and decision == before
    result['items'][0]['destination']['xy'][0] = .5
    assert decision == before
    decision['items'][-1]['destination']['name'] = 'invented'
    with pytest.raises(ValueError):
        llm.validate_decision(decision, scene)


@pytest.mark.parametrize('style', ['responses', 'chat_completions'])
def test_both_model_apis_receive_only_public_placement_context(monkeypatch, style):
    scene = observed()
    scene['selected_destination'] = deepcopy(POINT)
    scene['private_world'] = 'private-sentinel'
    scene['placement']['private_planner'] = 'private-sentinel'
    scene['objects'][0]['truth_object'] = 'private-sentinel'
    decision = pick(scene, POINT)
    text = json.dumps(decision, ensure_ascii=False)
    if style == 'responses':
        raw = completed(text)
    else:
        raw = (sse({'choices': [{'index': 0, 'delta': {'content': text}}]})
               + sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
               + b'data: [DONE]\n\n')
    requests = install_stream(monkeypatch, FakeStream([raw]))
    result = llm.ModelClient(settings(style)).decide('把苹果放到选定位置', scene, lambda _: None)
    assert result['destination'] == POINT and len(requests) == 1
    body = json.loads(requests[0][0].data)
    user = json.loads(body['input'] if style == 'responses' else body['messages'][1]['content'])
    assert user['scene']['selected_destination'] == POINT
    assert user['scene']['placement']['bounds'] == scene['placement']['bounds']
    assert user['scene']['placement']['pick_polygon'] == scene['placement']['pick_polygon']
    assert 'private-sentinel' not in requests[0][0].data.decode()


def test_selected_point_cannot_be_changed_by_the_model():
    scene = observed()
    scene['selected_destination'] = deepcopy(POINT)
    with pytest.raises(ValueError):
        llm.validate_decision(pick(scene, {'type': 'point', 'xy': [.31, .08]}), scene)


class PlacementSession(Session):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scene = observed(1)
        self.observing = threading.Event()
        self.release = None

    def observe(self):
        self.observing.set()
        if self.release is not None:
            assert self.release.wait(3)
        self.scene = observed(self.scene['scene_version'] + 1)
        return self.scene


class PlacementClient:
    def __init__(self, action='pick'):
        self.action = action
        self.seen = []

    def decide(self, instruction, scene, on_delta, cancel_event=None):
        self.seen.append(deepcopy(scene))
        if self.action == 'pick':
            return pick(scene, scene.get('selected_destination'))
        return {'scene_version': scene['scene_version'], 'action': self.action,
                'object_id': None, 'zone_name': None, 'message': '这里可以选择一个放置点。'}


def create(client=None):
    ctl = DashboardController(session_factory=PlacementSession, settings=Settings(),
                              client=client or PlacementClient(), realtime=False)
    assert ctl.wait_idle(3)
    return ctl


@pytest.mark.parametrize('version', [None, True, 0, 9, '1', 1.0])
def test_selection_requires_current_integer_version_before_any_scan_or_model_call(version):
    client = PlacementClient()
    ctl = create(client)
    try:
        with pytest.raises(ValueError):
            ctl.submit('把苹果放到选定位置', destination=deepcopy(POINT), scene_version=version)
        assert not client.seen and not ctl._session.observing.is_set()
        assert not ctl._session.executions
    finally:
        ctl.close()


def test_selection_is_detached_and_added_to_the_fresh_observation():
    client = PlacementClient()
    ctl = create(client)
    release = threading.Event()
    ctl._session.release = release
    selected = deepcopy(POINT)
    try:
        ctl.submit('把苹果放到选定位置', destination=selected, scene_version=1)
        assert ctl._session.observing.wait(2)
        selected['xy'][0] = .55
        release.set()
        assert ctl.wait_idle(3)
        assert client.seen[0]['scene_version'] == 2
        assert client.seen[0]['selected_destination'] == POINT
        assert ctl._session.executions[0]['destination'] == POINT
    finally:
        release.set()
        ctl.close()


def test_selected_point_does_not_turn_a_question_into_motion():
    client = PlacementClient(action='answer')
    ctl = create(client)
    try:
        ctl.submit('这个点能放苹果吗？', destination=deepcopy(POINT), scene_version=1)
        assert ctl.wait_idle(3)
        assert ctl.snapshot()['status'] == 'ready' and not ctl._session.executions
        assert client.seen[0]['selected_destination'] == POINT
    finally:
        ctl.close()


@pytest.mark.parametrize('payload', [
    {'instruction': '抓苹果', 'unexpected': True},
    {'instruction': '抓苹果', 'destination': REGION, 'scene_version': 1},
    {'instruction': '抓苹果', 'destination': POINT},
    {'instruction': '抓苹果', 'destination': POINT, 'scene_version': True},
    {'instruction': '抓苹果', 'destination': {'type': 'point', 'xy': [.7, 0.]},
     'scene_version': 1},
    '{"instruction":"抓苹果","instruction":"抓梨"}',
    '{"instruction":"抓苹果","destination":{"type":"point","xy":[0.3,NaN]},"scene_version":1}',
    '{"instruction":"抓苹果","destination":{"type":"point","xy":[0.3,0.08],"xy":[0.4,0.1]},"scene_version":1}',
])
def test_http_rejects_bad_selection_payload_with_a_chinese_4xx(payload):
    ctl = create()
    server = DashboardServer(('127.0.0.1', 0), ctl)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        request = Request('http://127.0.0.1:%d/api/command' % server.server_port,
                          data=(payload if isinstance(payload, str) else json.dumps(payload)).encode(),
                          headers={'Content-Type': 'application/json'})
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=3)
        assert 400 <= error.value.code < 500
        message = json.loads(error.value.read())['error']
        assert any('\u4e00' <= char <= '\u9fff' for char in message)
        assert not ctl._session.executions
    finally:
        server.shutdown()
        server.server_close()
        ctl.close()
