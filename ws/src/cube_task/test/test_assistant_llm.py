"""Offline model-bridge tests: fake streams, local config and motion gating."""
from __future__ import annotations

from copy import deepcopy
import http.client
import io
import json
import threading
import urllib.error

import pytest

from cube_task.assistant import llm


@pytest.fixture
def scene():
    return {
        'scene_version': 7,
        'objects': [{'object_id': 'scene7-red-1', 'label': 'red', 'score': 0.96,
                     'valid': True, 'center_base': [0.3, 0.02, 0.04]}],
        'zones': [{'name': 'red-bin', 'label': '红色区'},
                  {'name': 'other-bin', 'label': '备用区'}],
        'class_to_zone': {'red': 'red-bin'},
    }


@pytest.fixture
def decision():
    return {'scene_version': 7, 'action': 'pick', 'object_id': 'scene7-red-1',
            'zone_name': None, 'message': '将红色方块放入默认分拣区。'}


def settings(style='responses', **kwargs):
    return llm.LLMSettings(base_url='https://relay.invalid/v1', model='gpt-6.1-sol',
                           api_key='test-secret-only', api_style=style, **kwargs)


def sse(item, event=None):
    prefix = '' if event is None else 'event: ' + event + '\r\n'
    return (prefix + 'data: ' + json.dumps(item, ensure_ascii=False)
            + '\r\n\r\n').encode('utf-8')


def completed(text=None, **extra):
    response = {'status': 'completed', **extra}
    if text is not None:
        response['output'] = [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': text}]}]
    return sse({'type': 'response.completed', 'response': response})


class FakeStream:
    def __init__(self, chunks, on_read=None):
        self.chunks = iter(chunks)
        self.on_read = on_read
        self.closed = False

    def read(self, _size):
        if self.on_read:
            self.on_read()
        return next(self.chunks, b'')

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True


def install_stream(monkeypatch, stream):
    requests = []

    def fake_open(request, timeout):
        requests.append((request, timeout))
        return stream

    monkeypatch.setattr(llm, '_open', fake_open)
    return requests


def test_responses_split_utf8_streams_visible_output_only(monkeypatch, scene, decision):
    text = json.dumps(decision, ensure_ascii=False)
    first, second = text[:70], text[70:]
    raw = (b'\xef\xbb\xbf: heartbeat\r\n\r\n'
           + sse({'type': 'response.reasoning_summary_text.delta', 'delta': 'hidden thought'})
           + sse({'type': 'response.output_text.delta', 'delta': first})
           + sse({'type': 'response.output_text.delta', 'delta': second})
           + completed(text))
    stream = FakeStream([raw[i:i + 1] for i in range(len(raw))])
    requests = install_stream(monkeypatch, stream)
    deltas = []
    result = llm.ModelClient(settings()).decide('抓起红色方块', scene, deltas.append)
    assert result == dict(decision, zone_name='red-bin')
    assert deltas == [first, second]
    assert stream.closed
    assert len(requests) == 1
    request, timeout = requests[0]
    body = json.loads(request.data)
    assert request.full_url == 'https://relay.invalid/v1/responses'
    assert request.get_method() == 'POST'
    assert request.headers['Authorization'] == 'Bearer test-secret-only'
    assert timeout == 30
    assert body['stream'] is True and body['store'] is False
    assert body['max_output_tokens'] == 2048
    assert body['reasoning']['effort'] == 'low'
    assert 'test-secret-only' not in request.data.decode()
    assert 'scene7-red-1' in body['input']
    assert '多个物体' in body['instructions']


def test_responses_final_only_and_code_fence(monkeypatch, scene, decision):
    text = '```json\n' + json.dumps(decision, ensure_ascii=False) + '\n```'
    install_stream(monkeypatch, FakeStream([completed(text)]))
    deltas = []
    result = llm.ModelClient(settings()).decide('抓红色', scene, deltas.append)
    assert result['zone_name'] == 'red-bin'
    assert deltas == [text]


def test_placed_object_region_is_available_to_model_for_relocation(monkeypatch, scene, decision):
    scene['objects'][0].update(in_zone=True, zone='other-bin')
    text = json.dumps(decision, ensure_ascii=False)
    requests = install_stream(monkeypatch, FakeStream([completed(text)]))
    result = llm.ModelClient(settings()).decide('把红块搬到红区', scene, lambda _: None)
    body = json.loads(requests[0][0].data)
    measured = json.loads(body['input'])['scene']['objects'][0]
    assert measured['in_zone'] is True and measured['zone'] == 'other-bin'
    assert '已在放置区的物体允许再次抓取' in body['instructions']
    assert result['zone_name'] == 'red-bin'


def test_chat_stream_ignores_reasoning_and_other_choices(monkeypatch, scene, decision):
    text = json.dumps(decision, ensure_ascii=False)
    raw = (sse({'choices': [{'index': 0, 'delta': {'reasoning_content': 'private'}}]})
           + sse({'choices': [{'index': 1, 'delta': {'content': 'ignored'}},
                              {'index': 0, 'delta': {'content': text[:65]}}]})
           + sse({'choices': [{'index': 0, 'delta': {'content': text[65:]}}]})
           + sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
           + b'data: [DONE]\n\n')
    requests = install_stream(monkeypatch, FakeStream([raw[:51], raw[51:200], raw[200:]]))
    deltas = []
    result = llm.ModelClient(settings('chat_completions')).decide('抓红色', scene, deltas.append)
    assert result['action'] == 'pick'
    assert ''.join(deltas) == text
    request, _ = requests[0]
    body = json.loads(request.data)
    assert request.full_url.endswith('/chat/completions')
    assert body['max_completion_tokens'] == 800
    assert body['messages'][0]['role'] == 'system'


def test_multi_object_instruction_is_sent_to_actual_model(monkeypatch, scene):
    answer = {'scene_version': 7, 'action': 'clarify', 'object_id': None,
              'zone_name': None, 'message': '请一次指定一个物体。'}
    requests = install_stream(monkeypatch, FakeStream([completed(json.dumps(answer))]))
    result = llm.ModelClient(settings()).decide('把所有物体依次分拣', scene, lambda _: None)
    assert result['action'] == 'clarify'
    assert len(requests) == 1
    assert '把所有物体依次分拣' in json.loads(requests[0][0].data)['input']


@pytest.mark.parametrize('raw,code', [
    (sse({'type': 'response.output_text.delta', 'delta': '{}'}), 'incomplete'),
    (b'data: [DONE]\n\n', 'incomplete'),
    (completed(), 'empty_output'),
    (sse({'type': 'response.incomplete'}), 'incomplete'),
    (sse({'type': 'response.failed', 'error': {'message': 'secret echo'}}), 'provider_error'),
    (sse({'type': 'response.refusal.delta', 'delta': 'refused'}), 'refusal'),
    (b'data: not-json\n\n', 'invalid_output'),
    (completed('truncated {'), 'invalid_output'),
    (completed('[]'), 'invalid_output'),
    (completed('{"action":"pick","action":"clarify"}'), 'invalid_output'),
    (completed('{"scene_version":NaN}'), 'invalid_output'),
    (completed(extra='unused', status='incomplete'), 'incomplete'),
])
def test_bad_responses_never_return_decision(monkeypatch, scene, raw, code):
    requests = install_stream(monkeypatch, FakeStream([raw]))
    with pytest.raises(llm.LLMError) as exc:
        llm.ModelClient(settings()).decide('抓红色', scene, lambda _: None)
    assert exc.value.code == code
    assert 'secret echo' not in str(exc.value)
    assert len(requests) == 1


@pytest.mark.parametrize('raw', [
    b'data: [DONE]\n\n',
    sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}),
    sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]}),
    sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'content_filter'}]}),
    sse({'choices': [{'index': 0, 'delta': {'tool_calls': [{'name': 'move'}]}}]}),
    sse({'choices': [{'index': 0, 'delta': {'refusal': 'refused'}}]}),
])
def test_chat_needs_clean_stop_and_done(monkeypatch, scene, raw):
    install_stream(monkeypatch, FakeStream([raw]))
    with pytest.raises(llm.LLMError):
        llm.ModelClient(settings('chat_completions')).decide('抓红色', scene, lambda _: None)


def test_response_complete_text_must_match_stream(monkeypatch, scene, decision):
    text = json.dumps(decision)
    raw = sse({'type': 'response.output_text.delta', 'delta': text[:-3]}) + completed(text)
    install_stream(monkeypatch, FakeStream([raw]))
    with pytest.raises(llm.LLMError, match='不完整'):
        llm.ModelClient(settings()).decide('抓红色', scene, lambda _: None)


@pytest.mark.parametrize('error,code', [
    (urllib.error.HTTPError('https://relay.invalid', 401, 'test-secret-only', {},
                            io.BytesIO(b'test-secret-only')), 'authentication'),
    (urllib.error.HTTPError('https://relay.invalid', 429, 'test-secret-only', {},
                            io.BytesIO(b'test-secret-only')), 'rate_limit'),
    (urllib.error.HTTPError('https://relay.invalid', 500, 'test-secret-only', {},
                            io.BytesIO(b'test-secret-only')), 'provider_error'),
    (urllib.error.URLError('test-secret-only'), 'connection'),
    (TimeoutError('test-secret-only'), 'timeout'),
    (ValueError('invalid header containing test-secret-only'), 'invalid_output'),
    (http.client.IncompleteRead(b'test-secret-only'), 'incomplete'),
])
def test_remote_errors_are_redacted_and_never_retried(monkeypatch, scene, error, code):
    calls = []

    def fail(*_args):
        calls.append(True)
        raise error

    monkeypatch.setattr(llm, '_open', fail)
    with pytest.raises(llm.LLMError) as exc:
        llm.ModelClient(settings()).decide('抓红色', scene, lambda _: None)
    assert exc.value.code == code
    assert 'test-secret-only' not in str(exc.value)
    assert 'relay.invalid' not in str(exc.value)
    assert len(calls) == 1


def test_cancel_before_request_does_not_call_api(monkeypatch, scene):
    event = threading.Event()
    event.set()
    requests = install_stream(monkeypatch, FakeStream([]))
    with pytest.raises(llm.LLMError) as exc:
        llm.ModelClient(settings()).decide('抓红色', scene, lambda _: None, event)
    assert exc.value.code == 'cancelled'
    assert not requests


def test_cancel_during_stream_closes_connection(monkeypatch, scene):
    event = threading.Event()
    stream = FakeStream([sse({'type': 'response.output_text.delta', 'delta': '{}'})],
                        on_read=event.set)
    install_stream(monkeypatch, stream)
    deltas = []
    with pytest.raises(llm.LLMError) as exc:
        llm.ModelClient(settings()).decide('抓红色', scene, deltas.append, event)
    assert exc.value.code == 'cancelled'
    assert not deltas
    assert stream.closed


@pytest.mark.parametrize('phase', ['connect', 'read'])
def test_stop_interrupts_blocked_network_wait_without_late_output(monkeypatch, scene, phase):
    entered, release, done, cancelled = [threading.Event() for _ in range(4)]
    deltas, failures, calls = [], [], []

    class BlockingStream(FakeStream):
        def read(self, size):
            if phase == 'read':
                entered.set()
                release.wait(3)
            return super().read(size)

    stream = BlockingStream([completed('{}')])

    def blocked_open(*args):
        calls.append(1)
        if phase == 'connect':
            entered.set()
            release.wait(3)
        return stream

    monkeypatch.setattr(llm, '_open', blocked_open)

    def request():
        try:
            llm.ModelClient(settings()).decide('抓红色', scene, deltas.append, cancelled)
        except llm.LLMError as exc:
            failures.append(exc.code)
        finally:
            done.set()

    worker = threading.Thread(target=request)
    worker.start()
    try:
        assert entered.wait(2)
        cancelled.set()
        assert done.wait(.5), 'stop must not wait for the provider socket timeout'
        assert failures == ['cancelled']
        assert not deltas and calls == [1]
    finally:
        release.set()
        worker.join(3)
    assert not deltas


def test_output_and_duration_are_bounded(monkeypatch, scene):
    install_stream(monkeypatch, FakeStream([
        sse({'type': 'response.output_text.delta', 'delta': 'x' * 12001})]))
    with pytest.raises(llm.LLMError) as exc:
        llm.ModelClient(settings()).decide('抓红色', scene, lambda _: None)
    assert exc.value.code == 'invalid_output'
    with pytest.raises(llm.LLMError) as exc:
        llm.ModelClient(settings(max_duration=-1)).decide('抓红色', scene, lambda _: None)
    assert exc.value.code == 'timeout'


@pytest.mark.parametrize('field,value', [
    ('scene_version', 6), ('scene_version', True), ('action', 'move_xyz'),
    ('object_id', 'invented'), ('object_id', None), ('object_id', ['scene7-red-1']),
    ('zone_name', 'invented-bin'), ('zone_name', ['red-bin']),
    ('message', ''), ('message', 'x' * 501),
])
def test_decision_rejects_unknown_stale_or_invalid_fields(scene, decision, field, value):
    decision[field] = value
    with pytest.raises(ValueError):
        llm.validate_decision(decision, scene)


@pytest.mark.parametrize('center', [None, [0.2, 0.1], [float('nan'), 0.1, 0.04],
                                   [0.2, float('inf'), 0.04], [True, 0.1, 0.04],
                                   ['0.2', 0.1, 0.04], 'xyz'])
def test_decision_requires_finite_three_dimensional_observation(scene, decision, center):
    scene['objects'][0]['center_base'] = center
    with pytest.raises(ValueError):
        llm.validate_decision(decision, scene)


@pytest.mark.parametrize('change', ['invalid', 'missing_mapping', 'bad_mapping',
                                    'duplicate_object', 'duplicate_zone', 'invalid_zone',
                                    'invalid_scene', 'not_executable'])
def test_decision_cannot_bypass_runtime_scene_constraints(scene, decision, change):
    if change == 'invalid':
        scene['objects'][0]['valid'] = False
    elif change == 'missing_mapping':
        scene['class_to_zone'] = {}
    elif change == 'bad_mapping':
        scene['class_to_zone']['red'] = 'missing'
    elif change == 'duplicate_object':
        scene['objects'].append(deepcopy(scene['objects'][0]))
    elif change == 'duplicate_zone':
        scene['zones'].append(deepcopy(scene['zones'][0]))
    elif change == 'invalid_zone':
        scene['zones'][0]['valid'] = False
    elif change == 'invalid_scene':
        scene['valid'] = False
    else:
        scene['executable'] = False
    with pytest.raises(ValueError):
        llm.validate_decision(decision, scene)


def test_explicit_zone_override_and_default_are_validated(scene, decision):
    assert llm.validate_decision(decision, scene)['zone_name'] == 'red-bin'
    decision['zone_name'] = 'other-bin'
    assert llm.validate_decision(decision, scene)['zone_name'] == 'other-bin'
    assert decision['zone_name'] == 'other-bin'


def test_model_cannot_supply_geometry_or_commands(scene, decision):
    for key in ('center_base', 'command', 'trajectory'):
        bad = dict(decision, **{key: [1, 2, 3]})
        with pytest.raises(ValueError):
            llm.validate_decision(bad, scene)


@pytest.mark.parametrize('action', ['clarify', 'target_not_observed', 'answer'])
def test_nonpick_must_not_carry_executable_target(scene, action):
    decision = {'scene_version': 7, 'action': action, 'object_id': None,
                'zone_name': None, 'message': '请确认目标。'}
    assert llm.validate_decision(decision, scene) == decision
    for key, value in [('object_id', 'scene7-red-1'), ('zone_name', 'red-bin')]:
        with pytest.raises(ValueError):
            llm.validate_decision(dict(decision, **{key: value}), scene)


@pytest.fixture
def local_config(tmp_path, monkeypatch):
    for name in list(llm.os.environ):
        if name.startswith('ROBOT_LLM_') or name in ('OPENAI_API_KEY', 'RELAY_TOKEN'):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CODEX_HOME', str(tmp_path))
    (tmp_path / 'config.toml').write_text(
        'model = "local-test-model"\nmodel_provider = "relay"\n'
        '[model_providers.relay]\nname = "Relay"\n'
        'base_url = "https://relay.invalid/v1"\nwire_api = "responses"\n'
        'requires_openai_auth = true\nenv_key = "RELAY_TOKEN"\n', encoding='utf-8')
    (tmp_path / 'auth.json').write_text(json.dumps({'OPENAI_API_KEY': 'auth-test-secret'}),
                                      encoding='utf-8')
    return tmp_path


def test_settings_reuses_local_codex_without_exposing_credentials(local_config):
    config = llm.load_settings()
    assert config.configured
    assert config.api_key == 'auth-test-secret'
    assert config.base_url == 'https://relay.invalid/v1'
    assert config.public_info() == {'configured': True, 'provider': 'relay',
                                    'model': 'local-test-model', 'api_style': 'responses'}
    assert 'auth-test-secret' not in repr(config)
    assert 'relay.invalid' not in repr(config)
    assert 'auth-test-secret' not in json.dumps(config.public_info())


def test_settings_environment_overrides(local_config, monkeypatch):
    monkeypatch.setenv('RELAY_TOKEN', 'provider-env-secret')
    assert llm.load_settings().api_key == 'provider-env-secret'
    monkeypatch.setenv('ROBOT_LLM_BASE_URL', 'https://override.invalid/v1/')
    monkeypatch.setenv('ROBOT_LLM_MODEL', 'override-model')
    monkeypatch.setenv('ROBOT_LLM_API_KEY', 'override-secret')
    monkeypatch.setenv('ROBOT_LLM_API_STYLE', 'chat_completions')
    monkeypatch.setenv('ROBOT_LLM_REASONING_EFFORT', 'minimal')
    config = llm.load_settings()
    assert config.base_url == 'https://override.invalid/v1'
    assert config.model == 'override-model' and config.api_key == 'override-secret'
    assert config.api_style == 'chat_completions'
    assert config.reasoning_effort == 'minimal'
    assert config.provider == 'environment'


def test_python310_config_fallback_and_missing_auth(local_config, monkeypatch):
    monkeypatch.setattr(llm, 'tomllib', None)
    config = llm.load_settings()
    assert config.model == 'local-test-model' and config.api_key == 'auth-test-secret'
    (local_config / 'auth.json').write_text('{invalid json', encoding='utf-8')
    assert llm.load_settings().configured is False
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-env-secret')
    assert llm.load_settings().api_key == 'openai-env-secret'


@pytest.mark.parametrize('url', ['file:///tmp/api', 'https://user:password@relay.invalid/v1',
                               'https://relay.invalid/v1?token=secret', 'https://[bad'])
def test_unsafe_or_invalid_url_is_not_configured(url):
    config = llm.LLMSettings(base_url=url, model='test', api_key='test')
    assert config.configured is False


def test_observation_question_returns_nonmotion_answer(monkeypatch, scene):
    answer = {'scene_version': 7, 'action': 'answer', 'object_id': None,
              'zone_name': None, 'message': '桌上观察到了一个红色方块。'}
    requests = install_stream(monkeypatch, FakeStream([completed(json.dumps(answer))]))
    result = llm.ModelClient(settings()).decide('桌上有红色方块吗？', scene, lambda _: None)
    assert result == answer
    assert len(requests) == 1
    assert '问题严禁执行抓取' in json.loads(requests[0][0].data)['instructions']


def test_records_response_usage_and_stops_on_completion(monkeypatch, scene, decision):
    text = json.dumps(decision)
    stream = FakeStream([completed(text, usage={'input_tokens': 130, 'output_tokens': 72,
                                               'total_tokens': 202, 'private': 'secret'}),
                         b'data: broken event that must not be read\n\n'])
    install_stream(monkeypatch, stream)
    client = llm.ModelClient(settings())
    client.decide('抓红色', scene, lambda _: None)
    assert client.last_usage == {'input_tokens': 130, 'output_tokens': 72, 'total_tokens': 202}
    assert stream.closed
    monkeypatch.setattr(llm, '_open', lambda *_args: FakeStream([completed('broken')]))
    with pytest.raises(llm.LLMError):
        client.decide('再抓一次', scene, lambda _: None)
    assert client.last_usage == {}


def test_records_chat_usage_without_exposing_extra_fields(monkeypatch, scene, decision):
    raw = (sse({'choices': [{'index': 0, 'delta': {'content': json.dumps(decision)},
                             'finish_reason': 'stop'}]})
           + sse({'choices': [], 'usage': {'prompt_tokens': 90, 'completion_tokens': 55,
                                            'total_tokens': 145, 'debug': 'secret'}})
           + b'data: [DONE]\n\n')
    install_stream(monkeypatch, FakeStream([raw]))
    client = llm.ModelClient(settings('chat_completions'))
    client.decide('抓红色', scene, lambda _: None)
    assert client.last_usage == {'input_tokens': 90, 'output_tokens': 55, 'total_tokens': 145}


def test_redirect_does_not_forward_credential():
    assert llm._NoRedirect().redirect_request(None, None, 302, '', {},
                                            'https://other.invalid') is None


def test_existing_endpoint_path_is_used_exactly_once(monkeypatch, scene, decision):
    requests = install_stream(monkeypatch, FakeStream([completed(json.dumps(decision))]))
    config = llm.LLMSettings(base_url='https://relay.invalid/v1/responses',
                             model='gpt-4.1-mini', api_key='fake-test-key')
    llm.ModelClient(config).decide('抓红色', scene, lambda _: None)
    assert requests[0][0].full_url == 'https://relay.invalid/v1/responses'
    assert json.loads(requests[0][0].data)['max_output_tokens'] == 800


def test_finite_xyz_rejects_huge_integer(scene, decision):
    scene['objects'][0]['center_base'] = [10 ** 500, 0.1, 0.03]
    with pytest.raises(ValueError):
        llm.validate_decision(decision, scene)
