"""Recovery choices remain within locally offered, current, cancellable actions."""
from copy import deepcopy
import json
import threading
import urllib.error

import pytest

from cube_task.assistant import llm
from cube_task.assistant.recovery import model_recovery_context, validate_recovery_decision


@pytest.fixture
def context():
    return {
        'scene_version': 15, 'attempt': 1, 'max_attempts': 3,
        'goal': {'label': 'yellow', 'zone_name': 'zone_red'},
        'failure': {'stage': 'release', 'error': 'holding_object',
                    'holding': True, 'fault': True, 'aborted': False},
        'observations': [{'object_id': 'v14-red', 'label': 'red', 'valid': True,
                          'center_base': [.17, -.35, .05], 'in_zone': True, 'zone': 'zone_red'}],
        'options': [{'id': 'place-interior', 'description': '保持夹持，改用区域内部的可达位置放置',
                     'reachable': True, 'edge_margin': .02},
                    {'id': 'put-back', 'description': '安全放回原处后重新观察'}],
        'history': [],
    }


@pytest.fixture
def decision():
    return {'scene_version': 15, 'action': 'recover', 'option_id': 'place-interior',
            'message': '放置点接近工作空间边界，保持夹持并改用区域内部的可达位置。'}


class Stream:
    def __init__(self, content):
        self.chunks = iter(content)
        self.closed = False

    def read(self, _size):
        return next(self.chunks, b'')

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True


def sse(item):
    return ('data: ' + json.dumps(item, ensure_ascii=False) + '\n\n').encode('utf-8')


def client(style='responses'):
    return llm.ModelClient(llm.LLMSettings(
        base_url='https://relay.invalid/v1', model='gpt-6.1-sol',
        api_key='test-secret', api_style=style))


@pytest.mark.parametrize('style', ['responses', 'chat_completions'])
def test_actual_recovery_payload_and_stream_reuse_safe_transport(
        monkeypatch, context, decision, style):
    context['private_world'] = 'must-not-leave-process'
    context['failure']['provider_key'] = 'must-not-leave-process'
    context['observations'][0]['truth_object'] = 'must-not-leave-process'
    context['options'][0]['private_planner'] = 'must-not-leave-process'
    text = json.dumps(decision, ensure_ascii=False)
    a, b = text[:65], text[65:]
    if style == 'responses':
        chunks = [sse({'type': 'response.output_text.delta', 'delta': a}),
                  sse({'type': 'response.output_text.delta', 'delta': b}),
                  sse({'type': 'response.completed', 'response': {'status': 'completed'}})]
    else:
        chunks = [sse({'choices': [{'index': 0, 'delta': {'content': a}}]}),
                  sse({'choices': [{'index': 0, 'delta': {'content': b}}]}),
                  sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}),
                  b'data: [DONE]\n\n']
    stream, requests = Stream(chunks), []

    def fake_open(request, _timeout):
        requests.append(request)
        return stream

    monkeypatch.setattr(llm, '_open', fake_open)
    deltas = []
    result = client(style).recover('抓取黄色方块到红色放置区里', context, deltas.append)
    assert result == decision and deltas == [a, b] and stream.closed
    assert len(requests) == 1
    body = json.loads(requests[0].data)
    user = json.loads(body['input'] if style == 'responses' else body['messages'][1]['content'])
    prompt = body['instructions'] if style == 'responses' else body['messages'][0]['content']
    assert user['instruction'] == '抓取黄色方块到红色放置区里'
    assert user['recovery']['goal'] == {'label': 'yellow', 'zone_name': 'zone_red'}
    assert user['recovery']['failure']['holding'] is True
    assert user['recovery']['observations'][0]['zone'] == 'zone_red'
    assert {o['id'] for o in user['recovery']['options']} == {'place-interior', 'put-back'}
    assert '不能换物体、换放置区' in prompt and 'holding=true' in prompt
    assert 'must-not-leave-process' not in requests[0].data.decode()
    assert 'test-secret' not in requests[0].data.decode()


@pytest.mark.parametrize('field,value', [
    ('scene_version', 14), ('scene_version', True), ('action', 'pick'),
    ('action', 'execute_code'), ('option_id', 'invented'), ('option_id', None),
    ('option_id', {'id': 'place-interior'}), ('message', ''),
    ('message', 'x' * 501), ('position', [.1, .2, .3]),
    ('zone_name', 'zone_green'), ('object_id', 'different-object'),
])
def test_recovery_rejects_stale_invented_or_free_form_execution(context, decision, field, value):
    decision[field] = value
    with pytest.raises(ValueError):
        validate_recovery_decision(decision, context)


def test_stop_requires_no_operation_and_empty_options_can_only_stop(context, decision):
    context['options'] = []
    with pytest.raises(ValueError):
        validate_recovery_decision(decision, context)
    stop = dict(decision, action='stop', option_id=None, message='没有可用的恢复方案，保持当前状态。')
    assert validate_recovery_decision(stop, context) == stop
    with pytest.raises(ValueError):
        validate_recovery_decision(dict(stop, option_id='put-back'), context)


def test_cancelled_failure_cannot_restart_motion(context, decision):
    context['failure']['aborted'] = True
    with pytest.raises(ValueError, match='已取消'):
        validate_recovery_decision(decision, context)


@pytest.mark.parametrize('change', ['duplicate', 'bad_attempt', 'too_many_attempts', 'bad_goal'])
def test_ambiguous_or_unbounded_context_is_rejected(context, change):
    if change == 'duplicate':
        context['options'].append(deepcopy(context['options'][0]))
    elif change == 'bad_attempt':
        context['attempt'] = True
    elif change == 'too_many_attempts':
        context.update(attempt=4, max_attempts=4)
    else:
        context['goal']['zone_name'] = None
    with pytest.raises(ValueError):
        model_recovery_context(context)


def test_context_is_detached_and_omits_nonfinite_metrics(context):
    context['options'][0]['clearance'] = float('inf')
    context['options'][0]['distance'] = 10 ** 1000
    visible = model_recovery_context(context)
    assert 'clearance' not in visible['options'][0] and 'distance' not in visible['options'][0]
    visible['observations'][0]['center_base'][0] = 999
    visible['options'].clear()
    assert context['observations'][0]['center_base'][0] == .17
    assert len(context['options']) == 2


def test_stationary_observation_limits_and_history_remain_visible(context):
    context['observation_mode'] = 'stationary_with_last_scan'
    context['failure']['diagnostics'] = '上一放置点位于工作空间边界。'
    context['history'] = [{'attempt': 1, 'action': 'place_held_alternate',
                           'error': 'rejected', 'ok': False}]
    visible = model_recovery_context(context)
    assert visible['observation_mode'] == 'stationary_with_last_scan'
    assert visible['failure']['diagnostics'] == '上一放置点位于工作空间边界。'
    assert visible['history'][0]['action'] == 'place_held_alternate'
    context['observation_mode'] = 'ground_truth'
    with pytest.raises(ValueError):
        model_recovery_context(context)


def test_unavailable_observation_cannot_offer_motion(context, decision):
    context['observation_mode'] = 'unavailable'
    with pytest.raises(ValueError):
        model_recovery_context(context)
    context['options'] = []
    stop = dict(decision, action='stop', option_id=None, message='当前无可用观测，保持夹持并停止恢复。')
    assert validate_recovery_decision(stop, context) == stop


def test_recovery_cancel_before_request_does_not_spend_api_call(monkeypatch, context):
    event = threading.Event()
    event.set()
    calls = []
    monkeypatch.setattr(llm, '_open', lambda *args: calls.append(args))
    with pytest.raises(llm.LLMError) as exc:
        client().recover('继续原来的目标', context, lambda _: None, event)
    assert exc.value.code == 'cancelled' and not calls


def test_recovery_remote_failure_is_redacted_without_retry(monkeypatch, context):
    calls = []

    def fail(*args):
        calls.append(args)
        raise urllib.error.HTTPError('https://relay.invalid', 500, 'private-key-body', {}, None)

    monkeypatch.setattr(llm, '_open', fail)
    with pytest.raises(llm.LLMError) as exc:
        client().recover('继续原来的目标', context, lambda _: None)
    assert exc.value.code == 'provider_error' and 'private-key-body' not in str(exc.value)
    assert len(calls) == 1
