"""Streaming skill planning uses fake network responses and no billed calls."""
import json
import threading
import urllib.error

import pytest

from cube_task.assistant import llm
from test_assistant_llm import FakeStream, completed, install_stream, settings, sse
from test_assistant_skill_protocol import context, plan, step


@pytest.mark.parametrize('style', ['responses', 'chat_completions'])
def test_both_streaming_apis_receive_composable_skills_and_preserved_goals(monkeypatch, context, style):
    context['scene']['objects'][0]['truth'] = 'world-sentinel'
    context['options'] = [{'id': 'forbidden-macro-sentinel'}]
    raw_plan = plan(context, [step(1, 'move', 'banana', {'type': 'auto_free'}),
                              step(2, 'move', 'apple', {'type': 'goal', 'index': 0}),
                              step(3, 'move', 'banana', {'type': 'original', 'label': 'banana'}),
                              step(4, 'verify')])
    text = json.dumps(raw_plan, ensure_ascii=False)
    first, second = text[:80], text[80:]
    if style == 'responses':
        raw = (sse({'type': 'response.reasoning_summary_text.delta', 'delta': 'private-thought'})
               + sse({'type': 'response.output_text.delta', 'delta': first})
               + sse({'type': 'response.output_text.delta', 'delta': second}) + completed(text))
    else:
        raw = (sse({'choices': [{'index': 0, 'delta': {'reasoning_content': 'private-thought'}}]})
               + sse({'choices': [{'index': 0, 'delta': {'content': first}}]})
               + sse({'choices': [{'index': 0, 'delta': {'content': second}}]})
               + sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
               + b'data: [DONE]\n\n')
    stream = FakeStream([raw[i:i + 7] for i in range(0, len(raw), 7)])
    requests = install_stream(monkeypatch, stream)
    deltas = []
    result = llm.ModelClient(settings(style)).plan_skills('把苹果放进红区，受阻时自行组合技能', context, deltas.append)
    assert result == raw_plan and deltas == [first, second] and stream.closed
    assert len(requests) == 1
    body = json.loads(requests[0][0].data)
    visible = json.loads(body['input'] if style == 'responses' else body['messages'][1]['content'])
    prompt = body['instructions'] if style == 'responses' else body['messages'][0]['content']
    assert visible['context']['goals'] == context['goals']
    assert visible['context']['originals'][0]['length'] == .04
    assert visible['context']['images_provided'] is False
    assert '自行组合' in prompt and '逐步执行反馈' in prompt
    assert all(s not in requests[0][0].data.decode() for s in
               ('world-sentinel', 'forbidden-macro-sentinel', 'test-secret-only'))
    assert 'private-thought' not in ''.join(deltas)


def test_cancelled_plan_does_not_make_any_network_request(monkeypatch, context):
    event = threading.Event()
    event.set()
    calls = []
    monkeypatch.setattr(llm, '_open', lambda *args: calls.append(args))
    with pytest.raises(llm.LLMError) as error:
        llm.ModelClient(settings()).plan_skills('继续原任务', context, lambda _: None, event)
    assert error.value.code == 'cancelled' and not calls


@pytest.mark.parametrize('response', [
    sse({'type': 'response.output_text.delta', 'delta': '{}'}),
    completed('{"action":"plan","action":"stop"}'),
    completed('{"scene_version":NaN}'),
])
def test_incomplete_or_malformed_plans_never_return_an_executable_result(monkeypatch, context, response):
    calls = install_stream(monkeypatch, FakeStream([response]))
    with pytest.raises(llm.LLMError):
        llm.ModelClient(settings()).plan_skills('继续原任务', context, lambda _: None)
    assert len(calls) == 1


def test_model_cannot_change_final_goal_while_replanning_after_failure(monkeypatch, context):
    context['failure'] = {'stage': 'place', 'error': 'rejected', 'holding': True}
    context.update(holding=True, held_label='apple')
    result = plan(context, [step(1, 'place', 'apple', {'type': 'goal', 'index': 0})])
    result['goals'][0]['zone_name'] = 'zone_green'
    calls = install_stream(monkeypatch, FakeStream([completed(json.dumps(result))]))
    with pytest.raises(ValueError, match='最终目标'):
        llm.ModelClient(settings()).plan_skills('继续原任务', context, lambda _: None)
    assert len(calls) == 1


def test_remote_error_is_redacted_and_has_no_automatic_billing_retry(monkeypatch, context):
    calls = []

    def fail(*args):
        calls.append(args)
        raise urllib.error.HTTPError('https://relay.invalid', 500, 'remote-secret-body', {}, None)

    monkeypatch.setattr(llm, '_open', fail)
    with pytest.raises(llm.LLMError) as error:
        llm.ModelClient(settings()).plan_skills('继续原任务', context, lambda _: None)
    assert error.value.code == 'provider_error'
    assert 'remote-secret-body' not in str(error.value) and len(calls) == 1


@pytest.mark.parametrize('style', ['responses', 'chat_completions'])
def test_streamed_replan_can_move_valid_banana_beside_an_invalid_cropped_fragment(monkeypatch, context, style):
    context['scene']['objects'].insert(0, {
        'object_id': 'v4-banana-fragment', 'label': 'banana', 'valid': False,
        'reject_reason': 'touches_image_border', 'center_base': [.39, .05, .06],
    })
    context['failure'] = {'stage': 'place', 'error': 'placement_not_verified', 'holding': False}
    decision = plan(context, [step(1, 'move', 'banana', {'type': 'auto_free'})])
    text = json.dumps(decision, ensure_ascii=False)
    raw = (completed(text) if style == 'responses' else
           sse({'choices': [{'index': 0, 'delta': {'content': text}}]})
           + sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
           + b'data: [DONE]\n\n')
    requests = install_stream(monkeypatch, FakeStream([raw]))
    result = llm.ModelClient(settings(style)).plan_skills('继续完成原任务', context, lambda _: None)
    assert result == decision and len(requests) == 1
    body = json.loads(requests[0][0].data)
    visible = json.loads(body['input'] if style == 'responses' else body['messages'][1]['content'])
    bananas = [o for o in visible['context']['scene']['objects'] if o['label'] == 'banana']
    assert [o['valid'] for o in bananas] == [False, True]
    assert bananas[0]['reject_reason'] == 'touches_image_border'
