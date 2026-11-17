"""Whole-plan validation and streaming for bounded, sequential batch picks."""
from copy import deepcopy
import json

import pytest

from cube_task.assistant import llm
from test_assistant_llm import FakeStream, completed, install_stream, settings, sse


@pytest.fixture
def batch_scene():
    labels = ['red', 'green', 'yellow', 'apple', 'banana', 'pear']
    return {
        'scene_version': 12,
        'objects': [{'object_id': 'v12-object-%d' % i, 'label': label,
                     'score': .97, 'valid': True, 'center_base': [.20 + .04 * i, 0., .05]}
                    for i, label in enumerate(labels, 1)],
        'zones': [{'name': 'zone_red'}, {'name': 'zone_green'}, {'name': 'zone_yellow'}],
        'class_to_zone': dict(zip(labels, ['zone_red', 'zone_green', 'zone_yellow',
                                          'zone_red', 'zone_yellow', 'zone_green'])),
    }


@pytest.fixture
def batch_decision():
    return {'message': '先将苹果放到绿区，再将香蕉放入默认区域。',
            'scene_version': 12, 'action': 'pick_batch',
            'items': [{'object_id': 'v12-object-4', 'zone_name': 'zone_green'},
                      {'object_id': 'v12-object-5', 'zone_name': None}]}


def test_batch_normalizes_every_destination_preserves_order_and_inputs(batch_scene, batch_decision):
    before_scene, before_decision = deepcopy(batch_scene), deepcopy(batch_decision)
    result = llm.validate_decision(batch_decision, batch_scene)
    assert result['items'] == [
        {'object_id': 'v12-object-4', 'zone_name': 'zone_green'},
        {'object_id': 'v12-object-5', 'zone_name': 'zone_yellow'}]
    assert batch_scene == before_scene and batch_decision == before_decision
    result['items'][0]['zone_name'] = 'zone_red'
    assert batch_decision == before_decision


def test_batch_allows_distinct_instances_of_the_same_class_and_items_already_in_zone(
        batch_scene, batch_decision):
    batch_scene['objects'][4].update(label='apple', in_zone=True, zone='zone_red')
    result = llm.validate_decision(batch_decision, batch_scene)
    assert len(result['items']) == 2
    assert result['items'][1]['zone_name'] == 'zone_red'


@pytest.mark.parametrize('items', [None, {}, (), [], [{}], [{}] * 7, 'all'])
def test_batch_size_and_container_are_bounded(batch_scene, batch_decision, items):
    batch_decision['items'] = items
    with pytest.raises(ValueError):
        llm.validate_decision(batch_decision, batch_scene)


@pytest.mark.parametrize('item', [None, 'v12-object-5', {},
    {'object_id': 'v12-object-5'}, {'zone_name': None},
    {'object_id': 'v12-object-5', 'zone_name': None, 'center_base': [1, 2, 3]},
    {'object_id': 'v12-object-5', 'zone_name': None, 'command': 'move'},
    {'object_id': 'invented', 'zone_name': None},
    {'object_id': None, 'zone_name': None},
    {'object_id': ['v12-object-5'], 'zone_name': None},
    {'object_id': 'v12-object-5', 'zone_name': 'invented-zone'},
    {'object_id': 'v12-object-5', 'zone_name': ['zone_red']}])
def test_an_invalid_later_item_rejects_the_whole_plan_without_modifying_inputs(
        batch_scene, batch_decision, item):
    batch_decision['items'][1] = item
    before = deepcopy(batch_decision)
    with pytest.raises(ValueError):
        llm.validate_decision(batch_decision, batch_scene)
    assert batch_decision == before


@pytest.mark.parametrize('zone', [None, 'zone_red', 'zone_green'])
def test_repeated_object_is_rejected_even_with_another_destination(
        batch_scene, batch_decision, zone):
    batch_decision['items'][1] = {'object_id': 'v12-object-4', 'zone_name': zone}
    with pytest.raises(ValueError, match='重复'):
        llm.validate_decision(batch_decision, batch_scene)


@pytest.mark.parametrize('change', ['invalid_target', 'bad_geometry', 'duplicate_id',
    'unmapped_label', 'invalid_default_zone', 'invalid_explicit_zone', 'invalid_scene'])
def test_every_batch_item_uses_all_existing_single_pick_gates(
        batch_scene, batch_decision, change):
    target = batch_scene['objects'][4]
    if change == 'invalid_target':
        target['valid'] = False
    elif change == 'bad_geometry':
        target['center_base'] = [float('nan'), 0, .04]
    elif change == 'duplicate_id':
        batch_scene['objects'].append(deepcopy(target))
    elif change == 'unmapped_label':
        batch_scene['class_to_zone'].pop('banana')
    elif change == 'invalid_default_zone':
        batch_scene['class_to_zone']['banana'] = 'missing'
    elif change == 'invalid_explicit_zone':
        batch_scene['zones'][1]['valid'] = False
    else:
        batch_scene['executable'] = False
    with pytest.raises(ValueError):
        llm.validate_decision(batch_decision, batch_scene)


@pytest.mark.parametrize('field,value', [('scene_version', 11), ('scene_version', True),
    ('message', ''), ('message', 'x' * 501), ('object_id', None), ('zone_name', None),
    ('command', 'move'), ('center_base', [1, 2, 3])])
def test_batch_rejects_stale_or_extra_top_level_fields(batch_scene, batch_decision, field, value):
    batch_decision[field] = value
    with pytest.raises(ValueError):
        llm.validate_decision(batch_decision, batch_scene)


@pytest.mark.parametrize('style', ['responses', 'chat_completions'])
def test_six_item_batch_streams_once_over_both_apis(monkeypatch, batch_scene, style):
    decision = {'message': '逐件分拣全部六个物品。', 'scene_version': 12,
                'action': 'pick_batch', 'items': [
                    {'object_id': obj['object_id'], 'zone_name': None}
                    for obj in batch_scene['objects']]}
    text = json.dumps(decision, ensure_ascii=False)
    # Split inside the first item, its string, and the final array terminator.
    parts = [text[:85], text[85:100], text[100:-2], text[-2:]]
    if style == 'responses':
        raw = b''.join(sse({'type': 'response.output_text.delta', 'delta': part})
                       for part in parts) + completed(text)
    else:
        raw = b''.join(sse({'choices': [{'index': 0, 'delta': {'content': part}}]})
                       for part in parts)
        raw += (sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
                + b'data: [DONE]\n\n')
    stream = FakeStream([raw[i:i + 3] for i in range(0, len(raw), 3)])
    requests = install_stream(monkeypatch, stream)
    deltas = []
    result = llm.ModelClient(settings(style)).decide('把全部物品逐个分拣', batch_scene, deltas.append)
    assert result['action'] == 'pick_batch' and len(result['items']) == 6
    assert [item['object_id'] for item in result['items']] == [
        obj['object_id'] for obj in batch_scene['objects']]
    assert [item['zone_name'] for item in result['items']] == [
        batch_scene['class_to_zone'][obj['label']] for obj in batch_scene['objects']]
    assert deltas == parts and stream.closed and len(requests) == 1


@pytest.mark.parametrize('bad_item', [
    '{"object_id":"v12-object-5","object_id":"v12-object-6","zone_name":null}',
    '{"object_id":"v12-object-5","zone_name":NaN}',
])
def test_nested_duplicate_or_nonfinite_batch_json_is_rejected(monkeypatch, batch_scene, bad_item):
    text = ('{"message":"顺序抓取","scene_version":12,"action":"pick_batch",'
            '"items":[{"object_id":"v12-object-4","zone_name":null},' + bad_item + ']}')
    install_stream(monkeypatch, FakeStream([completed(text)]))
    with pytest.raises(llm.LLMError) as exc:
        llm.ModelClient(settings()).decide('先苹果再香蕉', batch_scene, lambda _: None)
    assert exc.value.code == 'invalid_output'


@pytest.mark.parametrize('style', ['responses', 'chat_completions'])
def test_partial_batch_stream_never_returns_a_plan(monkeypatch, batch_scene, batch_decision, style):
    text = json.dumps(batch_decision, ensure_ascii=False)
    if style == 'responses':
        raw = sse({'type': 'response.output_text.delta', 'delta': text[:-5]})
        raw += sse({'type': 'response.incomplete'})
    else:
        raw = sse({'choices': [{'index': 0, 'delta': {'content': text[:-5]}}]})
        raw += sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]})
    requests = install_stream(monkeypatch, FakeStream([raw]))
    with pytest.raises(llm.LLMError) as exc:
        llm.ModelClient(settings(style)).decide('先苹果再香蕉', batch_scene, lambda _: None)
    assert exc.value.code == 'incomplete' and len(requests) == 1


@pytest.mark.parametrize('action', ['answer', 'clarify', 'target_not_observed'])
def test_multi_object_question_and_unconfirmed_requests_keep_nonmotion_format(
        monkeypatch, batch_scene, action):
    answer = {'scene_version': 12, 'action': action, 'object_id': None,
              'zone_name': None, 'message': '请核对苹果和香蕉的状态。'}
    requests = install_stream(monkeypatch, FakeStream([completed(json.dumps(answer))]))
    result = llm.ModelClient(settings()).decide('苹果和香蕉在哪里？', batch_scene, lambda _: None)
    assert result == answer and len(requests) == 1
    prompt = json.loads(requests[0][0].data)['instructions']
    assert 'pick_batch' in prompt and '不得偷偷只执行可用子集' in prompt
    assert '问题严禁执行抓取' in prompt and '不得猜测' in prompt
