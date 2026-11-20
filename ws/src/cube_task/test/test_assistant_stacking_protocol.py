"""Semantic support references cannot become guessed coordinates or 2-D success."""
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest

from cube_task.assistant.destinations import (
    destination_label, validate_destination, whole_footprint_at_destination,
)
from cube_task.assistant.skill_orchestrator import SkillOrchestrator
from cube_task.assistant.skill_protocol import model_skill_context, validate_skill_plan
from test_assistant_skill_protocol import context, plan, step


@pytest.fixture
def stacking(context):
    scene = context['scene']
    source = deepcopy(scene['objects'][0])
    source.update(object_id='v4-red', label='red', center_base=[.3, -.1, .025], height=.05)
    scene['objects'].append(source)
    scene['class_to_zone']['red'] = 'zone_red'
    context['originals'] = deepcopy(scene['objects'])
    context['goals'] = [dict(label='apple', zone_name=None,
        destination={'type': 'on_object', 'label': 'red'})]
    return context


def test_stacking_uses_semantic_support_in_goal_move_and_observe(stacking):
    raw = plan(stacking, [step(1, 'observe', destination={'type': 'on_object', 'label': 'red'}),
        step(2, 'move', 'apple', {'type': 'on_object', 'label': 'red'}), step(3, 'finish')])
    assert validate_skill_plan(raw, stacking) == raw
    assert validate_destination({'type': 'on_object', 'label': 'red'}, stacking['scene']) == raw['goals'][0]['destination']
    assert destination_label(raw['goals'][0]['destination']) == '红色方块顶面'
    assert SkillOrchestrator._final_destination(raw['steps'][1], raw['goals'])
    assert '红色方块顶面' in SkillOrchestrator._describe(raw['steps'][1])


@pytest.mark.parametrize('value', [
    {'type': 'on_object', 'label': 'red', 'xy': [.3, -.1]},
    {'type': 'on_object', 'label': 'red', 'height': .05},
    {'type': 'on_object', 'label': 'zone_red'},
    {'type': 'on_object', 'label': []},
])
def test_support_reference_has_no_geometry_and_cannot_mean_a_zone(stacking, value):
    with pytest.raises(ValueError):
        validate_destination(value, stacking['scene'])


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'invalid', 'nonfinite'])
def test_new_stack_actions_require_current_unique_measured_support(stacking, fault):
    support = stacking['scene']['objects'][-1]
    if fault == 'missing':
        stacking['scene']['objects'].pop()
    elif fault == 'duplicate':
        stacking['scene']['objects'].append(deepcopy(support))
    elif fault == 'invalid':
        support['valid'] = False
    else:
        support['center_base'][0] = float('nan')
    raw = plan(stacking, [step(1, 'move', 'apple', {'type': 'goal', 'index': 0})])
    with pytest.raises(ValueError, match='支撑'):
        validate_skill_plan(raw, stacking)


def test_fixed_stack_goal_can_survive_occlusion_for_observe_and_feedback(stacking):
    stacking['scene']['objects'].pop()
    visible = model_skill_context(stacking)
    assert visible['goals'] == stacking['goals']
    raw = plan(stacking, [step(1, 'observe', destination={'type': 'on_object', 'label': 'red'})])
    assert validate_skill_plan(raw, stacking) == raw
    clarified = plan(stacking, action='clarify', message='底座尚未重新确认', steps=[])
    assert validate_skill_plan(clarified, stacking)['action'] == 'clarify'


def test_new_goal_cannot_rely_only_on_unseen_configured_support(stacking):
    goals = deepcopy(stacking['goals'])
    stacking['goals'] = []
    stacking['scene']['objects'].pop()
    with pytest.raises(ValueError, match='支撑'):
        validate_skill_plan(plan(stacking, [step(1, 'observe')], goals=goals), stacking)


def test_partial_camera_support_relation_allows_verify_goal_but_never_a_blind_move(stacking):
    goals = deepcopy(stacking['goals'])
    stacking['goals'] = []
    stacking['scene']['objects'][-1].update(valid=False, reject_reason='occluded_by_stack')
    upper = stacking['scene']['objects'][0]
    upper.update(support_label='red', support_z=.05, support_observed=True,
                 support_polygon=[[.27, -.12], [.33, -.12], [.33, -.08], [.27, -.08]],
                 truth={'secret': 'world-sentinel'})
    raw = plan(stacking, [step(1, 'verify'), step(2, 'finish')], goals=goals)
    assert validate_skill_plan(raw, stacking) == raw
    visible = model_skill_context(stacking)['scene']['objects'][0]
    assert visible['support_label'] == 'red' and visible['support_z'] == .05
    assert visible['support_observed'] and visible['support_polygon'] == upper['support_polygon']
    assert 'truth' not in visible
    raw['steps'] = [step(1, 'move', 'apple', {'type': 'goal', 'index': 0})]
    with pytest.raises(ValueError, match='支撑'):
        validate_skill_plan(raw, stacking)


@pytest.mark.parametrize('polygon', [
    [[0, 0]] * 65, [[float('nan'), 0], [0, 1], [1, 0]],
    [[0, 0, 0], [0, 1, 0], [1, 0, 0]], 'private-payload',
])
def test_invalid_support_patch_is_not_forwarded_as_a_confirmed_relation(stacking, polygon):
    upper = stacking['scene']['objects'][0]
    upper.update(support_label='red', support_z=.05, support_observed=True, support_polygon=polygon)
    visible = model_skill_context(stacking)['scene']['objects'][0]
    assert 'support_polygon' not in visible and visible['support_observed'] is False


@pytest.mark.parametrize('kind', ['goal', 'direct_step'])
def test_a_target_cannot_be_its_own_support(stacking, kind):
    if kind == 'goal':
        stacking['goals'][0]['destination']['label'] = 'apple'
    destination = {'type': 'on_object', 'label': 'apple'} if kind == 'direct_step' else {'type': 'goal', 'index': 0}
    with pytest.raises(ValueError, match='自身|自己'):
        validate_skill_plan(plan(stacking, [step(1, 'move', 'apple', destination)]), stacking)


def test_final_support_cycles_are_rejected_but_a_rooted_chain_is_allowed(stacking):
    stacking['goals'].append(dict(label='red', zone_name=None,
        destination={'type': 'on_object', 'label': 'apple'}))
    with pytest.raises(ValueError, match='循环'):
        model_skill_context(stacking)
    stacking['goals'][-1] = dict(label='red', zone_name='zone_red')
    assert model_skill_context(stacking)['goals'] == stacking['goals']


def test_old_table_footprint_check_cannot_prove_stacking(stacking):
    target = stacking['scene']['objects'][0]
    assert not whole_footprint_at_destination(target, stacking['scene'],
        {'type': 'on_object', 'label': 'red'})


def test_stack_proofs_are_not_reused_when_only_an_edit_revision_is_known():
    session = SimpleNamespace(_environment_revision=4, _interventions=[],
        snapshot=lambda: dict(scene_version=4, held_object=None, editor={}),
        runner=SimpleNamespace(has_held_recovery=False), cancel_event=Event())
    orchestrator = SkillOrchestrator(SimpleNamespace(_session=session, _cancel=Event()))
    proof = dict(goals=[dict(label='apple', zone_name=None,
        destination={'type': 'on_object', 'label': 'red'})],
        result={'ok': True, 'verified_revision': 4})
    assert orchestrator._cached_verification(proof, proof['goals']) is None
