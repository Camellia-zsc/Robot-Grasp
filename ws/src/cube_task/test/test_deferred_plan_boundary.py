"""A pending final-goal reference does not authorize an unobserved grasp."""
import copy

import pytest

from cube_task.assistant.skill_protocol import validate_skill_plan
from test_assistant_skill_protocol import context, plan, step
from test_assistant_target_deferral import executor


def mixed(context, observation):
    context['goals'] = [dict(label='pear', zone_name='zone_green'),
                        dict(label='apple', zone_name='zone_red')]
    for item in context['scene']['objects']:
        item['top_z'] = item['center_base'][2]
    source = next(o for o in context['scene']['objects'] if o['label'] == 'pear')
    if observation == 'missing':
        context['scene']['objects'].remove(source)
    elif observation == 'invalid':
        source['valid'] = False
    elif observation == 'ambiguous':
        context['scene']['objects'].append(copy.deepcopy(source))
    elif observation == 'incomplete':
        source['center_base'] = [float('nan'), .1, .05]
    return plan(context, [step('wait-pear', 'move', 'pear', dict(type='goal', index=0)),
                          step('do-apple', 'move', 'apple', dict(type='goal', index=1)),
                          step('done', 'finish')])


@pytest.mark.parametrize('observation', ['missing', 'invalid', 'ambiguous', 'incomplete'])
def test_final_move_reference_survives_validation_but_executor_defers_before_any_grasp(context, observation):
    proposal = mixed(context, observation)
    normalized = validate_skill_plan(proposal, context)
    assert normalized == proposal
    ex, _goals, _step = executor()
    ex.session.scene = copy.deepcopy(context['scene'])
    deferred = ex.execute(normalized['steps'][0], normalized['goals'])
    assert deferred['ok'] is False and deferred['deferred'] is True
    assert deferred['manipulation_started'] is False and deferred['holding'] is False
    ex._move.assert_not_called()
    result = ex.execute(normalized['steps'][1], normalized['goals'])
    assert result['ok']
    ex._move.assert_called_once()
    assert ex._move.call_args.args[0]['label'] == 'apple'


@pytest.mark.parametrize('skill,destination', [('pick', None), ('place', dict(type='goal', index=0)),
    ('move', dict(type='auto_free')), ('move', dict(type='original', label='pear')),
    ('move', dict(type='zone', name='zone_green')),
    ('move', dict(type='goal', index=1)), ('move', dict(type='goal', index=True)),
    ('move', dict(type='goal', index=0, xy=[.3, .1]))])
def test_unobserved_nonfinal_or_malformed_action_keeps_strict_identity_gate(context, skill, destination):
    mixed(context, 'missing')
    with pytest.raises(ValueError):
        validate_skill_plan(plan(context, [step(1, skill, 'pear', destination)]), context)


def test_unknown_category_cannot_be_smuggled_in_as_a_fixed_goal(context):
    context['goals'] = [dict(label='invented', zone_name='zone_red')]
    with pytest.raises(ValueError):
        validate_skill_plan(plan(context, [step(1, 'move', 'invented', dict(type='goal', index=0))]), context)
