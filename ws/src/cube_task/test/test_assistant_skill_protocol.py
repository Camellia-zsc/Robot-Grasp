"""Offline checks for composed skills, fixed goals and observed identities."""
from copy import deepcopy

import pytest

from cube_task.assistant.skill_protocol import model_skill_context, validate_skill_plan


@pytest.fixture
def context():
    objects = [
        {'object_id': 'v4-' + label, 'label': label, 'valid': True,
         'center_base': [.25 + index * .06, .04, .06],
         'length': .04, 'width': .03, 'height': .04, 'yaw': 0.,
         'bbox': [10, 20, 30, 40], 'n_points': 450, 'score_kind': 'knn', 'score': .95}
        for index, label in enumerate(('apple', 'banana', 'pear'))]
    return {
        'scene': {
            'scene_version': 4, 'objects': objects,
            'zones': [{'name': 'zone_red'}, {'name': 'zone_green'}],
            'class_to_zone': {'apple': 'zone_red', 'banana': 'zone_green', 'pear': 'zone_green'},
            'placement': {
                'bounds': {'x': [.08, .60], 'y': [-.40, .40], 'r_min': .10, 'r_max': .62},
                'table_bounds': {'x': [.08, .60], 'y': [-.40, .40]},
                'pick_polygon': [[.15, -.22], [.45, -.22], [.45, .22], [.15, .22]],
            },
        },
        'goals': [{'label': 'apple', 'zone_name': 'zone_red'}],
        'originals': deepcopy(objects), 'holding': False, 'held_label': None,
    }


def plan(context, steps=None, **changes):
    result = {'scene_version': context['scene']['scene_version'], 'action': 'plan',
              'message': '先处理遮挡，再继续完成苹果的原目标。',
              'goals': deepcopy(context['goals']), 'steps': steps or []}
    result.update(changes)
    return result


def step(number, skill, label=None, destination=None, **extra):
    result = {'id': str(number), 'skill': skill, **extra}
    if label is not None:
        result['label'] = label
    if destination is not None:
        result['destination'] = destination
    return result


def test_composes_two_obstacles_repeated_targets_and_original_restoration(context):
    raw = plan(context, [
        step(1, 'move', 'banana', {'type': 'auto_free', 'away_from': ['apple', 'pear']}),
        step(2, 'move', 'pear', {'type': 'auto_free'}),
        step(3, 'observe', 'apple'),
        step(4, 'pick', 'apple'),
        step(5, 'place', 'apple', {'type': 'region', 'name': 'pick_area'}),
        step(6, 'pick', 'apple'),
        step(7, 'place', 'apple', {'type': 'goal', 'index': 0}),
        step(8, 'move', 'pear', {'type': 'original', 'label': 'pear'}),
        step(9, 'move', 'banana', {'type': 'original', 'label': 'banana'}),
        step(10, 'verify', goal_index=0), step(11, 'finish'),
    ])
    before = deepcopy(raw)
    normalized = validate_skill_plan(raw, context)
    assert normalized == raw and raw == before
    normalized['steps'][0]['destination']['away_from'].clear()
    assert raw['steps'][0]['destination']['away_from'] == ['apple', 'pear']


def test_held_target_can_be_staged_even_when_not_currently_visible(context):
    context.update(holding=True, held_label='apple')
    context['scene']['objects'] = [o for o in context['scene']['objects'] if o['label'] != 'apple']
    raw = plan(context, [step(1, 'observe', 'apple'),
                         step(2, 'place', 'apple', {'type': 'auto_free'}),
                         step(3, 'observe', 'apple')])
    assert validate_skill_plan(raw, context) == raw


@pytest.mark.parametrize('observation', ['missing', 'cropped', 'ambiguous', 'original_only'])
def test_held_staging_avoidance_is_a_semantic_preference_not_a_motion_target(context, observation):
    context.update(holding=True, held_label='apple')
    banana = next(o for o in context['scene']['objects'] if o['label'] == 'banana')
    if observation in ('missing', 'original_only'):
        context['scene']['objects'].remove(banana)
    elif observation == 'cropped':
        banana.update(valid=False, reject_reason='touches_image_border')
    else:
        context['scene']['objects'].append(deepcopy(banana))
    if observation == 'original_only':
        del context['scene']['class_to_zone']['banana']
    raw = plan(context, [
        step(1, 'ask', question='允许先暂放苹果吗？', affected_goals=[0]),
        step(2, 'place', 'apple', {'type': 'auto_free', 'away_from': ['banana']}),
    ])
    assert validate_skill_plan(raw, context) == raw


def test_avoidance_preference_does_not_allow_grasping_an_unmeasured_object(context):
    context.update(holding=True, held_label='apple')
    context['scene']['objects'] = [o for o in context['scene']['objects'] if o['label'] != 'banana']
    raw = plan(context, [
        step(1, 'place', 'apple', {'type': 'auto_free', 'away_from': ['banana']}),
        step(2, 'move', 'banana', {'type': 'auto_free'}),
    ])
    with pytest.raises(ValueError, match='唯一且有效'):
        validate_skill_plan(raw, context)


def test_avoidance_cannot_use_an_unconfigured_invalid_original_fragment(context):
    context['originals'].append({'label': 'invented', 'valid': False, 'center_base': [.3, .1, .04]})
    raw = plan(context, [step(1, 'move', 'apple', {'type': 'auto_free', 'away_from': ['invented']})])
    with pytest.raises(ValueError):
        validate_skill_plan(raw, context)


def test_human_question_can_pause_between_steps_and_affect_no_final_goal(context):
    raw = plan(context, [step(1, 'observe'),
                         step(2, 'ask', question='可以暂时搬开香蕉并在任务后复位吗？', affected_goals=[]),
                         step(3, 'move', 'banana', {'type': 'auto_free'})])
    assert validate_skill_plan(raw, context) == raw


@pytest.mark.parametrize('steps', [
    [step(1, 'place', 'apple', {'type': 'goal', 'index': 0})],
    [step(1, 'pick', 'apple'), step(2, 'pick', 'pear')],
    [step(1, 'pick', 'apple'), step(2, 'move', 'pear', {'type': 'auto_free'})],
    [step(1, 'pick', 'apple'), step(2, 'place', 'pear', {'type': 'auto_free'})],
    [step(1, 'pick', 'apple'), step(2, 'finish')],
    [step(1, 'finish'), step(2, 'observe')],
    [step(1, 'observe'), step(1, 'observe')],
    [step(1, 'move', 'pear', {'type': 'goal', 'index': 0})],
    [step(1, 'move', 'pear', {'type': 'original', 'label': 'apple'})],
])
def test_gripper_state_and_identity_are_checked_across_whole_plan(context, steps):
    with pytest.raises(ValueError):
        validate_skill_plan(plan(context, steps), context)


@pytest.mark.parametrize('destination', [
    {'type': 'point', 'xy': [.3, .1]}, {'type': 'goal', 'index': True},
    {'type': 'goal', 'index': 1}, {'type': 'zone', 'name': 'invented'},
    {'type': 'zone', 'name': []}, {'type': 'auto_free', 'xy': [.3, 0]},
    {'type': 'auto_free', 'away_from': ['invented']},
    {'type': 'original', 'label': 'invented'},
    {'type': 'region', 'name': 'somewhere'},
])
def test_steps_cannot_supply_geometry_or_invent_semantic_references(context, destination):
    with pytest.raises(ValueError):
        validate_skill_plan(plan(context, [step(1, 'move', 'apple', destination)]), context)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'invalid', 'nonfinite'])
def test_motion_requires_unique_valid_current_measurement(context, change):
    if change == 'missing':
        context['scene']['objects'].pop(0)
    elif change == 'duplicate':
        context['scene']['objects'].append(deepcopy(context['scene']['objects'][0]))
    elif change == 'invalid':
        context['scene']['objects'][0]['valid'] = False
    else:
        context['scene']['objects'][0]['center_base'][0] = float('nan')
    with pytest.raises(ValueError):
        validate_skill_plan(plan(context, [step(1, 'pick', 'apple')]), context)
    assert validate_skill_plan(plan(context, [step(1, 'observe', 'apple')]), context)


def test_invalid_cropped_fragment_does_not_hide_one_complete_measured_object(context):
    fragment = deepcopy(context['scene']['objects'][1])
    fragment.update(object_id='v4-banana-cropped', valid=False,
                    reject_reason='touches_image_border', center_base=[.39, .05, .06])
    context['scene']['objects'].insert(0, fragment)
    raw = plan(context, [step(1, 'move', 'banana', {'type': 'auto_free'})])
    assert validate_skill_plan(raw, context) == raw
    visible = model_skill_context(context)
    assert [o['valid'] for o in visible['scene']['objects'] if o['label'] == 'banana'] == [False, True]
    # A second complete measurement still creates real identity ambiguity.
    fragment['valid'] = True
    with pytest.raises(ValueError, match='唯一'):
        validate_skill_plan(raw, context)


@pytest.mark.parametrize('action', ['plan', 'answer', 'clarify', 'stop'])
def test_fixed_final_goals_cannot_change_even_for_nonmotion_responses(context, action):
    raw = plan(context, [step(1, 'observe')] if action == 'plan' else [], action=action)
    raw['goals'][0]['zone_name'] = 'zone_green'
    with pytest.raises(ValueError, match='最终目标'):
        validate_skill_plan(raw, context)


def test_initial_plan_uses_default_zone_but_rejects_multiple_final_destinations(context):
    context['goals'] = []
    raw = plan(context, [step(1, 'move', 'apple', {'type': 'goal', 'index': 0})],
               goals=[{'label': 'apple', 'zone_name': None}])
    assert validate_skill_plan(raw, context)['goals'] == [{'label': 'apple', 'zone_name': 'zone_red'}]
    raw['goals'].append({'label': 'apple', 'zone_name': 'zone_green'})
    with pytest.raises(ValueError, match='最终目标'):
        validate_skill_plan(raw, context)


def test_unobserved_configured_target_can_be_observed_without_inventing_category(context):
    context['goals'] = []
    context['scene']['objects'].pop(0)
    raw = plan(context, [step(1, 'observe', 'apple')],
               goals=[{'label': 'apple', 'zone_name': 'zone_red'}])
    assert validate_skill_plan(raw, context) == raw
    raw['steps'][0]['label'] = 'invented'
    with pytest.raises(ValueError):
        validate_skill_plan(raw, context)


@pytest.mark.parametrize('change', ['stale', 'bool_version', 'cancelled', 'budget', 'extra', 'unknown_skill'])
def test_execution_gates_reject_invalid_plans(context, change):
    raw = plan(context, [step(1, 'observe'), step(2, 'observe')])
    if change == 'stale':
        raw['scene_version'] = 3
    elif change == 'bool_version':
        raw['scene_version'] = True
    elif change == 'cancelled':
        context['failure'] = {'aborted': True}
    elif change == 'budget':
        context['remaining_steps'] = 1
    elif change == 'extra':
        raw['robot_code'] = 'run()'
    else:
        raw['steps'][0]['skill'] = 'clear_restore'
    with pytest.raises(ValueError):
        validate_skill_plan(raw, context)


def test_context_exports_full_measured_geometry_and_bounded_feedback_only(context):
    context['scene']['truth'] = {'private': 'world-sentinel'}
    context['scene']['objects'][0]['truth_object'] = 'world-sentinel'
    context['scene']['placement']['private_planner'] = 'world-sentinel'
    context['originals'][0]['truth'] = 'world-sentinel'
    context['failure'] = {'stage': 'place', 'error': 'rejected', 'holding': False,
                          'diagnostics': '没有满足间隙的空位。', 'truth': 'world-sentinel'}
    context['history'] = [{'id': 'previous', 'skill': 'move', 'label': 'apple', 'ok': False,
                          'message': '目标区放置未完成。', 'world': 'world-sentinel'}]
    context['options'] = [{'id': 'legacy-fixed-macro'}]
    context['protected_goals'] = [{'label': 'pear', 'zone_name': 'zone_green'}]
    visible = model_skill_context(context)
    assert 'world-sentinel' not in str(visible) and 'legacy-fixed-macro' not in str(visible)
    assert visible['scene']['objects'][0]['length'] == .04
    assert visible['scene']['objects'][0]['bbox'] == [10, 20, 30, 40]
    assert visible['failure']['diagnostics'] == context['failure']['diagnostics']
    assert visible['history'][0]['skill'] == 'move'
    assert visible['protected_goals'] == context['protected_goals']
    assert visible['perception'] == 'structured_camera_observations'
    assert visible['images_provided'] is False
    visible['originals'][0]['center_base'][0] = 999
    assert context['originals'][0]['center_base'][0] == .25


def test_exact_point_goal_must_use_goal_reference_and_remains_fixed(context):
    point = {'type': 'point', 'xy': [.30, .08]}
    context['scene']['selected_destination'] = deepcopy(point)
    context['goals'][0] = {'label': 'apple', 'zone_name': None, 'destination': deepcopy(point)}
    raw = plan(context, [step(1, 'move', 'apple', {'type': 'goal', 'index': 0})])
    assert validate_skill_plan(raw, context) == raw
    raw['goals'][0]['destination']['xy'][0] = .31
    with pytest.raises(ValueError):
        validate_skill_plan(raw, context)


def test_protected_point_goal_can_differ_from_current_user_selection(context):
    context['scene']['selected_destination'] = {'type': 'point', 'xy': [.30, .08]}
    context['protected_goals'] = [{'label': 'pear', 'zone_name': None,
                                   'destination': {'type': 'point', 'xy': [.35, -.08]}}]
    assert model_skill_context(context)['protected_goals'] == context['protected_goals']
    raw = plan(context, [step(1, 'observe')])
    assert validate_skill_plan(raw, context) == raw


def test_replanning_sees_pending_restorations_and_semantic_step_destinations(context):
    context['pending_obligations'] = ['banana', {'label': 'pear', 'truth': 'private-sentinel'}]
    context['history'] = [
        {'skill': 'move', 'label': 'banana', 'ok': True,
         'destination': {'type': 'auto_free', 'away_from': ['apple'],
                         'xy': [.3, .1], 'truth': 'private-sentinel'}},
        {'skill': 'move', 'label': 'apple', 'ok': False,
         'destination': {'type': 'goal', 'index': 0}},
    ]
    visible = model_skill_context(context)
    assert visible['pending_obligations'] == ['banana', 'pear']
    assert visible['history'][0]['destination'] == {'type': 'auto_free', 'away_from': ['apple']}
    assert visible['history'][1]['destination'] == {'type': 'goal', 'index': 0}
    assert 'private-sentinel' not in str(visible)
    visible['history'][0]['destination']['away_from'].clear()
    assert context['history'][0]['destination']['away_from'] == ['apple']


@pytest.mark.parametrize('holding', [False, True])
@pytest.mark.parametrize('destination', [
    {'type': 'goal', 'index': 0}, {'type': 'zone', 'name': 'zone_green'},
    {'type': 'region', 'name': 'pick_area'}, {'type': 'original', 'label': 'banana'},
])
def test_active_observe_can_search_semantic_destinations_with_or_without_a_held_object(context, holding, destination):
    context.update(holding=holding, held_label='apple' if holding else None)
    raw = plan(context, [step(1, 'observe', 'banana', destination, mode='active')])
    assert validate_skill_plan(raw, context) == raw
    raw['steps'][0].pop('label')
    assert validate_skill_plan(raw, context) == raw


def test_stationary_observe_and_legacy_observe_remain_compatible(context):
    raw = plan(context, [step(1, 'observe'), step(2, 'observe', mode='stationary')])
    assert validate_skill_plan(raw, context) == raw


@pytest.mark.parametrize('extra', [
    {'mode': 'teleport'}, {'mode': None}, {'mode': []},
    {'destination': {'type': 'auto_free'}},
    {'destination': {'type': 'point', 'xy': [.3, 0]}},
    {'destination': {'type': 'goal', 'index': True}},
    {'destination': {'type': 'zone', 'name': 'invented'}},
    {'destination': {'type': 'original', 'label': 'invented'}},
    {'destination': {'type': 'region', 'name': 'pick_area', 'xy': [.3, 0]}},
    {'xy': [.3, 0]}, {'trajectory': [[0, 0, 0]]},
])
def test_observe_cannot_invent_modes_destinations_or_geometry(context, extra):
    with pytest.raises(ValueError):
        validate_skill_plan(plan(context, [step(1, 'observe', **extra)]), context)


@pytest.mark.parametrize('steps', [
    [step(1, 'verify')], [step(1, 'verify', goal_index=0)],
])
def test_held_object_must_be_placed_before_any_verification(context, steps):
    context.update(holding=True, held_label='apple')
    with pytest.raises(ValueError, match='持物'):
        validate_skill_plan(plan(context, steps), context)
    context.update(holding=False, held_label=None)
    with pytest.raises(ValueError, match='持物'):
        validate_skill_plan(plan(context, [step(0, 'pick', 'apple'), *steps]), context)


def test_model_context_exposes_capabilities_budget_and_action_failure_without_private_data(context):
    context['capabilities'] = {
        'skills': ['observe', 'pick', 'place', 'move', 'verify', 'ask', 'finish'],
        'observe_modes': ['stationary', 'active'],
        'observe_destination_types': ['goal', 'zone', 'region', 'original'],
        'destination_types': ['goal', 'zone', 'region', 'original', 'auto_free'],
        'holding_allowed_skills': ['observe', 'place', 'ask'],
        'active_observe_requires_geometry': True,
        'multi_object_rearrangement': True, 'restores_original_pose': True,
        'scene': {'truth': 'private-sentinel'}, 'private': 'private-sentinel',
    }
    context['planning_budget'] = {
        'plan_number': 3, 'max_plans': 8, 'remaining_plans': 5,
        'remaining_steps': 19, 'stagnation': 2, 'private': 'private-sentinel',
    }
    context['last_action_failure'] = {
        'stage': 'place', 'error': 'placement_infeasible', 'message': '没有安全空位',
        'diagnostics': '遮挡仍在', 'holding': True, 'duration': 1.5,
        'scene': {'objects': ['private-sentinel']}, 'truth': 'private-sentinel',
    }
    context['history'] = [{
        'skill': 'observe', 'diagnostics': '新视角仍遮挡', 'observation_mode': 'active',
        'changed': False, 'reason': 'no_information_gain',
        'scene': {'truth': 'private-sentinel'}, 'private': 'private-sentinel',
    }]
    visible = model_skill_context(context)
    assert visible['capabilities'] == {k: v for k, v in context['capabilities'].items()
                                       if k not in ('scene', 'private')}
    assert visible['planning_budget'] == {k: v for k, v in context['planning_budget'].items() if k != 'private'}
    assert visible['last_action_failure']['error'] == 'placement_infeasible'
    assert visible['last_action_failure']['duration'] == 1.5
    assert visible['history'][0]['changed'] is False
    assert visible['history'][0]['reason'] == 'no_information_gain'
    assert visible['history'][0]['observation_mode'] == 'active'
    assert 'private-sentinel' not in str(visible)
    visible['capabilities']['skills'].clear()
    assert context['capabilities']['skills'][0] == 'observe'


@pytest.mark.parametrize('value', [True, False, -1, 1.5, float('inf'), float('nan'), '3', None])
def test_planning_budget_does_not_coerce_nonfinite_or_noninteger_values(context, value):
    context['planning_budget'] = {key: value for key in
                                  ('plan_number', 'max_plans', 'remaining_plans', 'remaining_steps', 'stagnation')}
    assert model_skill_context(context)['planning_budget'] == {}


@pytest.mark.parametrize('detail', [None, {'truth': 'private-detail-sentinel'},
                                   ['private-detail-sentinel'], 3, True])
def test_refusal_detail_only_forwards_text_not_private_structures(context, detail):
    context['failure'] = {'error': 'invalid_destination', 'detail': detail}
    context['last_action_failure'] = dict(context['failure'])
    context['history'] = [dict(context['failure'], skill='move')]
    visible = model_skill_context(context)
    for record in (visible['failure'], visible['last_action_failure'], visible['history'][0]):
        assert record.get('detail') is None
    assert 'private-detail-sentinel' not in str(visible)


def test_capabilities_do_not_forward_nested_or_mistyped_values(context):
    context['capabilities'] = {
        'skills': [{'truth': 'private-sentinel'}], 'observe_modes': 'active',
        'holding_allowed_skills': ['observe', {'truth': 'private-sentinel'}],
        'multi_object_rearrangement': 'true', 'restores_original_pose': 1,
        'active_observe_requires_geometry': None,
    }
    assert model_skill_context(context)['capabilities'] == {}
