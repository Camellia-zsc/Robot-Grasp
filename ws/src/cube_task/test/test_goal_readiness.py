"""A missing target can wait while the complete immutable goal list remains."""
import copy
import json

import pytest

from cube_task.assistant.skill_protocol import model_skill_context, validate_skill_plan
from test_assistant_skill_protocol import context, plan, step


def readiness():
    from cube_task.assistant.goal_readiness import goal_worklist
    return goal_worklist


def measured(label, x=.17, y=-.35, **changes):
    return dict(label=label, valid=True, center_base=[x, y, .05], top_z=.05,
                height=.05, length=.05, width=.05, yaw=0., **changes)


@pytest.fixture
def table_scene():
    return dict(scene_version=4, objects=[measured('red'), measured('green', .17, .35),
                measured('apple'), measured('banana', .49, 0.),
                dict(measured('pear', .32, -.16), valid=False,
                     reject_reason='touches_image_border')],
                zones=[dict(name='zone_red', center=[.17, -.35, 0.], size=[.2, .2]),
                       dict(name='zone_green', center=[.17, .35, 0.], size=[.2, .2]),
                       dict(name='zone_yellow', center=[.49, 0., 0.], size=[.2, .2])],
                class_to_zone=dict(red='zone_red', green='zone_green', yellow='zone_yellow',
                                   apple='zone_red', banana='zone_yellow', pear='zone_green'))


def six_goals():
    return [dict(label=label, zone_name=zone) for label, zone in (
        ('yellow', 'zone_yellow'), ('green', 'zone_green'), ('red', 'zone_red'),
        ('apple', 'zone_red'), ('banana', 'zone_yellow'), ('pear', 'zone_green'))]


def test_missing_and_border_targets_are_deferred_without_hiding_four_observed_goals(table_scene):
    goals, before = six_goals(), copy.deepcopy(table_scene)
    result = readiness()(table_scene, goals)
    assert [r['goal_index'] for r in result] == list(range(6))
    assert [r['label'] for r in result] == [g['label'] for g in goals]
    assert [r['status'] for r in result] == ['deferred'] + ['observed_at_goal'] * 4 + ['deferred']
    assert result[0]['reason'] == 'target_not_observed'
    assert result[5]['reason'] == 'target_invalid'
    assert all(set(r) == {'goal_index', 'label', 'status', 'reason'} for r in result)
    assert table_scene == before and goals == six_goals()


def test_ready_goal_after_missing_goal_is_observed_pending_without_reordering_contract(table_scene):
    table_scene['objects'][0]['center_base'] = [.3, .1, .05]
    result = readiness()(table_scene, six_goals())
    assert result[0]['status'] == 'deferred'
    assert result[2]['status'] == 'observed_pending'
    assert result[2]['reason'] == 'goal_not_observed_satisfied'


@pytest.mark.parametrize('change', [dict(center_base=[float('nan'), 0., .05]),
    dict(center_base=[.17, -.35]), dict(length=None), dict(width=-.1), dict(height=0.),
    dict(top_z=float('inf')), dict(yaw=None)])
def test_incomplete_geometry_does_not_become_an_actionable_or_completed_measurement(table_scene, change):
    table_scene['objects'][0].update(change)
    result = readiness()(table_scene, [dict(label='red', zone_name='zone_red')])
    assert result[0]['status'] == 'deferred' and result[0]['reason'] == 'target_geometry_incomplete'


def test_invalid_fragment_does_not_hide_one_complete_identity_but_two_valid_instances_are_ambiguous(table_scene):
    goals = [dict(label='red', zone_name='zone_red')]
    fragment = dict(measured('red', .3, .1), valid=False)
    table_scene['objects'].append(fragment)
    assert readiness()(table_scene, goals)[0]['status'] == 'observed_at_goal'
    fragment['valid'] = True
    result = readiness()(table_scene, goals)[0]
    assert result['status'] == 'deferred' and result['reason'] == 'target_ambiguous'


def test_region_center_and_in_zone_flag_do_not_replace_full_footprint_check(table_scene):
    table_scene['objects'][0].update(center_base=[.265, -.35, .05], in_zone=True)
    assert readiness()(table_scene, [dict(label='red', zone_name='zone_red')])[0]['status'] == 'observed_pending'


@pytest.mark.parametrize('flag', ['valid', 'executable'])
def test_invalidated_scene_keeps_old_observations_as_hints_only(table_scene, flag):
    table_scene[flag] = False
    items = readiness()(table_scene, six_goals())
    assert len(items) == 6
    assert all(item['status'] == 'deferred' and item['reason'] == 'scene_unavailable' for item in items)


def test_point_destination_uses_existing_outline_and_tolerance(context):
    context['scene']['objects'] = [measured('apple', .3, .1)]
    goal = dict(label='apple', zone_name=None, destination=dict(type='point', xy=[.3, .1]))
    assert readiness()(context['scene'], [goal])[0]['status'] == 'observed_at_goal'
    context['scene']['objects'][0]['center_base'][0] += .03
    assert readiness()(context['scene'], [goal])[0]['status'] == 'observed_pending'


def test_worklist_context_only_forwards_complete_goal_identity_and_short_status(context):
    context['goal_worklist'] = [dict(goal_index=0, label='apple', status='deferred',
        reason='r' * 600, truth='private-sentinel', center_base=[99., 99., 99.])]
    value = model_skill_context(context)['goal_worklist']
    assert value == [dict(goal_index=0, label='apple', status='deferred', reason='r' * 200)]
    assert 'private-sentinel' not in json.dumps(model_skill_context(context))


@pytest.mark.parametrize('items', [[], [dict(goal_index=True, label='apple', status='deferred')],
    [dict(goal_index=1, label='apple', status='deferred')],
    [dict(goal_index=0, label='banana', status='deferred')],
    [dict(goal_index=0, label='apple', status='verified')]])
def test_worklist_cannot_change_scope_or_manufacture_verified_status(context, items):
    context['goal_worklist'] = items
    with pytest.raises(ValueError):
        model_skill_context(context)


def test_final_audit_context_exports_boolean_result_without_physics_or_geometry(context):
    context['final_audit'] = dict(phase='reviewed', scene_version=4, truth='private-sentinel',
        items=[dict(goal_index=0, label='apple', ok=False, camera_verified=True,
            error='not_at_goal', message='m' * 700, truth='private-sentinel',
            center_base=[99., 99., 99.])])
    value = model_skill_context(context)['final_audit']
    assert value == dict(phase='reviewed', scene_version=4,
        items=[dict(goal_index=0, label='apple', ok=False, camera_verified=True,
                    error='not_at_goal', message='m' * 500)])
    assert 'private-sentinel' not in json.dumps(model_skill_context(context))


@pytest.mark.parametrize('change', [dict(phase='complete'), dict(scene_version=True),
    dict(items=[]), dict(items=[dict(goal_index=0, label='apple', ok=1, camera_verified=True)]),
    dict(items=[dict(goal_index=0, label='apple', ok=False, camera_verified='true')])])
def test_final_audit_rejects_fabricated_completeness_and_nonboolean_conclusions(context, change):
    context['final_audit'] = dict(phase='reviewed', scene_version=4,
        items=[dict(goal_index=0, label='apple', ok=False, camera_verified=False)])
    context['final_audit'].update(change)
    with pytest.raises(ValueError):
        model_skill_context(context)


@pytest.mark.parametrize('flag', ['deferred', 'manipulation_started'])
def test_deferred_and_manipulation_started_are_strict_boolean_feedback(context, flag):
    context.update(failure={flag: True}, last_action_failure={flag: False}, history=[{flag: True}])
    value = model_skill_context(context)
    assert value['failure'][flag] is True and value['last_action_failure'][flag] is False
    assert value['history'][0][flag] is True
    context.update(failure={flag: 1}, history=[{flag: 'true'}])
    value = model_skill_context(context)
    assert flag not in value['failure'] and flag not in value['history'][0]


def test_worklist_hint_never_overrides_unique_valid_grasp_gate(context):
    context['scene']['objects'][0]['valid'] = False
    context['goal_worklist'] = [dict(goal_index=0, label='apple',
                                   status='observed_pending', reason='hint_only')]
    with pytest.raises(ValueError, match='唯一且有效'):
        validate_skill_plan(plan(context, [step(1, 'pick', 'apple')]), context)
