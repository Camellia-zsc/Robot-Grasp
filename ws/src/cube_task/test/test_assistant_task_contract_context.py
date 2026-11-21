"""Task identity metadata cannot grant motion or replace complete final proof."""
import copy
import threading
from types import SimpleNamespace

import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_orchestrator import SkillOrchestrator
from cube_task.assistant.skill_protocol import model_skill_context, validate_skill_plan
from cube_task.assistant.task_contract import bind_task_contract
from test_assistant_skill_protocol import context, plan, step
from test_assistant_approval_scope_boundaries import controller


def scoped(context):
    inventory = [dict(label=o['label'], count=1) for o in context['scene']['objects']]
    result = copy.deepcopy(context)
    result['task_contract'] = bind_task_contract('把所有物品放到各自默认区域',
                                                result['scene'], inventory)
    result['task_inventory'] = inventory
    result['goals'] = copy.deepcopy(result['task_contract']['goals'])
    return result


def test_inventory_only_contains_operator_identity_and_count():
    class Identity:
        label = 'pear'
        def __getattr__(self, name):
            raise AssertionError('inventory must never read geometry: '+name)
    session = object.__new__(SimulationSession)
    session._closed = False
    session._operation_lock = threading.RLock()
    session.world = SimpleNamespace(objects=[Identity(), Identity(), SimpleNamespace(label='apple')])
    assert session.task_inventory() == [dict(label='pear', count=2), dict(label='apple', count=1)]


def test_scope_keeps_unobserved_identity_without_turning_it_into_an_action_target(context):
    value = scoped(context)
    value['scene']['objects'] = [o for o in value['scene']['objects'] if o['label'] != 'pear']
    value['task_inventory'][0]['transform'] = [[999.]]
    visible = model_skill_context(value)
    assert len(visible['goals']) == 3 and 'pear' in visible['task_contract']['required_labels']
    assert all(set(item) == {'label', 'count'} for item in visible['task_inventory'])
    assert not any(o['label'] == 'pear' for o in visible['scene']['objects'])
    with pytest.raises(ValueError):
        validate_skill_plan(plan(value, [step(1, 'move', 'pear', dict(type='goal', index=2))]), value)
    assert validate_skill_plan(plan(value, [step(1, 'observe', 'pear')]), value)


def test_contract_and_goal_context_cannot_disagree(context):
    value = scoped(context)
    value['goals'] = value['goals'][:1]
    with pytest.raises(ValueError):
        model_skill_context(value)


def test_inventory_snapshot_does_not_shrink_when_scene_membership_changes(controller):
    ctl, session, _client = controller
    inventory = [dict(label=o['label'], count=1) for o in session.scene['objects']]
    session.task_inventory = lambda: copy.deepcopy(inventory)
    memory = dict(instruction='把所有物品放到各自默认区域', goals=[])
    orchestrator = SkillOrchestrator(ctl)
    orchestrator._bind_task_contract(memory)
    expected = copy.deepcopy(memory['goals'])
    inventory.pop()
    session._scene['objects'].pop()
    orchestrator._bind_task_contract(memory)
    assert memory['goals'] == expected


def test_inventory_adapter_is_normalized_before_planner_context(controller):
    ctl, session, _client = controller
    session.task_inventory = lambda: [o['label'] for o in session.scene['objects']]
    memory = dict(instruction='把所有物品放到各自默认区域', goals=[])
    SkillOrchestrator(ctl)._bind_task_contract(memory)
    assert all(set(item) == {'label', 'count'} for item in memory['task_inventory'])


def test_final_publication_refuses_success_proof_for_only_part_of_fixed_scope(controller):
    ctl, session, _client = controller
    partial = [dict(label='banana', zone_name='zone_yellow'), dict(label='apple', zone_name='zone_red')]
    memory = dict(instruction='把所有物品放到各自默认区域', goals=partial,
                  protected_goals=[], originals=session.scene['objects'],
                  obligations={}, restorations=[])
    proof = dict(ok=True, camera_verified=True, truth={'success': True}, verified_revision=0,
                 items=[dict(goal=goal, ok=True, truth={'success': True}) for goal in partial])
    result = SkillOrchestrator(ctl)._finish(True, '完成', memory, proof)
    assert not result['ok'] and len(result['goals']) == 6
    assert ctl.snapshot()['metrics']['successes'] == 0
    assert not session.calls
