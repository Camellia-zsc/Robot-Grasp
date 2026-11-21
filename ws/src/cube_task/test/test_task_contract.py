"""Task-wide intent must survive partial visibility and partial model plans."""
from copy import deepcopy

import pytest

from cube_task.assistant.task_contract import bind_task_contract


LABELS = ('red', 'green', 'yellow', 'apple', 'banana', 'pear')
MAPPING = dict(zip(LABELS, ('zone_red', 'zone_green', 'zone_yellow',
                          'zone_red', 'zone_yellow', 'zone_green')))


@pytest.fixture
def scene():
    return dict(
        scene_version=19,
        # Yellow and red are hidden; pear is only an unusable fragment.
        objects=[dict(label='apple', valid=True), dict(label='banana', valid=True),
                 dict(label='green', valid=False), dict(label='pear', valid=False)],
        class_to_zone=deepcopy(MAPPING),
        zones=[dict(name=name) for name in ('zone_red', 'zone_green', 'zone_yellow')])


@pytest.fixture
def inventory():
    return [dict(label=label, count=1) for label in LABELS]


@pytest.mark.parametrize('instruction', [
    '把所有物品放到各自默认区域',
    '请将全部物体送回对应的默认放置区',
    '每件物品都放回各自的默认区域',
    '每件都回默认区域',
    '将每一个物品移到默认区域',
    '把物品全部放到默认区域',
    '全部按默认区域归位',
    'Put all objects into their default zones.',
    'Please return all items to their default areas.',
    'Move every object to its default region.',
    'Restore each item to its default zone.',
    'Put all objects in their default zones.',
    '能否把所有物品放到各自默认区域？',
    '能不能把全部物品送回默认区？',
    '把所有物品回默认区，完成后检查',
    '把所有物品回默认区，完成后确认所有物品都已归位',
    'Can you put all objects in their default zones?',
    'Could you return all items to their default areas?',
    'Would you move every object to its default region?',
    'Move all objects to default zones, then verify the result.',
])
def test_all_default_keeps_occluded_and_invalid_inventory_members(scene, inventory, instruction):
    before = deepcopy((scene, inventory))
    contract = bind_task_contract(instruction, scene, inventory)
    assert contract == dict(kind='all_default', required_labels=list(LABELS),
                           goals=[dict(label=label, zone_name=MAPPING[label]) for label in LABELS])
    assert (scene, inventory) == before
    assert 'center_base' not in repr(contract)


@pytest.mark.parametrize('instruction', [
    '把苹果和香蕉放到默认区域',
    '把所有苹果放到默认区域',
    '全部苹果放到默认区域',
    '每个苹果都放默认区域',
    '只把所有可见物品放到默认区域',
    '把当前有效的全部物品放到默认区域',
    '把所有物品中有效的放到默认区域',
    '把除了梨子的所有物品放回默认区域',
    '把所有物品放默认区域，但不要动苹果',
    '把所有物品放默认区域，梨先不动',
    '不要把所有物品放到默认区域',
    '所有物品都放到默认区域了吗？',
    '检查所有物品是否放在默认区域',
    '告诉我所有物品应该放到哪个默认区域',
    '回答所有物品的默认区域',
    '把所有物品放回原位',
    '把所有物品从默认区域放回原来的位置',
    '全部复位',
    '把所有物品归位',
    '所有物品的默认区域',
    'Put all visible objects into their default zones.',
    'Move all valid items to default areas.',
    'Return all items except the pear to their default areas.',
    'Return all items apart from the pear to their default areas.',
    'Put all other objects in their default zones.',
    'Put all objects I can see in their default zones.',
    'Put all objects in their default zones but leave the pear alone.',
    'Do not move all objects to their default zones.',
    'Are all objects in their default zones?',
    'Did you move all objects to their default zones?',
    'Can you check whether all objects are put in their default zones?',
    'Can you confirm whether all objects were put in their default zones?',
    'Where should I put all objects in their default zones?',
    'Return all objects to their original positions.',
    'Move all objects from default zones to their initial positions.',
    'Move all apples into their default zones.',
])
def test_other_scopes_questions_and_original_pose_are_not_bound(scene, inventory, instruction):
    assert bind_task_contract(instruction, scene, inventory) is None


def test_actual_inventory_is_not_the_configured_catalog(scene):
    contract = bind_task_contract('把所有物品放到各自默认区域', scene,
                                  [dict(label='apple', count=1)])
    assert contract['required_labels'] == ['apple']
    assert contract['goals'] == [dict(label='apple', zone_name='zone_red')]


def test_legacy_partial_goals_keep_their_indices_and_gain_missing_members(scene, inventory):
    existing = [dict(label='banana', zone_name='zone_yellow'),
                dict(label='apple', zone_name='zone_red')]
    before = deepcopy(existing)
    contract = bind_task_contract('把所有物品放到各自默认区域', scene, inventory, existing)
    assert contract['goals'][:2] == existing == before
    assert [g['label'] for g in contract['goals']] == [
        'banana', 'apple', 'red', 'green', 'yellow', 'pear']
    contract['goals'][0]['zone_name'] = 'changed'
    assert existing == before


@pytest.mark.parametrize('existing', [
    [dict(label='apple', zone_name='zone_green')],
    [dict(label='apple', zone_name=None,
          destination=dict(type='on_object', label='green'))],
    [dict(label='apple', zone_name='zone_red'), dict(label='apple', zone_name='zone_red')],
    [dict(label='invented', zone_name='zone_red')],
])
def test_existing_promises_are_never_silently_rewritten(scene, inventory, existing):
    with pytest.raises(ValueError):
        bind_task_contract('把所有物品放到各自默认区域', scene, inventory, existing)


@pytest.mark.parametrize('inventory', [None, [], [dict(label='invented', count=1)],
                                        [dict(label='apple', count=2)],
                                        [dict(label='apple', count=True)]])
def test_missing_or_unaddressable_inventory_cannot_shrink_the_contract(scene, inventory):
    with pytest.raises(ValueError):
        bind_task_contract('把所有物品放到各自默认区域', scene, inventory)


@pytest.mark.parametrize('broken', ['missing_mapping', 'unknown_zone', 'invalid_zone'])
def test_every_member_requires_a_valid_configured_default(scene, inventory, broken):
    if broken == 'missing_mapping':
        del scene['class_to_zone']['pear']
    elif broken == 'unknown_zone':
        scene['class_to_zone']['pear'] = 'invented'
    else:
        scene['zones'][1]['valid'] = False
    with pytest.raises(ValueError):
        bind_task_contract('把所有物品放到各自默认区域', scene, inventory)
