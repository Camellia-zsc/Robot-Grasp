"""Scene edits are validated before any world geometry changes."""
import math

import numpy as np
import pytest

from cube_task.assistant.catalog import (CatalogEditor, catalog_entries,
                                         make_catalog_object)
from cube_task.sim.world import SimWorld

PICK = [[0.12, -0.25], [0.4, -0.25], [0.4, 0.25], [0.12, 0.25]]


def editor():
    world = SimWorld()
    return world, CatalogEditor(world, PICK)


def test_catalog_preserves_distinct_physical_shapes_and_dimensions():
    entries = {e['id']: e for e in catalog_entries()}
    assert set(entries) == {'red', 'green', 'yellow', 'apple', 'banana', 'pear'}
    assert entries['apple']['shape'] == 'sphere'
    assert entries['banana']['shape'] == 'compound'
    assert entries['pear']['dimensions'][2] > 0.12
    assert entries['red']['dimensions'] == pytest.approx([0.05] * 3)
    assert entries['apple']['dimensions'] == pytest.approx([0.062] * 3)
    assert entries['pear']['default_zone'] == 'zone_green'


def test_add_move_rotate_remove_and_clear():
    world, edit = editor()
    result = edit.apply({'action': 'add', 'catalog_id': 'banana', 'xy': [0.28, 0.0]})
    name = result['object_id']
    assert world.objects[0].name == name
    result = edit.apply({'action': 'move', 'object_id': name,
                         'xy': [0.26, 0.07], 'yaw': math.pi / 2})
    assert result['count'] == 1
    assert world.objects[0].center_xy() == pytest.approx([0.26, 0.07])
    assert world.objects[0].yaw() == pytest.approx(math.pi / 2)
    edit.apply({'action': 'remove', 'object_id': name})
    assert not world.objects
    edit.apply({'action': 'add', 'catalog_id': 'apple', 'xy': [0.28, 0]})
    edit.apply({'action': 'clear'})
    assert not world.objects


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), True, '0.2', 10**500])
def test_nonfinite_or_non_numeric_coordinates_cannot_change_world(bad):
    world, edit = editor()
    with pytest.raises(ValueError):
        edit.apply({'action': 'add', 'catalog_id': 'apple', 'xy': [bad, 0.0]})
    assert world.objects == []


def test_overlap_invalid_rotation_and_duplicate_are_atomic():
    world, edit = editor()
    first = edit.apply({'action': 'add', 'catalog_id': 'red', 'xy': [0.15, 0.0]})
    obj = world.objects[0]
    pose = obj.T_base_obj.copy()
    with pytest.raises(ValueError, match='一件'):
        edit.apply({'action': 'add', 'catalog_id': 'red', 'xy': [0.31, 0.13]})
    with pytest.raises(ValueError, match='8 mm'):
        edit.apply({'action': 'add', 'catalog_id': 'apple', 'xy': [0.18, 0.0]})
    with pytest.raises(ValueError, match='完整轮廓'):
        edit.apply({'action': 'move', 'object_id': first['object_id'], 'yaw': math.pi / 4})
    assert world.objects == [obj]
    assert np.array_equal(obj.T_base_obj, pose)


def test_the_whole_footprint_must_be_inside_not_only_the_center():
    world, edit = editor()
    with pytest.raises(ValueError, match='完整轮廓'):
        edit.apply({'action': 'add', 'catalog_id': 'banana', 'xy': [0.15, 0.0]})
    assert not world.objects
    edit.apply({'action': 'add', 'catalog_id': 'red', 'xy': [0.145, 0.0]})
    assert len(world.objects) == 1


def test_concave_boundary_cannot_cut_through_an_aabb_with_four_inside_corners():
    world = SimWorld()
    polygon = [[0.1, -0.1], [0.4, -0.1], [0.4, 0.1], [0.3, 0.1],
               [0.3, 0.0], [0.29, 0.0], [0.29, 0.1], [0.1, 0.1]]
    edit = CatalogEditor(world, polygon)
    with pytest.raises(ValueError, match='完整轮廓'):
        edit.apply({'action': 'add', 'catalog_id': 'apple', 'xy': [0.295, -0.005]})
    assert not world.objects


def test_eight_mm_gap_is_enforced_at_the_aabb_edges():
    world, edit = editor()
    edit.apply({'action': 'add', 'catalog_id': 'red', 'xy': [0.20, 0.0]})
    with pytest.raises(ValueError, match='8 mm'):
        edit.apply({'action': 'add', 'catalog_id': 'green', 'xy': [0.257, 0.0]})
    edit.apply({'action': 'add', 'catalog_id': 'green', 'xy': [0.258, 0.0]})
    assert len(world.objects) == 2


def test_rotating_a_banana_without_xy_preserves_its_footprint_center():
    world, edit = editor()
    result = edit.apply({'action': 'add', 'catalog_id': 'banana', 'xy': [0.27, 0.03]})
    for yaw in (0.6, -0.9, math.pi / 2, 0.0):
        edit.apply({'action': 'move', 'object_id': result['object_id'], 'yaw': yaw})
        assert world.objects[0].center_xy() == pytest.approx([0.27, 0.03], abs=1e-12)


def test_all_six_classes_fit_and_a_seventh_object_is_rejected():
    world, edit = editor()
    placements = [('red', [0.17, -0.18]), ('green', [0.27, -0.18]),
                  ('yellow', [0.37, -0.18]), ('banana', [0.27, 0.0]),
                  ('apple', [0.17, 0.15]), ('pear', [0.32, 0.15])]
    for label, xy in placements:
        edit.apply({'action': 'add', 'catalog_id': label, 'xy': xy})
    before = list(world.objects)
    with pytest.raises(ValueError, match='六件'):
        edit.apply({'action': 'add', 'catalog_id': 'red', 'xy': [0.20, -0.10]})
    assert world.objects == before


def test_held_objects_block_edits_and_metadata_is_a_detached_copy():
    world, edit = editor()
    obj = make_catalog_object('apple', 'held', [0.28, 0])
    world.objects = [obj]
    world.held = obj
    with pytest.raises(ValueError, match='夹持'):
        edit.apply({'action': 'clear'})
    assert world.objects == [obj]
    metadata = edit.metadata()
    metadata['catalog'][0]['name'] = 'changed'
    assert edit.metadata()['catalog'][0]['name'] == '红色方块'
    assert metadata['max_objects'] == 6


def test_missing_object_unknown_fields_and_unknown_catalog_are_rejected():
    world, edit = editor()
    for request in ({'action': 'remove', 'object_id': 'missing'},
                    {'action': 'clear', 'unexpected': 1},
                    {'action': 'add', 'catalog_id': 'watermelon', 'xy': [0.28, 0]},
                    {'action': 'unsupported'}):
        with pytest.raises(ValueError):
            edit.apply(request)
    assert not world.objects
