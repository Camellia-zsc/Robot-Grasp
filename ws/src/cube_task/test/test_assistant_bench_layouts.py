"""Random trials must vary real layouts and cover more than the central strip."""
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.catalog import CatalogEditor
from cube_task.tools.assistant_bench import LABELS, random_objects
from cube_task.sim.scenarios import DEFAULT_PICK_POLYGON


def session():
    return SimpleNamespace(editor=CatalogEditor(None, DEFAULT_PICK_POLYGON,
        edit_polygon=[[-.3,-.7],[.9,-.7],[.9,.7],[-.3,.7]]))


@pytest.mark.parametrize('scenario', ['tabletop', 'stacking', 'intervention'])
def test_seed_changes_each_object_but_same_seed_replays_identically(scenario):
    s = session()
    layouts = [random_objects(s, seed, scenario=scenario) for seed in (19, 53, 19)]
    mapped = [{o.label:o.T_base_obj for o in objects} for objects in layouts]
    for label in LABELS:
        np.testing.assert_array_equal(mapped[0][label], mapped[2][label])
        assert not np.allclose(mapped[0][label], mapped[1][label])
    for objects in layouts:
        assert sorted(o.label for o in objects) == sorted(LABELS)
        for i, obj in enumerate(objects):
            s.editor._check_placement(obj, objects[:i])


def test_random_coverage_includes_both_sides_of_legacy_strip_and_near_far():
    s = session()
    centers = np.array([o.center_xy() for seed in range(20)
                       for o in random_objects(s, seed)])
    assert centers[:,1].min() < -.40 and centers[:,1].max() > .40
    radius = np.linalg.norm(centers, axis=1)
    assert radius.min() < .26 and radius.max() > .54


def test_stack_base_does_not_start_inside_its_destination_and_skip_unstacking():
    for seed in range(20):
        base = next(o for o in random_objects(session(),seed,scenario='stacking') if o.label=='yellow')
        assert base.aabb()[0][0] < .49-.30/2  # Dashboard yellow card left edge.


def test_intervention_fruits_start_wholly_in_pick_area():
    from cube_task.perception.geometry import points_in_polygon
    for seed in range(20):
        for obj in random_objects(session(), seed, scenario='intervention'):
            if obj.label not in ('banana','pear'):
                continue
            lo,hi = obj.aabb()
            corners = [[lo[0],lo[1]],[lo[0],hi[1]],[hi[0],lo[1]],[hi[0],hi[1]]]
            assert np.all(points_in_polygon(np.array(corners), np.array(DEFAULT_PICK_POLYGON)))
