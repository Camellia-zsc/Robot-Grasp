"""A fresh scan must retain both height layers and visible support evidence."""
import copy

import pytest

from cube_task.assistant.observations import merge_observations
from cube_task.task.views import merge_view_detections, tag_view
from test_task_fakes import make_det


@pytest.mark.parametrize('same_view', [True, False])
def test_upper_and_base_at_same_xy_remain_two_measured_objects(same_view):
    lower = make_det('green', .3, .1, size=.05, height=.05)
    upper = make_det('pear', .3, .1, size=.085, height=.112, table_z=.05)
    views = {0: tag_view([lower, upper], 0)} if same_view else {
        0: tag_view([lower], 0), 1: tag_view([upper], 1)}
    merged, _ = merge_observations(views)
    assert {d.label for d in merged} == {'pear', 'green'}


def test_two_views_of_same_height_object_still_merge_with_depth_noise():
    first = make_det('pear', .3, .1, size=.085, height=.112, table_z=.05)
    second = copy.deepcopy(first)
    second.top_z += .003
    second.n_points = first.n_points + 100
    merged = merge_view_detections({0: tag_view([first], 0), 1: tag_view([second], 1)})
    assert merged == [second]


def test_support_evidence_is_not_discarded_for_a_denser_view_without_support():
    supported = make_det('pear', .3, .1, size=.085, height=.112, table_z=.05)
    supported.extra.update(support_observed=True, support_label='green', support_z=.05)
    dense = copy.deepcopy(supported)
    dense.extra.clear()
    dense.height += .007  # The neck preference must not undo support selection.
    dense.n_points = supported.n_points + 1000
    merged, _ = merge_observations({0: tag_view([dense], 0), 1: tag_view([supported], 1)})
    assert merged == [supported]
    assert merge_view_detections({0: [dense], 1: [supported]}, stale={1}) == [dense]


def test_a_second_same_label_at_another_height_is_not_silently_deleted():
    first = make_det('green', .3, .1, size=.05, height=.05)
    second = make_det('green', .3, .1, size=.05, height=.05, table_z=.05)
    merged, _ = merge_observations({0: tag_view([first, second], 0)})
    assert len(merged) == 2  # Semantic commands must see this ambiguity.
