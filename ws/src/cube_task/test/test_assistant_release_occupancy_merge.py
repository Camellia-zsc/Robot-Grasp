"""Fresh complete singleton measurements supersede cropped release obstacles."""
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.core.types import Detection, PickResult
from test_assistant_release_phases import release_session


def banana(xy, *, valid=True, complete=True):
    return Detection('banana', .99, center_base=np.array([*xy, .045]),
                     length=.18, width=.04, height=.045, top_z=.045, yaw=0.,
                     valid=valid, extra={'catalog_complete': complete})


@pytest.mark.parametrize('one_per_class,kind,replaces', [
    (True, 'complete', True),
    (False, 'complete', False),
    (True, 'incomplete', False),
    (True, 'invalid', False),
    (True, 'two_valid', False),
    (True, 'valid_and_invalid', False),
])
def test_release_merge_rebinds_only_a_fresh_complete_unique_catalog_object(
        monkeypatch, one_per_class, kind, replaces):
    session, target, grasp, place, _now, _moves, _calls = release_session(monkeypatch)
    session.editor = SimpleNamespace(metadata=lambda: {'one_per_class': one_per_class})
    # A cropped centroid lies beyond normal 4 cm association from the later
    # complete camera measurement. Occluded old space cannot disprove it.
    old = banana((.2121, .2229), valid=False, complete=False)
    full = banana((.205, .152))
    fresh = [full]
    if kind == 'incomplete':
        full.extra['catalog_complete'] = False
    elif kind == 'invalid':
        full.valid = False
    elif kind == 'two_valid':
        fresh.append(banana((.32, .15)))
    elif kind == 'valid_and_invalid':
        fresh.append(banana((.32, .15), valid=False, complete=False))
    uncertain_neighbor = Detection('pear', .2, center_base=np.array([.14, .35, .05]),
                                   valid=False)
    fresh.append(uncertain_neighbor)
    session._observed_detections = [target, old]
    session._stationary_recovery_observation = lambda: fresh
    monkeypatch.setattr('cube_task.assistant.observation_policy.measurement_visible',
                        lambda *a: False)
    planned_neighbors = []

    def replan(_target, _grasp, _place, _width, neighbors, **kwargs):
        planned_neighbors.extend(neighbors)
        return place, ''

    session.runner.replan_release = replan
    result = PickResult(False, detail={'holding': True})
    assert session._before_release(grasp, place, .05, result) is place
    assert any(d is old for d in planned_neighbors) is not replaces
    assert all(any(d is observation for d in planned_neighbors) for observation in fresh)
    assert not any(d is target for d in planned_neighbors)
    assert result.detail['holding']
