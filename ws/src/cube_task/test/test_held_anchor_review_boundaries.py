"""Review regressions for carried-reference and held-transit boundaries."""
import numpy as np

from cube_task.assistant.held_anchor import apply_translation_anchor, source_anchor
from cube_task.core.transforms import inv_T
from cube_task.core.types import PickResult
from test_assistant_held_transit_route import setup_route
from test_held_spherical_anchor import measured_grasp


def test_seated_held_continuation_never_checks_sweep_with_unconfirmed_source_anchor(monkeypatch):
    session, _grasp, place, routes = setup_route(monkeypatch)
    grasp = measured_grasp()
    grasp.target.extra['_closed_grip_width'] = .0464
    session._observed_detections = [grasp.target]
    session.runner.nominal['pear'] = (.085, .085)
    session.runner._held_transfer_feedback = lambda det: ({'grasp': grasp}, .0385, '')
    # The current capture contains no held target. A 7.9 mm seating change
    # therefore cannot establish the rigid body envelope used by the sweep.
    session._stationary_recovery_observation = lambda **kwargs: []
    result = PickResult(False, grasp_width=.0385,
                        detail={'holding': True, 'steps': []})
    route = session._before_held_transit(grasp, place, .0385, result)
    assert route is None
    assert not routes
    assert result.detail['holding']
    assert result.detail['held_transit_planning_reason'].startswith('held_anchor_unconfirmed')


def test_repeated_translation_refinement_uses_original_source_frame_without_double_correction():
    grasp = measured_grasp()
    executed = grasp.T_grasp.copy()
    source = source_anchor(grasp)
    local = np.array([.004, -.009, .061])
    pair = dict(ok=True, translation_only=True, center=local)
    first = apply_translation_anchor(grasp, source, pair, 1)
    assert first['ok'] and first['changed']
    corrected = grasp.T_grasp.copy()
    second = apply_translation_anchor(grasp, source_anchor(grasp), pair, 2)
    assert second['ok'] and not second['changed']
    np.testing.assert_array_equal(grasp.T_grasp, corrected)
    np.testing.assert_array_equal(grasp.target.extra['_executed_grasp_T'], executed)
    np.testing.assert_allclose((inv_T(grasp.T_grasp) @ np.r_[source['center'], 1.])[:3], local)
