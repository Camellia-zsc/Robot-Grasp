"""Held continuation plans its actual entry route after a camera-view change."""
from types import SimpleNamespace

import numpy as np

from cube_task.core.types import PickResult
from test_assistant_release_phases import release_session


def setup_route(monkeypatch):
    session, target, grasp, place, now, moves, calls = release_session(monkeypatch)
    # The release fixture bypasses entry planning. These tests need the real
    # current-pose -> checked-route seam rather than that release-only stub.
    del session._plan_carried_entry
    session._active_revision = session._environment_revision
    session._stage = 'transit'
    session.runner._held_transfer_feedback = lambda det: ({'grasp': grasp}, .05, '')
    session.camera._last = SimpleNamespace()
    session.arm.get_tcp_pose = lambda: SimpleNamespace(value=np.eye(4), stamp=0.)
    session.clock = SimpleNamespace(now=lambda: 0.)
    session.runner.skill.feedback_max_age = .25
    session.runner.T_flange_tcp = np.eye(4)
    routes = []

    def checked_route(*args, **kwargs):
        routes.append((args, kwargs))
        return [place.T_pre.copy()], ''

    monkeypatch.setattr('cube_task.assistant.placement_observation.safe_inspection_route', checked_route)
    return session, grasp, place, routes


def test_held_entry_uses_current_measured_pose_and_full_swept_path(monkeypatch):
    session, grasp, place, routes = setup_route(monkeypatch)
    result = PickResult(False, detail={'holding': True})
    route = session._before_held_transit(grasp, place, .05, result)
    np.testing.assert_array_equal(route[-1], place.T_pre)
    args, options = routes[0]
    np.testing.assert_array_equal(args[2], np.eye(4))
    np.testing.assert_array_equal(args[3], place.T_pre)
    assert options['opening_uncertainty'] == .001
    assert result.detail['held_transit_opening_uncertainty'] == .001


def test_held_entry_intervention_during_capture_never_plans_old_route(monkeypatch):
    session, grasp, place, routes = setup_route(monkeypatch)
    def changed():
        session._environment_revision += 1
        return []
    session._stationary_recovery_observation = changed
    result = PickResult(False, detail={'holding': True})
    assert session._before_held_transit(grasp, place, .05, result) is None
    assert not routes
    assert result.detail['held_transit_planning_reason'] == 'scene_changed'


def test_held_entry_cannot_use_old_robot_pose_or_changed_width(monkeypatch):
    session, grasp, place, routes = setup_route(monkeypatch)
    session.runner._held_transfer_feedback = lambda det: ({'grasp': grasp}, .052, '')
    result = PickResult(False, detail={'holding': True})
    assert session._before_held_transit(grasp, place, .05, result) is None
    assert not routes
    assert result.detail['held_transit_planning_reason'] == 'held_opening_changed'


def test_held_route_guard_applies_intervention_before_each_segment(monkeypatch):
    session, grasp, place, routes = setup_route(monkeypatch)
    session._apply_interventions = lambda: setattr(session, '_environment_revision', 2)
    assert session._execution_guard('held_transit') == 'scene_changed'


def test_missing_camera_frame_keeps_measured_grasp_and_never_plans_blindly(monkeypatch):
    session, grasp, place, routes = setup_route(monkeypatch)
    def unavailable():
        raise RuntimeError('no fresh RGB-D frame')
    session._stationary_recovery_observation = unavailable
    result = PickResult(False, detail={'holding': True})
    assert session._before_held_transit(grasp, place, .05, result) is None
    assert result.detail['holding'] and not result.detail.get('fault')
    assert not routes
    assert result.detail['held_transit_planning_reason'].startswith('fresh_observation_unavailable:')


def test_lift_intervention_replans_the_landing_before_entering_an_occupied_old_slot(monkeypatch):
    import copy
    session, grasp, place, routes = setup_route(monkeypatch)
    old_xy = place.T_pre[:2, 3].copy()
    session._motion_replan_required = True
    session.runner.skill._verify_held = lambda *a: .05
    session._calibrate_carried_anchor = lambda *a: (True, '', [])
    updated = copy.deepcopy(place)
    for pose in (updated.T_pre, updated.T_release, updated.T_retreat):
        pose[0, 3] += .10
    updated.inspection_entry_route = [updated.T_pre.copy()]
    calls = []
    def refine(g, p, width, result, budget):
        calls.append((p.T_pre[:2, 3].copy(), width))
        return updated
    session._refine_release = refine
    ok, why, route = session._before_carry(grasp, place, PickResult(False))
    assert ok and not why and len(calls) == 1
    np.testing.assert_allclose(place.T_pre, route[-1])
    assert not np.allclose(place.T_pre[:2, 3], old_xy)
    assert not routes  # No route search towards the now occupied old slot.


def changed_body_route(monkeypatch):
    from cube_task.core.types import Detection
    session, grasp, place, routes = setup_route(monkeypatch)
    unknown = Detection('unobserved_pear', 0., valid=False, top_z=.14)
    observer = SimpleNamespace(invalidated={'pear'}, height_priors={'pear': .12},
        uncertainty_obstacles=lambda **kw: [unknown] if observer.invalidated else [])
    session._motion_observer = observer
    session.runner.skill._verify_held = lambda *a: .05
    session._active_revision = 1
    session._move_inspection_waypoint = lambda *a, **kw: True
    captures = []
    def capture(**kw):
        captures.append(1)
        if len(captures) > 1:
            observer.invalidated.clear()
        return []
    session._stationary_recovery_observation = capture
    inspection = np.eye(4)
    inspection[2, 3] = .35
    def poses(*args, **kw):
        assert kw['focus_z'] == .06
        assert args[5] >= .14  # Candidate height must clear the exclusion volume.
        return iter([inspection])
    monkeypatch.setattr('cube_task.assistant.placement_observation.iter_safe_inspection_poses', poses)
    return session, grasp, place, routes, observer, captures


def test_entry_locates_moved_body_before_destination_without_a_recovery_context(monkeypatch):
    session, grasp, place, routes, observer, captures = changed_body_route(monkeypatch)
    assert not hasattr(session.runner, '_held_recovery_context')
    result = PickResult(False, detail={'holding': True})
    assert session._before_held_transit(grasp, place, .05, result)
    assert len(captures) == 2 and len(routes) == 2
    assert routes[0][0][4][0].label == 'unobserved_pear'
    assert not any(d.label.startswith('unobserved_') for d in routes[1][0][4])
    assert not observer.invalidated


def test_unknown_body_is_not_waived_when_no_inspection_path_is_safe(monkeypatch):
    session, grasp, place, routes, observer, captures = changed_body_route(monkeypatch)
    moves = []
    session._move_inspection_waypoint = lambda *a, **kw: moves.append(a)
    monkeypatch.setattr('cube_task.assistant.placement_observation.safe_inspection_route',
        lambda *a, **kw: (None, 'unknown body intersects sweep'))
    result = PickResult(False, detail={'holding': True})
    assert session._before_held_transit(grasp, place, .05, result) is None
    assert not moves and observer.invalidated == {'pear'} and len(captures) == 1
    assert 'changed_object_not_observed' in result.detail['held_transit_planning_reason']


def test_second_change_during_inspection_never_continues_old_entry(monkeypatch):
    session, grasp, place, routes, observer, captures = changed_body_route(monkeypatch)
    session._move_inspection_waypoint = lambda *a, **kw: setattr(session, '_environment_revision', 2)
    result = PickResult(False, detail={'holding': True})
    assert session._before_held_transit(grasp, place, .05, result) is None
    assert len(routes) == 1
    assert result.detail['held_transit_planning_reason'] == 'scene_changed'


def test_entry_merge_preserves_other_complete_targets_beside_new_crops(monkeypatch):
    import copy
    from test_assistant_targeted_scan import block
    session, grasp, place, routes = setup_route(monkeypatch)
    other = block('red', xy=(.4,.3))
    crop = copy.deepcopy(other); crop.valid = False
    session._observed_detections.append(other)
    session._motion_observer = SimpleNamespace(invalidated=set(),
        uncertainty_obstacles=lambda **kw: [])
    session._stationary_recovery_observation = lambda: [crop]
    result = PickResult(False, detail={'holding': True})
    assert session._before_held_transit(grasp, place, .05, result)
    peers = routes[-1][0][4]
    assert any(d.label == 'red' and d.valid for d in peers)
    assert any(d.label == 'red' and d.valid for d in session._observed_detections)
