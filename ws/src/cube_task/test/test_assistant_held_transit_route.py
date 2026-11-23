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
