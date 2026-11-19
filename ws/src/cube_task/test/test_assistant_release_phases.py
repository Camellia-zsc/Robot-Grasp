"""A difficult held-green release must leave time for a checked new view."""
from contextlib import contextmanager
from types import SimpleNamespace
import math
import threading

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.core.transforms import rot_y, top_down_T
from cube_task.core.types import Detection, GraspPlan, PickResult, PlacePlan, Zone


def release_session(monkeypatch, capacity=True):
    now, moves, calls = [0.], [], []
    monkeypatch.setattr('cube_task.assistant.simulation.time.monotonic', lambda: now[0])
    target = Detection('green', .998, center_base=np.array([.2137, .3138, .0511]),
                       top_z=.0511, height=.0511, length=.0498, width=.0498, yaw=-.4694)
    grasp_pose = top_down_T(.2143, .314, .0111, 2.6722)
    grasp = GraspPlan(target, grasp_pose.copy(), grasp_pose, grasp_pose.copy(),
                      .1, 0., .050638421257518655, 2.6722)
    pose = top_down_T(.2253, -.0755, .0211, 1.2490457723982544)
    place = PlacePlan('pick_area', pose.copy(), pose.copy(), pose.copy(), .07063842125751865)
    session = object.__new__(SimulationSession)
    session.cancel_event = threading.Event()
    session._environment_revision = 1
    session._observed_detections = [target]
    session._apply_interventions = lambda: None
    session._emit = lambda *a: None
    session._on_stage = lambda *a: None
    session._refresh_placement_occupancy = lambda *a: None
    session._stationary_recovery_observation = lambda: []
    session.cfg = SimpleNamespace(get=lambda key, default=None: default)
    session.camera = SimpleNamespace(_last=SimpleNamespace(), intrinsics=lambda: None)

    @contextmanager
    def scope(cancelled):
        yield

    session.arm = SimpleNamespace(planning_scope=scope,
        get_flange_pose=lambda: SimpleNamespace(value=np.eye(4)),
        get_tcp_pose=lambda: SimpleNamespace(value=np.eye(4)))
    zone = Zone('pick_area', np.array([.26, 0., 0.]), (.28, .5))
    options = dict(placement_zone=zone, placement_polygon=zone.corners_xy(), exact_xy=False)
    session._active_placement = (target, {'type': 'region', 'name': 'pick_area'}, 'pick_area', options)

    def replan(*args, cancelled=None, **kwargs):
        calls.append(kwargs)
        if moves:
            now[0] += .5
            return place, ''
        # The incident consumed all 45 seconds in this first visibility pass.
        while not cancelled():
            now[0] += 1.
        return None, 'placement_planning_budget_exceeded'

    session.runner = SimpleNamespace(replan_release=replan,
        has_release_capacity=lambda *a, **k: (capacity, 'no_clear_supported_release' if not capacity else ''),
        ext=SimpleNamespace(parent_frame='flange', T=np.eye(4)), T_flange_tcp=np.eye(4),
        nominal={'green': (.05, .05)}, free_tilt=40.,
        placer=SimpleNamespace(open_width=.1, last_info={'xy': [.225, -.075]}),
        skill=SimpleNamespace(_move=lambda *a: moves.append(a), _verify_held=lambda *a: None))
    monkeypatch.setattr('cube_task.assistant.placement_observation.candidate_visibility_filter',
                        lambda *a, **k: lambda *a, **k: True)
    monkeypatch.setattr('cube_task.assistant.placement_observation.iter_safe_inspection_poses',
                        lambda *a, **k: iter([np.eye(4)]))
    monkeypatch.setattr('cube_task.assistant.placement_observation.safe_inspection_route',
                        lambda *a, **k: ([np.eye(4)], ''))
    return session, target, grasp, place, now, moves, calls


@pytest.mark.parametrize('capacity', [True, None])
def test_visibility_search_timeout_reserves_time_for_checked_view_and_fresh_replan(monkeypatch, capacity):
    session, target, grasp, place, now, moves, calls = release_session(monkeypatch, capacity)
    result = PickResult(False, detail={'holding': True})
    updated = session._before_release(grasp, place, .05093465497112733, result)
    assert updated is place and len(moves) == 1
    assert len(calls) == 2 and now[0] < 45.
    assert result.detail['holding'] and not session.cancel_event.is_set()


def test_proven_geometric_capacity_failure_does_not_move_to_another_view(monkeypatch):
    session, target, grasp, place, now, moves, calls = release_session(monkeypatch, capacity=False)
    result = PickResult(False, detail={'holding': True})
    assert session._before_release(grasp, place, .05093465497112733, result) is None
    assert not moves and now[0] < 45.
    assert 'no_clear_supported_release' in result.detail['placement_planning_reason']


def test_timed_slice_with_completed_rejections_continues_before_camera_motion(monkeypatch):
    session, target, grasp, place, now, moves, calls = release_session(monkeypatch)
    states = []

    def replan(*args, cancelled=None, search_state=None, **kwargs):
        states.append(search_state)
        assert search_state is not None
        if len(states) == 1:
            search_state['avoided'] = [(.17, .34), (.155, .34), (.185, .34)]
            while not cancelled():
                now[0] += 1.
            return None, 'placement_planning_budget_exceeded'
        assert search_state['avoided'] == [(.17, .34), (.155, .34), (.185, .34)]
        now[0] += .5
        return place, ''

    session.runner.replan_release = replan
    result = PickResult(False, detail={'holding': True})
    assert session._before_release(grasp, place, .05, result) is place
    assert len(states) == 2 and states[0] is states[1]
    assert not moves and now[0] < 17.


def test_release_search_rejections_do_not_survive_a_new_camera_measurement(monkeypatch):
    session, target, grasp, place, now, moves, calls = release_session(monkeypatch)
    states = []

    def replan(*args, cancelled=None, search_state=None, **kwargs):
        states.append(search_state)
        assert search_state is not None
        if moves:
            assert search_state == {}
            return place, ''
        search_state['avoided'] = [(.17, .34)]
        return None, 'leaned finger blocked'

    session.runner.replan_release = replan
    result = PickResult(False, detail={'holding': True})
    assert session._before_release(grasp, place, .05, result) is place
    assert len(states) == 2 and states[0] is not states[1]
    assert len(moves) == 1


def test_geometry_capacity_checks_do_not_run_ik_or_return_an_executable_plan(monkeypatch):
    from test_runner_local_release import scene
    runner, target, grasp, place = scene()
    monkeypatch.setattr('cube_task.task.reach.adjust_plan_poses',
                        lambda *a, **k: pytest.fail('capacity must not spend IK'))
    capacity, why = runner.has_release_capacity(target, grasp, place, .05, [target])
    assert capacity is True and why == ''
    assert not runner.events


def test_release_attempt_limit_preserves_full_checks_and_default_retry_contract():
    from test_runner_local_release import scene
    runner, target, grasp, place = scene()
    checks = []
    runner._adjust_strict_release = lambda *a, **k: checks.append(1) or (False, 'blocked final fingers')
    updated, why = runner.replan_release(target, grasp, place, .05, [target], max_attempts=1)
    assert updated is None and why == 'blocked final fingers' and checks == [1]
    assert not runner.events


def test_stop_in_first_release_phase_never_starts_capacity_or_camera_motion(monkeypatch):
    session, target, grasp, place, now, moves, calls = release_session(monkeypatch)
    def stopped(*args, **kwargs):
        session.cancel_event.set()
        return None, 'aborted'
    session.runner.replan_release = stopped
    session.runner.has_release_capacity = lambda *a, **k: pytest.fail('cancelled capacity search')
    result = PickResult(False, detail={'holding': True})
    assert session._before_release(grasp, place, .05093465497112733, result) is None
    assert not moves and result.detail['holding']
    assert result.detail['placement_planning_reason'] == 'aborted'


def test_capacity_rejects_an_occupied_fixed_point_without_ik():
    from test_runner_local_release import scene
    runner, target, grasp, place = scene()
    blocker = Detection('green', 1., center_base=np.array([.4, 0., .05]),
                        top_z=.05, height=.05, length=.05, width=.05, yaw=0.)
    capacity, why = runner.has_release_capacity(target, grasp, place, .05,
                                               [target, blocker], exact_xy=True)
    assert capacity is False and why == 'no_clear_supported_release'
    assert not runner.events


def test_capacity_keeps_missing_neighbor_geometry_unknown():
    from test_runner_local_release import scene
    runner, target, grasp, place = scene()
    blocker = Detection('unknown', .2, center_base=np.array([.4, 0., .05]), valid=False)
    capacity, why = runner.has_release_capacity(target, grasp, place, .05, [target, blocker])
    assert capacity is None and 'neighbor footprint' in why


@pytest.mark.parametrize('outcome', ['success', 'no_capacity', 'cancelled', 'unknown', 'exception'])
def test_capacity_probe_preserves_the_previous_executable_metadata(monkeypatch, outcome):
    from test_runner_local_release import scene
    runner, target, grasp, place = scene()
    previous = {'xy': [.4, 0.], 'tier': 'previous_executable_release'}
    runner.placer.last_info = previous
    interrupted = [False]

    def probe(*args, **kwargs):
        runner.placer.last_info = {'xy': [.31, -.08], 'tier': 'geometry_only_probe'}
        if outcome == 'exception':
            raise RuntimeError('probe failed')
        interrupted[0] = outcome == 'cancelled'
        return SimpleNamespace(feasible=outcome != 'no_capacity', reason='full')

    monkeypatch.setattr(runner.placer, 'plan', probe)
    neighbors = ([Detection('unknown', .2, center_base=np.array([.4, 0., .05]), valid=False)]
                 if outcome == 'unknown' else [target])
    if outcome == 'exception':
        with pytest.raises(RuntimeError, match='probe failed'):
            runner.has_release_capacity(target, grasp, place, .05, neighbors)
    else:
        capacity, _why = runner.has_release_capacity(
            target, grasp, place, .05, neighbors, cancelled=lambda: interrupted[0])
        assert capacity is {'success': True, 'no_capacity': False,
                            'cancelled': None, 'unknown': None}[outcome]
    assert runner.placer.last_info is previous
    assert runner.placer.last_info == {'xy': [.4, 0.], 'tier': 'previous_executable_release'}


def test_new_inspection_targets_the_measured_planned_landing_not_region_center(monkeypatch):
    session, target, grasp, place, _now, _moves, _calls = release_session(monkeypatch)
    target.center_base = np.array([.20, .30, .06])
    target.top_z = target.height = .06
    grasp.T_grasp = top_down_T(.18, .30, .01, 0.)
    tilt = math.radians(30.)
    place.T_release = top_down_T(.34, -.15, .02, 0.)
    place.T_release[:3, :3] = rot_y(tilt) @ grasp.T_grasp[:3, :3]
    focused = []

    def capacity(*args, **kwargs):
        # Even an injected capacity implementation's unrelated metadata must
        # not determine the view of this executable release's actual landing.
        session.runner.placer.last_info = {'xy': [.9, .9]}
        return True, ''

    def views(arm, gp, camera, tcp, xy, *args, **kwargs):
        focused.append(np.asarray(xy).copy())
        return iter([np.eye(4)])

    session.runner.has_release_capacity = capacity
    monkeypatch.setattr('cube_task.assistant.placement_observation.iter_safe_inspection_poses', views)
    result = PickResult(False, detail={'holding': True})
    assert session._before_release(grasp, place, .05, result) is place
    # Relative to the measured grasp the object centre is 20 mm in X and Z.
    # The release's 30 degree lean rotates both into the projected landing X.
    np.testing.assert_allclose(focused, [[.34 + .02 * math.cos(tilt) + .02 * math.sin(tilt), -.15]])


def test_intervention_during_capture_never_uses_mixed_scene_depth_for_release(monkeypatch):
    session, target, grasp, place, _now, moves, calls = release_session(monkeypatch)
    def capture():
        session._environment_revision += 1
        return []
    session._stationary_recovery_observation = capture
    session.runner.has_release_capacity = lambda *a, **k: pytest.fail('mixed-scene frame must not plan')
    result = PickResult(False, detail={'holding': True})
    assert session._before_release(grasp, place, .05, result) is None
    assert not calls and not moves and result.detail['placement_planning_reason'] == 'scene_changed'


def test_release_support_views_are_accumulated_only_within_one_scene_revision(monkeypatch):
    session, target, grasp, place, _now, moves, _calls = release_session(monkeypatch)
    seen = []
    def visible(*args, observations=(), **kwargs):
        seen.append(len(observations))
        return lambda *a, **k: True
    monkeypatch.setattr('cube_task.assistant.placement_observation.candidate_visibility_filter', visible)
    session.runner.replan_release = lambda *a, **k: (place, '') if moves else (None, 'not_visible')
    def move(*args):
        moves.append(args)
        session._environment_revision += 1
    session.runner.skill._move = move
    result = PickResult(False, detail={'holding': True})
    assert session._before_release(grasp, place, .05, result) is place
    assert seen == [0, 0]
