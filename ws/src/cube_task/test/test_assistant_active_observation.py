"""Observation modes must execute the requested capability, never silently downgrade."""
import copy
import threading
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant import active_observation
from cube_task.assistant.placement_observation import safe_inspection_route
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.core.transforms import top_down_T
from cube_task.core.types import Detection, GraspPlan
from cube_task.task.skills import _Lost


def fake_executor():
    stop = threading.Event()
    observed = {'scene_version': 1, 'objects': []}
    session = SimpleNamespace(
        _operation_lock=threading.RLock(), _check_open=lambda: None,
        cancel_event=stop, runner=SimpleNamespace(has_held_recovery=True, skill=None),
        snapshot=lambda: {'held_object': 'green'}, scene=observed,
        _publish_snapshot=lambda: None)
    executor = SkillExecutor(session)
    executor._stationary = lambda: []
    return executor


def test_explicit_active_observation_reaches_active_executor_instead_of_stationary():
    executor, calls = fake_executor(), []
    requested = {'id': 'look', 'skill': 'observe', 'mode': 'active',
                 'destination': {'type': 'goal', 'index': 0}}
    goals = [{'label': 'green', 'zone_name': 'zone_green'}]
    executor._active_observe = lambda step, goals: calls.append(copy.deepcopy(step)) or {
        'ok': True, 'skill': 'observe', 'holding': True, 'changed': True,
        'observation_mode': 'active_local', 'message': '已从受检新视角观察'}
    result = executor.execute(requested, goals)
    assert calls == [requested]
    assert result['changed'] and result['observation_mode'] == 'active_local'


def test_static_observation_of_same_scene_reports_no_information_change():
    executor = fake_executor()
    result = executor.execute({'id': 'look', 'skill': 'observe'}, [])
    assert result['ok'] and result['holding']
    assert result['changed'] is False
    assert result['observation_mode'] == 'stationary'


def test_explicit_stationary_with_empty_gripper_never_uses_active_observation():
    executor = fake_executor()
    executor.session.runner.has_held_recovery = False
    executor.session.snapshot = lambda: {'held_object': None}
    calls = []
    executor._stationary = lambda: calls.append('stationary') or []
    executor.session.observe = lambda **kw: pytest.fail('stationary mode must not command a view change')
    observed = executor.execute({'id': 'look', 'skill': 'observe', 'mode': 'stationary'}, [])
    assert calls == ['stationary'] and observed['ok'] and not observed['holding']
    assert observed['observation_mode'] == 'stationary' and observed['changed'] is False


def test_capabilities_advertise_implemented_modes_and_no_arbitrary_robot_code():
    executor = fake_executor()
    capabilities = executor.capabilities()
    assert capabilities['observe_modes'] == ['stationary', 'active']
    assert capabilities['active_observe_requires_geometry'] is True
    assert set(capabilities['skills']) == {'observe', 'pick', 'place', 'move', 'verify', 'ask', 'finish'}


def test_explicit_observation_label_takes_precedence_over_held_recovery_goal():
    executor = fake_executor()
    executor.session._recovery_goal = {'label': 'green', 'zone_name': 'old_goal'}
    executor.session.scene['objects'] = [
        {'label': 'banana', 'valid': True, 'center_base': [.28, -.12, .04]}]

    def no_recovery_destination(value, goals):
        pytest.fail('an explicit observed label must not be replaced by the recovery destination')

    executor._resolve = no_recovery_destination
    xy, polygon = executor._observation_focus({'label': 'banana'}, [])
    np.testing.assert_allclose(xy, [.28, -.12])
    assert polygon.shape == (4, 2)


def test_explicit_observation_destination_keeps_its_full_region_polygon():
    executor = fake_executor()
    polygon = np.array([[.1, -.2], [.1, .1], [.5, .1], [.5, -.2]])
    executor._resolve = lambda value, goals: ({'type': 'named_region', 'name': 'goal'}, None)
    executor._options = lambda destination: {
        'placement_zone': SimpleNamespace(center=[.3, -.05, 0.]),
        'placement_polygon': polygon}
    xy, focused = executor._observation_focus(
        {'label': 'banana', 'destination': {'type': 'goal', 'index': 0}}, [])
    np.testing.assert_allclose(xy, [.3, -.05])
    np.testing.assert_array_equal(focused, polygon)


def test_unknown_observation_label_requires_a_measured_location():
    executor = fake_executor()
    with pytest.raises(ValueError, match='实测观察位置'):
        executor._observation_focus({'label': 'banana'}, [])


@pytest.fixture
def active_harness(monkeypatch):
    """Exercise the real active executor without a simulator, IK, or motion."""
    h = SimpleNamespace(moves=[], verifies=[], routes=[], feedback_ok=True, feedback_width=.05,
                        on_move=lambda: None, on_apply=lambda: None,
                        on_route=lambda: None, on_verify=lambda: None,
                        coverage_calls=0, gain_coverage=False)
    current, first, second = (np.eye(4) for _ in range(3))
    current[:3, 3], first[:3, 3], second[:3, 3] = (
        [.25, .05, .25], [.25, .10, .30], [.25, .15, .35])
    h.poses, h.route = [second], [first, second]
    target = SimpleNamespace(label='green')
    context = {'det': target, 'grasp': {'measured_anchor': [1., 2., 3.]}, 'width': .05}
    h.neighbors = [SimpleNamespace(label='banana', valid=False)]

    def move(result, stage, waypoint, linear):
        h.moves.append((stage, waypoint.copy(), linear))
        current[:] = waypoint
        h.on_move()

    def verify(result, stage):
        h.verifies.append(stage)
        h.on_verify()

    skill = SimpleNamespace(_move=move, _verify_held=verify, execution_guard=None)
    runner = SimpleNamespace(
        has_held_recovery=True, skill=skill, _held_recovery_context=context,
        ext=SimpleNamespace(parent_frame='flange', T=np.eye(4)),
        T_flange_tcp=np.eye(4), free_tilt=40., nominal={'green': [.05, .05, .05]})

    def feedback(target):
        return ((runner._held_recovery_context, h.feedback_width, '') if h.feedback_ok else
                (None, None, 'held grasp feedback missing or changed'))

    runner._held_transfer_feedback = feedback
    session = SimpleNamespace(
        _operation_lock=threading.RLock(), _check_open=lambda: None,
        cancel_event=threading.Event(), runner=runner,
        arm=SimpleNamespace(get_tcp_pose=lambda: SimpleNamespace(value=current)),
        camera=SimpleNamespace(intrinsics=lambda: None), cfg={},
        scene={'scene_version': 1, 'objects': []}, _environment_revision=1,
        _interventions=[], _on_stage=lambda stage: None,
        _emit=lambda kind, fields: None, _publish_snapshot=lambda: None,
        snapshot=lambda: {'held_object': 'green', 'editor': {}})
    session._apply_interventions = lambda: h.on_apply()

    def route(*args, **kwargs):
        h.routes.append((args, kwargs))
        h.on_route()
        return h.route, '' if h.route else 'incomplete neighbor geometry'

    def coverage(session, polygon):
        h.coverage_calls += 1
        return np.array([h.gain_coverage and h.coverage_calls > 1, False])

    monkeypatch.setattr(active_observation, 'iter_safe_inspection_poses',
                        lambda *args, **kwargs: iter(h.poses))
    monkeypatch.setattr(active_observation, 'safe_inspection_route', route)
    monkeypatch.setattr(active_observation, '_coverage', coverage)
    monkeypatch.setattr(active_observation, 'remember_inspection', lambda *args: None)
    monkeypatch.setattr(active_observation, 'inspection_history', lambda *args: [])
    executor = SkillExecutor(session)
    executor._stationary = lambda: h.neighbors
    executor._observation_focus = lambda step, goals: (
        np.array([.25, .05]), np.array([[.2, 0.], [.2, .1], [.3, .1], [.3, 0.]]))
    h.session, h.executor, h.context = session, executor, context
    h.execute = lambda: executor.execute(
        {'id': 'look', 'skill': 'observe', 'mode': 'active'}, [], expected_revision=1)
    return h


def test_active_observation_requires_full_route_before_any_motion(active_harness):
    h = active_harness
    h.route = []
    result = h.execute()
    assert not result['ok'] and result['error'] == 'no_safe_observation_path'
    assert h.moves == []
    args, kwargs = h.routes[0]
    assert args[4] is h.neighbors
    assert args[5] == .05
    assert args[1] == h.context['grasp']
    assert kwargs['neighbor_nominal'] is h.session.runner.nominal
    assert result['holding']


def test_explicit_active_observation_preserves_unlocalized_moved_body_exclusion(active_harness):
    from cube_task.core.types import Detection
    h = active_harness
    unknown = Detection('unobserved_pear', 0., valid=False, top_z=.14)
    h.session._motion_observer = SimpleNamespace(invalidated={'pear'},
        uncertainty_obstacles=lambda **kw: [unknown])
    h.route = []
    result = h.execute()
    assert not result['ok'] and not h.moves
    assert any(d is unknown for d in h.routes[0][0][4])


def test_stop_during_candidate_search_never_reaches_route_or_motion(active_harness, monkeypatch):
    h = active_harness

    def stopped_search(*args, **kwargs):
        h.session.cancel_event.set()
        yield h.poses[0]

    monkeypatch.setattr(active_observation, 'iter_safe_inspection_poses', stopped_search)
    result = h.execute()
    assert result['error'] == 'aborted' and result['aborted']
    assert h.routes == [] and h.moves == []


def test_stop_after_first_waypoint_prevents_later_waypoints(active_harness):
    h = active_harness
    h.on_move = h.session.cancel_event.set
    result = h.execute()
    assert result['error'] == 'aborted' and result['holding']
    assert len(h.moves) == 1


def test_intervention_after_route_check_invalidates_it_before_motion(active_harness):
    h = active_harness

    def intervene():
        h.session._environment_revision += 1

    h.on_apply = intervene
    result = h.execute()
    assert result['error'] == 'scene_changed'
    assert h.moves == []
    assert h.session.runner.skill.execution_guard is None


@pytest.mark.parametrize('when,expected_moves', [('route_check', 0), ('first_move', 1)])
def test_changed_held_feedback_prevents_unexecuted_motion(active_harness, when, expected_moves):
    h = active_harness
    setattr(h, 'on_route' if when == 'route_check' else 'on_move',
            lambda: setattr(h, 'feedback_ok', False))
    result = h.execute()
    assert not result['ok']
    assert len(h.moves) == expected_moves


def test_normal_submillimetre_grip_feedback_is_included_in_the_checked_finger_sweep(active_harness):
    h = active_harness
    h.gain_coverage = True
    h.on_route = lambda: setattr(h, 'feedback_width', .05007)
    result = h.execute()
    assert result['ok'] and result['holding'], result
    assert len(h.moves) == 2
    assert h.routes[0][1]['opening_uncertainty'] == pytest.approx(.001)


@pytest.mark.parametrize('case', ['carried_only', 'obstacle_moved', 'dropped',
                                 'stopped', 'capture_exception'])
def test_prepared_physics_observation_keeps_motion_state_across_real_scene_publication(active_harness, case):
    from cube_task.assistant.simulation import SimulationSession
    h = active_harness
    carried = SimpleNamespace(name='held-green', label='green', T_base_obj=np.eye(4))
    obstacle = SimpleNamespace(name='other-banana', label='banana', T_base_obj=np.eye(4))
    session = h.session
    session.world = SimpleNamespace(objects=[carried, obstacle])
    session._prepared = True
    session._status = 'ready'
    session._physics = SimpleNamespace(expected_labels=set())
    session._remember_physics = MethodType(SimulationSession._remember_physics, session)
    session._remember_physics()
    session._observed_detections, session._scan_stats, session._placement = [], {}, {}
    session._scene_version, session.backend = 0, 'mujoco'
    session._zones = lambda: []
    session._publish_scene = MethodType(SimulationSession._publish_scene, session)
    publications = []
    session._publish_snapshot = lambda: publications.append(session._status)
    session.arm.get_flange_pose = lambda: SimpleNamespace(value=np.eye(4))
    session.camera._last = None
    session.runner._last_stamp, session.runner.class_to_zone = 0., {}

    def capture():
        if case == 'capture_exception':
            raise RuntimeError('camera unavailable')
        return []

    session._stationary_recovery_observation = capture
    # Retain the real capture -> merge -> scene publication call chain. Scene
    # publication sets ready even though the held observation is still moving.
    h.executor._stationary = MethodType(SkillExecutor._stationary, h.executor)

    def physics_tick():
        carried.T_base_obj[0, 3] += .02
        if case == 'obstacle_moved':
            obstacle.T_base_obj[0, 3] += .02
        if case == 'stopped':
            session.cancel_event.set()
        SimulationSession._check_physical_changes(session)

    h.on_move = physics_tick
    if case == 'dropped':
        def lost():
            raise _Lost('dropped_in_transit', .001)
        h.on_verify = lost
    if case == 'capture_exception':
        with pytest.raises(RuntimeError, match='camera unavailable'):
            h.execute()
        assert not h.moves and session._status == 'ready'
        assert publications[-1] == 'ready'
        assert session.runner._held_recovery_context is h.context
        return

    result = h.execute()
    if case == 'obstacle_moved':
        assert result['error'] == 'scene_changed' and len(h.moves) == 1
        assert session._environment_revision == 2
    elif case == 'dropped':
        assert result['error'] == 'dropped_in_transit' and len(h.moves) == 1
        assert not result['holding'] and session.runner._held_recovery_context is None
    elif case == 'stopped':
        assert result['error'] == 'aborted' and len(h.moves) == 1
        assert result['holding'] and session.runner._held_recovery_context is h.context
    else:
        assert result['ok'] and len(h.moves) == 2
        assert session._environment_revision == 1
    expected_status = 'stopped' if case == 'stopped' else 'ready'
    assert session._status == publications[-1] == expected_status


@pytest.mark.parametrize('width', [.0489, .0511])
def test_width_outside_checked_geometry_requires_replan_without_moving_or_faulting(active_harness, width):
    h = active_harness
    h.on_route = lambda: setattr(h, 'feedback_width', width)
    result = h.execute()
    assert not result['ok'] and result['holding'], result
    assert result['error'] == 'held_feedback_changed'
    assert result['reason'] == 'opening_outside_checked_sweep'
    assert not result.get('fault')
    assert h.moves == []
    assert h.session.runner._held_recovery_context is h.context


def test_changed_grasp_context_cannot_reuse_an_old_checked_route(active_harness):
    h = active_harness
    h.on_route = lambda: setattr(h.session.runner, '_held_recovery_context', copy.deepcopy(h.context))
    result = h.execute()
    assert not result['ok'] and result['holding'], result
    assert result['reason'] == 'held_grasp_context_changed'
    assert h.moves == []


def test_lost_grip_stops_route_and_expires_measured_recovery(active_harness):
    h = active_harness

    def lost():
        raise _Lost('dropped_in_transit', .001)

    h.on_verify = lost
    result = h.execute()
    assert not result['ok'] and not result['holding']
    assert result['error'] == 'dropped_in_transit'
    assert len(h.moves) == 1
    assert h.session.runner._held_recovery_context is None


def test_useful_new_coverage_finishes_one_view_without_changing_grasp(active_harness):
    h = active_harness
    h.gain_coverage = True
    original = copy.deepcopy(h.context['grasp'])
    result = h.execute()
    assert result['ok'] and result['changed'] and result['views'] == 1
    assert result['holding'] and result['camera_verified']
    assert len(h.moves) == len(h.route) == 2
    assert h.context['grasp'] == original
    assert all(stage == 'inspect_place' and linear for stage, _, linear in h.moves)


def test_repeated_views_in_later_plan_do_not_claim_new_coverage(active_harness):
    h = active_harness
    h.gain_coverage = True
    assert h.execute()['changed'] is True
    h.coverage_calls = 0
    h.session.arm.get_tcp_pose().value[:3, 3] = [.25, .05, .25]
    repeated = h.execute()
    assert repeated['ok'] and repeated['changed'] is False
    # A real user edit invalidates coverage accumulated for the old geometry.
    h.session._environment_revision += 1
    h.coverage_calls = 0
    h.session.arm.get_tcp_pose().value[:3, 3] = [.25, .05, .25]
    h.executor._expected_revision = None
    refreshed = active_observation.inspect_held(h.session, np.array([.25, .05]),
        np.array([[.2, 0.], [.2, .1], [.3, .1], [.3, 0.]]), h.executor._stationary, lambda: None)
    assert refreshed['changed'] is True


@pytest.mark.parametrize('case', ['incomplete_neighbor', 'swept_fingers', 'native_path'])
def test_real_active_route_keeps_geometry_and_native_path_checks(active_harness, monkeypatch, case):
    h = active_harness
    target = Detection('green', 1., center_base=np.array([0., 0., .05]), top_z=.05,
                       height=.05, length=.05, width=.05, yaw=0.)
    grasp = GraspPlan(target, top_down_T(0., 0., .15, 0.), top_down_T(0., 0., .02, 0.),
                      top_down_T(0., 0., .2, 0.), .06, 0., .05, 0.)
    h.context.update(det=target, grasp=grasp)
    h.poses = [top_down_T(0., .4, .15, 0.)]
    h.session.arm.get_tcp_pose = lambda: SimpleNamespace(value=top_down_T(0., 0., .15, 0.))
    h.session.arm.check_reachable = lambda *args, **kwargs: (True, '')
    if case == 'native_path':
        h.neighbors = []
        h.session.arm._check_linear_path = lambda *args: (False, 'joint path unavailable')
    else:
        # The 6 mm object lies outside the carried object's XY footprint;
        # its location still intersects the closed finger sweep.
        h.neighbors = [Detection('banana', 1., valid=False,
            center_base=np.array([.05, .2, .2]), top_z=.2,
            height=.2, length=.006, width=None if case == 'incomplete_neighbor' else .006,
            yaw=0.)]
    monkeypatch.setattr(active_observation, 'safe_inspection_route', safe_inspection_route)
    result = h.execute()
    assert result['error'] == 'no_safe_observation_path'
    assert h.moves == []
    expected = {'incomplete_neighbor': 'incomplete neighbor geometry',
                'swept_fingers': 'overlaps', 'native_path': 'joint path unavailable'}
    assert expected[case] in result['reason']
