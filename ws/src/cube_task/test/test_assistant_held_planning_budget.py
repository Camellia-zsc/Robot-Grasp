"""The first held-place search shares a finite budget across every candidate."""
import copy
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.core.types import Detection, Zone
from test_planning_cancellation import planning_arm

SEARCH_SECONDS = 15.


def held_search(monkeypatch, *, candidate_seconds=40., count=1, stop=False, succeed=False):
    now, events, calls = [0.], [], []
    monkeypatch.setattr('cube_task.assistant.simulation.time.monotonic', lambda: now[0])
    det = Detection('green', 1., center_base=np.array([.3, .1, .05]),
                    height=.05, length=.05, width=.05, top_z=.05, yaw=0.)
    context = dict(det=det, grasp=SimpleNamespace(T_grasp=np.eye(4)), width=.05)
    zone = Zone('temporary', np.array([.3, -.1, 0.]), (.2, .2))
    session = SimpleNamespace(cancel_event=threading.Event(), arm=planning_arm(),
        _environment_revision=2, _recovery_evaluation=None,
        _emit=lambda kind, fields: events.append((kind, fields)),
        _on_stage=Mock(), _publish_snapshot=Mock(),
        _finish_attempt=lambda *a: dict(ok=True, holding=False))
    session.snapshot = lambda: dict(held_object='green')

    def plan(target, neighbors, **kwargs):
        calls.append(kwargs)
        cancelled, progress = kwargs.get('cancelled'), kwargs.get('progress')
        for _ in range(round(candidate_seconds * 4)):
            if stop:
                session.cancel_event.set()
            if cancelled is not None and cancelled():
                return None, 'aborted' if stop else 'placement_planning_budget_exceeded'
            if session.arm._planning_cancelled():
                return None, 'aborted' if stop else 'placement_planning_budget_exceeded'
            now[0] += .25
            if progress:
                progress(dict(phase='release_candidates', attempt=1, max_attempts=1))
        return (dict(det=copy.deepcopy(target)), '') if succeed else (None, 'no_clear_supported_release')

    def execute(bundle):
        # Physical movement belongs outside the computation deadline.
        now[0] += 30.
        assert not session.arm._planning_cancelled()
        return SimpleNamespace(ok=True), ''

    session.runner = SimpleNamespace(_held_recovery_context=context, has_held_recovery=True,
                                    plan_held_transfer=plan, execute_held_transfer=Mock(side_effect=execute))
    executor = SkillExecutor(session)
    executor._stationary = lambda: []
    executor._resolve = lambda destination, goals: (copy.deepcopy(destination), None)
    executor._options = lambda destination: dict(placement_zone=zone,
                                               placement_polygon=zone.corners_xy(), exact_xy=True)
    executor._candidates = lambda *args: [dict(type='point', xy=[.3 + .01*i, -.1]) for i in range(count)]
    action = dict(id='place-held', skill='place', label='green', destination=dict(type='auto_free'))
    return executor, session, action, now, events, calls


def test_fifteen_second_search_timeout_preserves_grasp_and_does_not_latch_stop(monkeypatch):
    executor, session, action, now, events, calls = held_search(monkeypatch)
    context = session.runner._held_recovery_context
    anchor = context['grasp'].T_grasp.copy()
    result = executor._place(action, [], [], None)
    assert result['error'] == 'placement_planning_budget_exceeded'
    assert result['holding'] and result['fault'] is False and result['aborted'] is False
    assert now[0] == SEARCH_SECONDS and len(calls) == 1
    assert session.runner._held_recovery_context is context
    np.testing.assert_array_equal(context['grasp'].T_grasp, anchor)
    assert not session.cancel_event.is_set() and not session.arm._planning_cancelled()
    session.runner.execute_held_transfer.assert_not_called()
    assert events and calls[0].get('max_attempts') is None


def test_candidate_loop_shares_fifteen_seconds_and_throttles_ui_progress(monkeypatch):
    executor, session, action, now, events, calls = held_search(
        monkeypatch, candidate_seconds=3., count=10)
    result = executor._place(action, [], [], None)
    assert result['error'] == 'placement_planning_budget_exceeded'
    assert now[0] == SEARCH_SECONDS and len(calls) == 5
    assert all(call.get('max_attempts') is None for call in calls)
    progress = [fields for kind, fields in events if kind == 'placement_progress']
    assert 2 <= len(progress) <= SEARCH_SECONDS + 2
    assert all(fields['message'] for fields in progress)
    session.runner.execute_held_transfer.assert_not_called()


def test_stop_during_initial_search_never_tries_another_candidate(monkeypatch):
    executor, session, action, _now, _events, calls = held_search(monkeypatch, count=10, stop=True)
    result = executor._place(action, [], [], None)
    assert result['error'] == 'aborted' and result['aborted'] and result['holding']
    assert session.cancel_event.is_set() and len(calls) == 1
    session.runner.execute_held_transfer.assert_not_called()


def test_successful_initial_search_releases_planning_scope_before_motion(monkeypatch):
    executor, session, action, now, _events, calls = held_search(
        monkeypatch, candidate_seconds=.5, succeed=True)
    result = executor._place(action, [], [], None)
    assert result['ok'] and now[0] == 30.5
    assert calls[0].get('max_attempts') is None
    session.runner.execute_held_transfer.assert_called_once()
    assert not session.cancel_event.is_set() and not session.arm._planning_cancelled()


def test_initial_search_can_complete_twelve_seconds_of_geometry_without_latching_stop(monkeypatch):
    executor, session, action, now, _events, calls = held_search(
        monkeypatch, candidate_seconds=12., succeed=True)
    result = executor._place(action, [], [], None)
    assert result['ok'] and now[0] == 42. and len(calls) == 1
    session.runner.execute_held_transfer.assert_called_once()
    assert not session.cancel_event.is_set() and not session.arm._planning_cancelled()


def test_held_transfer_forwards_optional_phase_budget_without_changing_default(monkeypatch):
    from test_runner_held_transfer import transfer_scene
    runner, target, _grasp, zone = transfer_scene()
    runner.replan_release = Mock(return_value=(None, 'placement_planning_budget_exceeded'))
    cancelled, progress = lambda: False, Mock()
    bundle, why = runner.plan_held_transfer(target, [], zone, zone.corners_xy(),
        cancelled=cancelled, progress=progress, max_attempts=1)
    assert bundle is None and why == 'placement_planning_budget_exceeded'
    kwargs = runner.replan_release.call_args.kwargs
    assert kwargs['cancelled'] is cancelled and kwargs['progress'] is progress
    assert kwargs['max_attempts'] == 1
    assert runner.has_held_recovery


def fixed_search(monkeypatch):
    from test_runner_held_transfer import transfer_scene
    executor, session, action, now, events, _calls = held_search(monkeypatch)
    runner, target, _grasp, zone = transfer_scene()
    runner.execute_held_transfer = Mock(return_value=(SimpleNamespace(ok=True), ''))
    session.runner = runner
    executor._resolve = lambda *a: (dict(type='point', xy=zone.center[:2].tolist()), None)
    executor._options = lambda *a: dict(placement_zone=zone,
                                       placement_polygon=zone.corners_xy(), exact_xy=True)
    action.update(label=target.label, destination=dict(type='goal', index=0))
    return executor, session, action, now, events, runner


def test_fixed_point_can_try_a_second_orientation_inside_the_same_budget(monkeypatch):
    executor, session, action, _now, _events, runner = fixed_search(monkeypatch)
    orientations, centers = [], []
    def final_check(_det, _gp, pp, *_args, **_kwargs):
        orientations.append(np.arctan2(pp.T_release[1, 1], pp.T_release[0, 1]))
        centers.append(pp.T_release[:2, 3].copy())
        return (False, 'first orientation blocked') if len(orientations) == 1 else (True, '')
    runner._adjust_strict_release = final_check
    result = executor._place(action, [], [], None)
    assert result['ok'] and len(orientations) == 2
    assert orientations[0] != pytest.approx(orientations[1])
    np.testing.assert_array_equal(centers[0], centers[1])
    runner.execute_held_transfer.assert_called_once()
    assert not session.cancel_event.is_set()


def test_fixed_point_orientation_retries_still_share_fifteen_second_deadline(monkeypatch):
    executor, session, action, now, _events, runner = fixed_search(monkeypatch)
    checks = []
    def final_check(*_args, **_kwargs):
        checks.append(True)
        for _ in range(20):
            if now[0] >= SEARCH_SECONDS:
                break
            now[0] += .25
        return False, 'orientation blocked'
    runner._adjust_strict_release = final_check
    result = executor._place(action, [], [], None)
    assert result['error'] == 'placement_planning_budget_exceeded'
    assert result['holding'] and result['fault'] is False and now[0] == SEARCH_SECONDS
    assert len(checks) == 3
    runner.execute_held_transfer.assert_not_called()
    assert not session.cancel_event.is_set()


def test_region_can_advance_after_a_blocked_first_candidate_without_restarting(monkeypatch):
    executor, session, action, _now, _events, runner = fixed_search(monkeypatch)
    zone = Zone('temporary', np.array([.3, -.1, 0.]), (.2, .2))
    executor._resolve = lambda *a: (dict(type='named_region', name=zone.name), None)
    executor._options = lambda *a: dict(placement_zone=zone,
                                       placement_polygon=zone.corners_xy(), exact_xy=False)
    poses = []
    def final_check(_det, _gp, pp, *_args, **_kwargs):
        poses.append(pp.T_release.copy())
        return (False, 'first candidate blocked') if len(poses) == 1 else (True, '')
    runner._adjust_strict_release = final_check
    result = executor._place(action, [], [], None)
    assert result['ok'] and len(poses) == 2
    # A second wrist direction at the same centre is also a distinct candidate.
    assert not np.array_equal(poses[0], poses[1])
    runner.execute_held_transfer.assert_called_once()
    assert not session.cancel_event.is_set()


def test_physics_planning_budget_counts_worker_cpu_but_keeps_stop_and_wall_limit(monkeypatch):
    from cube_task.assistant.simulation import _PlacementBudget, _PlacementInterrupted
    cpu, wall = [0.], [0.]
    monkeypatch.setattr('cube_task.assistant.simulation.time.monotonic',lambda:wall[0])
    arm = SimpleNamespace(planning_time=lambda:cpu[0])
    stop = threading.Event()
    budget = _PlacementBudget(arm,stop,seconds=15.)
    def waited():
        wall[0] += 20.
        cpu[0] += 1.
        return 'planned'
    assert budget.run(waited) == 'planned' and budget.remaining == 14.
    def excessive():
        cpu[0] += 15.
    with pytest.raises(_PlacementInterrupted):budget.run(excessive)
    budget = _PlacementBudget(arm,stop,seconds=15.)
    def no_cpu_progress():wall[0] += 91.
    with pytest.raises(_PlacementInterrupted):budget.run(no_cpu_progress)
    budget = _PlacementBudget(arm,stop,seconds=15.)
    stop.set()
    with pytest.raises(_PlacementInterrupted):budget.run(lambda:None)


@pytest.mark.parametrize('confirmed',[True,False])
def test_held_place_calibrates_actual_grip_before_selecting_landing(monkeypatch,confirmed):
    executor,session,action,now,events,calls=held_search(monkeypatch,candidate_seconds=1.,succeed=True)
    context=session.runner._held_recovery_context
    order=[]
    session.runner._held_transfer_feedback=lambda target:(context,.05,'')
    def calibrate(grasp,result,width,first):
        order.append('measure')
        if confirmed:grasp.T_grasp[2,3]+=.006
        return confirmed,'source unconfirmed' if not confirmed else '',None
    session._calibrate_carried_anchor=calibrate
    original=session.runner.plan_held_transfer
    def plan(*args,**kwargs):
        order.append('plan')
        assert context['grasp'].T_grasp[2,3]==pytest.approx(.006)
        return original(*args,**kwargs)
    session.runner.plan_held_transfer=plan
    result=executor._place(action,[],[],None)
    assert order==(['measure','plan'] if confirmed else ['measure'])
    assert result['ok'] is confirmed
    if not confirmed:session.runner.execute_held_transfer.assert_not_called()
