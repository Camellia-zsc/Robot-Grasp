"""Placement computation is bounded without clearing stop or releasing a grasp."""
from contextlib import contextmanager
from types import SimpleNamespace
import threading

import pytest

from cube_task.assistant.simulation import (
    SimulationSession, _PlacementBudget, _PlacementInterrupted,
)
from cube_task.core.types import PickResult


def test_budget_counts_computation_and_excludes_observation_motion(monkeypatch):
    now, scopes = [0.], []
    monkeypatch.setattr('cube_task.assistant.simulation.time.monotonic', lambda: now[0])
    stop = threading.Event()

    @contextmanager
    def planning_scope(cancelled):
        scopes.append(cancelled)
        try:
            yield
        finally:
            scopes.pop()

    budget = _PlacementBudget(SimpleNamespace(planning_scope=planning_scope), stop, seconds=2.)
    def compute():
        now[0] += .5
        assert not scopes[-1]()
        return 'checked'
    assert budget.run(compute) == 'checked'
    now[0] += 30.  # Robot/camera movement takes real time, not computation budget.
    assert budget.run(compute) == 'checked'
    assert budget.remaining == 1.
    def expensive():
        now[0] += 1.1
        assert scopes[-1]()
    with pytest.raises(_PlacementInterrupted):
        budget.run(expensive)
    assert not scopes and not stop.is_set()  # Timeout cannot masquerade as user stop.


def test_stop_at_release_callback_does_no_observation_or_planning():
    session = object.__new__(SimulationSession)
    session.cancel_event = threading.Event()
    session.cancel_event.set()
    assert session._before_release(None, None, None, None) is None
    assert session.cancel_event.is_set()


@pytest.mark.parametrize('user_stop', [False, True])
def test_interrupted_release_keeps_held_result_and_reports_only_timeout(monkeypatch, user_stop):
    session = object.__new__(SimulationSession)
    session.cancel_event = threading.Event()
    session.arm = SimpleNamespace()
    events = []
    session._emit = lambda kind, fields: events.append((kind, fields))
    def interrupted(*args):
        if user_stop:
            session.cancel_event.set()
        raise _PlacementInterrupted()
    session._refine_release = interrupted
    result = PickResult(False, detail={'holding': True})
    assert session._before_release(None, None, .041, result) is None
    assert result.detail['holding']
    assert result.detail['placement_planning_reason'] == (
        'aborted' if user_stop else 'placement_planning_budget_exceeded')
    assert bool(events) is not user_stop
    assert session.cancel_event.is_set() == user_stop


def test_deadline_during_grid_search_is_returned_as_planning_feedback():
    from test_runner_local_release import scene
    runner, target, grasp, previous = scene()
    expired = []
    def visibility(*args, **kwargs):
        expired.append(True)
        return False
    result, reason = runner.replan_release(target, grasp, previous, .05, [target],
        observation_filter=visibility, cancelled=lambda: bool(expired))
    assert result is None and reason == 'placement_planning_budget_exceeded'
    assert len(expired) == 1 and not runner.events
