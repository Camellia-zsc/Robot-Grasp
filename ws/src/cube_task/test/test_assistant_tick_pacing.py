"""Viewer pacing must not add display work to every simulation wait."""
from types import SimpleNamespace

import pytest

from cube_task.assistant import simulation


def test_tick_limits_display_publication_but_applies_every_intervention(monkeypatch):
    wall = [0.]
    monkeypatch.setattr(simulation.time, 'monotonic', lambda: wall[0])
    session = object.__new__(simulation.SimulationSession)
    session._environment_revision = 0
    session._last_snapshot_wall = float('-inf')
    applied, published = [], []

    def intervene():
        applied.append(wall[0])
        if len(applied) == 12:
            session._environment_revision += 1

    def publish():
        published.append(wall[0])
        session._last_snapshot_wall = wall[0]

    session._apply_interventions = intervene
    session._publish_snapshot = publish
    for _ in range(100):
        session._tick()
        wall[0] += .001
    assert len(applied) == 100
    assert len(published) <= 4
    assert published[0] == 0.
    # A human edit is visible immediately, even between display intervals.
    assert published[1] == pytest.approx(.011)


@pytest.mark.parametrize('speed', [1., 2., 4.])
def test_display_work_is_included_in_playback_time_instead_of_added_to_it(monkeypatch, speed):
    wall, waits = [0.], []
    monkeypatch.setattr(simulation.time, 'monotonic', lambda: wall[0])

    def wait(dt):
        waits.append(dt)
        wall[0] += dt

    event = SimpleNamespace(is_set=lambda: False, wait=wait)
    clock = simulation._ViewerClock(event, realtime=True)
    clock.paced, clock.speed = True, speed
    clock.on_tick = lambda: wall.__setitem__(0, wall[0]+.005)
    before = clock.now()
    clock.sleep(.1)
    assert clock.now()-before == pytest.approx(.1)
    assert wall[0] == pytest.approx(.1/speed)
    assert all(dt >= 0 for dt in waits)


def test_display_overrun_never_causes_a_negative_wait_or_skips_ticks(monkeypatch):
    wall, ticks, waits = [0.], [], []
    monkeypatch.setattr(simulation.time, 'monotonic', lambda: wall[0])
    event = SimpleNamespace(is_set=lambda: False, wait=waits.append)
    clock = simulation._ViewerClock(event, realtime=True)
    clock.paced, clock.speed = True, 4.

    def tick():
        ticks.append(True)
        wall[0] += .1

    clock.on_tick = tick
    clock.sleep(.1)
    assert len(ticks) == 2
    assert not waits


@pytest.mark.parametrize('physics_cost', [.004, .03])
def test_physics_computation_is_included_in_pacing_and_every_step_is_preserved(monkeypatch, physics_cost):
    wall, waits, advances = [0.], [], []
    monkeypatch.setattr(simulation.time, 'monotonic', lambda: wall[0])
    def wait(dt):
        waits.append(dt)
        wall[0] += dt
    event = SimpleNamespace(is_set=lambda: False, wait=wait)
    clock = simulation._ViewerClock(event, realtime=True)
    clock.paced, clock.speed = True, 4.
    def physics(old, new):
        advances.append(new-old)
        wall[0] += physics_cost
    clock.add_listener(physics)
    clock.sleep(.1)
    assert advances == pytest.approx([.05, .05])
    assert wall[0] == pytest.approx(2*max(.05/4, physics_cost))
    if physics_cost > .05/4:
        assert not waits
