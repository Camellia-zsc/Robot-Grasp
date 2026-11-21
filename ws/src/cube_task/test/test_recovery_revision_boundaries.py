"""Legacy recovery offers use the same environment gate as semantic skills."""
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.core.types import PickResult
from test_assistant_recovery_simulation import choose
from test_task_fakes import make_det


@pytest.fixture
def held_recovery(monkeypatch, tmp_path):
    with SimulationSession(realtime=False, log_dir=tmp_path) as sim:
        sim._prepared = True
        sim.world.held = next(o for o in sim.world.objects if o.label == 'red')
        target = make_det('red', .2, 0.)
        sim._recovery_target = target
        sim._recovery_goal = dict(label='red', zone_name='zone_red')
        sim.runner._held_recovery_context = dict(det=target, width=.03)
        sim._environment_revision, sim._active_revision = 7, 6
        monkeypatch.setattr(sim, '_stationary_recovery_observation', lambda: [])
        monkeypatch.setattr('cube_task.assistant.held_recovery.propose_held_recovery',
                            lambda *_args: [])
        calls = []

        def retry(*_args, **_kwargs):
            calls.append(sim._execution_guard('held_transit'))
            return PickResult(False, error='scene_changed', detail={'holding': True}), ''

        monkeypatch.setattr(sim.runner, 'retry_held_placement', retry)
        monkeypatch.setattr(sim, '_finish_attempt', lambda *_args: {'ok': False, 'holding': True})
        yield sim, calls


def test_fresh_recovery_rebinds_active_revision_without_discarding_held_grasp(held_recovery):
    sim, calls = held_recovery
    original = sim.runner._held_recovery_context
    context = sim.prepare_recovery({'ok': False}, sim._recovery_goal, 1, [])
    assert context['environment_revision'] == 7
    sim.execute_recovery(choose(context, 'place_held_alternate'))
    assert calls == [None]
    assert sim._active_revision == sim._environment_revision == 7
    assert sim.runner._held_recovery_context is original


def test_physics_change_invalidates_recovery_even_without_a_new_camera_scene(held_recovery):
    sim, calls = held_recovery
    context = sim.prepare_recovery({'ok': False}, sim._recovery_goal, 1, [])
    scene_version = sim._scene_version
    sim._environment_revision += 1
    with pytest.raises(ValueError, match='已变化'):
        sim.execute_recovery(choose(context, 'place_held_alternate'))
    assert sim._scene_version == scene_version and not calls
    assert sim._active_revision == 6 and sim.world.held is not None


def test_change_during_recovery_camera_capture_cannot_create_a_fresh_offer(held_recovery, monkeypatch):
    sim, calls = held_recovery

    def capture():
        sim._environment_revision += 1
        return []

    monkeypatch.setattr(sim, '_stationary_recovery_observation', capture)
    with pytest.raises(ValueError, match='已变化'):
        sim.prepare_recovery({'ok': False}, sim._recovery_goal, 1, [])
    assert sim._recovery_context is None and not calls
    assert sim.world.held is not None


def test_queued_change_applied_after_recovery_validation_cannot_dispatch(held_recovery, monkeypatch):
    sim, calls = held_recovery
    context = sim.prepare_recovery({'ok': False}, sim._recovery_goal, 1, [])
    applications = []

    def apply():
        applications.append(True)
        if len(applications) == 2:
            sim._environment_revision += 1

    monkeypatch.setattr(sim, '_apply_interventions', apply)
    with pytest.raises(ValueError, match='已变化'):
        sim.execute_recovery(choose(context, 'place_held_alternate'))
    assert not calls and sim.world.held is not None
