"""A camera proof cannot conceal physical support loss below pose thresholds."""
import threading
from types import SimpleNamespace

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_orchestrator import SkillOrchestrator
from test_assistant_physics import build, no_network


def test_cached_stack_proof_rechecks_contact_below_physical_revision_threshold(build):
    physics, _base, upper, _events = build()
    physics.settle(1.2)
    session = SimpleNamespace(_physics=physics, world=physics.world, _prepared=True,
        _status='ready', _environment_revision=0, _interventions=[],
        cancel_event=threading.Event(),
        runner=SimpleNamespace(has_held_recovery=False, _held_recovery_context=None))
    session.snapshot = lambda: dict(scene_version=1, held_object=None, editor={})
    session._remember_physics = lambda: SimulationSession._remember_physics(session)
    session._emit = lambda *args: None
    session._remember_physics()
    controller = SimpleNamespace(_session=session, _cancel=threading.Event())
    orchestrator = SkillOrchestrator(controller)
    goals = [dict(label='apple', zone_name=None, destination=dict(type='on_object', label='red'))]
    proof = orchestrator._proof(dict(ok=True, camera_verified=True, truth={'success': True},
        verified_revision=0, items=[dict(goal=goals[0], ok=True, truth={'success': True})]), goals)
    assert physics.support_status(upper.name, 'red')['success']
    assert orchestrator._cached_verification(proof, goals) is not None

    # The revision monitor tolerates 6 mm, while a 4 mm separation already
    # loses the real MuJoCo support contact. No camera call is involved here.
    pose = upper.T_base_obj.copy()
    pose[2, 3] += .004
    physics.mujoco_world.set_object_pose(upper.name, pose)
    physics.sync()
    SimulationSession._check_physical_changes(session)
    assert session._environment_revision == 0
    assert not physics.support_status(upper.name, 'red')['contact']
    assert orchestrator._cached_verification(proof, goals) is None
