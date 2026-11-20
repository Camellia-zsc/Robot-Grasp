"""Contact-solver accuracy must not let a resting stack creep across the table."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from test_assistant_physics import build, no_network
from cube_task.assistant.simulation import SimulationSession


def session_tick(physics, status='ready'):
    """Use the production phase selector with real physical contact state."""
    session = object.__new__(SimulationSession)
    session._physics, session._status = physics, status
    session.runner = SimpleNamespace(has_held_recovery=False)
    session._environment_revision = 0
    session._last_snapshot_wall = time.monotonic()
    session._check_physical_changes = session._apply_interventions = lambda: None
    session._publish_snapshot = lambda: None
    return session


def test_off_center_stack_stays_put_but_still_falls_when_support_moves(build):
    physics, base, apple, _events = build(offset=.013)
    assert physics.mujoco_world.model.opt.noslip_iterations == 5
    physics.settle(1.5)
    session = session_tick(physics, 'settling')
    session._tick()
    assert physics.mujoco_world.model.opt.noslip_iterations == 30
    assert physics.mujoco_world.model.opt.noslip_tolerance == pytest.approx(1e-10, rel=1e-6, abs=0.)
    physics.settle(.5)
    assert physics.support_status(apple.name, base.label)['success']
    before = {obj.name: obj.T_base_obj[:3, 3].copy() for obj in (base, apple)}
    started = time.perf_counter()
    physics.settle(20.)
    elapsed = time.perf_counter() - started
    displacement = {obj.name: float(np.linalg.norm(obj.T_base_obj[:2, 3] - before[obj.name][:2]))
                    for obj in (base, apple)}
    print('20 s stack drift:', displacement, 'wall seconds:', elapsed, flush=True)
    assert max(displacement.values()) < .0005, displacement
    assert physics.support_status(apple.name, base.label)['success']
    assert physics.world.held is None and physics.world._T_tcp_obj is None
    assert not any(int(t) == int(physics.mujoco_world._mj.mjtEq.mjEQ_WELD)
                   for t in physics.mujoco_world.model.eq_type)

    # Explicit human interference moves only the base. The upper free body
    # must lose its real contact and fall under gravity, rather than be frozen.
    apple_before = apple.T_base_obj.copy()
    base.place_upright(.45, .15, 0., physics.world.table_z)
    physics.apply_scene_changes()
    np.testing.assert_allclose(apple.T_base_obj, apple_before, atol=1e-12)
    assert not physics.support_status(apple.name, base.label)['contact']
    session._tick()
    assert not physics.state()['contact_solver']['rest_precision']
    assert physics.mujoco_world.model.opt.noslip_iterations == 5
    physics.settle(.08)
    assert apple.T_base_obj[2, 3] < apple_before[2, 3] - .015
    assert physics.mujoco_world.object_speed(apple.name)[0] > .1
    physics.settle(1.)
    assert apple.bottom_z() == pytest.approx(physics.world.table_z, abs=.003)
    assert not physics.support_status(apple.name, base.label)['success']


def test_rest_precision_ends_before_manipulation_and_never_applies_to_holding(build):
    physics, _base, apple, _events = build(offset=.013)
    physics.settle(.6)
    session = session_tick(physics)
    for phase, active in [('ready', True), ('executing', False), ('error', True),
                          ('observing', True), ('initializing', False), ('settling', True)]:
        session._status = phase
        session._tick()
        assert physics.state()['contact_solver']['rest_precision'] is active
        assert physics.mujoco_world.model.opt.noslip_iterations == (30 if active else 5)
    session.runner.has_held_recovery = True
    session._tick()
    assert not physics.state()['contact_solver']['rest_precision']
    # The numerical adapter itself also rejects a held-state input, even if a
    # caller mistakenly labels that phase as resting.
    physics.world.held = apple
    physics.set_rest_contact_precision(True)
    assert not physics.state()['contact_solver']['rest_precision']
    physics.sync()


@pytest.mark.parametrize('iterations,tolerance', [(7, 1e-8), (60, 1e-12)])
def test_rest_precision_preserves_configuration_through_rebuild_and_cleanup(build, iterations, tolerance):
    physics, _base, _apple, _events = build(offset=.013)
    physics.settle(.6)
    physics.cfg.set('sim.mujoco.noslip_iterations', iterations)
    physics.cfg.set('sim.mujoco.noslip_tolerance', tolerance)
    physics._build()
    physics.sync()
    physics.set_rest_contact_precision(True)
    assert physics.mujoco_world.model.opt.noslip_iterations == max(30, iterations)
    assert physics.mujoco_world.model.opt.noslip_tolerance == min(1e-10, tolerance)
    physics.set_rest_contact_precision(False)
    assert physics.mujoco_world.model.opt.noslip_iterations == iterations
    assert physics.mujoco_world.model.opt.noslip_tolerance == tolerance
    physics.set_rest_contact_precision(True)
    physics._build()
    assert not physics.state()['contact_solver']['rest_precision']
    assert physics.mujoco_world.model.opt.noslip_iterations == iterations
    assert physics.mujoco_world.model.opt.noslip_tolerance == tolerance
    physics.sync()
    physics.set_rest_contact_precision(True)
    try:
        raise RuntimeError('simulated caller failure')
    except RuntimeError:
        physics.close()
    assert not physics.state()['contact_solver']['rest_precision']
    assert physics.mujoco_world.model.opt.noslip_iterations == iterations
    assert physics.mujoco_world.model.opt.noslip_tolerance == tolerance
