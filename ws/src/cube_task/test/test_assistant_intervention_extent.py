"""Whole-table human moves keep geometry validation at the simulation boundary."""
import math
import threading

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession


@pytest.fixture
def tabletop(tmp_path):
    events = []
    with SimulationSession(realtime=False, log_dir=tmp_path,
                           on_event=lambda kind, fields: events.append((kind, fields))) as sim:
        target = next(obj for obj in sim.world.objects if obj.label == 'red')
        sim.world.objects = [target]
        # Only the queue/geometry boundary is under test; no perception or model
        # call is needed to mark this isolated simulation as prepared.
        sim._prepared = True
        sim._publish_snapshot()
        yield sim, target, events


def apply_while_running(sim, request):
    """A UI request queues while the execution worker owns the simulation."""
    answers, failures = [], []

    def send():
        try:
            answers.append(sim.intervene_scene(dict(request,
                scene_version=sim.snapshot()['scene_version'])))
        except Exception as exc:
            failures.append(exc)

    with sim._operation_lock:
        before = [obj.T_base_obj.copy() for obj in sim.world.objects]
        worker = threading.Thread(target=send)
        worker.start()
        worker.join(3.)
        assert not worker.is_alive(), 'human intervention must not wait for the running task'
        assert not failures, failures
        assert answers[0]['queued']
        for obj, pose in zip(sim.world.objects, before):
            np.testing.assert_array_equal(obj.T_base_obj, pose)
        sim._tick()


@pytest.mark.parametrize('x_side,y_side', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_running_move_reaches_all_table_corners_outside_robot_and_pick_area(
        tabletop, x_side, y_side):
    sim, target, events = tabletop
    table = sim._placement['table_bounds']
    xy = [table['x'][x_side] + (.04 if x_side == 0 else -.04),
          table['y'][y_side] + (.04 if y_side == 0 else -.04)]
    # This human placement is outside the robot's work area; it must still be
    # supported on the table. Ordinary scene editing keeps its pick-area rule.
    assert not sim._placement['bounds']['x'][0] <= xy[0] <= sim._placement['bounds']['x'][1]
    with pytest.raises(ValueError, match='抓取区'):
        sim.editor.apply(dict(action='move', object_id=target.name, xy=xy, yaw=math.pi / 4))
    identity = id(target)
    apply_while_running(sim, dict(action='move', object_id=target.name, xy=xy, yaw=math.pi / 4))
    np.testing.assert_allclose(target.center_xy(), xy, atol=1e-10)
    assert id(sim.world.objects[0]) == identity and len(sim.world.objects) == 1
    assert any(kind == 'scene_intervention_applied' for kind, _ in events)
    assert not any(kind == 'scene_intervention_rejected' for kind, _ in events)


@pytest.mark.parametrize('axis,side', [('x', 0), ('x', 1), ('y', 0), ('y', 1)])
def test_running_move_refuses_rotated_footprint_over_every_table_edge(tabletop, axis, side):
    sim, target, events = tabletop
    # The center and an unrotated 50 mm cube fit. At 45 degrees its complete
    # footprint extends past the table, so the queued mutation must be atomic.
    xy = [.5, 0.]
    xy[1 if axis == 'y' else 0] = sim._placement['table_bounds'][axis][side] + (.03 if side == 0 else -.03)
    before, revision = target.T_base_obj.copy(), sim._environment_revision
    apply_while_running(sim, dict(action='move', object_id=target.name, xy=xy, yaw=math.pi / 4))
    np.testing.assert_array_equal(target.T_base_obj, before)
    assert sim._environment_revision == revision
    assert any(kind == 'scene_intervention_rejected' for kind, _ in events)


def test_running_move_outside_pick_area_still_checks_object_gap(tabletop):
    sim, target, events = tabletop
    neighbor = target.copy()
    neighbor.name = 'table-obstacle'
    neighbor.place_upright(.75, .5, 0., sim.world.table_z)
    sim.world.objects.append(neighbor)
    sim._publish_snapshot()
    before = target.T_base_obj.copy()
    # 55 mm center separation leaves only 5 mm between the cube footprints.
    apply_while_running(sim, dict(action='move', object_id=target.name, xy=[.695, .5]))
    np.testing.assert_array_equal(target.T_base_obj, before)
    assert len(sim.world.objects) == 2
    assert any(kind == 'scene_intervention_rejected' and '8 mm' in fields['message']
               for kind, fields in events)


def test_human_move_cannot_reposition_held_object(tabletop):
    sim, target, _ = tabletop
    sim.world.held = target
    sim._publish_snapshot()
    before = target.T_base_obj.copy()
    with pytest.raises(ValueError, match='未夹持'):
        sim.intervene_scene(dict(action='move', object_id=target.name, xy=[.75, .5],
                                 scene_version=sim.snapshot()['scene_version']))
    np.testing.assert_array_equal(target.T_base_obj, before)
    assert not sim._interventions
