"""A crowded zone still needs a release slot inside the arm's workspace."""
from itertools import combinations
import copy

import numpy as np

from cube_task.assistant.simulation import SimulationSession
from cube_task.core.types import ArmHealth, MotionResult
from cube_task.core.config import load_config
from cube_task.core.types import Zone
from cube_task.perception.grasp import PlacePlanner


def _polygons_overlap(a, b):
    for poly in (a, b):
        for edge in np.roll(poly, -1, axis=0) - poly:
            normal = np.array([-edge[1], edge[0]])
            pa, pb = a @ normal, b @ normal
            if max(pa.min(), pb.min()) >= min(pa.max(), pb.max()) - 1e-8:
                return False
    return True


def test_workspace_filter_searches_other_free_grid_points_before_ik_retry_budget():
    planner = PlacePlanner(load_config(use_overlays=False))
    zone = Zone(name='red', center=(.17, -.35, 0.0), size=(.2, .2))
    held = dict(length=.05, width=.05, elongated=False, grip_width=.04809, grip_z=.0082)
    occupied = [(.1697248, -.349474, .05, .0506, .04899, .04849, -1.1182),
                (.1198994, -.2997266, .05, .0511, .049967, .0491217, -1.1913)]
    plan = planner.plan(zone, .0082, .15, footprint=.05, occupied=occupied, held=held,
                        avoid=[(.215, -.385)],
                        candidate_filter=lambda x, y: .082 <= x <= .598 and -.398 <= y <= .398)
    assert plan.feasible
    x, y = plan.T_release[:2, 3]
    assert .082 <= x <= .598 and -.398 <= y <= .398
    assert np.linalg.norm(np.array([x, y]) - [.215, -.385]) > .012
    assert planner.last_info['footprint_clearance'] > 0.0
    assert planner.last_info['finger_clearance'] > 0.0


def test_workspace_filter_cannot_fall_back_to_a_forbidden_center():
    planner = PlacePlanner(load_config(use_overlays=False))
    zone = Zone(name='red', center=(.17, -.35, 0.0), size=(.2, .2))
    held = dict(length=.05, width=.05, elongated=False, grip_width=.05, grip_z=.02)
    plan = planner.plan(zone, .02, .15, held=held, candidate_filter=lambda x, y: False)
    assert not plan.feasible and plan.reason == 'no_release_in_workspace'


def test_three_camera_measured_cubes_fit_red_zone_without_boundary_release(tmp_path):
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False) as sim:
        sim.prepare()
        previous = {}
        for label in ('red', 'green', 'yellow'):
            scene = sim.scene
            selected = next(o for o in scene['objects'] if o['valid'] and o['label'] == label)
            result = sim.execute({'action': 'pick', 'scene_version': scene['scene_version'],
                                  'object_id': selected['object_id'], 'zone_name': 'zone_red'})
            assert result['ok'], result
            assert result['truth']['success'] and result['truth']['actual_zone'] == 'zone_red'
            assert sim.world.held is None
            for obj in sim.world.objects:
                if obj.name in previous:
                    np.testing.assert_allclose(obj.T_base_obj, previous[obj.name], atol=1e-8)
                if sim.world.zone_of(obj) is not None:
                    previous[obj.name] = obj.T_base_obj.copy()
        assert len(previous) == 3
        for obj in sim.world.objects:
            assert sim.world.zone_of(obj).name == 'zone_red'
        for a, b in combinations(sim.world.objects, 2):
            assert not _polygons_overlap(a.footprint_polygons()[0], b.footprint_polygons()[0])


def test_held_release_retry_uses_new_slot_without_opening_or_regrasping(tmp_path, monkeypatch):
    """A motion rejection can be recovered while preserving the verified grasp."""
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False) as sim:
        sim.prepare()
        for label in ('red', 'green'):
            scene = sim.scene
            selected = next(o for o in scene['objects'] if o['valid'] and o['label'] == label)
            placed = sim.execute({'action': 'pick', 'scene_version': scene['scene_version'],
                                  'object_id': selected['object_id'], 'zone_name': 'zone_red'})
            assert placed['ok'], placed
        before = {o.name: o.T_base_obj.copy() for o in sim.world.objects if o.label != 'yellow'}
        det = copy.deepcopy(next(d for d in sim._detections.values() if d.label == 'yellow'))
        all_measured = list(sim._detections.values())
        sim._refresh_placement_occupancy(det)
        move = sim.arm.move_tcp
        remaining = {'release', 'recover_up'}

        def reject_once(T, linear=False, **kwargs):
            if sim._stage in remaining:
                remaining.remove(sim._stage)
                return MotionResult(ok=False, error='rejected', detail='injected guarded path')
            return move(T, linear=linear, **kwargs)

        monkeypatch.setattr(sim.arm, 'move_tcp', reject_once)
        result, why = sim.runner.pick_detection(det, zone_name='zone_red', neighbors=all_measured)
        assert not why and result.detail['holding'] and sim.world.held is not None
        assert sim.runner.has_held_recovery
        assert result.detail['last_motion_detail'] == 'injected guarded path'
        failed_xy = tuple(sim.runner._held_recovery_context['place'].T_release[:2, 3])
        calls = sim.arm.inner.n_calls
        refusal, why = sim.runner.retry_held_placement(det, 'zone_green', all_measured)
        assert refusal is None and 'changed' in why and sim.arm.inner.n_calls == calls
        health = sim.arm.health
        monkeypatch.setattr(sim.arm, 'health', lambda: ArmHealth(ok=False, reason='test fault'))
        refusal, why = sim.runner.retry_held_placement(det, 'zone_red', all_measured)
        assert refusal is None and 'unhealthy' in why and sim.arm.inner.n_calls == calls
        monkeypatch.setattr(sim.arm, 'health', health)
        widths = []
        gripper = sim.arm.gripper

        def record_gripper(width, **kwargs):
            widths.append(width)
            return gripper(width, **kwargs)

        monkeypatch.setattr(sim.arm, 'gripper', record_gripper)
        retried, why = sim.runner.retry_held_placement(det, 'zone_red', all_measured)
        assert retried.ok, (why, retried)
        assert retried.detail['continuation'] and not retried.detail['holding']
        assert not sim.runner.has_held_recovery
        assert sim.world.held is None
        yellow = next(o for o in sim.world.objects if o.label == 'yellow')
        assert sim.world.zone_of(yellow).name == 'zone_red'
        steps = retried.detail['steps']
        assert all(s['stage'] not in ('open', 'close', 'pre', 'descend') for s in steps)
        assert [s['stage'] for s in steps if 'command' in s] == ['open_release']
        assert len(widths) == 1 and widths[0] > retried.grasp_width
        released = next(s['target'] for s in steps if s['stage'] == 'release')
        assert np.linalg.norm(np.array(released[:2]) - failed_xy) > 0.01
        for obj in sim.world.objects:
            assert sim.world.zone_of(obj).name == 'zone_red'
            if obj.name in before:
                np.testing.assert_allclose(obj.T_base_obj, before[obj.name], atol=1e-8)
        for a, b in combinations(sim.world.objects, 2):
            assert not _polygons_overlap(a.footprint_polygons()[0], b.footprint_polygons()[0])
