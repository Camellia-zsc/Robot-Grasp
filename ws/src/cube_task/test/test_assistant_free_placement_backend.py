"""Free placement uses measured geometry and the existing guarded motion path."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.destinations import placement_metadata, validate_destination
from cube_task.assistant.simulation import SimulationSession
from cube_task.core.config import load_config
from cube_task.core.types import Detection, MotionResult, Zone
from cube_task.core.transforms import top_down_T
from cube_task.perception.grasp import PlacePlanner
from cube_task.task.runner import TaskRunner
from cube_task.robot.kinematics import tilt_outward


def decision(sim, label, destination=None, zone_name=None):
    scene = sim.scene
    target = next(o for o in scene['objects'] if o['valid'] and o['label'] == label)
    result = dict(action='pick', scene_version=scene['scene_version'],
                  object_id=target['object_id'], zone_name=zone_name)
    if destination is not None:
        result['destination'] = destination
    return result


@pytest.mark.parametrize('xy', [[True, .1], ['.3', .1], [float('nan'), 0],
                               [float('inf'), 0], [10**1000, 0], [.01, 0], [.6, .4]])
def test_destination_validation_refuses_nonfinite_or_unconfigured_points(xy):
    cfg = load_config(use_overlays=False)
    scene = {'placement': placement_metadata(cfg, [[.12, -.25], [.4, -.25], [.4, .25], [.12, .25]])}
    with pytest.raises(ValueError):
        validate_destination({'type': 'point', 'xy': xy}, scene)


def test_exact_point_does_not_move_to_another_clear_slot():
    planner = PlacePlanner(load_config(use_overlays=False))
    zone = Zone('point', np.array([.3, 0, 0]), (.3, .3))
    held = dict(length=.05, width=.05, elongated=False, grip_width=.05, grip_z=.02)
    occupied = [(.3, 0, .06, .05)]
    exact = planner.plan(zone, .02, .15, held=held, occupied=occupied, strict=True, fixed_xy=True)
    assert not exact.feasible
    region = planner.plan(zone, .02, .15, held=held, occupied=occupied, strict=True)
    assert region.feasible
    assert np.linalg.norm(region.T_release[:2, 3] - [.3, 0]) > .05


def test_strict_region_rejects_corners_even_when_inscribed_circle_fits():
    planner = PlacePlanner(load_config(use_overlays=False))
    zone = Zone('point', np.array([.3, 0, 0]), (.05, .05))
    held = dict(length=.05, width=.05, elongated=False, grip_width=.05, grip_z=.02)
    blocked = planner.plan(zone, .02, .15, held=held, strict=True, fixed_xy=True,
                           pose_filter=lambda x, y, angle: False)
    assert not blocked.feasible


def test_rotated_square_corners_and_unpickable_partial_still_block_placement():
    planner = PlacePlanner(load_config(use_overlays=False))
    runner = SimpleNamespace(arm=SimpleNamespace(), nominal={'red': (.05, .05), 'green': (.05, .05)},
                             placer=planner)
    target = Detection('red', 1., center_base=np.array([.3, .1, .05]),
                       top_z=.05, height=.05, length=.05, width=.05, yaw=math.pi/4)
    grasp = SimpleNamespace(T_grasp=top_down_T(.3, .1, .02, 0), closing_angle=0,
                            expected_width=.05)
    polygon = np.array([[.12, -.25], [.4, -.25], [.4, .25], [.12, .25]])
    allowed = TaskRunner._strict_place_filter(runner, target, grasp, [target], polygon)
    assert not allowed(.37, 0, 0)  # Circle radius .025 fits; rotated square corner does not.
    partial = Detection('green', 1., center_base=np.array([.28, 0, .05]),
                        top_z=.05, height=.05, length=.05, width=.05, yaw=0,
                        valid=False, reject_reason='partial_object')
    allowed = TaskRunner._strict_place_filter(runner, target, grasp, [target, partial], polygon)
    assert not allowed(.28, 0, 0)


def test_exact_point_can_try_another_yaw_without_changing_center():
    planner = PlacePlanner(load_config(use_overlays=False))
    zone = Zone('point', np.array([.3, .1, 0]), (.3, .3))
    held = dict(length=.05, width=.05, elongated=False, grip_width=.05, grip_z=.02)
    first = planner.plan(zone, .02, .15, held=held, strict=True, fixed_xy=True)
    rejected = (*planner.last_info['xy'], planner.last_info['closing_angle'])
    second = planner.plan(zone, .02, .15, held=held, strict=True, fixed_xy=True,
                          avoid_poses=[rejected])
    assert first.feasible and second.feasible
    np.testing.assert_allclose(second.T_release[:2, 3], [.3, .1])
    assert planner.last_info['closing_angle'] != rejected[2]


def test_region_recovery_feedback_updates_actual_planned_object_center():
    sim = object.__new__(SimulationSession)
    sim._placement_center = [.25, -.1]
    sim._record_feedback('decision', {'action': 'retry_held_placement',
                                     'release_xyz': [.31, -.1, .08],
                                     'place': {'xy': [.30, -.1]}})
    assert sim._placement_center == [.30, -.1]


def test_leaned_finger_upper_corner_cannot_overlap_a_tall_measured_neighbor():
    planner = PlacePlanner(load_config(use_overlays=False))
    runner = SimpleNamespace(arm=SimpleNamespace(), nominal={'red': (.05, .05)}, placer=planner)
    target = Detection('red', 1., center_base=np.array([.3, 0, .05]), top_z=.05,
                       height=.05, length=.05, width=.05, yaw=0)
    obstacle = Detection('green', 1., center_base=np.array([.43, .08, .14]), top_z=.14,
                         height=.14, length=.10, width=.08, yaw=0)
    grasp = SimpleNamespace(T_grasp=top_down_T(.3, 0, .02, math.pi/2),
                            closing_angle=math.pi/2, expected_width=.05)
    allowed = TaskRunner._strict_place_filter(runner, target, grasp, [target, obstacle],
                                              np.array([[.1, -.3], [.6, -.3], [.6, .3], [.1, .3]]))
    release = tilt_outward(top_down_T(.5, 0, .055, math.pi/2), math.radians(15))
    # The old flat band misses this collision at the raised finger corner.
    assert allowed(.5, 0, math.pi/2, final_yaw=0, tcp_xy=[.5, 0])
    assert not allowed(.5, 0, math.pi/2, final_yaw=0, tcp_xy=[.5, 0],
                       gripper_pose=release, gripper_open_width=.07)


def test_color_zone_to_pick_area_then_exact_point_then_color_zone(tmp_path):
    with SimulationSession(seed=0, realtime=False, log_dir=tmp_path) as sim:
        sim.prepare()
        original_zones = [z.name for z in sim.world.zones]
        steps = [dict(zone_name='zone_red'),
                 dict(destination={'type': 'region', 'name': 'pick_area'}),
                 dict(destination={'type': 'point', 'xy': [.32, .16]}),
                 dict(zone_name='zone_green')]
        for step in steps:
            before = {o.name: o.T_base_obj.copy() for o in sim.world.objects if o.label != 'red'}
            result = sim.execute(decision(sim, 'red', **step))
            assert result['ok'], result
            assert result['controller_ok'] and result['truth']['success']
            assert next(o for o in sim.scene['objects'] if o['valid'] and o['label'] == 'red')
            assert [z.name for z in sim.world.zones] == original_zones
            for obj in sim.world.objects:
                if obj.name in before:
                    np.testing.assert_allclose(obj.T_base_obj, before[obj.name], atol=1e-8)


def test_occupied_free_point_refused_before_grasp(tmp_path):
    with SimulationSession(seed=0, realtime=False, log_dir=tmp_path) as sim:
        sim.prepare()
        green = next(o for o in sim.scene['objects'] if o['valid'] and o['label'] == 'green')
        before = {o.name: o.T_base_obj.copy() for o in sim.world.objects}
        stages = []
        sim.on_event = lambda kind, fields: stages.append(fields.get('stage')) if kind == 'stage' else None
        result = sim.execute(decision(sim, 'red', destination={'type': 'point', 'xy': green['center_base'][:2]}))
        assert not result['ok'] and result['stage'] == 'plan'
        assert 'close' not in stages and sim.world.held is None
        for obj in sim.world.objects:
            np.testing.assert_allclose(obj.T_base_obj, before[obj.name], atol=1e-8)


def test_conflicting_or_outside_destination_refused_without_commands(tmp_path):
    with SimulationSession(seed=0, realtime=False, log_dir=tmp_path) as sim:
        sim.prepare()
        calls = sim.arm.inner.n_calls
        for request in [decision(sim, 'red', destination={'type': 'point', 'xy': [.3, 0]}, zone_name='zone_red'),
                        decision(sim, 'red', destination={'type': 'point', 'xy': [20, 0]})]:
            result = sim.execute(request)
            assert not result['ok'] and result['error'] == 'invalid_destination'
            assert sim.arm.inner.n_calls == calls


def test_held_point_recovery_keeps_original_point_and_rejects_goal_change(tmp_path, monkeypatch):
    destination = {'type': 'point', 'xy': [.32, .16]}
    with SimulationSession(seed=0, realtime=False, log_dir=tmp_path) as sim:
        sim.prepare()
        move = sim.arm.move_tcp
        blocked = {'release', 'recover_up'}

        def block_once(T, linear=False, **kwargs):
            if sim._stage in blocked:
                blocked.remove(sim._stage)
                return MotionResult(ok=False, error='rejected', detail='injected release path failure')
            return move(T, linear=linear, **kwargs)

        monkeypatch.setattr(sim.arm, 'move_tcp', block_once)
        result = sim.execute(decision(sim, 'red', destination=destination))
        assert not result['ok'] and result['holding']
        goal = {'label': 'red', 'zone_name': None, 'destination': destination}
        with pytest.raises(ValueError, match='不一致'):
            sim.prepare_recovery(result, dict(goal, destination={'type': 'point', 'xy': [.32, .18]}), 1, [])
        context = sim.prepare_recovery(result, goal, 1, [])
        assert context['goal'] == goal and context['placement'] == sim.scene['placement']
        assert any(o['id'] == 'place_held_alternate' for o in context['options'])
        recovered = sim.execute_recovery({'scene_version': context['scene_version'],
                                         'action': 'recover', 'option_id': 'place_held_alternate',
                                         'message': '在原指定点重试放置'})
        assert recovered['ok'], recovered
        assert recovered['destination'] == destination
        assert np.linalg.norm(sim.world.object('red_cube').center_xy() - destination['xy']) < .015


def test_banana_center_reaches_point_despite_offset_grasp(tmp_path):
    with SimulationSession(realtime=False, log_dir=tmp_path) as sim:
        sim.prepare()
        sim.edit_scene(dict(scene_version=sim.scene['scene_version'], action='clear'))
        sim.edit_scene(dict(scene_version=sim.scene['scene_version'], action='add',
                            catalog_id='banana', xy=[.28, 0], yaw=-.6))
        sim.observe()
        destination = {'type': 'point', 'xy': [.33, .12]}
        result = sim.execute(decision(sim, 'banana', destination=destination))
        assert result['ok'], result
        banana = next(o for o in sim.world.objects if o.label == 'banana')
        assert np.linalg.norm(banana.center_xy() - destination['xy']) < .015
        assert any(o['valid'] and o['label'] == 'banana' for o in sim.scene['objects'])


def test_banana_regrasped_from_yellow_uses_rotated_cloud_bounds_for_point_center(tmp_path):
    """Recorded post-sort pose: camera rect and curved-object AABB differ."""
    from cube_task.assistant.catalog import make_catalog_object
    with SimulationSession(realtime=False, log_dir=tmp_path) as sim:
        sim.world.objects = [make_catalog_object('banana', 'regression-banana', [.5, 0], 0)]
        sim.world.objects[0].T_base_obj = np.array([
            [.699368722234062, -.7147610722197282, 0, .4955489648394764],
            [.7147610722197282, .699368722234062, 0, .0005384077475879753],
            [0, 0, 1, 0], [0, 0, 0, 1]])
        sim.prepare()
        destination = {'type': 'point', 'xy': [.33, .12]}
        result = sim.execute(decision(sim, 'banana', destination=destination))
        assert result['ok'], result
        assert np.linalg.norm(sim.world.objects[0].center_xy() - destination['xy']) < .015


def test_region_retries_another_center_after_final_pose_clearance_failure(tmp_path, monkeypatch):
    with SimulationSession(realtime=False, log_dir=tmp_path) as sim:
        sim.prepare()
        original = sim.runner._adjust_strict_release
        centers = []

        def reject_first(det, gp, pp, neighbors, polygon, desired_xy, held, **kwargs):
            centers.append(list(desired_xy))
            if len(centers) == 1:
                return False, 'injected final leaned clearance rejection'
            return original(det, gp, pp, neighbors, polygon, desired_xy, held, **kwargs)

        monkeypatch.setattr(sim.runner, '_adjust_strict_release', reject_first)
        result = sim.execute(decision(sim, 'red', destination={'type': 'region', 'name': 'pick_area'}))
        assert result['ok'], result
        assert len(centers) >= 2
        assert np.linalg.norm(np.asarray(centers[1]) - centers[0]) > .01
