"""Measured local grasps must avoid neighbors without changing object geometry."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cube_task.assistant.catalog_perception import configure_sandbox
from cube_task.core.types import Detection, Zone
from cube_task.perception.grasp import (GraspPlanner, PlacePlanner,
                                        annotate_grasp_neighbors, section_profile, section_width)
from cube_task.sim.world import SimObject, SimWorld
from test_task_fakes import make_cfg, make_det


FIXTURES = Path(__file__).with_name('fixtures')


def recorded_banana_neighbors():
    fixture = json.loads((FIXTURES / 'assistant_banana_neighbor_scene.json').read_text(encoding='utf-8'))
    clouds = np.load(FIXTURES / 'assistant_banana_neighbor_cloud.npz')
    detections = []
    for data in fixture['measured_detections']:
        data = copy.deepcopy(data)
        data['center_base'] = np.asarray(data['center_base'])
        det = Detection(**data)
        for key in ('_points_base', '_points_xy', '_placement_points_base'):
            if det.label + key in clouds:
                det.extra[key] = clouds[det.label + key]
        detections.append(det)
    cfg = make_cfg(robot__gripper__max_width=.1)
    configure_sandbox(cfg)
    return fixture, detections, cfg


def test_exact_banana_collision_uses_a_measured_local_section_and_keeps_yellow_untouched():
    fixture, detections, cfg = recorded_banana_neighbors()
    target = next(d for d in detections if d.label == 'banana')
    annotate_grasp_neighbors(target, detections)
    world = SimWorld(objects=[SimObject(o['id'], o['label'], o['shape'], o['size'],
                                       o['color_rgb'][::-1], o['transform'])
                              for o in fixture['objects']])
    original = {o.name: o.T_base_obj.copy() for o in world.objects}

    cfg.set('grasp.adaptive_neighbors', False)
    baseline = GraspPlanner(cfg, 'Sandbox').plan(copy.deepcopy(target))
    assert baseline.feasible
    ok, failure = world.check_grasp(baseline.T_grasp, baseline.open_width, baseline.close_width)
    assert not ok and failure['reason'] == 'collision_finger'
    assert failure['collided'] == 'yellow_cube'
    # Merely reducing the opening on the original approach cannot solve this scene.
    for opening in (.09, .08, .07, .06):
        assert not world.check_grasp(baseline.T_grasp, opening, baseline.close_width)[0]

    cfg.set('grasp.adaptive_neighbors', True)
    planner = GraspPlanner(cfg, 'Sandbox')
    plan = planner.plan(target)
    assert plan.feasible, plan.reason
    assert target.extra['grasp']['strategy'] == 'local_section_and_angle'
    assert target.extra['grasp']['neighbor_hits'] == 0
    assert plan.open_width < baseline.open_width
    # Opening covers the whole descending pad column, including material above TCP.
    profile = section_profile(target.extra['_points_base'], plan.T_grasp[:2, 3],
                              plan.closing_angle, target.extra['table_z'], with_support=True)
    passed = section_width(profile, plan.T_grasp[2, 3] - planner.pad_half,
                           float(profile[0][-1]))
    assert plan.open_width >= passed + 2 * planner.finger_clearance - 1e-9
    ok, feedback = world.check_grasp(plan.T_grasp, plan.open_width, plan.close_width)
    assert ok, feedback
    assert feedback['object'] == 'catalog-banana-1'
    assert world.try_grasp(plan.T_grasp, plan.open_width, plan.close_width) > 0
    assert world.held.name == 'catalog-banana-1'
    for obj in world.objects:
        if obj is not world.held:
            np.testing.assert_array_equal(obj.T_base_obj, original[obj.name])


def test_a_narrow_opening_avoids_neighbors_without_an_unnecessary_wrist_turn():
    cfg = make_cfg(robot__gripper__max_width=.1, grasp__open_full_tasks=['Sandbox'])
    target = make_det('red', .3, 0., size=.03)
    target.extra['_neighbor_xy'] = np.array([[.357, 0.], [.3, .057]])
    plan = GraspPlanner(cfg, 'Sandbox').plan(target)
    assert plan.feasible, plan.reason
    assert target.extra['grasp']['strategy'] == 'narrow_opening'
    assert plan.open_width == pytest.approx(.04)
    assert plan.open_width >= plan.expected_width + .01 - 1e-9


def test_a_real_blocker_is_identified_and_removing_it_invalidates_old_neighbor_points():
    cfg = make_cfg(robot__gripper__max_width=.1, grasp__open_full_tasks=['Sandbox'])
    target = make_det('red', .3, 0.)
    other = make_det('green', .33, .03)
    other.extra.update(object_id='measured-green',
                       _points_xy=np.array([[.33, 0.], [.3, .03]]))
    annotate_grasp_neighbors(target, [target, other])
    plan = GraspPlanner(cfg, 'Sandbox').plan(target)
    assert not plan.feasible
    assert plan.reason.startswith('blocked_by_neighbor:')
    assert target.extra['grasp']['blockers'][0]['object_id'] == 'measured-green'
    annotate_grasp_neighbors(target, [target])
    assert '_neighbor_xy' not in target.extra
    assert GraspPlanner(cfg, 'Sandbox').plan(target).feasible


def test_original_yaw_restoration_refuses_an_alternative_release_angle():
    planner = PlacePlanner(make_cfg(robot__gripper__max_width=.1))
    zone = Zone('temporary-origin', np.array([.3, 0., 0.]), (.3, .3))
    required = .37
    held = dict(length=.07, width=.03, elongated=True, grip_width=.03, grip_z=.02,
                axis_offset=-.8, _required_closing=required)
    plan = planner.plan(zone, .02, .15, held=held, strict=True, fixed_xy=True)
    assert plan.feasible
    assert planner.last_info['closing_angle'] == required
    assert planner.last_info['axis'] == pytest.approx(required - .8)
    attempted = []

    def refuse_original_angle(x, y, closing):
        attempted.append(closing)
        return abs(closing - required) > .1

    blocked = planner.plan(zone, .02, .15, held=held, strict=True, fixed_xy=True,
                           pose_filter=refuse_original_angle)
    assert not blocked.feasible
    assert attempted and all(a == required for a in attempted)
