"""Keep tight measured edge pockets without reducing any placement margin."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.destinations import ZONE_COMPLETION_MARGIN
from cube_task.core.transforms import top_down_T
from cube_task.core.types import Detection, Zone
from cube_task.perception.grasp import PlacePlanner, SLOT_GRID_STEP, capsule_edge_margin, footprint_capsule
from cube_task.task.reach import release_workspace_reason
from cube_task.task.runner import TaskRunner
from test_task_fakes import make_cfg


def on_grid(point, center):
    return all(abs((a-b)/SLOT_GRID_STEP-round((a-b)/SLOT_GRID_STEP)) < 1e-7
               for a, b in zip(point, center))


def test_measured_pear_and_green_cube_have_a_legal_pocket_between_grid_points():
    # Camera summaries from the supported-pear incident, with catalog lower
    # bounds for incomplete outlines. No physical object transform is used.
    pear = Detection('pear', 1., center_base=np.array([.3353353557, -.1722351323, .1563]),
        top_z=.1563, height=.1059, length=.0754, width=.0460, yaw=1.4662,
        extra={'resting_pose': 'upright', 'table_z': .05040329695})
    green = Detection('green', .981, center_base=np.array([.1663147512, .3498362021, .0511]),
        top_z=.0511, height=.0511, length=.0498, width=.0495, yaw=-.4338989465)
    planner = PlacePlanner(make_cfg(robot__gripper__max_width=.1))
    zone = Zone('zone_green', np.array([.17, .35, 0.]), (.2, .2))
    grasp = SimpleNamespace(T_grasp=top_down_T(.3356537103, -.1714347553, .1449544818, 1.0963117408),
                            closing_angle=1.0963117408, expected_width=.051159611)
    guard = SimpleNamespace(x=(.08, .60), y=(-.40, .40), r_min=.10, r_max=.62)
    runner = SimpleNamespace(nominal={'pear': [.085, .085], 'green': [.05, .05]},
                             placer=planner, arm=SimpleNamespace(guard=guard))
    allowed = TaskRunner._strict_place_filter(runner, pear, grasp, [green], zone.corners_xy(),
                                               completion_zone=zone)
    shape = dict(length=.085, width=.085, elongated=True, axis_offset=.3698738567,
                 grip_width=grasp.expected_width, grip_z=.0945511849,
                 center_offset=pear.center_base[:2]-grasp.T_grasp[:2, 3],
                 grasp_closing=grasp.closing_angle)
    occupied = [(green.center_base[0], green.center_base[1], .05, green.height, .05, .05, green.yaw)]
    assert not allowed(.220, .300, grasp.closing_angle)
    assert allowed(.22125, .29875, grasp.closing_angle)

    result = planner.plan(zone, .095, .2, occupied=occupied, held=shape, strict=True,
        pose_filter=allowed, candidate_filter=lambda x, y: not release_workspace_reason(runner.arm, (x,y)))

    assert result.feasible, result.reason
    xy, closing = planner.last_info['xy'], planner.last_info['closing_angle']
    assert not on_grid(xy, zone.center[:2])
    assert allowed(*xy, closing)
    assert planner.last_info['tier'] == 'strict_boundary'
    assert capsule_edge_margin(zone, footprint_capsule(*xy, .085, .085, None)) > ZONE_COMPLETION_MARGIN


@pytest.mark.parametrize('yaw', [0., math.pi/5])
@pytest.mark.parametrize('length,width,elongated', [(.085, .085, False), (.12, .03, True)],
                         ids=['round', 'elongated'])
def test_rotated_support_and_thin_capsule_keep_narrow_boundary_pocket(yaw, length, width, elongated):
    planner = PlacePlanner(make_cfg(robot__gripper__max_width=.1))
    zone = Zone('support', np.array([.3, 0., 0.]), (.2, .2), yaw=yaw)
    shape = dict(length=length, width=width, grip_width=.03, grip_z=.08,
                 elongated=elongated, axis_offset=0., _required_closing=yaw)
    extent = np.array([length/2, width/2])
    local = np.array([.1-extent[0]-.00625, -(.1-extent[1]-.00625)])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    target = zone.center[:2]+rotation@local

    result = planner.plan(zone, .08, .2, held=shape, strict=True,
        pose_filter=lambda x, y, closing: np.linalg.norm(np.array([x,y])-target) < .0001)

    assert result.feasible, result.reason
    assert planner.last_info['xy'] == pytest.approx(target)
    assert planner.last_info['tier'] == 'strict_boundary'
    cap = footprint_capsule(*target, length, width, yaw if elongated else None)
    assert capsule_edge_margin(zone, cap) == pytest.approx(.00625)


def test_exact_point_never_uses_boundary_pocket_or_preferred_points():
    planner = PlacePlanner(make_cfg(robot__gripper__max_width=.1))
    zone = Zone('point', np.array([.3, 0., 0.]), (.2, .2))
    shape = dict(length=.085, width=.085, grip_width=.04, grip_z=.06,
                 elongated=False, _required_closing=.2)
    visits = []

    def only_boundary(x, y, closing):
        visits.append((x,y))
        return abs(x-.35125) < .0001 and abs(y+.05125) < .0001

    result = planner.plan(zone, .06, .2, held=shape, strict=True, fixed_xy=True,
                          preferred_xy=([.35125, -.05125],), pose_filter=only_boundary)
    assert not result.feasible and visits
    assert all(point == pytest.approx(zone.center[:2]) for point in visits)


def test_boundary_fallback_keeps_filter_and_cancellation_budget():
    planner = PlacePlanner(make_cfg(robot__gripper__max_width=.1))
    zone = Zone('support', np.array([.3, 0., 0.]), (.2, .2))
    shape = dict(length=.085, width=.085, grip_width=.04, grip_z=.06,
                 elongated=False, _required_closing=.2)
    boundary_calls = []

    def reject(x, y, closing):
        if not on_grid((x,y), zone.center[:2]):
            boundary_calls.append((x,y))
        return False

    result = planner.plan(zone, .06, .2, held=shape, strict=True, pose_filter=reject,
                          cancelled=lambda: len(boundary_calls) >= 3)
    assert not result.feasible
    assert len(boundary_calls) == 3


def test_boundary_refinement_does_not_densify_the_support_interior():
    planner = PlacePlanner(make_cfg(robot__gripper__max_width=.1))
    zone = Zone('support', np.array([.3, 0., 0.]), (.2, .2))
    shape = dict(length=.085, width=.085, grip_width=.04, grip_z=.06,
                 elongated=False, _required_closing=.2)
    off_grid = []

    def interior_pocket(x, y, closing):
        if not on_grid((x,y), zone.center[:2]):
            off_grid.append((x,y))
        return abs(x-.3175) < .0001 and abs(y-.0125) < .0001

    result = planner.plan(zone, .06, .2, held=shape, strict=True, pose_filter=interior_pocket)
    assert not result.feasible
    assert off_grid
    assert all(capsule_edge_margin(zone, footprint_capsule(x,y,.085,.085,None)) <= .010001
               for x,y in off_grid)
