"""A cropped region occupant is incomplete evidence that the region is full."""
from types import SimpleNamespace

import numpy as np

from cube_task.assistant.catalog import catalog_entries
from cube_task.assistant.catalog_perception import configure_sandbox
from cube_task.core.transforms import top_down_T
from cube_task.core.types import Detection, GraspPlan, PlacePlan, Zone
from cube_task.perception.grasp import PlacePlanner
from test_runner_local_release import scene


def measured_green_region():
    runner, _, _, _ = scene()
    configure_sandbox(runner.cfg)
    # This fixture deliberately models the old narrow custom guard/card.
    # Keep candidate enumeration and the guard in the same workspace.
    runner.cfg.set('robot.workspace.y', [-.40,.40])
    runner.placer = PlacePlanner(runner.cfg)
    runner.nominal = {entry['label']: entry['dimensions'][:2] for entry in catalog_entries()}
    runner.arm.guard = SimpleNamespace(x=[.08, .60], y=[-.40, .40], r_max=.62)
    zone = Zone('zone_green', np.array([.17, .35, 0.]), (.2, .2))
    runner.zones = {zone.name: zone}
    target = Detection('red', .997, center_base=np.array([.3221, .0973, .0505]),
                       top_z=.0505, height=.0505, length=.0494, width=.0491, yaw=-1.1479)
    grasp_pose = top_down_T(.3228, .0972, .013, 1.9937)
    lift = grasp_pose.copy()
    lift[2, 3] = .15
    grasp = GraspPlan(target, lift.copy(), grasp_pose, lift, .1, 0., .0502054774, 1.9937)
    release = top_down_T(.12, .30, .017, 1.9937)
    pre = release.copy()
    pre[2, 3] = .15
    place = PlacePlan(zone.name, pre, release, pre.copy(), .075)
    green = Detection('green', 1., center_base=np.array([.2198, .2991, .051]),
                      top_z=.051, height=.051, length=.0496, width=.0494, yaw=0.)
    pear = Detection('pear', 1., center_base=np.array([.1566, .3231, .0563]),
                     top_z=.0563, height=.0563, length=.0559, width=.0164, yaw=0.,
                     valid=False, reject_reason='touches_image_border',
                     extra={'resting_pose': 'uncertain'})
    return runner, target, grasp, place, green, pear


def test_cropped_pear_cannot_prove_green_region_has_no_release_capacity():
    runner, target, grasp, place, green, pear = measured_green_region()
    neighbors = [green, pear]
    capacity, why = runner.has_release_capacity(target, grasp, place, .0502, neighbors)
    assert capacity is None and why == 'release_occupancy_unconfirmed'
    assert runner._neighbors == neighbors and runner._neighbors[-1] is pear
    # Incomplete occupancy still blocks executable release geometry. The only
    # permission gained here is to look from another fully checked viewpoint.
    search, _ = runner._release_search_geometry(target, grasp, place, .0502, neighbors)
    candidate = runner.placer.plan(
        search['zone'], search['held'], search['transit'], held=search['shape'],
        occupied=search['occupied'], strict=True,
        pose_filter=runner._strict_place_filter(target, search['grasp'], neighbors, search['polygon']))
    assert not candidate.feasible


def test_complete_current_pear_measurement_keeps_strict_capacity_checks():
    runner, target, grasp, place, green, _ = measured_green_region()
    pear = Detection('pear', 1., center_base=np.array([.1409, .3522, .113]),
                     top_z=.113, height=.113, length=.0753, width=.0524, yaw=0.,
                     extra={'resting_pose': 'upright'})
    capacity, why = runner.has_release_capacity(target, grasp, place, .0502, [green, pear])
    # At the current 15 mm measurement reserve the old 20 cm fixture is
    # genuinely full. Completeness makes this a definite result, not permission
    # to bypass collision/outline checks. The real dashboard uses 30 cm cards.
    assert capacity is False and why == 'no_clear_supported_release'
    runner.zones[place.zone].size = (.30,.30)
    capacity, why = runner.has_release_capacity(target, grasp, place, .0502, [green, pear])
    assert capacity is True and why == ''


def test_distant_cropped_fragment_does_not_hide_a_completely_occupied_region():
    runner, target, grasp, place, green, pear = measured_green_region()
    green.center_base = np.array([.17, .35, .05])
    green.length = green.width = .30
    pear.center_base = np.array([.55, -.35, .0563])
    capacity, why = runner.has_release_capacity(target, grasp, place, .0502, [green, pear])
    assert capacity is False and why == 'no_clear_supported_release'
