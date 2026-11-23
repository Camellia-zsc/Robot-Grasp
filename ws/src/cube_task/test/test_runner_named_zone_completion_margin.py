"""Executable named-zone releases must satisfy the camera completion margin."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.batch import whole_footprint_in_zone
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.core.types import Zone
from cube_task.task.runner import TaskRunner
from test_runner_local_release import scene


def test_interactive_carry_cannot_lower_below_a_tall_body_outside_the_old_route():
    from dataclasses import replace
    runner, target, grasp, place = scene()
    runner.min_approach, runner.transit_clear = .02, .03
    runner._neighbors = [target, replace(target, label='pear',
        center_base=np.array([.3, -.3, .125]), top_z=.125, height=.125)]
    static_floor = TaskRunner._floors(runner, target, grasp, place, .02)['lift']
    runner.cfg.set('assistant.dynamic_scene_clearance', True)
    interactive_floor = TaskRunner._floors(runner, target, grasp, place, .02)['lift']
    assert static_floor < .195
    assert interactive_floor >= .195


def zone_scene(size=(.2, .2), yaw=0.):
    runner, target, grasp, place = scene()
    zone = Zone('zone_red', np.array([.4, 0., 0.]), size, yaw=yaw)
    runner.zones = {zone.name: zone}
    camera_scene = {'zones': [{'name': zone.name, 'center': zone.center.tolist(),
                              'size': list(zone.size), 'yaw': zone.yaw}]}
    return runner, target, grasp, place, zone, camera_scene


def test_recorded_case08_outline_with_2_618_mm_gap_is_not_a_complete_release():
    runner, target, grasp, _place, zone, camera_scene = zone_scene()
    zone.center = np.array([.17, .35, 0.])
    camera_scene['zones'][0]['center'] = zone.center.tolist()
    target.yaw = 1.1768
    observed = target.summary() | {'center_base': [.1207, .2853, .0505],
                                   'length': .0498, 'width': .0487}
    assert not whole_footprint_in_zone(observed, camera_scene, zone.name)
    allowed = runner._strict_place_filter(target, grasp, [], zone.corners_xy(),
                                          completion_zone=zone)
    assert not allowed(.1207, .2853, grasp.closing_angle)
    # Moving inward satisfies the same existing five-millimetre contract.
    assert allowed(.1207, .2893, grasp.closing_angle)


@pytest.mark.parametrize('yaw', [0., math.pi / 4, math.pi / 3])
def test_named_zone_margin_uses_zone_axes_and_matches_completion(yaw):
    runner, target, grasp, _place, zone, camera_scene = zone_scene(yaw=yaw)
    target.yaw = yaw
    allowed = runner._strict_place_filter(target, grasp, [], zone.corners_xy(),
                                          completion_zone=zone)
    direction = np.array([math.cos(yaw), math.sin(yaw)])
    for gap, expected in ((.004, False), (.0051, True), (.006, True)):
        xy = zone.center[:2] + direction * (.1 - .025 - gap)
        observed = target.summary() | {'center_base': [*xy, .05]}
        assert whole_footprint_in_zone(observed, camera_scene, zone.name) is expected
        assert allowed(*xy, grasp.closing_angle) is expected


def test_capacity_and_executable_replan_reject_the_same_three_mm_only_zone(monkeypatch):
    runner, target, grasp, place, _zone, _camera_scene = zone_scene(size=(.056, .056))
    monkeypatch.setattr('cube_task.task.reach.adjust_plan_poses',
                        lambda *a, **k: pytest.fail('no legal completed footprint may reach IK'))
    assert runner.has_release_capacity(target, grasp, place, .05, []) == (
        False, 'no_clear_supported_release')
    updated, why = runner.replan_release(target, grasp, place, .05, [])
    assert updated is None and why == 'no_clear_supported_release'


def test_final_leaned_pose_keeps_the_same_named_zone_completion_check():
    runner, target, grasp, place, zone, _camera_scene = zone_scene(size=(.056, .056))
    ok, why = runner._adjust_strict_release(target, grasp, place, [], zone.corners_xy(),
        zone.center[:2], .02, held_only=True, completion_zone=zone)
    assert not ok and why == 'leaned release has no full footprint or finger clearance'


def test_dashboard_reserves_measurement_error_without_weakening_camera_completion():
    runner, target, grasp, _place, zone, camera_scene = zone_scene()
    runner.cfg.set('assistant.named_zone_measurement_margin', .004)
    allowed = runner._strict_place_filter(target, grasp, [], zone.corners_xy(), completion_zone=zone)
    near_edge = zone.center[:2] + [.1-.025-.008, 0.]
    assert whole_footprint_in_zone(target.summary() | {'center_base': [*near_edge, .05]}, camera_scene, zone.name)
    assert not allowed(*near_edge, grasp.closing_angle)
    safer = zone.center[:2] + [.1-.025-.0091, 0.]
    assert allowed(*safer, grasp.closing_angle)


def test_recorded_long_fruit_cross_view_shift_does_not_force_a_second_grasp():
    runner, target, grasp, _place, zone, camera_scene = zone_scene(size=(.24,.24))
    zone.center = np.array([.17,-.35,0.])
    camera_scene['zones'][0]['center'] = zone.center.tolist()
    target.label = 'banana'
    target.extra['resting_pose'] = 'flat'
    target.length, target.width = .1335,.0451
    runner.nominal['banana'] = (.154785,.052385)
    runner.cfg.set('assistant.named_zone_measurement_margin', .015)
    allowed = runner._strict_place_filter(target, grasp, [], zone.corners_xy(), completion_zone=zone)
    assert not allowed(.17,-.266,grasp.closing_angle)
    assert not allowed(.17,-.270,grasp.closing_angle)
    assert allowed(.17,-.281,grasp.closing_angle)
    # The same observed 11 mm centre difference failed at the old seed. The
    # interior seed keeps the unchanged five-millimetre camera postcondition.
    measured = target.summary() | {'length': .1421,'width': .0395,'yaw': -.0165}
    assert not whole_footprint_in_zone(measured | {'center_base': [.1655,-.255,.0398]},camera_scene,zone.name)
    assert whole_footprint_in_zone(measured | {'center_base': [.1655,-.270,.0398]},camera_scene,zone.name)


def test_real_named_region_options_keep_completion_contract_for_held_transfer(monkeypatch):
    runner, target, grasp, place, zone, _camera_scene = zone_scene(size=(.056, .056))
    executor = object.__new__(SkillExecutor)
    executor.session = SimpleNamespace(runner=runner)
    options = executor._options({'type': 'named_region', 'name': zone.name})
    assert options['placement_zone'] is not zone
    runner._held_transfer_feedback = lambda det: (
        {'det': target, 'grasp': grasp, 'place': place}, .05, '')
    monkeypatch.setattr('cube_task.task.reach.adjust_plan_poses',
                        lambda *a, **k: pytest.fail('incomplete named-zone footprint reached IK'))
    planned, why = runner.plan_held_transfer(target, [], **options)
    assert planned is None and why == 'no_clear_supported_release'


@pytest.mark.parametrize('name', ['pick_area', 'desktop_point', 'on_object_green'])
def test_explicit_supports_keep_their_existing_two_mm_contract(name):
    runner, target, grasp, place, zone, _camera_scene = zone_scene(size=(.056, .056))
    zone.name = place.zone = name
    search, why = runner._release_search_geometry(target, grasp, place, .05, [],
                                                 zone, zone.corners_xy())
    assert search is not None and why == '' and search['completion_zone'] is None
    allowed = runner._strict_place_filter(target, grasp, [], search['polygon'],
                                          completion_zone=search['completion_zone'])
    assert allowed(*zone.center[:2], grasp.closing_angle)
