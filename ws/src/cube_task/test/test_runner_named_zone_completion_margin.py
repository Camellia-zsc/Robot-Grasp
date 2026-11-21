"""Executable named-zone releases must satisfy the camera completion margin."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.batch import whole_footprint_in_zone
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.core.types import Zone
from test_runner_local_release import scene


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
