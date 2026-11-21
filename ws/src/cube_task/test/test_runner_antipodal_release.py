"""Parallel jaws share an axis at pi, but their full tilted tool poses differ."""
import math

import numpy as np
import pytest

from cube_task.core.types import Detection, Zone
from cube_task.robot.kinematics import tilt_outward
from test_runner_stack_opening import stacked_scene


def shifted_axis_scene():
    parts = list(stacked_scene())
    runner, target, grasp, previous, _support, _zone, polygon = parts
    target.center_base = np.array([.4972282929, .0150290585, .1126])
    target.top_z = target.height = .1126
    target.length, target.width, target.yaw = .0754, .0512, -1.469
    target.extra.pop('_restore_yaw')
    support = dict(label='yellow', xy=[.3361368907, -.1634126142],
        top_z=.0481517628, length=.0466806998, width=.0458290105, yaw=-.4747256958)
    target.extra['_stack_support'] = support
    source = [.5000081814, .0149978172, .0423651764]
    target.extra['_held_translation_anchor'] = dict(source_center=source,
        translation_only=True, pair=dict(ok=True, translation_only=True))
    grasp.T_grasp = np.array([
        [-.0991643437, .9608539317, .2587008987, .4865312087],
        [.9948989902, .1005726410, .0078194081, .0138189788],
        [-.0185049236, .2581566693, -.9659258263, .0970688249],
        [0., 0., 0., 1.]])
    grasp.closing_angle = .1017750605
    grasp.expected_width = .0485007331
    target.extra['grasp'] = {'kind': 'elongated'}
    runner.nominal['yellow'] = (.05, .05)
    zone = Zone('on_object_yellow', np.r_[support['xy'], support['top_z']],
                (.3, .3), release_height=.004)
    previous.zone = zone.name
    measured = Detection('yellow', 1., center_base=zone.center.copy(),
        top_z=support['top_z'], height=support['top_z'],
        length=support['length'], width=support['width'], yaw=support['yaw'])
    # The incident's source and candidate release are both actually reachable
    # at these leans. Isolate full-tool yaw enumeration; all body/finger/
    # source-anchor/certificate geometry remains the production implementation.
    runner.arm.check_reachable = lambda *args, **kwargs: (True, '')
    def reachable_tilt(pose, **kwargs):
        current = math.acos(np.clip(-float(pose[2, 2]), -1., 1.))
        if current >= math.radians(9.99):
            return pose.copy(), current, ''
        return tilt_outward(pose, math.radians(10.)), math.radians(10.), ''
    runner.arm.find_reachable_tilt = reachable_tilt
    return runner, target, grasp, previous, measured, zone, polygon, np.asarray(source)


def test_real_replan_tries_antipodal_tool_pose_across_parallel_jaw_axis_boundary():
    runner, target, grasp, previous, support, zone, polygon, source = shifted_axis_scene()
    candidate, why = runner.replan_release(target, grasp, previous, .038515,
        [target, support], placement_zone=zone, placement_polygon=polygon,
        exact_xy=True, observation_filter=lambda *args, **kwargs: True)
    assert candidate is not None, why
    closing = math.atan2(candidate.T_release[1, 1], candidate.T_release[0, 1])
    assert closing == pytest.approx(-.4525055382, abs=.001)
    rotation = candidate.T_release[:3, :3] @ grasp.T_grasp[:3, :3].T
    tilt = math.degrees(math.acos(np.clip(rotation[2, 2], -1., 1.)))
    assert tilt < 6.
    landing = candidate.T_release[:3, 3] + rotation @ (source-grasp.T_grasp[:3, 3])
    np.testing.assert_allclose(landing[:2], support.center_base[:2], atol=.001)
    assert hasattr(candidate, 'release_opening_checked')


def test_explicit_pear_restore_yaw_does_not_gain_an_antipodal_exemption():
    runner, target, grasp, previous, support, zone, polygon, _source = shifted_axis_scene()
    target.extra['_restore_yaw'] = target.yaw + 2.6890871154-grasp.closing_angle
    candidate, why = runner.replan_release(target, grasp, previous, .038515,
        [target, support], placement_zone=zone, placement_polygon=polygon,
        exact_xy=True, observation_filter=lambda *args, **kwargs: True)
    assert candidate is None and why


def test_previous_successful_source_side_keeps_its_safe_full_tool_heading():
    runner, target, grasp, previous, support, zone, polygon, _source = shifted_axis_scene()
    target.center_base = np.array([.5069095496, -.0100500188, .1123717278])
    target.top_z, target.height, target.yaw = .1123717278, .1123673572, 1.5223880940
    grasp.closing_angle = 3.0931844208
    grasp.T_grasp = np.array([
        [-.0474156503, -.9647749885, .2587681928, .4970273513],
        [-.9988478545, .0477141808, -.0051303536, -.0096693091],
        [-.0073972755, -.2587133132, -.9659258263, .0972268697],
        [0., 0., 0., 1.]])
    target.extra['_held_translation_anchor']['source_center'] = [.5100025995, -.0099829945, .0423781738]
    candidate, why = runner.replan_release(target, grasp, previous, .038515,
        [target, support], placement_zone=zone, placement_polygon=polygon,
        exact_xy=True, observation_filter=lambda *args, **kwargs: True)
    assert candidate is not None, why
    closing = math.atan2(candidate.T_release[1, 1], candidate.T_release[0, 1])
    assert closing == pytest.approx(2.6890871154, abs=.001)
    assert hasattr(candidate, 'release_opening_checked')
