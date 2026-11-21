"""A leaned supported grasp should not pinch a tapered body at unequal heights."""
import copy
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.core.transforms import top_down_T
from cube_task.core.types import Detection, GraspPlan
from cube_task.robot.kinematics import tilt_outward
from cube_task.task.grasp_contact import supported_grasp


def contact_harness():
    detection = Detection('any_supported_object', 1., center_base=np.array([.3, 0., .15]),
        top_z=.15, height=.1, length=.06, width=.05, yaw=0.,
        extra=dict(support_observed=True, support_label='any_base', support_z=.05,
                   grasp=dict(mode='profile')))
    T = top_down_T(.3, 0., .13, 0.)
    baseline = GraspPlan(detection, T.copy(), T.copy(), T.copy(), .1, 0., .05, 0.)
    calls = []
    def plan(target, *, closing_angle):
        calls.append((target, closing_angle))
        pose = top_down_T(.3, 0., .13, closing_angle)
        return GraspPlan(target, pose.copy(), pose.copy(), pose.copy(), .1, 0., .055, closing_angle)
    planner = SimpleNamespace(plan=plan)
    arm = SimpleNamespace(abort_event=threading.Event(), find_reachable_tilt=lambda T, **kwargs:
        (tilt_outward(T, math.radians(10.)), math.radians(10.), ''))
    return detection, baseline, planner, arm, calls


def test_balanced_axis_replans_profile_and_ik_without_changing_source_or_thresholds():
    detection, baseline, planner, arm, calls = contact_harness()
    old_pose = baseline.T_grasp.copy()
    selected = supported_grasp(planner, arm, detection, baseline)
    assert selected is not baseline and selected.target is detection
    assert selected.closing_angle == pytest.approx(math.pi / 2)
    np.testing.assert_array_equal(baseline.T_grasp, old_pose)
    assert selected.close_width == baseline.close_width
    assert selected.T_grasp[2, 3] == baseline.T_grasp[2, 3]
    assert len(calls) == 1 and calls[0][0] is not detection
    info = detection.extra['grasp']['supported_contact']
    assert info['before_gap_mm'] > 8.
    assert info['after_gap_mm'] < .01


@pytest.mark.parametrize('missing', ['support_observed', 'support_label', 'support_z', 'grasp'])
def test_missing_camera_support_or_profile_keeps_original_grasp(missing):
    detection, baseline, planner, arm, calls = contact_harness()
    detection.extra.pop(missing)
    assert supported_grasp(planner, arm, detection, baseline) is baseline
    assert not calls


@pytest.mark.parametrize('case', ['invalid', 'no_ik', 'no_lean', 'already_balanced', 'nan_support'])
def test_unconfirmed_or_already_balanced_grasp_does_not_search(case):
    detection, baseline, planner, arm, calls = contact_harness()
    if case == 'invalid':
        detection.valid = False
    elif case == 'no_ik':
        arm.find_reachable_tilt = lambda T, **kwargs: (None, 0., 'no IK')
    elif case == 'no_lean':
        arm.find_reachable_tilt = lambda T, **kwargs: (T, 0., '')
    elif case == 'already_balanced':
        baseline.T_grasp = top_down_T(.3, 0., .13, math.pi / 2)
        baseline.closing_angle = math.pi / 2
    else:
        detection.extra['support_z'] = float('nan')
    assert supported_grasp(planner, arm, detection, baseline) is baseline
    assert not calls


@pytest.mark.parametrize('rejection', ['asymmetric', 'blocked', 'shifted', 'unreachable', 'bad_width'])
def test_alternative_cannot_bypass_section_neighbor_or_reach_checks(rejection):
    detection, baseline, planner, arm, calls = contact_harness()
    original = planner.plan
    def rejected(target, *, closing_angle):
        candidate = original(target, closing_angle=closing_angle)
        if rejection == 'asymmetric':
            candidate.expected_width = .12  # A long axis is not a round section.
        elif rejection == 'blocked':
            candidate.feasible = False
            candidate.reason = 'blocked_by_neighbor'
        elif rejection == 'shifted':
            candidate.T_grasp[0, 3] += .03
        elif rejection == 'bad_width':
            candidate.expected_width = None
        return candidate
    planner.plan = rejected
    if rejection == 'unreachable':
        original_reach = arm.find_reachable_tilt
        arm.find_reachable_tilt = lambda T, **kwargs: (
            original_reach(T) if np.allclose(T, baseline.T_grasp) else (None, 0., 'no IK'))
    old = copy.deepcopy(detection.extra)
    assert supported_grasp(planner, arm, detection, baseline) is baseline
    assert detection.extra == old


def test_cancellation_does_not_return_an_unchecked_new_grasp():
    detection, baseline, planner, arm, calls = contact_harness()
    arm.abort_event.set()
    assert supported_grasp(planner, arm, detection, baseline) is baseline
    assert not calls


def test_real_measured_profile_candidate_retains_depth_and_pad_stability_checks():
    from cube_task.perception.grasp import GraspPlanner
    from test_grasp_pad_stability import _plan
    from test_task_fakes import make_cfg
    detection, _ = _plan([0., .04, .08, .10], [.08, .08, .06, .05])
    detection.extra['_points_base'][:, 2] += .05
    detection.center_base[2] += .05
    detection.top_z += .05
    detection.length, detection.width, detection.yaw = .08, .06, -math.pi / 2
    detection.extra.update(table_z=.05, support_observed=True, support_label='base', support_z=.05)
    planner = GraspPlanner(make_cfg('L2', robot__gripper__max_width=.1,
                                   grasp__max_grasp_depth=.025), 'L2')
    baseline = planner.plan(detection)
    assert baseline.feasible, baseline.reason
    arm = SimpleNamespace(abort_event=threading.Event(), find_reachable_tilt=lambda T, **kwargs:
        (tilt_outward(T, math.radians(10.)), math.radians(10.), ''))
    selected = supported_grasp(planner, arm, detection, baseline)
    assert selected is not baseline
    assert selected.target.extra['grasp']['mode'] == 'profile'
    assert selected.target.extra['grasp']['neck_taper_deg'] <= 20.
    assert selected.T_grasp[2, 3] >= detection.top_z - .025 - 1e-9
    assert selected.expected_width <= planner.max_width
    invalid = planner.plan(copy.deepcopy(detection), closing_angle=float('nan'))
    assert not invalid.feasible
