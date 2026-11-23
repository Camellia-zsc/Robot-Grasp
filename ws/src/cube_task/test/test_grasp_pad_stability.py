"""A wide opening does not make a steep shoulder a stable pad contact."""
import math

import numpy as np
import pytest

from cube_task.core.transforms import closing_angle_of
from cube_task.perception.grasp import GraspPlanner
from test_task_fakes import make_cfg, make_det


def _plan(heights, widths, *, min_depth=.025, top=.100, pad_half=.015,
          observed_heights=None):
    # A measured surface of revolution, without a simulator or true object pose.
    z = (np.asarray(observed_heights) if observed_heights is not None else
         np.linspace(heights[0], heights[-1], round((heights[-1] - heights[0]) / .00025) + 1))
    angle = np.linspace(0., 2 * math.pi, 90, endpoint=False)
    radius = np.interp(z, heights, widths) / 2
    cloud = np.stack(np.broadcast_arrays(
        .30 + radius[:, None] * np.cos(angle),
        radius[:, None] * np.sin(angle), z[:, None]), axis=-1).reshape(-1, 3)
    det = make_det('pear', .30, 0., size=max(widths), height=top)
    det.extra.update(_points_base=cloud, cap_center_xy=[.30, 0.])
    cfg = make_cfg('L2', robot__gripper__max_width=.100,
                   grasp__max_grasp_depth=.025, grasp__min_grasp_depth=min_depth,
                   grasp__pad_half=pad_half)
    return det, GraspPlanner(cfg, 'L2').plan(det)


def test_lowest_tcp_does_not_hide_the_steep_lower_half_of_the_pads():
    det, plan = _plan([0., .040, .100], [.090, .090, .015])

    assert not plan.feasible
    assert 'dome' in plan.reason
    assert det.extra['grasp']['neck_taper_deg'] > 20.


@pytest.mark.parametrize('upper_width', [.074, .0795])
def test_a_short_steep_shoulder_cannot_be_hidden_by_a_wider_section_above_it(upper_width):
    # The first 10 mm above the pad bottom is steep; the next 5 mm widens again.
    # Averaging to the TCP, or taking the highest near-maximum, hides that drop.
    det, plan = _plan([0., .061, .070, .075, .080, .100],
                      [.080, .080, .064, upper_width, upper_width, .012])

    assert not plan.feasible
    assert 'dome' in plan.reason
    assert det.extra['grasp']['neck_taper_deg'] > 20.


def test_a_width_peak_above_the_pad_band_does_not_certify_its_lower_shoulder():
    det, plan = _plan([0., .040, .090, .110, .120, .130],
                      [.085, .085, .025, .025, .090, .020])

    assert not plan.feasible
    assert 'dome' in plan.reason
    assert det.extra['grasp']['neck_taper_deg'] > 20.


def test_a_later_candidate_with_an_equator_inside_its_pad_band_is_used():
    det, plan = _plan([0., .061, .070, .075, .080, .100],
                      [.080, .080, .064, .074, .074, .012], min_depth=.008)

    assert plan.feasible, plan.reason
    assert plan.T_grasp[2, 3] >= .080 - 1e-9
    assert not det.extra['grasp'].get('neck')


def test_a_gently_tapered_neck_remains_feasible():
    det, plan = _plan([0., .040, .080, .100], [.080, .080, .060, .050])

    assert plan.feasible, plan.reason
    assert det.extra['grasp']['neck']
    assert det.extra['grasp']['neck_taper_deg'] <= 20.


def test_fingertip_tcp_uses_only_the_contact_pad_above_the_tip():
    # The steep shoulder BELOW the fingertip cannot make the actual straight
    # 30 mm pad above it unstable. Depth limits still keep the palm clear.
    det, _ = _plan([0., .060, .075, .105, .120],
                   [.085, .085, .054, .054, .010], top=.120, min_depth=.008)
    planner = GraspPlanner(make_cfg('L2', robot__gripper__max_width=.1,
        grasp__max_grasp_depth=.040, grasp__pad_center_offset=.015), 'L2')
    plan = planner.plan(det)
    assert plan.feasible, plan.reason
    assert plan.T_grasp[2, 3] == pytest.approx(.080)
    assert plan.expected_width == pytest.approx(.054, abs=.001)


def test_shifted_contact_pad_still_rejects_a_steep_shoulder_above_the_tip():
    det, _ = _plan([0., .075, .120], [.085, .085, .010], top=.120, min_depth=.008)
    planner = GraspPlanner(make_cfg('L2', robot__gripper__max_width=.1,
        grasp__max_grasp_depth=.040, grasp__min_grasp_depth=.040,
        grasp__pad_center_offset=.015), 'L2')
    plan = planner.plan(det)
    assert not plan.feasible and 'dome' in plan.reason


def test_a_widest_section_inside_the_actual_pad_band_remains_feasible():
    det, plan = _plan([0., .060, .075, .090, .100],
                      [.040, .050, .080, .050, .020])

    assert plan.feasible, plan.reason
    assert not det.extra['grasp'].get('neck')


@pytest.mark.parametrize('height, pad_half', [(.010, .015), (.030, .004)])
def test_measured_straight_walls_do_not_need_a_full_10mm_of_pad_contact(height, pad_half):
    det, plan = _plan([0., height], [.030, .030], top=height,
                      min_depth=.008, pad_half=pad_half)

    assert plan.feasible, plan.reason
    assert plan.expected_width == pytest.approx(.030, abs=.001)
    assert not det.extra['grasp'].get('neck')


@pytest.mark.parametrize('top, pad_half', [(.010, .015), (.0101, .015), (.030, .004)])
def test_a_thin_rim_or_filled_profile_is_not_evidence_of_straight_sidewalls(top, pad_half):
    # A narrow ring can straddle a 5 mm bin edge. Filled lower bins must not
    # turn those few observed heights into an apparently tall, constant wall.
    det, plan = _plan([top - .0002, top], [.030, .030], top=top,
                      min_depth=.008, pad_half=pad_half)

    assert not plan.feasible
    assert 'dome' in plan.reason


def test_four_isolated_rings_are_not_a_continuous_short_sidewall():
    # Each 5 mm bin has a large measured min/max span, but the surface between
    # those samples is missing for 4.8 mm inside both bins.
    det, plan = _plan([0., .010], [.030, .030], top=.010, min_depth=.008,
                      observed_heights=[.0001, .0049, .0051, .0099])

    assert not plan.feasible
    assert 'dome' in plan.reason


def _partly_observed_square(*, sidewall=True, blocked=False):
    # An overhead RGB-D view sees the cap and one face. The neutral closing
    # axis slices through the cap only; the perpendicular pair sees the wall.
    xy = np.linspace(-.024, .024, 49)
    xx, yy = np.meshgrid(xy, xy)
    cap = np.column_stack((.30 + xx.ravel(), yy.ravel(),
                           np.full(xx.size, .048)))
    cloud = cap
    if sidewall:
        yy, zz = np.meshgrid(xy, np.linspace(.008, .048, 161))
        wall = np.column_stack((np.full(yy.size, .324), yy.ravel(), zz.ravel()))
        cloud = np.vstack((cap, wall))
    det = make_det('unlisted_box', .30, 0., size=.048, height=.048)
    det.extra.update(_points_base=cloud, cap_center_xy=[.30, 0.])
    if blocked:
        det.extra['_neighbor_xy'] = np.array([[.34, 0.]])
    return det


def test_a_partial_square_uses_the_face_pair_with_measured_sidewall_contact():
    det = _partly_observed_square()
    planner = GraspPlanner(make_cfg(robot__gripper__max_width=.100), 'L1')
    plan = planner.plan(det)

    assert plan.feasible, plan.reason
    assert abs(math.sin(plan.closing_angle)) < 1e-9
    assert plan.expected_width == pytest.approx(.048, abs=.001)
    for pose in (plan.T_pre, plan.T_grasp, plan.T_lift):
        assert closing_angle_of(pose) == pytest.approx(plan.closing_angle)


def test_measured_sidewall_fallback_does_not_turn_into_a_neighbor():
    det = _partly_observed_square(blocked=True)
    plan = GraspPlanner(make_cfg(robot__gripper__max_width=.100), 'L1').plan(det)

    assert not plan.feasible
    assert 'dome' in plan.reason


def test_a_square_cap_alone_does_not_establish_sidewall_contact():
    det = _partly_observed_square(sidewall=False)
    plan = GraspPlanner(make_cfg(robot__gripper__max_width=.100), 'L1').plan(det)

    assert not plan.feasible
    assert 'dome' in plan.reason
