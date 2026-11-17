"""A wide opening does not make a steep shoulder a stable pad contact."""
import math

import numpy as np
import pytest

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
