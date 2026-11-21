"""Balanced finger heights must not override a measured neighboring obstacle."""
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.perception.grasp import GraspPlanner
from cube_task.robot.kinematics import tilt_outward
from cube_task.task.grasp_contact import supported_grasp
from test_task_fakes import make_cfg, make_det


@pytest.mark.parametrize('adaptive', [False, True])
def test_supported_axis_rechecks_real_neighbor_sweep_even_when_adaptive_search_is_disabled(adaptive):
    cfg = make_cfg('L2', robot__gripper__max_width=.1, grasp__max_grasp_depth=.025,
                   grasp__min_grasp_depth=.025, grasp__pad_half=.015,
                   grasp__adaptive_neighbors=adaptive)
    heights = np.linspace(.05, .15, 401)
    angles = np.linspace(0., 2 * math.pi, 90, endpoint=False)
    cloud = np.stack(np.broadcast_arrays(.3 + .025 * np.cos(angles),
        .025 * np.sin(angles), heights[:, None]), axis=-1).reshape(-1, 3)
    detection = make_det('apple', .3, 0., size=.05, height=.1, table_z=.05)
    detection.extra.update(_points_base=cloud, _neighbor_xy=np.array([[.3, .045]]),
        support_observed=True, support_label='measured_base', support_z=.05)
    planner = GraspPlanner(cfg, 'L2')
    baseline = planner.plan(detection)
    assert baseline.feasible
    neighbors = detection.extra['_neighbor_xy']

    def hits(plan):
        return int(np.count_nonzero(planner._sweep_mask(neighbors, plan.closing_angle,
            *plan.T_grasp[:2, 3], plan.expected_width, plan.open_width)))

    assert hits(baseline) == 0
    arm = SimpleNamespace(abort_event=threading.Event(), find_reachable_tilt=lambda T, **kwargs:
        (tilt_outward(T, math.radians(10.)), math.radians(10.), ''))
    selected = supported_grasp(planner, arm, detection, baseline)
    assert selected.feasible
    assert hits(selected) == 0
