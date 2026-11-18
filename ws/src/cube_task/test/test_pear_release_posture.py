"""Reach adjustments must preserve an observed upright fruit at release."""
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.core.transforms import rot_y, top_down_T
from cube_task.task.reach import adjust_plan_poses


@pytest.mark.parametrize('label,pose,tilt,allowed', [
    ('pear', 'upright', 5., True), ('pear', 'upright', 25., False),
    ('banana', 'flat', 25., True), ('apple', None, 25., True),
])
def test_final_reach_tilt_is_checked_before_mutating_plan(label, pose, tilt, allowed):
    original = top_down_T(.3, 0., .1, 0.)
    release = top_down_T(.3, .1, .1, 0.)
    class Arm:
        def find_reachable_tilt(self, target, **kwargs):
            out = target.copy()
            if np.allclose(target[:3, 3], release[:3, 3]):
                out[:3, :3] = rot_y(np.deg2rad(tilt)) @ target[:3, :3]
            return out, 0., ''
        def check_reachable(self, target, **kwargs):
            return True, ''
    target = SimpleNamespace(label=label, extra={'resting_pose': pose})
    gp = SimpleNamespace(target=target, T_grasp=original.copy(),
                         T_pre=original.copy(), T_lift=original.copy())
    pp = SimpleNamespace(T_release=release.copy(), T_pre=release.copy(), T_retreat=release.copy())
    ok, reason, _ = adjust_plan_poses(Arm(), gp, pp, free_tilt_deg=40.)
    assert ok is allowed
    if not allowed:
        assert 'upright_pear_tilt' in reason
        np.testing.assert_array_equal(pp.T_release, release)


def test_held_release_uses_the_actual_completed_grasp_orientation():
    original = top_down_T(.3, 0., .1, 0.)
    original[:3, :3] = rot_y(np.deg2rad(20.)) @ original[:3, :3]
    release = top_down_T(.3, .1, .1, 0.)
    release[:3, :3] = rot_y(np.deg2rad(25.)) @ release[:3, :3]
    class Arm:
        def check_reachable(self, target, **kwargs):
            return True, ''
    gp = SimpleNamespace(target=SimpleNamespace(label='pear', extra={'resting_pose': 'upright'}),
        T_grasp=original.copy(), T_pre=original.copy(), T_lift=original.copy())
    pp = SimpleNamespace(T_release=release.copy(), T_pre=release.copy(), T_retreat=release.copy())
    ok, _, _ = adjust_plan_poses(Arm(), gp, pp, free_tilt_deg=40., held_only=True)
    assert ok  # 25 deg tool lean changes this already tilted grasp by only 5 deg.
    np.testing.assert_array_equal(gp.T_grasp, original)
