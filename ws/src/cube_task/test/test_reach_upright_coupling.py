"""Contact IK must preserve the fruit's body-up across different wrist leans."""
from types import SimpleNamespace

import numpy as np

from cube_task.core.transforms import rot_y, top_down_T
from cube_task.core.types import Detection, GraspPlan, PlacePlan
from cube_task.task.reach import adjust_plan_poses


def test_initial_plan_couples_source_lean_when_vertical_release_is_unreachable():
    target = Detection('pear',1.,extra={'resting_pose':'upright'})
    pose = top_down_T(.3,0.,.075,0.)
    grasp = GraspPlan(target,pose.copy(),pose.copy(),pose.copy(),.1,0.,.05,0.)
    release = top_down_T(.4,0.,.13,0.)
    place = PlacePlan('stack',release.copy(),release.copy(),release.copy(),.1)
    def check(T, **kwargs):
        if T[0,3]>.35 and abs(T[2,2]+1)<1e-6:
            return False,'no IK solution within joint limits'
        return True,''
    def find(T, **kwargs):
        result = T.copy()
        if not check(T)[0]:
            result[:3,:3] = rot_y(np.radians(10))@T[:3,:3]
        return result, np.arccos(-result[2,2]), ''
    arm = SimpleNamespace(find_reachable_tilt=find,check_reachable=check)
    ok,why,_ = adjust_plan_poses(arm,grasp,place)
    assert ok,why
    assert not np.allclose(grasp.T_grasp[:3,:3],pose[:3,:3])
    np.testing.assert_allclose(place.T_release[:3,:3]@grasp.T_grasp[:3,:3].T@[0.,0.,1.],
                               [0.,0.,1.],atol=1e-9)
    frozen = grasp.T_grasp.copy()
    assert adjust_plan_poses(arm,grasp,place,held_only=True)[0]
    np.testing.assert_array_equal(grasp.T_grasp,frozen)
