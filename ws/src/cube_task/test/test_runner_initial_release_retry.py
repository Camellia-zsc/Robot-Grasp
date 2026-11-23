"""Initial release planning must explore wrist poses before discarding a free slot."""
import math

import numpy as np

from cube_task.core.transforms import closing_angle_of, top_down_T
from cube_task.assistant.simulation import SimulationSession
from test_task_fakes import Rig, make_det, make_cfg


def test_release_ik_failure_retries_orientation_at_the_same_clear_position(monkeypatch):
    from cube_task.task import reach
    rig = Rig('L1')
    target = rig.world.add(make_det('red', .3, 0.))
    runner = rig.runner()
    zone = runner.zones['zone_red']
    original = reach.adjust_plan_poses
    tried = []
    def adjust(arm, grasp, place, *args, **kwargs):
        angle = closing_angle_of(place.T_release)
        tried.append((place.T_release[:2,3].copy(), angle))
        if abs(math.remainder(angle-tried[0][1], math.pi)) < math.radians(10):
            return False, 'release: no IK solution within joint limits', {}
        return original(arm, grasp, place, *args, **kwargs)
    monkeypatch.setattr(reach, 'adjust_plan_poses', adjust)
    result, why = runner.pick_detection(target, neighbors=[target],
        placement_zone=zone, placement_polygon=zone.corners_xy())
    assert result is not None and result.ok, why
    assert len(tried) >= 2
    np.testing.assert_allclose(tried[0][0], tried[-1][0])


def test_interactive_workspace_drops_the_legacy_competition_y_box_only(tmp_path):
    before = make_cfg()
    assert before.get('robot.workspace.y') == [-.4,.4]
    with SimulationSession(realtime=False, log_dir=tmp_path) as session:
        radius = session.arm.guard.r_max
        assert session.arm.guard.y == (-radius, radius)
        pose = top_down_T(.1396, .4075, .02412, 1.246)
        ok, why = session.arm.check_reachable(pose)
        assert ok, why
        pose[0,3] = 0.
        assert not session.arm.check_reachable(pose)[0]
    assert before.get('robot.workspace.y') == [-.4,.4]
