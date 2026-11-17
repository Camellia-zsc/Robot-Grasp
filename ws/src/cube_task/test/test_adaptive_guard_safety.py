"""Execution boundaries must retain grasps and require current camera evidence."""
import copy

import numpy as np
import pytest

from cube_task.assistant.server import DashboardController
from cube_task.assistant.simulation import SimulationSession
from cube_task.core.types import PickResult
from test_assistant_batch import BatchClient, BatchSession
from test_assistant_server import Settings
from test_task_fakes import make_det
from test_task_skills import _setup


@pytest.mark.parametrize('changed_stage,expected_motion_count', [('release', 4), ('open_release', 5)])
def test_changed_scene_after_grasp_keeps_object_without_using_old_drop_or_putback(
        changed_stage, expected_motion_count):
    _cfg, _clock, world, arm, _log, skill, grasp, place = _setup()
    skill.execution_guard = lambda stage: 'scene_changed' if stage == changed_stage else None

    result = skill.execute(grasp, place)

    assert not result.ok and result.error == 'scene_changed'
    assert result.detail['holding'] and not result.detail['fault']
    assert world.held is not None and world.held.label == 'red'
    assert not world.dropped and not world.placed
    # The only gripper commands are the initial opening and successful closure.
    assert arm.gripper_cmds() == [grasp.open_width, grasp.close_width]
    assert len(arm.motions()) == expected_motion_count
    assert np.allclose(arm.motions()[-1][1],
                       place.T_pre if changed_stage == 'release' else place.T_release)


def test_failed_post_place_capture_cannot_reuse_a_pre_pick_image(monkeypatch):
    with SimulationSession(realtime=False) as sim:
        sim._prepared = True
        physical = next(obj for obj in sim.world.objects if obj.label == 'red')
        zone = sim.runner.zones['zone_red']
        physical.place_upright(*zone.center[:2], 0., sim.world.table_z)
        old_measurement = make_det('red', *zone.center[:2], size=.05, height=.05)
        # The old image really does satisfy this goal. That is exactly why it
        # must expire when the post-release acquisition fails.
        sim._last_fresh = [old_measurement]
        monkeypatch.setattr(sim.runner, '_observe', lambda _view: ('timeout', None, None))
        skill_result = PickResult(ok=True, stage='done', label='red', zone='zone_red',
                                  grasp_width=.05, detail={'fault': False})

        result = sim._finish_attempt(skill_result, '', 'old-target', 'zone_red',
                                     old_measurement, physical)

        assert result['controller_ok'] and result['truth']['success']
        assert result.get('observation_error')
        assert not sim._last_fresh
        assert not result['camera_verified'] and not result['ok']
        assert result['error'] == 'placement_not_observed'


def test_occluded_cached_completed_object_cannot_make_queue_report_a_skip():
    class OccludedCompletedSession(BatchSession):
        targeted_observation = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            apple = self.scene['objects'][0]
            apple['center_base'][:2] = [.17, -.35]
            apple['zone'] = 'zone_red'
            pear = self.scene['objects'][1]
            pear['center_base'][:2] = [.17, .35]
            pear['zone'] = 'zone_green'
            self._last_fresh = []

        def observe(self, target=None, purpose='search', forget=()):
            assert purpose == 'confirm_completed'
            self.observe_calls += 1
            # Camera coverage is occluded, so the observation policy retains
            # the old obstacle in the merged scene, with no fresh confirmation.
            self._last_fresh = []
            return copy.deepcopy(self.scene)

    ctl = DashboardController(session_factory=OccludedCompletedSession,
                              client=BatchClient(), settings=Settings(), realtime=False)
    try:
        assert ctl.wait_idle(3)
        ctl.submit('把苹果放到红色区域')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert ctl._session.observe_calls == 1
        assert not ctl._session.executions
        assert not state['last_result']['ok']
        assert state['batch']['completed'] == 0
        assert state['batch']['items'][0]['status'] == 'failed'
        assert not state['batch']['items'][0]['result'].get('skipped')
    finally:
        ctl.close()


def test_intervention_accepts_clear_space_below_a_held_object_but_refuses_3d_overlap():
    events = []
    with SimulationSession(realtime=False,
                           on_event=lambda kind, fields: events.append((kind, fields))) as sim:
        held = next(obj for obj in sim.world.objects if obj.label == 'red')
        movable = next(obj for obj in sim.world.objects if obj.label == 'green')
        sim.world.objects = [held, movable]
        held.place_upright(.3, 0., 0., .20)
        movable.place_upright(.45, .2, 0., sim.world.table_z)
        sim.world.held = held
        sim._prepared = True
        sim._publish_snapshot()

        accepted = sim.intervene_scene(dict(action='move', object_id=movable.name,
                         xy=[.3, 0.], scene_version=sim.snapshot()['scene_version']))
        assert accepted['queued']
        np.testing.assert_allclose(movable.center_xy(), [.3, 0.], atol=1e-10)
        assert any(kind == 'scene_intervention_applied' for kind, _fields in events)
        revision = sim._environment_revision
        prior_pose = movable.T_base_obj.copy()
        events.clear()

        # Lowering the held object removes that vertical separation. Rotating
        # the table object at the same XY would now be a real collision.
        held.place_upright(.3, 0., 0., sim.world.table_z)
        sim._publish_snapshot()
        sim.intervene_scene(dict(action='move', object_id=movable.name,
                        yaw=.3, scene_version=sim.snapshot()['scene_version']))
        np.testing.assert_array_equal(movable.T_base_obj, prior_pose)
        assert sim._environment_revision == revision
        assert any(kind == 'scene_intervention_rejected' for kind, _fields in events)
