"""Real RGB-D, shape planning, guarded IK and feedback for editable fruit scenes."""
import copy

import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor


def edit(session, action, **fields):
    return session.edit_scene(dict(scene_version=session.scene['scene_version'],
                                   action=action, **fields))


@pytest.mark.parametrize('label,yaw', [('apple', 0.4), ('banana', -0.6), ('pear', 0.6)])
def test_catalog_fruit_is_camera_measured_grasped_and_verified(tmp_path, label, yaw):
    with SimulationSession(realtime=False, log_dir=tmp_path) as session:
        session.prepare()
        edit(session, 'clear')
        old_version = session.scene['scene_version']
        edit(session, 'add', catalog_id=label, xy=[0.28, 0.0], yaw=yaw)
        assert session.scene['scene_version'] > old_version
        assert session.scene['objects'] == []
        assert session.observation_views() == []
        assert session.frame_jpeg() is None
        scene = session.observe()
        targets = [o for o in scene['objects'] if o['valid'] and o['label'] == label]
        assert len(targets) == 1, scene['objects']
        target = targets[0]
        assert target['center_base'] is not None
        result = session.execute(dict(scene_version=scene['scene_version'], action='pick',
                                      object_id=target['object_id'], zone_name=None))
        assert result['ok'], result
        assert result['truth']['success'] and not result['holding']
        feedback = session.snapshot()['feedback']
        assert feedback['grasp_plan']['label'] == label
        assert feedback['grasp_plan']['kind'] != 'square'
        checks = {check['stage']: check for check in feedback['checks']}
        assert {'close', 'dropped', 'dropped_in_transit', 'placement'} <= checks.keys()
        assert all(checks[stage]['ok'] for stage in
                   ('close', 'dropped', 'dropped_in_transit', 'placement'))
        assert checks['close']['measured_width'] > 0


def test_layout_edit_invalidates_old_decision_and_rejects_bad_geometry_atomically(tmp_path):
    with SimulationSession(realtime=False, log_dir=tmp_path) as session:
        scene = session.prepare()
        target = next(o for o in scene['objects'] if o['valid'])
        decision = dict(scene_version=scene['scene_version'], action='pick',
                        object_id=target['object_id'], zone_name=None)
        edit(session, 'clear')
        calls = session.arm.inner.n_calls
        assert session.execute(decision)['error'] == 'stale_scene'
        assert session.arm.inner.n_calls == calls
        before = copy.deepcopy(session.snapshot()['objects'])
        version = session.scene['scene_version']
        with pytest.raises(ValueError):
            edit(session, 'add', catalog_id='banana', xy=[99.0, 0.0], yaw=0)
        assert session.snapshot()['objects'] == before
        assert session.scene['scene_version'] == version
        with pytest.raises(ValueError, match='版本'):
            session.edit_scene(dict(scene_version=version - 1, action='clear'))
        assert session.snapshot()['objects'] == before


def test_mixed_fruit_layout_is_detected_and_banana_can_be_sorted(tmp_path):
    with SimulationSession(realtime=False, log_dir=tmp_path) as session:
        session.prepare()
        edit(session, 'clear')
        for label, xy, yaw in [('banana', [0.26, -0.12], -0.4),
                               ('pear', [0.26, 0.07], 0.6),
                               ('apple', [0.35, 0.05], 0.3)]:
            edit(session, 'add', catalog_id=label, xy=xy, yaw=yaw)
        scene = session.observe()
        visible = {o['label']: o for o in scene['objects'] if o['valid']}
        assert {'banana', 'pear', 'apple'} <= visible.keys(), scene['objects']
        # The web flow confirms the contact section from a local source view;
        # a broad scene detection alone is not a grasp-contact measurement.
        result = SkillExecutor(session, originals=scene['objects']).execute(
            dict(id='banana', skill='move', label='banana',
                 destination=dict(type='goal', index=0)),
            [dict(label='banana', zone_name='zone_yellow')])
        assert result['ok'], result
