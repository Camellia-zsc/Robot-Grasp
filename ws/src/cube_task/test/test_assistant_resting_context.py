"""Measured resting poses and actual righting capability reach the planner."""
from cube_task.assistant.skill_protocol import model_skill_context


def test_planner_receives_side_pear_and_unavailable_automatic_righting():
    context = {
        'scene': {'scene_version': 1, 'objects': [
            {'object_id': 'pear', 'label': 'pear', 'valid': True,
             'resting_pose': 'side', 'center_base': [.3, 0., .08]}],
            'zones': [], 'class_to_zone': {}},
        'goals': [], 'originals': [], 'holding': False,
        'capabilities': {'automatic_righting': False},
    }
    visible = model_skill_context(context)
    assert visible['scene']['objects'][0]['resting_pose'] == 'side'
    assert visible['capabilities']['automatic_righting'] is False
