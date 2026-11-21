"""A refined carried reference is not an executable source put-back pose."""
import pytest

from cube_task.core.types import PickResult
from test_task_skills import _setup


@pytest.mark.parametrize('from_transit', [False, True])
def test_calibrated_source_reference_never_drives_legacy_automatic_putback(from_transit):
    _cfg, _clock, _world, arm, _log, skill, grasp, _place = _setup()
    grasp.target.extra['_executed_grasp_T'] = grasp.T_grasp.copy()
    grasp.target.extra['_held_translation_anchor'] = {'translation_only': True}
    grasp.T_grasp[0, 3] += .008
    result = PickResult(False, error='timeout', grasp_width=.03,
                        detail={'holding': True, 'fault': False, 'steps': []})
    skill._put_back(result, grasp, from_transit)
    assert result.detail['holding'] and not result.detail['fault']
    assert result.error == 'timeout'
    assert not arm.commands
    assert 'calibrated' in result.detail['recovery']


def test_unchanged_source_reference_retains_existing_putback_behavior():
    _cfg, _clock, _world, arm, _log, skill, grasp, _place = _setup()
    grasp.target.extra['_executed_grasp_T'] = grasp.T_grasp.copy()
    grasp.target.extra['_held_translation_anchor'] = {'translation_only': True}
    result = PickResult(False, error='timeout', grasp_width=.03,
                        detail={'holding': True, 'fault': False, 'steps': []})
    skill._put_back(result, grasp, False)
    assert not result.detail['holding'] and not result.detail['fault']
    assert arm.commands and result.detail['recovery'] == 'put back'
