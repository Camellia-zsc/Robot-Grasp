"""A supported source may seat in the grip even with a flat-region destination."""
import pytest

from cube_task.assistant.held_anchor import seating_requires_feedback
from cube_task.core.types import PickResult
from test_held_spherical_anchor import measured_grasp


def supported_source():
    grasp = measured_grasp()
    grasp.target.extra.pop('_stack_support')
    grasp.target.extra.update(support_observed=True, support_label='yellow', support_z=.047)
    result = PickResult(False, detail={'steps': [dict(stage='close', ok=True, width=.048)]})
    return grasp, result


def test_large_seating_of_a_measured_supported_source_requires_rgbd_feedback():
    grasp, result = supported_source()
    assert seating_requires_feedback(grasp, result, .038)
    assert not seating_requires_feedback(grasp, result, .0475)
    assert '_stack_support' not in grasp.target.extra


@pytest.mark.parametrize('evidence', [dict(support_observed=False), dict(support_label=None),
                                        dict(support_z=None), dict(support_z=float('nan'))])
def test_unconfirmed_support_does_not_bypass_a_measured_large_seating_change(evidence):
    grasp, result = supported_source()
    grasp.target.extra.update(evidence)
    assert seating_requires_feedback(grasp, result, .038)


@pytest.mark.parametrize('pose,label', [('side', 'pear'), ('flat', 'banana'), ('upright', 'red')])
def test_supported_source_without_a_known_spherical_section_keeps_existing_behavior(pose, label):
    grasp, result = supported_source()
    grasp.target.label = label
    grasp.target.extra['resting_pose'] = pose
    assert not seating_requires_feedback(grasp, result, .038)


def test_plain_table_source_only_requests_calibration_after_material_seating():
    grasp, result = supported_source()
    for key in ('support_observed', 'support_label', 'support_z'):
        grasp.target.extra.pop(key)
    assert seating_requires_feedback(grasp, result, .038)
    assert not seating_requires_feedback(grasp, result, .0475)
