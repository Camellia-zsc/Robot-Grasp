"""A support image must not discard the already measured upper-body grasp."""
import copy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor, _PlanChanged
from test_assistant_targeted_scan import block, scan_session


def stack_scan(*, support_valid=True):
    pear = block('pear', xy=(.51, -.01), top=.112, height=.112)
    pear.extra.update(_cloud_views=2, _observed_revision=1,
                      _placement_points_base=np.zeros((40, 3)))
    yellow = block('yellow', xy=(.337, -.164), top=.048, height=.048)
    cropped = copy.deepcopy(yellow)
    cropped.valid, cropped.reject_reason = False, 'touches_image_border'
    support = yellow if support_valid else cropped
    banana = block('banana', xy=(.17, -.32), top=.04, height=.04)
    session, seen = scan_session([[pear, cropped], [banana], [support], []])
    capture = session.runner._observe
    def observe(view):
        status, result, frame = capture(view)
        frame.depth = None
        return status, result, frame
    session.runner._observe = observe
    session._observed_detections = [copy.deepcopy(pear), copy.deepcopy(yellow)]
    session._scene['objects'] = [d.summary() for d in session._observed_detections]
    session.runner.planner = SimpleNamespace(plan=lambda det: SimpleNamespace(feasible=True))
    session._stable_cloud = MethodType(SimulationSession._stable_cloud, session)
    # The old first scan covered the upper body. The second required entity is
    # outside it and has a known best view; the banana view adds no evidence.
    session._view_order = lambda target: ([2, 0, 1, 3] if target.get('label') == 'yellow'
                                          else [0, 1, 2, 3])
    session._reusable_target = lambda reference: False
    return session, seen, pear, yellow


def test_fresh_target_callsite_keeps_upper_geometry_while_seeking_only_its_support():
    session, seen, _pear, _yellow = stack_scan()
    det, observed = SkillExecutor(session)._fresh_target(
        'pear', 'recover_queued_target', support_label='yellow')
    assert det.label == observed['label'] == 'pear'
    assert seen == [0, 2]
    assert {d.label for d in session._last_fresh if d.valid} == {'pear', 'yellow'}
    assert session._scan_stats['frames'] == 2


def test_missing_support_does_not_turn_cached_upper_geometry_into_joint_success():
    session, seen, _pear, _yellow = stack_scan(support_valid=False)
    with pytest.raises(ValueError, match='同时完整确认'):
        SkillExecutor(session)._fresh_target('pear', 'recover_queued_target', support_label='yellow')
    assert set(seen) == {0, 1, 2, 3}
    assert len(seen) == 4


@pytest.mark.parametrize('change', ['revision', 'intervention'])
def test_change_while_seeking_support_invalidates_the_retained_upper_measurement(change):
    session, seen, _pear, _yellow = stack_scan()
    observe = session.runner._observe
    def capture(view):
        result = observe(view)
        if view == 2:
            if change == 'revision':
                session._environment_revision += 1
            else:
                session._interventions.append({'action': 'move'})
        return result
    session.runner._observe = capture
    with pytest.raises(_PlanChanged if change == 'revision' else ValueError, match='新相机|同时完整确认'):
        SkillExecutor(session)._fresh_target('pear', 'recover_queued_target', support_label='yellow')
    assert seen == [0, 2]


def test_same_batch_target_geometry_does_not_need_to_appear_in_the_support_frame():
    session, seen, pear, yellow = stack_scan()
    session._view_order = lambda target: [0, 2, 1, 3]
    session.observe(target=pear.summary() | dict(require_grasp=True,
                    required_support=yellow.summary()), purpose='recover_queued_target')
    assert seen == [0, 2]
    assert session._support_observed({'required_support': yellow.summary()}, session._last_fresh, 1)
