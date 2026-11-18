"""A stopped camera motion can yield an image; it can never authorize release."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.core.types import PickResult, Stamped
from cube_task.task.skills import _Lost, _StepFailed


def stopped_inspection(*, physical=True, error='timeout', fault=False):
    skill = SimpleNamespace(_move=Mock(side_effect=_StepFailed('inspect_place', error, 'stopped', fault)),
        _check_abort=Mock(), _healthy=lambda: True, _verify_held=Mock(), feedback_max_age=1.)
    session = SimpleNamespace(_physics=object() if physical else None,
        runner=SimpleNamespace(skill=skill), clock=SimpleNamespace(now=lambda: 3.),
        arm=SimpleNamespace(get_tcp_pose=lambda: Stamped(np.eye(4), 3.)), _emit=Mock())
    return session, skill


def test_stopped_physical_inspection_reobserves_actual_pose_with_fresh_held_feedback():
    session, skill = stopped_inspection()
    reached = SimulationSession._move_inspection_waypoint(session, PickResult(False), np.eye(4))
    assert not reached
    skill._verify_held.assert_called_once()
    assert session._emit.call_args.args[0] == 'inspection_motion_incomplete'
    assert session._emit.call_args.args[1]['actual_tcp'] == [0., 0., 0.]


@pytest.mark.parametrize('physical,error,fault', [(False, 'timeout', False),
    (True, 'comm', True), (True, 'aborted', True), (True, 'arm_error', True)])
def test_failed_inspection_never_swallows_fault_or_unsupported_driver_semantics(physical, error, fault):
    session, _skill = stopped_inspection(physical=physical, error=error, fault=fault)
    with pytest.raises(_StepFailed):
        SimulationSession._move_inspection_waypoint(session, PickResult(False), np.eye(4))


def test_stopped_inspection_cannot_continue_after_a_dropped_object():
    session, skill = stopped_inspection()
    skill._verify_held.side_effect = _Lost('dropped_in_transit', 0.)
    with pytest.raises(_Lost):
        SimulationSession._move_inspection_waypoint(session, PickResult(False), np.eye(4))
