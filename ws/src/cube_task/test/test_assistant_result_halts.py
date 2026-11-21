"""A verified physical release cannot conceal a later controller halt."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.core.types import Detection, PickResult, Zone
from test_assistant_batch import BatchSession, create, scene


def reporting_session(*, fault=False, aborted=False):
    session = object.__new__(SimulationSession)
    session.cancel_event = threading.Event()
    if aborted:
        session.cancel_event.set()
    session._physics = None
    session._scene = scene()
    zone = Zone('zone_red', np.array([.17, -.35, 0.]), (.2, .2), label='red')
    session._environment_revision = session._observation_revision = 4
    session.world = SimpleNamespace(held=None, score=lambda *_: {}, zone_of=lambda _: zone)
    session.runner = SimpleNamespace(class_to_zone={'apple': zone.name}, zones={zone.name: zone})
    session._verify_zone = lambda *_: True
    target = Detection('apple', 1., center_base=np.array([.17, -.35, .06]),
                       top_z=.06, height=.06, length=.06, width=.06, yaw=0.)
    session._last_fresh = [target]
    session._append_check = Mock()
    session.observe = Mock()
    session._emit = Mock()
    session._publish_snapshot = Mock()
    session.logger = Mock()
    motion = PickResult(True, stage='retreat', label='apple', zone=zone.name,
                        error='driver_error' if fault else 'aborted' if aborted else '',
                        detail={'fault': fault, 'holding': False})
    return session, target, motion


def finish(session, target, motion):
    return session._finish_attempt(motion, '', 'apple-id', 'zone_red', target,
                                   SimpleNamespace(name='apple'))


@pytest.mark.parametrize('fault,aborted', [(False, False), (True, False),
                                          (False, True), (True, True)])
def test_verified_release_keeps_controller_detail_but_public_success_requires_no_halt(fault, aborted):
    session, target, motion = reporting_session(fault=fault, aborted=aborted)
    result = finish(session, target, motion)

    assert result['controller_ok'] and result['truth']['success'] and result['camera_verified']
    assert result['fault'] is fault and result['aborted'] is aborted
    assert result['ok'] is (not fault and not aborted)
    assert not result['holding']
    assert result['error'] == motion.error
    assert session.observe.call_count == (0 if fault or aborted else 1)
    if fault and not aborted:
        assert '异常' in result['message'] and '停止' in result['message']


def test_legacy_batch_stops_after_real_finish_summary_reports_a_release_then_driver_fault():
    class ReleaseThenFault(BatchSession):
        def execute(self, decision, cancel_event=None):
            result = super().execute(decision, cancel_event)
            if len(self.executions) == 1:
                reporter, target, motion = reporting_session(fault=True)
                result = finish(reporter, target, motion)
            return result

    ctl = create(ReleaseThenFault)
    try:
        ctl.submit('依次抓苹果和梨')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert len(ctl._session.executions) == 1
        assert [item['status'] for item in state['batch']['items']] == ['failed', 'cancelled']
        assert state['batch']['completed'] == 0 and state['metrics']['successes'] == 0
        assert not state['last_result']['ok'] and state['last_result']['fault']
        assert state['last_result']['controller_ok'] and state['last_result']['truth']['success']
        assert state['needs_reset']
    finally:
        ctl.close()
