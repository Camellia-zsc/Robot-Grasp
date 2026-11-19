"""Stopping relinquishes the worker before rearming this same scene."""
import copy
import threading

import pytest

from cube_task.assistant.server import BusyError, DashboardController
from test_assistant_server import Client, Settings
from test_assistant_scene_server import EditableSession


class ResumableSession(EditableSession):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cancel_event = threading.Event()
        self.resume_calls = 0
        self.old_operation_active = False

    def stop(self):
        super().stop()
        self.cancel_event.set()

    def resume_after_stop(self):
        assert not self.old_operation_active, 'old worker must finish before rearming'
        self.resume_calls += 1
        self.stopped = False
        self.cancel_event.clear()
        return True


def make_controller(client):
    ctl = DashboardController(session_factory=ResumableSession, settings=Settings(), client=client)
    assert ctl.wait_idle(3)
    return ctl


def test_stopped_worker_releases_same_scene_for_edit_and_new_command():
    gate = threading.Event()
    client = Client(gate=gate)
    ctl = make_controller(client)
    try:
        session = ctl._session
        ctl.submit('抓红色方块')
        assert client.streaming.wait(2)
        session.old_operation_active = True
        ctl.stop()
        assert ctl.snapshot()['busy'] and not ctl.snapshot()['simulation']['editor']['editable']
        with pytest.raises((BusyError, ValueError)):
            ctl.edit_scene(dict(action='clear', scene_version=1))
        assert session.resume_calls == 0
        session.old_operation_active = False
        gate.set()
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'stopped'
        assert not state['needs_reset'], 'normal stop must not require destructive scene reset'
        assert state['simulation']['editor']['editable']
        assert not ctl._cancel.is_set() and not session.cancel_event.is_set()
        assert not session.executions and ctl._session is session
        ctl.edit_scene(dict(action='clear', scene_version=1))
        assert ctl.wait_idle(3)
        ctl._scene = copy.deepcopy(session.scene)
        # Restore only this test's observed target after testing editing.
        session.scene['objects'] = [dict(object_id='obj_1', label='red', valid=True)]
        ctl._scene = copy.deepcopy(session.scene)
        ctl.submit('重新观察')
        assert ctl.wait_idle(3)
        assert len(state['messages']) >= 2
    finally:
        gate.set()
        ctl.close()


def test_stop_preserves_held_object_and_restore_debt_and_enables_explicit_resume():
    gate = threading.Event()
    client = Client(gate=gate)
    client.plan_skills = lambda *args, **kwargs: None
    ctl = make_controller(client)
    try:
        ctl._session.held_object = 'pear'
        ctl._session.runner = type('Runner', (), {'has_held_recovery': True})()
        ctl._skill_memory = dict(goals=[dict(label='banana', zone_name='zone_red')],
            protected_goals=[], obligations={'pear': {'status': 'pending'}}, instruction='原任务')
        ctl._skill_task['status'] = 'stopped'
        ctl._last_result = dict(ok=False, holding=True, pending_obligations={'pear': 'pending'})
        ctl._launch(lambda: gate.wait(3), 'executing', '测试持物')
        ctl.stop()
        gate.set()
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert not state['needs_reset']
        assert state['simulation']['held_object'] == 'pear'
        assert not state['simulation']['editor']['editable']
        assert state['can_resume_recovery']
        assert state['last_result']['pending_obligations'] == {'pear': 'pending'}
        assert ctl._skill_memory['obligations']['pear']['status'] == 'pending'
    finally:
        gate.set()
        ctl.close()


def test_real_session_rearm_holds_pose_and_invalidates_old_observation(tmp_path):
    from cube_task.assistant.simulation import SimulationSession
    import numpy as np
    with SimulationSession(realtime=False, log_dir=tmp_path) as session:
        session.prepare()
        before = session.snapshot()
        calls = session.arm.inner.n_calls
        revision = session._environment_revision
        session.stop()
        assert session.snapshot()['status'] == 'stopped'
        assert session.resume_after_stop()
        after = session.snapshot()
        assert after['editor']['editable']
        assert after['editor']['intervention_available']
        assert not session.cancel_event.is_set()
        assert session._environment_revision > revision
        assert session.arm.inner.n_calls == calls
        np.testing.assert_allclose(after['joints'], before['joints'])
        np.testing.assert_allclose(after['tcp_xyz'], before['tcp_xyz'])
        session.arm.inner.emergency_stop()
        session.stop()
        assert not session.resume_after_stop()
        assert session.cancel_event.is_set()


def test_stop_during_real_model_network_wait_releases_controller_without_late_writes(monkeypatch):
    import json
    from cube_task.assistant import llm
    from test_assistant_llm import FakeStream, completed
    entered, release = threading.Event(), threading.Event()

    def blocked_open(*args):
        entered.set()
        release.wait(3)
        return FakeStream([completed('{"late":"must be discarded"}')])

    monkeypatch.setattr(llm, '_open', blocked_open)
    settings = llm.LLMSettings(base_url='https://relay.invalid/v1', model='test', api_key='fake')
    from types import SimpleNamespace
    model = llm.ModelClient(settings)
    ctl = DashboardController(session_factory=ResumableSession, settings=settings,
                              client=SimpleNamespace(decide=model.decide, last_usage={}))
    assert ctl.wait_idle(3)
    try:
        session = ctl._session
        ctl.submit('旧请求')
        assert entered.wait(2)
        ctl.stop()
        assert ctl.wait_idle(.5), 'network wait must relinquish the sole robot worker'
        assert not ctl.snapshot()['needs_reset']
        answer = dict(scene_version=1, action='answer', object_id=None, zone_name=None,
                      message='新请求已完成')
        monkeypatch.setattr(llm, '_open', lambda *args: FakeStream([completed(json.dumps(answer))]))
        ctl.submit('新请求')
        assert ctl.wait_idle(2)
        before = ctl.snapshot()
        release.set()
        # Wait for the old network producer; it owns no callbacks or scene state.
        import time
        time.sleep(.1)
        after = ctl.snapshot()
        assert before['messages'] == after['messages']
        assert before['status'] == after['status'] == 'ready'
        assert before['stage'] == after['stage'] == '新请求已完成'
        assert not session.executions and ctl._session is session
    finally:
        release.set()
        ctl.close()


@pytest.mark.parametrize('invalid', ['health', 'joints_missing', 'joints_value', 'joints_nan',
    'tcp_shape', 'tcp_nan', 'tcp_stale', 'tcp_future', 'tcp_nan_stamp', 'width_missing'])
def test_rearm_refuses_unhealthy_or_invalid_feedback(invalid):
    import numpy as np
    from types import SimpleNamespace as NS
    from cube_task.assistant.simulation import SimulationSession
    session = SimulationSession.__new__(SimulationSession)
    session._operation_lock = threading.RLock()
    session._closed, session._prepared = False, True
    session.clock = NS(now=lambda: 10.)
    session.runner = NS(skill=NS(feedback_max_age=.5))
    health = NS(ok=True, enabled=True)
    joints, tcp, width = NS(value=np.zeros(6), stamp=10.), NS(value=np.eye(4), stamp=10.), NS(value=.04, stamp=10.)
    if invalid == 'health': health = None
    elif invalid == 'joints_missing': joints = None
    elif invalid == 'joints_value': joints.value = None
    elif invalid == 'joints_nan': joints.value[0] = np.nan
    elif invalid == 'tcp_shape': tcp.value = np.zeros(3)
    elif invalid == 'tcp_nan': tcp.value[0, 0] = np.nan
    elif invalid == 'tcp_stale': tcp.stamp = 9.
    elif invalid == 'tcp_future': tcp.stamp = 11.
    elif invalid == 'tcp_nan_stamp': tcp.stamp = np.nan
    elif invalid == 'width_missing': width = None
    session.arm = NS(health=lambda: health, get_joints=lambda: joints,
                     get_tcp_pose=lambda: tcp, gripper_width=lambda: width)
    assert not session.resume_after_stop()


def test_stop_does_not_clear_a_preexisting_real_fault():
    gate = threading.Event()
    ctl = make_controller(Client())
    try:
        ctl._launch(lambda: gate.wait(3), 'executing', '故障等待')
        ctl._last_result = dict(ok=False, fault=True, error='collision', aborted=False)
        ctl.stop()
        gate.set()
        assert ctl.wait_idle(2)
        assert ctl.snapshot()['needs_reset']
        assert ctl._session.resume_calls == 0
        assert ctl._cancel.is_set()
    finally:
        gate.set()
        ctl.close()


def test_repeated_stop_during_rearm_is_idempotent():
    operation, rearmed, finish = threading.Event(), threading.Event(), threading.Event()

    class SlowRearm(ResumableSession):
        def resume_after_stop(self):
            result = super().resume_after_stop()
            rearmed.set()
            finish.wait(3)
            return result

    ctl = DashboardController(session_factory=SlowRearm, settings=Settings(), client=Client())
    assert ctl.wait_idle(2)
    try:
        ctl._launch(lambda: operation.wait(3), 'executing', '执行')
        ctl.stop()
        operation.set()
        assert rearmed.wait(2)
        assert ctl.snapshot()['busy']
        ctl.stop()
        finish.set()
        assert ctl.wait_idle(2)
        state = ctl.snapshot()
        assert not state['needs_reset'] and state['simulation']['editor']['editable']
        assert not ctl._session.cancel_event.is_set()
    finally:
        operation.set()
        finish.set()
        ctl.close()


def test_rearm_waits_for_first_physical_stop_to_finish():
    operation, stop_entered, stop_finish = threading.Event(), threading.Event(), threading.Event()

    class SlowStop(ResumableSession):
        def stop(self):
            stop_entered.set()
            stop_finish.wait(3)
            super().stop()

    ctl = DashboardController(session_factory=SlowStop, settings=Settings(), client=Client())
    assert ctl.wait_idle(2)
    stopper = threading.Thread(target=ctl.stop)
    try:
        ctl._launch(lambda: operation.wait(3), 'executing', '执行')
        stopper.start()
        assert stop_entered.wait(2)
        operation.set()
        assert not ctl.wait_idle(.1)
        assert ctl._session.resume_calls == 0
        stop_finish.set()
        stopper.join(2)
        assert ctl.wait_idle(2)
        state = ctl.snapshot()
        assert not state['needs_reset'] and state['simulation']['editor']['editable']
        assert not ctl._session.cancel_event.is_set()
    finally:
        operation.set()
        stop_finish.set()
        stopper.join(3)
        ctl.close()

