"""Recovery orchestration, cancellation and bounded paid-call behavior."""
import copy
import threading

import pytest

from cube_task.assistant.server import DashboardController
from test_assistant_server import Client, SCENE, Session, Settings


class RecoveringSession(Session):
    succeed_at = 1
    offer_options = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scene = copy.deepcopy(SCENE)
        self.repairs = []
        self.contexts = []

    def execute(self, decision, cancel_event=None):
        self.executions.append(decision)
        return {'ok': False, 'stage': 'release', 'error': 'holding_object',
                'holding': True, 'fault': True, 'message': '下降未完成'}

    def prepare_recovery(self, failure, goal, attempt, history):
        assert goal == {'label': 'red', 'zone_name': 'zone_red'}
        assert not self.stopped
        context = {'scene_version': attempt + 1, 'attempt': attempt,
                   'max_attempts': 3, 'goal': copy.deepcopy(goal), 'failure': failure,
                   'observations': [], 'history': copy.deepcopy(history),
                   'observation_mode': 'stationary_with_last_scan',
                   'options': [{'id': 'place_held_alternate', 'description': '保持夹持换空位'}]
                              if self.offer_options else []}
        self.contexts.append(context)
        return context

    def execute_recovery(self, chosen, cancel_event=None):
        assert not self.stopped and not cancel_event.is_set()
        self.repairs.append(chosen)
        ok = self.succeed_at is not None and len(self.repairs) >= self.succeed_at
        return {'ok': ok, 'holding': not ok, 'fault': not ok,
                'error': '' if ok else 'rejected', 'stage': 'done' if ok else 'release',
                'truth': {'success': ok}, 'message': '落位核验通过' if ok else '仍不能下降'}


class RecoveryClient(Client):
    def __init__(self, *, gate=None, action='recover', invented=False, error=False):
        super().__init__()
        self.recovery_gate = gate
        self.recovery_streaming = threading.Event()
        self.recovery_calls = []
        self.recovery_action = action
        self.invented = invented
        self.error = error

    def recover(self, instruction, context, on_delta, cancel_event=None):
        self.recovery_calls.append(copy.deepcopy(context))
        on_delta('{"message":"保持夹持并重新规划放置"}')
        self.recovery_streaming.set()
        if self.recovery_gate:
            self.recovery_gate.wait(3)
        if self.error:
            raise RuntimeError('模型连接失败')
        return {'scene_version': context['scene_version'], 'action': self.recovery_action,
                'option_id': ('invented' if self.invented else 'place_held_alternate')
                             if self.recovery_action == 'recover' else None,
                'message': '保持夹持并选择其他空位' if self.recovery_action == 'recover'
                           else '没有可继续执行的方案'}


def create(client=None, session_factory=RecoveringSession):
    ctl = DashboardController(session_factory=session_factory, settings=Settings(),
                              client=client or RecoveryClient())
    assert ctl.wait_idle(3)
    return ctl


def test_one_failed_attempt_is_repaired_and_counted_as_one_user_success():
    ctl = create()
    try:
        ctl.submit('抓红色方块')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'complete' and state['last_result']['ok']
        assert state['recovery']['status'] == 'succeeded'
        assert state['recovery']['attempt'] == 1
        assert not state['needs_reset']
        assert state['metrics'] == {'requests': 1, 'successes': 1, 'model_requests': 2}
        assert len(ctl._session.executions) == len(ctl._session.repairs) == 1
        reply = state['messages'][-1]
        assert reply['kind'] == 'recovery' and reply['attempt'] == 1
        assert reply['status'] == 'complete' and '保持夹持' in reply['content']
        assert state['last_result']['recovery_history'][0]['ok']
    finally:
        ctl.close()


def test_three_unsuccessful_repairs_stop_without_unbounded_paid_requests():
    class AlwaysFail(RecoveringSession):
        succeed_at = None

    ctl = create(session_factory=AlwaysFail)
    try:
        ctl.submit('抓红色方块')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'error' and state['needs_reset']
        assert state['recovery']['status'] == 'exhausted'
        assert len(ctl._session.repairs) == len(ctl._client.recovery_calls) == 3
        assert state['metrics'] == {'requests': 1, 'successes': 0, 'model_requests': 4}
        assert len(state['last_result']['recovery_history']) == 3
    finally:
        ctl.close()


@pytest.mark.parametrize('client', [RecoveryClient(action='stop'),
                                   RecoveryClient(invented=True),
                                   RecoveryClient(error=True)])
def test_stop_invalid_option_or_api_error_keeps_held_object_without_motion(client):
    ctl = create(client=client)
    try:
        ctl.submit('抓红色方块')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'error' and state['needs_reset']
        assert state['last_result']['holding']
        assert ctl._session.repairs == []
        assert len(client.recovery_calls) == 1
    finally:
        ctl.close()


def test_cancel_during_recovery_stream_preserves_stop_and_never_moves():
    gate = threading.Event()
    client = RecoveryClient(gate=gate)
    ctl = create(client=client)
    try:
        ctl.submit('抓红色方块')
        assert client.recovery_streaming.wait(2)
        assert ctl.snapshot()['messages'][-1]['status'] == 'streaming'
        ctl.stop()
        gate.set()
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['status'] == 'stopped' and state['needs_reset']
        assert state['recovery']['status'] == 'stopped'
        assert state['messages'][-1]['status'] == 'cancelled'
        assert ctl._session.repairs == []
    finally:
        gate.set()
        ctl.close()


def test_no_available_local_repair_does_not_call_model_or_move():
    class NoOptions(RecoveringSession):
        offer_options = False

    ctl = create(session_factory=NoOptions)
    try:
        ctl.submit('抓红色方块')
        assert ctl.wait_idle(3)
        assert not ctl._session.repairs and not ctl._client.recovery_calls
        assert ctl.snapshot()['needs_reset']
    finally:
        ctl.close()


def test_recovery_exception_with_stopped_session_requires_reset_even_if_initially_empty():
    class StoppedDuringRetry(RecoveringSession):
        def execute(self, decision, cancel_event=None):
            return {'ok': False, 'holding': False, 'stage': 'grasp', 'error': 'grasp_miss'}

        def execute_recovery(self, decision, cancel_event=None):
            self.stopped = True
            raise RuntimeError('重试动作发生异常')

        def snapshot(self):
            return {'status': 'stopped' if self.stopped else 'ready', 'held_object': None}

    ctl = create(session_factory=StoppedDuringRetry)
    try:
        ctl.submit('抓红色方块')
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert state['needs_reset'] and state['last_result']['fault']
        assert not state['last_result']['ok'] and not state['last_result']['holding']
        with pytest.raises(ValueError):
            ctl.submit('再次抓取')
    finally:
        ctl.close()


def test_no_progress_planning_refusal_is_not_repeated_three_times():
    class PlanRefusal(RecoveringSession):
        def execute_recovery(self, chosen, cancel_event=None):
            self.repairs.append(chosen)
            return {'ok': False, 'holding': True, 'stage': 'plan',
                    'error': 'no alternative', 'message': '没有新的可行放置点'}

    ctl = create(session_factory=PlanRefusal)
    try:
        ctl.submit('抓红色方块')
        assert ctl.wait_idle(3)
        assert len(ctl._client.recovery_calls) == len(ctl._session.repairs) == 1
        assert '没有找到新的可执行动作' in ctl.snapshot()['stage']
        assert ctl.snapshot()['needs_reset']
    finally:
        ctl.close()
