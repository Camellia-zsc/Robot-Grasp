"""Stopping a clearance transaction must retain its outstanding restoration."""
import copy
import threading

import pytest

from test_assistant_batch import BatchClient, BatchSession, create


PENDING_RESTORE = {
    'label': 'yellow', 'destination': {'type': 'point', 'xy': [.54, -.05]},
    'yaw': .3, 'message': '临时挪开的物品尚未确认回到原位置和朝向',
}


class ClearanceSession(BatchSession):
    fail_at = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clearance_started = threading.Event()
        self.clearance_return = threading.Event()
        self.recovery_calls = 0

    def prepare_recovery(self, failure, goal, attempt, history, reference_xy=None):
        return dict(scene_version=self.scene['scene_version'], attempt=attempt,
                    max_attempts=3, goal=goal, failure=failure, history=history,
                    observations=[], observation_mode='target_search',
                    options=[{'id': 'clear_restore_yellow', 'description': '移障后复位'}])

    def execute_recovery(self, chosen, cancel_event=None):
        self.recovery_calls += 1
        self.clearance_started.set()  # The blocker has already left its origin.
        assert self.clearance_return.wait(3)
        assert self.stopped and cancel_event.is_set()
        return dict(ok=False, aborted=True, holding=False, stage='obstacle_restore',
                    error='pending_restore', pending_restore=copy.deepcopy(PENDING_RESTORE),
                    clearance_steps=[{'step': 'move_obstacle', 'result': {'ok': True}}],
                    message='任务已停止，临时移开的物品尚未复位')


class ClearanceClient(BatchClient):
    def __init__(self, batch):
        super().__init__()
        self.batch = batch

    def decide(self, instruction, observed, on_delta, cancel_event=None):
        decision = super().decide(instruction, observed, on_delta, cancel_event)
        if not self.batch:
            decision.update(action='pick', object_id=observed['objects'][0]['object_id'],
                            zone_name='zone_red')
            decision.pop('items')
        return decision

    def recover(self, instruction, context, on_delta, cancel_event=None):
        return dict(scene_version=context['scene_version'], action='recover',
                    option_id='clear_restore_yellow', message='移障后完成原任务并复位')


@pytest.mark.parametrize('batch', [False, True], ids=['single', 'batch'])
def test_stop_after_obstacle_move_preserves_restore_and_prevents_more_motion(batch):
    ctl = create(ClearanceSession, ClearanceClient(batch))
    try:
        ctl.submit('移动苹果和梨' if batch else '移动苹果')
        assert ctl._session.clearance_started.wait(2)
        ctl.stop()
        ctl._session.clearance_return.set()
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        result = state['last_result']
        assert state['status'] == 'stopped' and state['needs_reset']
        assert result['pending_restore'] == PENDING_RESTORE
        assert result['clearance_steps'][0]['step'] == 'move_obstacle'
        assert result['recovery_history'][0]['option_id'] == 'clear_restore_yellow'
        assert state['recovery']['status'] == 'stopped'
        assert state['recovery']['history'] == result['recovery_history']
        assert '黄色方块' in state['stage'] and '尚未复位' in state['stage']
        assert '尚未复位' in state['recovery']['message']
        assert any(event['kind'] == 'stopped' and '尚未复位' in event['message']
                   for event in state['events'])
        assert len(ctl._session.executions) == 1 and ctl._session.recovery_calls == 1
        if batch:
            assert state['batch']['status'] == 'stopped'
            assert all(item['status'] == 'cancelled' for item in state['batch']['items'])
    finally:
        ctl._session.clearance_return.set()
        ctl.close()
