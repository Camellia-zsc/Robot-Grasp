"""Human confirmation gates a locally checked recovery, never model text alone."""
import threading
import time
from types import SimpleNamespace

import pytest

from test_assistant_batch import RecoveringBatch, RecoveryClient, create


class ConfirmedSession(RecoveringBatch):
    def prepare_recovery(self, *args, **kwargs):
        context = super().prepare_recovery(*args, **kwargs)
        context['options'][0].update(requires_confirmation=True,
            question='是否暂放苹果并调整黄色方块？',
            steps=['安全暂放苹果', '调整黄色方块', '重新抓取苹果完成原任务'])
        return context

    def execute_recovery(self, chosen, cancel_event=None, confirmed=False):
        assert confirmed is True
        return super().execute_recovery(chosen, cancel_event)


def pending(controller):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = controller.snapshot()
        if state.get('recovery_approval', {}).get('status') == 'pending':
            return state['recovery_approval']
        threading.Event().wait(.01)
    pytest.fail('confirmation card was not offered')


@pytest.mark.parametrize('approve', [True, False])
def test_rearrangement_waits_for_explicit_answer(approve):
    ctl = create(ConfirmedSession, RecoveryClient())
    try:
        ctl.submit('依次抓苹果和梨')
        request = pending(ctl)
        assert not ctl._session.repairs
        assert ctl.snapshot()['status'] == 'awaiting_approval'
        ctl.approve_recovery(dict(request_id=request['request_id'],
            scene_version=request['scene_version'], approved=approve))
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert len(ctl._session.repairs) == int(approve)
        assert state['recovery_approval']['status'] == ('approved' if approve else 'declined')
        if not approve:
            assert len(ctl._session.executions) == 1
            assert not state['last_result']['ok']
    finally:
        ctl.close()


def test_stop_unblocks_confirmation_without_moving():
    ctl = create(ConfirmedSession, RecoveryClient())
    try:
        ctl.submit('依次抓苹果和梨')
        pending(ctl)
        ctl.stop()
        assert ctl.wait_idle(3)
        assert not ctl._session.repairs
        assert ctl.snapshot()['status'] == 'stopped'
    finally:
        ctl.close()


def test_confirmation_rejects_changed_scene_and_duplicate_answer():
    ctl = create(ConfirmedSession, RecoveryClient())
    try:
        ctl.submit('依次抓苹果和梨')
        request = pending(ctl)
        ctl._session.scene['scene_version'] += 1
        with pytest.raises(ValueError, match='变化|过期'):
            ctl.approve_recovery(dict(request_id=request['request_id'],
                scene_version=request['scene_version'], approved=True))
        assert ctl.wait_idle(3)
        assert not ctl._session.repairs
        with pytest.raises(ValueError):
            ctl.approve_recovery(dict(request_id=request['request_id'],
                scene_version=request['scene_version'], approved=True))
    finally:
        ctl.close()


def test_explicit_resume_preserves_goal_and_does_not_restart_cancelled_items():
    class Resumable(ConfirmedSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.runner = SimpleNamespace(has_held_recovery=True)
            self.cancel_event = threading.Event()
            self._recovery_goal = {'label': 'apple', 'zone_name': 'zone_red'}
    ctl = create(Resumable, RecoveryClient())
    try:
        ctl.submit('依次抓苹果和梨')
        request = pending(ctl)
        ctl.approve_recovery(dict(request_id=request['request_id'],
            scene_version=request['scene_version'], approved=False))
        assert ctl.wait_idle(3)
        assert ctl.snapshot()['can_resume_recovery']
        ctl.resume_recovery()
        second = pending(ctl)
        assert second['request_id'] != request['request_id']
        assert not ctl._session.repairs
        ctl.approve_recovery(dict(request_id=second['request_id'],
            scene_version=second['scene_version'], approved=True))
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert not state['can_resume_recovery']
        assert state['batch']['items'][0]['status'] == 'succeeded'
        assert state['batch']['items'][1]['status'] == 'cancelled'
        assert len(ctl._session.executions) == 1 and len(ctl._session.repairs) == 1
    finally:
        ctl.close()
