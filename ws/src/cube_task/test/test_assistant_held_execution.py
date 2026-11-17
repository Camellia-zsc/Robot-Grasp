"""Replay the actual held banana through human-gated recovery, without an API."""
import copy
import json
import socket
import threading
import time

import numpy as np

from cube_task.assistant.server import DashboardController
from held_scene_fixture import reconstruct_session


class OfflineSettings:
    def public_info(self):
        return dict(configured=True, provider='offline', model='recorded-intent', api_style='responses')


class RearrangementClient:
    def __init__(self):
        self.calls = 0

    def recover(self, instruction, context, on_delta, cancel_event=None):
        self.calls += 1
        assert any(o['id'] == 'stage_rearrange_yellow' for o in context['options'])
        decision = dict(scene_version=context['scene_version'], action='recover',
                        option_id='stage_rearrange_yellow', message='先安全暂放香蕉，再经确认调整黄块并完成原任务')
        on_delta(json.dumps(decision, ensure_ascii=False))
        return decision


def test_recorded_held_banana_stages_rearranges_and_finishes_only_after_confirmation(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('independent execution must not call any network/API')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    def factory(**kwargs):
        session = reconstruct_session(log_dir=kwargs['log_dir'])
        session.on_event = kwargs.get('on_event')
        session.prepare = lambda: session.scene  # Restoring a held session must not open or home.
        return session
    client = RearrangementClient()
    ctl = DashboardController(realtime=False, session_factory=factory, client=client,
                              settings=OfflineSettings(), log_dir=tmp_path)
    try:
        assert ctl.wait_idle(30)
        session = ctl._session
        initial = session.snapshot()
        untouched = {o.name: o.T_base_obj.copy() for o in session.world.objects
                     if o.label not in ('banana', 'yellow')}
        yellow = next(o for o in session.world.objects if o.label == 'yellow')
        yellow_origin = yellow.center_xy().copy()
        ctl._last_result = dict(ok=False, holding=True, stage='inspect_lift', error='rejected',
            recovery_history=[dict(option_id='place_held_alternate', ok=False)] * 2)
        done = [('apple', 'zone_red'), ('red', 'zone_red'), ('green', 'zone_green'), ('pear', 'zone_green')]
        ctl._batch = dict(status='failed', total=5, completed=4, current_index=5, message='4/5',
                         items=[dict(index=i+1, label=label, zone_name=zone, status='skipped')
                                for i, (label, zone) in enumerate(done)] +
                               [dict(index=5, label='banana', zone_name='zone_yellow', status='failed')])
        ctl.resume_recovery()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = ctl.snapshot()
            if state['recovery_approval']['status'] == 'pending':
                break
            threading.Event().wait(.02)
        assert state['status'] == 'awaiting_approval', state
        # Until a real button answer, only stationary sensing is allowed.
        np.testing.assert_allclose(session.arm.get_joints().value, initial['joints'], atol=1e-10)
        for old, new in zip(initial['objects'], session.snapshot()['objects']):
            np.testing.assert_allclose(old['transform'], new['transform'], atol=1e-10)
        approval = state['recovery_approval']
        ctl.approve_recovery(dict(request_id=approval['request_id'],
            scene_version=approval['scene_version'], approved=True))
        assert ctl.wait_idle(240), 'held transaction timed out'
        state = ctl.snapshot()
        (tmp_path/'held_rearrangement.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        assert state['last_result']['ok'], state['last_result']
        assert state['batch']['completed'] == state['batch']['total'] == 5
        assert state['batch']['status'] == 'complete'
        assert client.calls == 1
        assert session.world.held is None
        assert session.world.zone_of(yellow).name == 'zone_yellow'
        assert np.linalg.norm(yellow.center_xy() - yellow_origin) > .025
        banana = next(o for o in session.world.objects if o.label == 'banana')
        assert session.world.zone_of(banana).name == 'zone_yellow'
        for obj in session.world.objects:
            if obj.name in untouched:
                np.testing.assert_array_equal(obj.T_base_obj, untouched[obj.name])
    finally:
        ctl.close()
