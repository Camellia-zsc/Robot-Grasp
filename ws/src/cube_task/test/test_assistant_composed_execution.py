"""Recorded held incident through the composable controller, without networking.

The scripted offline planner tests the execution contract, not the judgement of
an online model. Every motion uses the real camera, geometry, IK and task layer.
"""
import copy
import json
import socket
import time

import numpy as np

from cube_task.assistant.server import DashboardController
from held_scene_fixture import reconstruct_session
from test_assistant_server import Settings
from test_assistant_skill_protocol import step


class Planner:
    def __init__(self):
        self.calls = []

    def plan_skills(self, instruction, context, on_delta, cancel_event=None):
        self.calls.append(copy.deepcopy(context))
        history = context.get('history', [])
        def succeeded(skill, label, destination_type=None):
            return any(item.get('ok') and item.get('skill') == skill
                       and item.get('label') == label
                       and (destination_type is None or
                            (item.get('destination') or {}).get('type') == destination_type)
                       for item in history)
        # This offline planner sees the same semantic feedback as a model. A
        # reachable release can lack a reachable wrist-camera view, so after
        # that refusal choose the configured observed region instead of asking
        # for another arbitrary free point until the replan budget runs out.
        unobserved_point = any(item.get('label') == 'yellow'
                               and item.get('error') == 'destination_not_observed'
                               for item in history)
        temporary = ({'type': 'region', 'name': 'pick_area'} if unobserved_point else
                     {'type': 'auto_free', 'away_from': ['banana']})
        if context['holding']:
            if context['held_label'] == 'banana' and succeeded('pick', 'banana'):
                steps = [step(1, 'place', 'banana', {'type': 'goal', 'index': 0})]
            elif context['held_label'] == 'yellow':
                destination = ({'type': 'zone', 'name': 'zone_yellow'}
                               if succeeded('place', 'banana', 'goal') else temporary)
                steps = [step(1, 'place', 'yellow', destination)]
            else:
                steps = [step(1, 'place', 'banana', {'type': 'auto_free'}),
                         step(2, 'observe', 'yellow')]
        else:
            steps = []
            if not succeeded('place', 'banana', 'goal'):
                if not (succeeded('move', 'yellow', 'auto_free') or
                        succeeded('move', 'yellow', 'region') or
                        succeeded('place', 'yellow', 'region')):
                    steps.append(step(1, 'move', 'yellow', temporary))
                steps += [step(len(steps)+1, 'pick', 'banana'),
                          step(len(steps)+2, 'place', 'banana', {'type': 'goal', 'index': 0})]
            if not succeeded('move', 'yellow', 'zone'):
                steps.append(step(len(steps)+1, 'move', 'yellow', {'type': 'zone', 'name': 'zone_yellow'}))
            steps += [step(len(steps)+1, 'verify'), step(len(steps)+2, 'finish')]
        on_delta('{"message":"先暂放，再根据新观测组合搬移与重抓步骤"}')
        return dict(scene_version=context['scene']['scene_version'], action='plan',
                    message='先暂放，再根据新观测组合搬移与重抓步骤', goals=context['goals'], steps=steps)


def test_recorded_held_scene_uses_composed_skills_and_live_feedback(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('independent execution must not call network or a model API')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)

    def factory(**kwargs):
        s = reconstruct_session(log_dir=kwargs['log_dir'])
        s.on_event = kwargs.get('on_event')
        s.prepare = lambda: s.scene
        return s

    planner = Planner()
    ctl = DashboardController(realtime=False, session_factory=factory, client=planner,
                              settings=Settings(), log_dir=tmp_path)
    try:
        assert ctl.wait_idle(30)
        s = ctl._session
        original = s.snapshot()
        untouched = {o.name: o.T_base_obj.copy() for o in s.world.objects
                     if o.label not in ('banana', 'yellow')}
        ctl._batch = dict(status='failed', total=5, completed=4, current_index=5,
            message='4/5', items=[dict(index=i+1, label=label, zone_name=zone, status='skipped')
            for i, (label, zone) in enumerate([('apple', 'zone_red'), ('red', 'zone_red'),
                                            ('green', 'zone_green'), ('pear', 'zone_green')])]
            + [dict(index=5, label='banana', zone_name='zone_yellow', status='failed')])
        ctl._last_result = dict(ok=False, holding=True, stage='inspect_lift', message='记录现场的失败')
        ctl.resume_recovery()
        # Full camera/IK validation includes four physical transfers and all
        # protected goals; allow it to finish on a shared CPU.
        answered, deadline = set(), time.monotonic()+720
        approval_count = 0
        while time.monotonic() < deadline:
            state = ctl.snapshot()
            approval = state['recovery_approval']
            if approval.get('status') == 'pending' and approval['request_id'] not in answered:
                if approval_count == 0:
                    np.testing.assert_allclose(s.snapshot()['joints'], original['joints'], atol=1e-10)
                    for a, b in zip(original['objects'], s.snapshot()['objects']):
                        np.testing.assert_allclose(a['transform'], b['transform'], atol=1e-10)
                answered.add(approval['request_id'])
                ctl.approve_recovery(dict(request_id=approval['request_id'],
                    scene_version=approval['scene_version'], approved=True))
                approval_count += 1
            if not state['busy']:
                break
            time.sleep(.03)
        idle = ctl.wait_idle(1)
        final = ctl.snapshot()
        (tmp_path/'composed_held_result.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
        assert idle, 'composed execution timed out'
        assert final['last_result']['ok'], final['last_result']
        assert final['batch']['completed'] == final['batch']['total'] == 5
        assert s.world.held is None and approval_count >= 2
        yellow = next(obj for obj in s.world.objects if obj.label == 'yellow')
        assert s.world.zone_of(yellow).name == 'zone_yellow'
        for obj in s.world.objects:
            if obj.name in untouched:
                np.testing.assert_array_equal(obj.T_base_obj, untouched[obj.name])
        assert all(c['goals'] == [{'label': 'banana', 'zone_name': 'zone_yellow'}] for c in planner.calls)
        history = final['skill_task']['history']
        assert any(h['skill'] == 'pick' for h in history)
        assert any(h['skill'] == 'place' for h in history)
        assert any(h['skill'] == 'move' and h['label'] == 'yellow' for h in history)
    finally:
        ctl.close()
