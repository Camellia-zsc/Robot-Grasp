"""Composition, feedback and approval boundaries; never call an online model."""
import copy
import threading
import time
from types import SimpleNamespace

import pytest

from cube_task.assistant.server import DashboardController
from cube_task.assistant.skill_orchestrator import SkillOrchestrator
from test_assistant_server import Settings
from test_assistant_skill_protocol import context, step

GOALS = [dict(label='apple', zone_name='zone_red')]


class Session:
    def __init__(self, observed):
        self._scene = copy.deepcopy(observed)
        self.cancel_event = threading.Event()
        self._environment_revision = 0
        self.runner = SimpleNamespace(_held_recovery_context=None, has_held_recovery=False)
        self.calls, self.holding = [], None
        self.hook = None
        self.fail_place = False
        self.bad_verification = False

    @property
    def scene(self):
        return copy.deepcopy(self._scene)

    def snapshot(self):
        return dict(scene_version=self._scene['scene_version'], held_object=self.holding,
                    objects=[], sim_time=0, editor={})

    def prepare(self):
        return self.scene

    def observe(self):
        return self.scene

    def set_speed(self, speed):
        pass

    def stop(self):
        self.cancel_event.set()

    def close(self):
        pass


class Executor:
    def __init__(self, session, originals=()):
        self.session, self.originals = session, copy.deepcopy(originals)

    def execute(self, action, goals, protected_goals=(), cancel_event=None, expected_revision=None):
        s = self.session
        s.calls.append(copy.deepcopy(action))
        skill, label = action['skill'], action.get('label')
        result = dict(ok=True, camera_verified=True, truth={'success': True},
                      verified_revision=s._environment_revision, message='实测步骤通过')
        if skill == 'pick':
            s.holding = label
            s.runner._held_recovery_context = dict(det=SimpleNamespace(label=label))
            s.runner.has_held_recovery = True
        elif skill in ('place', 'move'):
            if s.fail_place and skill == 'place':
                s.fail_place = False
                return dict(ok=False, stage='place', holding=True, error='occupied', message='落点发生占用')
            target = next(o for o in s._scene['objects'] if o['label'] == label)
            destination = action['destination']
            if destination['type'] == 'original':
                original = next(o for o in self.originals if o['label'] == label)
                target.update(center_base=copy.deepcopy(original['center_base']), yaw=original['yaw'])
            elif destination['type'] == 'goal':
                target['zone'] = goals[destination['index']]['zone_name']
            else:
                target['center_base'][0] += .04
            s.holding = None
            s.runner._held_recovery_context = None
            s.runner.has_held_recovery = False
        elif skill == 'verify':
            items = []
            for goal in goals:
                target = next(o for o in s._scene['objects'] if o['label'] == goal['label'])
                if goal.get('destination'):
                    ok = all(abs(a-b) < .015 for a, b in zip(target['center_base'], goal['destination']['xy']))
                else:
                    ok = target.get('zone') == goal['zone_name']
                items.append(dict(goal=goal, ok=ok, camera_verified=ok, truth={'success': ok}))
            ok = bool(items) and all(i['ok'] for i in items) and not s.bad_verification
            result.update(ok=ok, items=items, truth={'success': ok}, camera_verified=ok)
        if s.hook:
            s.hook(action, result)
        return dict(result, holding=bool(s.holding))


class Client:
    def __init__(self, fn):
        self.fn, self.contexts = fn, []

    def plan_skills(self, instruction, context, on_delta, cancel_event=None):
        self.contexts.append(copy.deepcopy(context))
        on_delta('{"message":"模型实际步骤"}')
        return dict(scene_version=context['scene']['scene_version'], action='plan', message='模型实际步骤',
                    goals=copy.deepcopy(context['goals'] or GOALS), steps=self.fn(context, len(self.contexts)))

    def decide(self, *args, **kwargs):
        raise AssertionError('default dashboard must not use old macro/decision route')


@pytest.fixture
def create(context, monkeypatch):
    from cube_task.assistant import skill_execution
    monkeypatch.setattr(skill_execution, 'SkillExecutor', Executor)
    controllers = []

    def factory(fn):
        s, c = Session(context['scene']), Client(fn)
        ctl = DashboardController(session_factory=lambda **kwargs: s, client=c, settings=Settings())
        assert ctl.wait_idle(2)
        controllers.append(ctl)
        return ctl, s, c

    yield factory
    for ctl in controllers:
        ctl.close()


def pending(ctl):
    deadline = time.monotonic()+3
    while time.monotonic() < deadline:
        state = ctl.snapshot()
        if state['recovery_approval']['status'] == 'pending':
            return state['recovery_approval']
        if not state['busy']:
            pytest.fail(str(state['last_result']))
        threading.Event().wait(.01)
    pytest.fail('no explicit approval request')


def test_incremental_goal_completion_ends_queue_before_redundant_observe(create):
    ctl, session, client = create(lambda context, number: [
        step('m', 'move', 'apple', {'type': 'goal', 'index': 0}),
        dict(id='extra', skill='observe', label='apple'),
        dict(id='v', skill='verify'), dict(id='f', skill='finish')])
    session._motion_observer = SimpleNamespace(current=lambda: [object()])
    session._camera_goal_confirmed = lambda goal, fresh, revision: bool(fresh) and any(
        o['label'] == goal['label'] and o.get('zone') == goal['zone_name']
        for o in session._scene['objects'])
    ctl.submit('把苹果放到红色区域')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert state['status'] == 'complete', state['last_result']
    assert len(client.contexts) == 1
    assert [call['skill'] for call in session.calls] == ['move', 'verify']
    assert all(s['status'] == 'skipped' for s in state['skill_task']['steps'][1:])


def approve(ctl, yes=True):
    p = pending(ctl)
    ctl.approve_recovery(dict(request_id=p['request_id'], scene_version=p['scene_version'], approved=yes))


def normal(_context, _number):
    return [step(1, 'move', 'apple', {'type': 'goal', 'index': 0}), step(2, 'verify'), step(3, 'finish')]


def test_explicit_question_before_rearrangement_needs_only_one_button_approval(create):
    def actions(c, n):
        return [step(1, 'ask', question='允许把香蕉留在安全空位，再归位苹果吗？', affected_goals=[]),
                step(2, 'move', 'banana', {'type': 'auto_free'}),
                step(3, 'move', 'apple', {'type': 'goal', 'index': 0}), step(4, 'finish')]
    ctl, session, _client = create(actions)
    ctl.submit('把苹果放回红色区域')
    pending(ctl)
    assert not session.calls
    approve(ctl)
    assert ctl.wait_idle(2), 'A prior explicit ask must not be preceded by a duplicate blanket approval'
    assert ctl.snapshot()['status'] == 'complete'
    assert [s['skill'] for s in session.calls[:2]] == ['move', 'move']


def test_default_planner_executes_composed_skills_and_verifies_once(create):
    ctl, session, client = create(normal)
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert state['status'] == 'complete', state['last_result']
    assert [s['skill'] for s in session.calls] == ['move', 'verify']
    assert len(client.contexts) == 1 and state['metrics']['successes'] == 1
    assert state['skill_task']['executed'] == 3


@pytest.mark.parametrize('halt_flag', ['fault', 'aborted'])
def test_completed_release_with_later_controller_halt_never_dispatches_next_step(create, halt_flag):
    # Production can finish opening before retreat fails: ok describes release,
    # while fault/aborted still forbids any subsequent movement or verification.
    ctl, session, client = create(normal)
    def release_then_halt(action, result):
        if action['skill'] == 'move':
            result.update(ok=True, **{halt_flag: True})
    session.hook = release_then_halt
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert [action['skill'] for action in session.calls] == ['move']
    assert not state['last_result']['ok']
    assert state['last_result'][halt_flag] is True
    assert state['needs_reset']
    assert state['metrics']['successes'] == 0
    assert len(client.contexts) == 1
    if halt_flag == 'fault':
        assert not state['can_resume_recovery']


def test_arbitrary_two_obstacles_and_repeated_object_require_real_approval(create):
    def compose(_context, _number):
        return [step(1, 'move', 'banana', {'type': 'auto_free'}),
                step(2, 'move', 'pear', {'type': 'auto_free'}),
                step(3, 'pick', 'apple'), step(4, 'place', 'apple', {'type': 'goal', 'index': 0}),
                step(5, 'move', 'pear', {'type': 'original', 'label': 'pear'}),
                step(6, 'move', 'banana', {'type': 'original', 'label': 'banana'}), step(7, 'finish')]
    ctl, session, client = create(compose)
    before = session.scene
    ctl.submit('苹果放红区，允许先询问再处理多个障碍')
    pending(ctl)
    assert session.calls == [] and session.scene == before
    approve(ctl)
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert state['last_result']['ok'], state['last_result']
    assert not state['last_result']['pending_obligations']
    assert [s['label'] for s in session.calls if s['skill'] == 'move'] == ['banana', 'pear', 'pear', 'banana']
    assert len(client.contexts) == 1


def test_decline_preserves_scene_and_resume_does_not_imply_approval(create):
    ctl, session, client = create(lambda c, n: [step(1, 'move', 'banana', {'type': 'auto_free'})])
    ctl.submit('苹果放红区')
    approve(ctl, False)
    assert ctl.wait_idle(3) and not session.calls
    assert ctl.snapshot()['can_resume_recovery']
    ctl.resume_recovery()
    pending(ctl)
    assert not session.calls and len(client.contexts) == 2
    approve(ctl, False)
    assert ctl.wait_idle(3)


def test_resume_after_stop_rescan_keeps_task_start_restore_position_and_yaw(create):
    def compose(c, n):
        return [step(1, 'move', 'pear', {'type': 'original', 'label': 'pear'}),
                step(2, 'move', 'apple', {'type': 'goal', 'index': 0}), step(3, 'finish')]
    ctl, session, client = create(compose)
    originals = session.scene['objects']
    original = copy.deepcopy(next(o for o in originals if o['label'] == 'pear'))
    displaced = session.scene
    pear = next(o for o in displaced['objects'] if o['label'] == 'pear')
    pear['center_base'][0] += .04
    pear['yaw'] += .4
    session._scene['objects'] = []  # stop invalidates old observations, not task obligations
    def fresh_observation():
        session._scene = copy.deepcopy(displaced)
        return session.scene
    session.observe = fresh_observation
    ctl._skill_memory = dict(instruction='苹果放红区，再将临时移开的梨复位',
        goals=copy.deepcopy(GOALS), protected_goals=[], originals=copy.deepcopy(originals),
        obligations={'pear': 'pending'}, restorations=[])
    ctl._skill_task['status'] = 'stopped'
    ctl._last_result = dict(ok=False, aborted=True, pending_obligations={'pear': 'pending'})
    ctl.resume_recovery()
    pending(ctl)
    approve(ctl)
    assert ctl.wait_idle(3)
    measured = next(o for o in session.scene['objects'] if o['label'] == 'pear')
    assert measured['center_base'] == original['center_base']
    assert measured['yaw'] == original['yaw']
    assert ctl._skill_memory['originals'] == originals
    assert ctl.snapshot()['last_result']['ok'], ctl.snapshot()['last_result']


def test_model_can_ask_mid_plan_and_continue_after_button_answer(create):
    def ask(c, n):
        return [step(1, 'observe', 'apple'), step(2, 'ask', question='是否继续本次操作？', affected_goals=[0])] + normal(c, n)
    # IDs must be unique even when plans combine fragments.
    def numbered(c, n):
        return [dict(s, id=str(i)) for i, s in enumerate(ask(c, n))]
    ctl, session, client = create(numbered)
    ctl.submit('苹果放红区')
    pending(ctl)
    assert [s['skill'] for s in session.calls] == ['observe']
    approve(ctl)
    assert ctl.wait_idle(3) and ctl.snapshot()['last_result']['ok']
    approval_feedback = next(item for item in ctl.snapshot()['skill_task']['history']
                             if item['skill'] == 'ask')
    assert approval_feedback['changed'] is False
    assert approval_feedback['reason'] == 'approval_only_no_scene_change'


def test_failed_place_replans_with_real_holding_and_unchanged_final_goal(create):
    def compose(c, n):
        return ([step(1, 'pick', 'apple')] if n == 1 else []) + [
            step(2, 'place', 'apple', {'type': 'goal', 'index': 0}), step(3, 'verify'), step(4, 'finish')]
    ctl, session, client = create(compose)
    session.fail_place = True
    ctl.submit('苹果放红区')
    assert ctl.wait_idle(3)
    assert ctl.snapshot()['last_result']['ok'], ctl.snapshot()['last_result']
    assert len(client.contexts) == 2
    retry = client.contexts[1]
    assert retry['holding'] and retry['held_label'] == 'apple' and retry['goals'] == GOALS
    assert retry['failure']['error'] == 'occupied' and retry['history'][-1]['ok'] is False
    assert [s['skill'] for s in session.calls].count('pick') == 1


def test_stop_during_first_obstacle_move_keeps_feedback_and_restore_obligation(create):
    ctl, session, _ = create(lambda c, n: [
        step(1, 'move', 'banana', {'type': 'auto_free'}),
        step(2, 'move', 'banana', {'type': 'original', 'label': 'banana'}), step(3, 'finish')])
    def stop(action, result):
        ctl.stop()
        result.update(ok=False, aborted=True, message='运动中停止')
    session.hook = stop
    ctl.submit('苹果放红区')
    approve(ctl)
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert state['status'] == 'stopped' and len(session.calls) == 1
    assert state['last_result']['pending_obligations'] == {'banana': 'pending'}
    assert state['skill_task']['history'][0]['ok'] is False
    assert not state['can_resume_recovery']
    with pytest.raises(ValueError):
        ctl.resume_recovery()


def test_failed_final_verification_never_counts_success(create):
    ctl, session, client = create(normal)
    session.bad_verification = True
    ctl.submit('苹果放红区')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert not state['last_result']['ok'] and state['metrics']['successes'] == 0
    assert len(client.contexts) == state['skill_task']['max_plans'] and state['can_resume_recovery']


def test_completed_task_cannot_be_replayed_through_resume_endpoint(create):
    ctl, _, _ = create(normal)
    ctl.submit('苹果放红区')
    assert ctl.wait_idle(3)
    with pytest.raises(ValueError):
        ctl.resume_recovery()


def test_model_receives_actual_executor_capabilities_and_remaining_plan_budget(create, monkeypatch):
    capabilities = dict(skills=['observe', 'pick', 'place', 'move', 'verify', 'ask', 'finish'],
                        observe_modes=['stationary', 'active'],
                        observe_destination_types=['goal', 'zone', 'region', 'original'],
                        holding_allowed_skills=['observe', 'place', 'ask'],
                        active_observe_requires_geometry=True)
    monkeypatch.setattr(Executor, 'capabilities', lambda self: copy.deepcopy(capabilities), raising=False)
    ctl, _session, client = create(normal)
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    first = client.contexts[0]
    assert first['capabilities'] == capabilities
    assert first['planning_budget'] == dict(plan_number=1, max_plans=8,
        remaining_plans=7, remaining_steps=64, stagnation=0)
    assert ctl.snapshot()['last_result']['ok']


def test_action_failure_diagnostics_survive_observe_only_replanning(create, monkeypatch):
    original = Executor.execute
    def execute(self, action, *args, **kwargs):
        result = original(self, action, *args, **kwargs)
        if action['skill'] == 'place' and not result['ok']:
            result['diagnostics'] = 'placement_planning_budget_exceeded'
        if action['skill'] == 'observe':
            result.update(observation_mode='stationary_with_last_scan', changed=False,
                          reason='same_view_no_new_coverage')
        return result
    monkeypatch.setattr(Executor, 'execute', execute)
    def compose(c, n):
        if n == 1:
            return [step(1, 'pick', 'apple'), step(2, 'place', 'apple', {'type': 'goal', 'index': 0})]
        if n == 2:
            return [step(3, 'observe')]
        return [step(4, 'place', 'apple', {'type': 'goal', 'index': 0}), step(5, 'finish')]
    ctl, session, client = create(compose)
    session.fail_place = True
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    assert ctl.snapshot()['last_result']['ok']
    last = client.contexts[2]
    assert last['failure']['error'] == 'goal_not_finished'
    assert last['last_action_failure']['error'] == 'occupied'
    assert last['last_action_failure']['diagnostics'] == 'placement_planning_budget_exceeded'
    assert last['history'][-2]['diagnostics'] == 'placement_planning_budget_exceeded'
    assert last['history'][-1]['observation_mode'] == 'stationary_with_last_scan'
    assert last['history'][-1]['changed'] is False
    assert last['history'][-1]['reason'] == 'same_view_no_new_coverage'
    assert last['planning_budget']['stagnation'] == 1


def test_three_observe_only_rounds_without_information_stop_and_preserve_hold(create):
    ctl, session, client = create(lambda c, n: [step(n, 'pick', 'apple')] if n == 1 else [step(n, 'observe')])
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert len(client.contexts) == 4
    assert [c['planning_budget']['stagnation'] for c in client.contexts] == [0, 0, 1, 2]
    assert state['last_result']['error'] == 'planning_stagnation'
    assert state['last_result']['holding'] and session.holding == 'apple'
    assert state['metrics']['successes'] == 0 and state['can_resume_recovery']
    assert state['needs_reset'] and not state['simulation']['editor']['editable']


def test_planning_stagnation_without_held_object_or_fault_allows_scene_adjustment(create):
    ctl, session, client = create(lambda c, n: [step(n, 'observe')])
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    # An unchanged view now receives one complete audit and one model
    # decision, ending without the old redundant observation-only rounds.
    assert state['last_result']['error'] == 'goal_not_finished'
    assert [action['skill'] for action in session.calls] == ['observe', 'verify', 'observe']
    assert state['skill_task']['final_audit']['items'][0]['ok'] is False
    assert not state['last_result']['ok'] and state['metrics']['successes'] == 0
    assert state['can_resume_recovery'] and ctl._skill_memory['goals'] == GOALS
    assert not state['needs_reset']
    assert state['simulation']['editor']['editable']
    adjusted = []
    def edit(request):
        adjusted.append(request)
        session._scene['scene_version'] += 1
        session._environment_revision += 1
        return session.scene
    session.edit_scene = edit
    ctl.edit_scene(dict(action='move', object_id='v4-apple', xy=[.3, .04]))
    assert ctl.wait_idle(3)
    assert adjusted and ctl.snapshot()['simulation']['editor']['editable']
    assert ctl.snapshot()['can_resume_recovery']
    assert ctl._skill_memory['goals'] == GOALS


@pytest.mark.parametrize('fault', [False, True])
def test_model_stop_after_unheld_geometry_failure_preserves_task_and_only_locks_real_faults(create, fault):
    ctl, session, client = create(normal)
    def failure(action, result):
        if action['skill'] == 'move':
            result.update(ok=False, fault=fault, stage='plan',
                          error='no_clear_supported_release', message='当前落点不可行')
    session.hook = failure
    original_plan = client.plan_skills
    def plan(*args, **kwargs):
        result = original_plan(*args, **kwargs)
        if len(client.contexts) > 1:
            result.update(action='stop', message='请先调整现场，再继续原任务', steps=[])
        return result
    client.plan_skills = plan
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    # This fake physically reaches the goal before its hook changes the action
    # report to a geometry failure. A healthy final audit can prove completion;
    # a real arm fault must stop before any audit camera command.
    assert state['skill_task']['status'] == ('failed' if fault else 'complete')
    assert state['last_result']['ok'] is not fault
    assert [action['skill'] for action in session.calls] == (['move'] if fault else ['move', 'verify'])
    assert ctl._skill_memory['goals'] == GOALS
    assert not state['can_resume_recovery']
    assert state['needs_reset'] is fault
    assert state['simulation']['editor']['editable'] is not fault


def test_real_fault_cannot_be_cleared_by_resuming_a_model_request(create):
    ctl, session, client = create(normal)
    def failure(action, result):
        if action['skill'] == 'move':
            result.update(ok=False, fault=True, error='comm', message='机械臂通信故障')
    session.hook = failure
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    calls = len(client.contexts)
    assert ctl.snapshot()['needs_reset']
    assert not ctl.snapshot()['can_resume_recovery']
    with pytest.raises(ValueError, match='故障'):
        ctl.resume_recovery()
    with pytest.raises(ValueError):
        ctl.edit_scene(dict(action='move', object_id='v4-apple', xy=[.3, .04]))
    assert len(client.contexts) == calls
    assert not ctl.snapshot()['simulation']['editor']['editable']


@pytest.mark.parametrize('fault', [False, True])
@pytest.mark.parametrize('source', ['memory', 'incoming'])
def test_recovery_model_error_preserves_only_a_recorded_real_fault(create, fault, source):
    ctl, session, client = create(lambda c, n: [step(n, 'observe')])
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    recorded = dict(ok=False, fault=fault, skill='move', error='comm' if fault else 'placement_infeasible')
    ctl._skill_memory['last_action_failure'] = recorded if source == 'memory' else {}
    def unavailable(*args, **kwargs):
        raise RuntimeError('模型连接中断')
    client.plan_skills = unavailable
    # Exercise the orchestrator seam directly: the public resume entry rejects
    # known real faults before launching any model request.
    SkillOrchestrator(ctl, Executor).run('把苹果放红区', resume=True, failure=recorded)
    state = ctl.snapshot()
    assert state['last_result']['error'] == 'skill_task_error'
    assert state['needs_reset'] is fault
    assert bool(state['last_result'].get('fault')) is fault
    assert state['simulation']['editor']['editable'] is not fault


def test_successful_final_verification_clears_historical_action_failure(create):
    ctl, session, client = create(normal)
    ctl._skill_memory = dict(instruction='把苹果放红区', goals=copy.deepcopy(GOALS),
        protected_goals=[], originals=session.scene['objects'], obligations={}, restorations=[],
        last_action_failure=dict(ok=False, fault=True, error='old_fault'))
    result = SkillOrchestrator(ctl, Executor).run('把苹果放红区', resume=True)
    assert result['ok'] and not ctl.snapshot()['needs_reset']
    assert not ctl._skill_memory['last_action_failure']


def test_actual_observation_information_resets_stagnation(create, monkeypatch):
    original, observed = Executor.execute, []
    def execute(self, action, *args, **kwargs):
        result = original(self, action, *args, **kwargs)
        if action['skill'] == 'observe':
            observed.append(True)
            result.update(changed=len(observed) == 2, observation_mode='target_search')
        return result
    monkeypatch.setattr(Executor, 'execute', execute)
    ctl, _session, client = create(lambda c, n: [step(n, 'observe')])
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    # New information permits another audit. The following unchanged view
    # cannot repeat that camera audit or prolong the post-audit decision.
    assert [c['planning_budget']['stagnation'] for c in client.contexts] == [0, 1, 0]
    assert [a['skill'] for a in _session.calls] == ['observe', 'verify', 'observe', 'verify', 'observe']
    assert ctl.snapshot()['last_result']['error'] == 'goal_not_finished'


def test_plan_limit_scales_with_goals_and_initial_steps_but_remains_finite():
    from cube_task.assistant.skill_orchestrator import plan_limit
    assert plan_limit([GOALS[0]]) == 8
    assert plan_limit([GOALS[0]] * 6) == 16
    assert plan_limit([GOALS[0]], [None] * 24) == 16
    assert plan_limit([GOALS[0]] * 100, [None] * 100) == 16


def test_actual_run_adapts_budget_after_first_plan_fixes_task_size(create, monkeypatch):
    goals = [dict(label=label, zone_name=zone) for label, zone in
             (('apple', 'zone_red'), ('banana', 'zone_green'), ('pear', 'zone_green'))]
    original = Executor.execute
    def execute(self, action, *args, **kwargs):
        return dict(original(self, action, *args, **kwargs), changed=True)
    monkeypatch.setattr(Executor, 'execute', execute)
    ctl, _session, client = create(lambda c, n: [step(n, 'observe')])
    original_plan = client.plan_skills
    def plan(*args, **kwargs):
        return dict(original_plan(*args, **kwargs), goals=copy.deepcopy(goals))
    client.plan_skills = plan
    ctl.submit('把三个物品放入各自区域')
    assert ctl.wait_idle(3)
    assert len(client.contexts) == 10
    assert client.contexts[0]['planning_budget']['max_plans'] == 8
    assert [c['planning_budget']['max_plans'] for c in client.contexts[1:]] == [10] * 9
    assert client.contexts[-1]['planning_budget']['remaining_plans'] == 0
    assert ctl.snapshot()['skill_task']['max_plans'] == 10
    assert not ctl.snapshot()['last_result']['ok']


@pytest.mark.parametrize('memory_kind', ['legacy', 'old', 'recorded'])
def test_resume_keeps_original_action_diagnostics_after_observe_only_plan(create, memory_kind):
    def compose(c, n):
        return ([step(1, 'observe')] if n == 1 else
                [step(2, 'place', 'apple', {'type': 'goal', 'index': 0}), step(3, 'finish')])
    ctl, session, client = create(compose)
    session.holding = 'apple'
    session.runner._held_recovery_context = dict(det=SimpleNamespace(label='apple'))
    session.runner.has_held_recovery = True
    session._recovery_goal = copy.deepcopy(GOALS[0])
    original = dict(ok=False, holding=True, stage='inspect_place', error='placement_changed',
                    diagnostics='placement_planning_budget_exceeded', message='仍保持夹持')
    newer = dict(original, diagnostics='newer_measured_placement_failure')
    ctl._last_result = copy.deepcopy(original)
    if memory_kind != 'legacy':
        ctl._skill_memory = dict(instruction='把苹果放红区', goals=copy.deepcopy(GOALS),
            protected_goals=[], originals=session.scene['objects'], obligations={}, restorations=[])
        ctl._skill_task['status'] = 'failed'
        if memory_kind == 'recorded':
            ctl._skill_memory['last_action_failure'] = copy.deepcopy(newer)
    ctl.resume_recovery()
    assert ctl.wait_idle(3)
    assert ctl.snapshot()['last_result']['ok']
    assert len(client.contexts) == 2
    retry = client.contexts[1]
    assert retry['failure']['error'] == 'goal_not_finished'
    assert retry['last_action_failure'] == (newer if memory_kind == 'recorded' else original)


def test_rejected_recovery_plan_is_returned_to_model_without_losing_held_failure(create):
    def compose(c, n):
        if n == 1:
            return [step(1, 'pick', 'apple'),
                    step(2, 'place', 'apple', {'type': 'goal', 'index': 0})]
        if n == 2:
            return [step(3, 'unsupported_skill', 'apple')]
        return [step(4, 'place', 'apple', {'type': 'goal', 'index': 0}), step(5, 'finish')]
    ctl, session, client = create(compose)
    session.fail_place = True
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    assert ctl.snapshot()['last_result']['ok'], ctl.snapshot()['last_result']
    assert len(client.contexts) == 3
    corrected = client.contexts[2]
    assert corrected['failure']['error'] == 'invalid_skill_plan'
    assert corrected['failure']['stage'] == 'plan_validation'
    assert corrected['last_action_failure']['error'] == 'occupied'
    assert corrected['held_label'] == 'apple' and corrected['goals'] == GOALS
    assert [s['skill'] for s in session.calls] == ['pick', 'place', 'place', 'verify']
    assert ctl.snapshot()['skill_task']['executed'] == 4
    assert any(m.get('status') == 'error' for m in ctl.snapshot()['messages'])


def test_repeated_invalid_model_plans_stop_with_no_robot_commands(create):
    ctl, session, client = create(lambda c, n: [step(n, 'unsupported_skill', 'apple')])
    ctl.submit('把苹果放红区')
    assert ctl.wait_idle(3)
    state = ctl.snapshot()
    assert len(client.contexts) == 3
    assert session.calls == [] and state['skill_task']['executed'] == 0
    assert not state['last_result']['ok']
    assert state['last_result']['error'] == 'planning_stagnation'
    assert state['metrics']['successes'] == 0
