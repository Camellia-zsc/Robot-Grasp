"""Offline regressions for approval, task coverage, and resumed final checks."""
import copy
import socket
import time

import pytest

from cube_task.assistant import skill_execution
from cube_task.assistant.server import DashboardController
from test_assistant_skill_orchestrator import Client, Executor, Session, Settings
from test_assistant_skill_protocol import context, step


INSTRUCTION = '把所有物品放到各自默认区域'
DEFAULTS = dict(red='zone_red', green='zone_green', yellow='zone_yellow',
                apple='zone_red', banana='zone_yellow', pear='zone_green')
PARTIAL = [dict(label='banana', zone_name='zone_yellow'),
           dict(label='apple', zone_name='zone_red')]


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('approval-scope regressions must never access the network')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def observed_objects(context, labels):
    scene = copy.deepcopy(context['scene'])
    template = copy.deepcopy(scene['objects'][0])
    scene['objects'] = [dict(copy.deepcopy(template), object_id='v4-'+label,
                             label=label, center_base=[.20+.04*i, .04, .06])
                        for i, label in enumerate(labels)]
    scene['class_to_zone'] = copy.deepcopy(DEFAULTS)
    scene['zones'] = [dict(name=zone) for zone in dict.fromkeys(DEFAULTS.values())]
    return scene


@pytest.fixture
def controller(context, monkeypatch):
    monkeypatch.setattr(skill_execution, 'SkillExecutor', Executor)
    session = Session(observed_objects(context, DEFAULTS))
    client = Client(lambda _context, _number: [])
    ctl = DashboardController(session_factory=lambda **kwargs: session,
                              client=client, settings=Settings())
    assert ctl.wait_idle(2)
    yield ctl, session, client
    ctl.close()


def approve_until_idle(ctl):
    """Exercise real approval messages, allowing a fixed validator to refuse early."""
    deadline = time.monotonic()+3
    while time.monotonic() < deadline:
        state = ctl.snapshot()
        if not state['busy']:
            return state
        pending = state['recovery_approval']
        if pending['status'] == 'pending':
            ctl.approve_recovery(dict(request_id=pending['request_id'],
                                      scene_version=pending['scene_version'], approved=True))
        else:
            time.sleep(.01)
    pytest.fail('offline task did not finish')


def install_partial_plan(client, ask=True, goals=PARTIAL):
    def plan(instruction, context, on_delta, cancel_event=None):
        client.contexts.append(copy.deepcopy(context))
        actions = ([step('a', 'ask', question='是否授权继续所列操作？',
                         affected_goals=list(range(len(goals))))]
                   if ask else [])
        actions += [step('m'+str(i), 'move', goal['label'], dict(type='goal', index=i))
                    for i, goal in enumerate(goals)]
        actions += [step('d', 'finish')]
        return dict(scene_version=context['scene']['scene_version'], action='plan',
                    message='先完成局部目标', goals=copy.deepcopy(goals), steps=actions)
    client.plan_skills = plan


@pytest.mark.parametrize('labels,partial', [
    (['apple', 'banana', 'pear'], [dict(label='apple', zone_name='zone_red')]),
    (list(DEFAULTS), PARTIAL),
], ids=['one_of_three', 'two_of_six'])
def test_approved_partial_initial_plan_cannot_complete_all_default_regions(
        controller, context, labels, partial):
    ctl, session, client = controller
    session._scene = observed_objects(context, labels)
    install_partial_plan(client, goals=partial)
    before = session.scene
    ctl.submit(INSTRUCTION)
    state = approve_until_idle(ctl)

    assert state['status'] != 'complete', (
        'Human approval authorized listed moves, but omitted requested objects '
        'were never included in final verification, so the all-objects task is incomplete')
    assert state['metrics']['successes'] == 0
    assert not state['last_result']['ok']
    for label in set(labels)-{g['label'] for g in partial}:
        assert next(o for o in session.scene['objects'] if o['label'] == label) == next(
            o for o in before['objects'] if o['label'] == label)


def test_approved_partial_plan_cannot_shrink_scope_after_scene_change(controller):
    ctl, session, client = controller
    install_partial_plan(client)
    original_await = ctl._await_recovery_approval
    edits = []

    def approve_then_scene_changes(*args, **kwargs):
        accepted = original_await(*args, **kwargs)
        if accepted and not edits:
            edits.append(True)
            session._environment_revision += 1
            session._scene['scene_version'] += 1
        return accepted

    ctl._await_recovery_approval = approve_then_scene_changes
    ctl.submit(INSTRUCTION)
    state = approve_until_idle(ctl)

    assert state['status'] != 'complete', (
        'An edit invalidating the approved plan cannot turn the remaining two '
        'goals into coverage for the original all-objects instruction')
    assert state['metrics']['successes'] == 0 and not state['last_result']['ok']
    if edits:
        assert len(client.contexts) > 1
        assert client.contexts[1]['failure']['error'] == 'scene_changed'
        assert not client.contexts[1]['history'], 'expired moves must never be dispatched'


def test_scope_expansion_preserves_partial_plan_indices_and_requires_all_objects(controller):
    ctl, session, client = controller

    def plan(instruction, context, on_delta, cancel_event=None):
        client.contexts.append(copy.deepcopy(context))
        goals = copy.deepcopy(PARTIAL if len(client.contexts) == 1 else context['goals'])
        steps = [step('m'+str(i), 'move', goal['label'], dict(type='goal', index=i))
                 for i, goal in enumerate(goals)] + [step('done', 'finish')]
        return dict(scene_version=context['scene']['scene_version'], action='plan',
                    message='按反馈继续完成全部任务', goals=goals, steps=steps)

    client.plan_skills = plan
    ctl.submit(INSTRUCTION)
    state = approve_until_idle(ctl)

    assert state['status'] == 'complete', state['last_result']
    assert {g['label'] for g in state['last_result']['goals']} == set(DEFAULTS)
    assert {item['goal']['label'] for item in state['last_result']['items']} == set(DEFAULTS)
    assert all(o.get('zone') == DEFAULTS[o['label']] for o in session.scene['objects'])


def test_explicit_default_region_subset_is_not_expanded_to_other_objects(controller):
    ctl, session, client = controller
    install_partial_plan(client)
    before = session.scene
    ctl.submit('把香蕉和苹果放到各自默认区域')
    state = approve_until_idle(ctl)

    assert state['status'] == 'complete', state['last_result']
    assert state['last_result']['goals'] == PARTIAL
    assert [a['label'] for a in session.calls if a['skill'] == 'move'] == ['banana', 'apple']
    assert all(next(o for o in session.scene['objects'] if o['label'] == label) == next(
        o for o in before['objects'] if o['label'] == label)
        for label in set(DEFAULTS)-{'banana', 'apple'})


def test_resuming_partial_saved_goals_cannot_complete_all_default_regions(controller):
    ctl, session, client = controller
    install_partial_plan(client)
    ctl._skill_memory = dict(instruction=INSTRUCTION, goals=copy.deepcopy(PARTIAL),
                            protected_goals=[], originals=session.scene['objects'],
                            obligations={}, restorations=[])
    ctl._skill_task['status'] = 'failed'
    ctl._last_result = dict(ok=False, error='placement_infeasible')
    ctl.resume_recovery()
    state = approve_until_idle(ctl)

    assert state['status'] != 'complete', (
        'Resume must retain the user all-objects task, including objects omitted '
        'by a previously saved partial goal list')
    assert state['metrics']['successes'] == 0
    assert not state['last_result']['ok']


def test_approval_preserves_pending_restore_and_cannot_claim_scene_progress(controller):
    ctl, session, client = controller
    goals = [dict(label='apple', zone_name='zone_red')]
    calls = []

    def plan(instruction, context, on_delta, cancel_event=None):
        client.contexts.append(copy.deepcopy(context))
        calls.append(True)
        return dict(scene_version=context['scene']['scene_version'],
                    action='plan' if len(calls) == 1 else 'stop',
                    message='请求授权' if len(calls) == 1 else '保留现场停止',
                    goals=copy.deepcopy(goals),
                    steps=[step('a', 'ask', question='是否授权继续？', affected_goals=[0])]
                    if len(calls) == 1 else [])

    client.plan_skills = plan
    ctl._skill_memory = dict(instruction='苹果归红区并将香蕉复位', goals=copy.deepcopy(goals),
                            protected_goals=[], originals=session.scene['objects'],
                            obligations={'banana': 'pending'}, restorations=['banana'])
    ctl._skill_task['status'] = 'failed'
    ctl._last_result = dict(ok=False, pending_obligations={'banana': 'pending'})
    before = session.scene
    ctl.resume_recovery()
    state = approve_until_idle(ctl)

    assert state['status'] != 'complete' and not state['last_result']['ok']
    # Natural task termination may now perform its required final audit.
    # Approval itself still changes no scene and dispatches no manipulation.
    assert session.scene == before and all(call['skill'] == 'verify' for call in session.calls)
    assert state['last_result']['pending_obligations'] == {'banana': 'pending'}
    assert ctl._skill_memory['goals'] == goals
    feedback = state['skill_task']['history'][0]
    assert feedback['skill'] == 'ask' and feedback['ok']
    assert feedback['changed'] is False
    assert feedback['reason'] == 'approval_only_no_scene_change'
    assert client.contexts[1]['pending_obligations'] == ['banana']
    assert client.contexts[1]['planning_budget']['stagnation'] == 1


def test_ask_after_one_cached_goal_requires_fresh_verification_of_all_goals(controller, monkeypatch):
    ctl, session, client = controller
    goals = [dict(label='apple', zone_name='zone_red'),
             dict(label='banana', zone_name='zone_yellow')]
    next(o for o in session._scene['objects'] if o['label'] == 'apple')['zone'] = 'zone_red'
    verified_goal_lists = []

    class SelectingExecutor(Executor):
        def execute(self, action, current_goals, *args, **kwargs):
            if action['skill'] == 'verify':
                selected = ([current_goals[action['goal_index']]]
                            if 'goal_index' in action else current_goals)
                verified_goal_lists.append(copy.deepcopy(selected))
                return super().execute(action, selected, *args, **kwargs)
            return super().execute(action, current_goals, *args, **kwargs)

    monkeypatch.setattr(skill_execution, 'SkillExecutor', SelectingExecutor)

    def plan(instruction, context, on_delta, cancel_event=None):
        client.contexts.append(copy.deepcopy(context))
        first = len(client.contexts) == 1
        return dict(scene_version=context['scene']['scene_version'],
                    action='plan' if first else 'stop', message='核验目标', goals=copy.deepcopy(goals),
                    steps=[step('v', 'verify', goal_index=0),
                           step('a', 'ask', question='是否授权结束前的全部核验？', affected_goals=[0, 1]),
                           step('f', 'finish')] if first else [])

    client.plan_skills = plan
    ctl.submit('把苹果和香蕉分别放到红区和黄区')
    state = approve_until_idle(ctl)

    assert state['status'] != 'complete' and not state['last_result']['ok']
    assert verified_goal_lists == [[goals[0]], goals]
    assert state['metrics']['successes'] == 0
    assert ctl._skill_memory['goals'] == goals


def test_legacy_task_start_inventory_is_not_shrunk_by_deletion_before_resume(controller):
    ctl, session, client = controller
    originals = session.scene['objects']
    # Legacy tasks have no task_inventory field. Their saved task-start camera
    # records still prove six identities existed before the user removed four.
    session._scene['objects'] = [o for o in originals if o['label'] in {'banana', 'apple'}]
    session.task_inventory = lambda: [dict(label=o['label'], count=1)
                                      for o in session._scene['objects']]
    install_partial_plan(client)
    ctl._skill_memory = dict(instruction=INSTRUCTION, goals=copy.deepcopy(PARTIAL),
                            protected_goals=[], originals=copy.deepcopy(originals),
                            obligations={}, restorations=[])
    ctl._skill_task['status'] = 'failed'
    ctl._last_result = dict(ok=False, error='placement_infeasible')
    ctl.resume_recovery()
    state = approve_until_idle(ctl)

    assert state['status'] != 'complete', (
        'Removing objects after a legacy task started cannot silently narrow '
        'its original all-objects instruction when that task resumes')
    assert not state['last_result']['ok'] and state['metrics']['successes'] == 0


def test_frozen_inventory_is_not_shrunk_by_current_editor_inventory_on_resume(controller):
    ctl, session, client = controller
    originals = session.scene['objects']
    frozen = [dict(label=o['label'], count=1) for o in originals]
    session._scene['objects'] = [o for o in originals if o['label'] in {'banana', 'apple'}]
    session.task_inventory = lambda: [dict(label=o['label'], count=1)
                                      for o in session._scene['objects']]
    install_partial_plan(client)
    ctl._skill_memory = dict(instruction=INSTRUCTION, goals=copy.deepcopy(PARTIAL),
                            task_inventory=copy.deepcopy(frozen),
                            protected_goals=[], originals=copy.deepcopy(originals),
                            obligations={}, restorations=[])
    ctl._skill_task['status'] = 'failed'
    ctl._last_result = dict(ok=False, error='placement_infeasible')
    ctl.resume_recovery()
    state = approve_until_idle(ctl)

    assert state['status'] != 'complete' and not state['last_result']['ok']
    assert ctl._skill_memory['task_inventory'] == frozen
    assert {g['label'] for g in ctl._skill_memory['goals']} == set(DEFAULTS)


def test_contract_final_check_rejects_aggregate_success_with_partial_items(controller, monkeypatch):
    ctl, session, client = controller
    for obj in session._scene['objects']:
        obj['zone'] = DEFAULTS[obj['label']]

    class IncompleteVerifier(Executor):
        def execute(self, action, goals, *args, **kwargs):
            result = super().execute(action, goals, *args, **kwargs)
            if action['skill'] == 'verify':
                result['items'] = result['items'][:2]
            return result

    monkeypatch.setattr(skill_execution, 'SkillExecutor', IncompleteVerifier)

    def plan(instruction, context, on_delta, cancel_event=None):
        client.contexts.append(copy.deepcopy(context))
        return dict(scene_version=context['scene']['scene_version'], action='plan',
                    message='请求全部最终核验', goals=copy.deepcopy(context['goals']),
                    steps=[step('done', 'finish')])

    client.plan_skills = plan
    ctl.submit(INSTRUCTION)
    state = approve_until_idle(ctl)

    assert state['status'] != 'complete' and not state['last_result']['ok']
    assert state['metrics']['successes'] == 0
    assert {g['label'] for g in ctl._skill_memory['goals']} == set(DEFAULTS)


def test_legacy_restore_evidence_can_coexist_with_full_contract_completion(controller):
    ctl, session, client = controller
    # The old partial planner treated the red cube as an obstacle although the
    # user's original task included it. Its original position already met its
    # default-zone goal, so both requirements can be measured at the same time.
    for obj in session._scene['objects']:
        obj['zone'] = DEFAULTS[obj['label']]
    originals = session.scene['objects']
    ctl._skill_memory = dict(instruction=INSTRUCTION, goals=copy.deepcopy(PARTIAL),
                            protected_goals=[], originals=copy.deepcopy(originals),
                            obligations={'red': 'pending'}, restorations=['red'])
    ctl._skill_task['status'] = 'failed'
    ctl._last_result = dict(ok=False, pending_obligations={'red': 'pending'})

    def plan(instruction, context, on_delta, cancel_event=None):
        client.contexts.append(copy.deepcopy(context))
        return dict(scene_version=context['scene']['scene_version'], action='plan',
                    message='核验全部目标与先前承诺的复位', goals=copy.deepcopy(context['goals']),
                    steps=[step('done', 'finish')])

    client.plan_skills = plan
    ctl.resume_recovery()
    state = approve_until_idle(ctl)

    assert state['status'] == 'complete', state['last_result']
    assert state['last_result']['ok'] and not state['last_result']['pending_obligations']
    assert {g['label'] for g in state['last_result']['goals']} == set(DEFAULTS)
    assert len(state['last_result']['items']) == len(DEFAULTS)+1
