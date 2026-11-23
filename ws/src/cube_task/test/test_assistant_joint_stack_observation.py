"""Preparing a stack must not trade the upper object's image for a clipped base."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.assistant.observation_policy import rank_views
from test_assistant_targeted_scan import block, scan_session as base_scan_session, fresh_session
from test_assistant_observation_policy import INTR, camera


@pytest.fixture(autouse=True)
def measured_body_anchor(monkeypatch):
    # These synthetic detections exercise view scheduling and joint support
    # evidence. Real source-body fitting has separate RGB-D/physics tests.
    monkeypatch.setattr('cube_task.assistant.held_anchor.source_anchor',
                        lambda grasp: dict(ok=True))


def scan_session(frames):
    session, seen = base_scan_session(frames)
    session._prepare_observation_motion = lambda *a, **kw: True
    return session, seen


def reference():
    return dict(label='apple', center_base=[.30, 0., .06], length=.06,
                width=.06, height=.06, require_grasp=True,
                required_support=dict(label='green', center_base=[.18, .14, .05],
                                      length=.05, width=.05, height=.05))


def pair():
    return block('apple', xy=(.30, 0.), top=.06, height=.06), block('green', xy=(.18, .14))


def clipped(det):
    result = copy.deepcopy(det)
    result.valid, result.reject_reason = False, 'touches_image_border'
    return result


def test_stack_preparation_combines_fresh_target_and_support_before_stopping():
    apple, green = pair()
    session, seen = scan_session([[apple, clipped(green)], [clipped(apple), green], []])
    session.runner.planner = SimpleNamespace(plan=lambda d: SimpleNamespace(feasible=True))
    session.observe(target=reference(), purpose='recover_queued_target')
    assert seen == [0, 1]
    assert {d.label for d in session._last_fresh if d.valid} == {'apple', 'green'}


@pytest.mark.parametrize('change', ['missing', 'ambiguous', 'revision', 'intervention'])
def test_stack_preparation_never_accepts_missing_ambiguous_or_invalidated_support(change):
    apple, green = pair()
    lower = [] if change == 'missing' else [green]
    if change == 'ambiguous':
        lower.append(block('green', xy=(.3, .2)))
    session, seen = scan_session([[apple] + lower] * 3)
    session.runner.planner = SimpleNamespace(plan=lambda d: SimpleNamespace(feasible=True))
    original = session.runner._observe
    def observe(view):
        result = original(view)
        if change == 'revision':
            session._environment_revision += 1
        elif change == 'intervention':
            session._interventions.append({})
        return result
    session.runner._observe = observe
    session.observe(target=reference(), purpose='recover_queued_target')
    assert not session._support_observed(reference(), session._last_fresh, 1)
    assert len(seen) > 1 or session._environment_revision != 1 or session._interventions


def test_one_joint_view_is_enough_for_stack_preparation():
    apple, green = pair()
    session, seen = scan_session([[apple, green], [], []])
    session.runner.planner = SimpleNamespace(plan=lambda d: SimpleNamespace(feasible=True))
    session.observe(target=reference(), purpose='recover_queued_target')
    assert seen == [0]


def test_joint_coverage_precedes_a_preferred_upper_only_view():
    views = [camera(0.), camera(.25)]
    assert rank_views(views, INTR, [0., 0.], preferred=0,
                      required_targets=[([.55, 0.], .04, .04)]) == [1, 0]


@pytest.mark.parametrize('support_state', ['missing', 'clipped', 'old', 'valid'])
def test_reuse_of_upper_capture_also_requires_fresh_complete_support(support_state):
    session = fresh_session()
    green = block('green', xy=(.3, .2))
    green.extra.update(_capture_stamp=10., _capture_flange=np.eye(4))
    if support_state == 'clipped':
        green = clipped(green)
    elif support_state == 'old':
        green.extra['_capture_stamp'] = 8.
    if support_state != 'missing':
        session._last_fresh.append(green)
    session._support_observed = lambda *a, **kw: SimulationSession._support_observed(session, *a, **kw)
    target = dict(label='red', required_support=dict(label='green', length=.05, width=.05, height=.05))
    assert session._reusable_target(target) is (support_state == 'valid')


def execution_session(frames, initial, *, preconfirmed=False):
    session, seen = scan_session(frames)
    observe = session.runner._observe
    def capture(view):
        status, result, frame = observe(view)
        frame.depth = None
        return status, result, frame
    session.runner._observe = capture
    session._status = 'ready'
    session._scene_version = 1
    session._observed_detections = copy.deepcopy(initial)
    session._last_fresh = copy.deepcopy(initial)
    session._detections = {'upper': copy.deepcopy(initial[0])}
    session._scene.update(scene_version=1, class_to_zone={'apple': 'zone_red', 'green': 'zone_green'},
                          objects=[d.summary() for d in initial])
    session._placement = {'table_bounds': {'x': [-.3, .9], 'y': [-.7, .7]}}
    session._physics = object()
    session._last_observation_purpose = 'recover_queued_target' if preconfirmed else 'search'
    session.runner.class_to_zone = session._scene['class_to_zone']
    session.runner._tracks = []
    session.runner.planner = SimpleNamespace(plan=lambda d: SimpleNamespace(feasible=True))
    calls = []
    session.runner.pick_detection = lambda det, **kw: (calls.append((det, kw)) or (None, 'probe'))
    session._evaluation_object = lambda d: None
    session._refresh_placement_occupancy = lambda *a, **kw: None
    session._finish_attempt = lambda *a, **kw: {'ok': True}
    session.editor = SimpleNamespace(metadata=lambda: {'one_per_class': True})
    return session, seen, calls


@pytest.mark.parametrize('preconfirmed', [False, True])
def test_direct_stack_execute_refreshes_missing_support_before_dispatch(preconfirmed):
    apple, green = pair()
    session, seen, calls = execution_session(
        [[apple, clipped(green)], [clipped(apple), green], []],
        [apple, clipped(green)], preconfirmed=preconfirmed)
    result = session.execute(dict(object_id='upper', scene_version=1,
                                  destination=dict(type='on_object', label='green')))
    assert result['ok'] and seen == [0, 1] and len(calls) == 1
    assert calls[0][0].extra['_stack_support']['label'] == 'green'


def test_joint_preconfirmed_move_does_not_repeat_target_scan_or_discard_support():
    apple, green = pair()
    session, seen, calls = execution_session([[], [], []], [apple, green], preconfirmed=True)
    result = session.execute(dict(object_id='upper', scene_version=1,
                                  destination=dict(type='on_object', label='green')))
    assert result['ok'] and not seen and len(calls) == 1


@pytest.mark.parametrize('change', ['missing', 'revision', 'cancel'])
def test_direct_stack_execute_never_dispatches_without_current_joint_evidence(change):
    apple, green = pair()
    session, seen, calls = execution_session([[apple]] * 3, [apple, clipped(green)])
    original = session.runner._observe
    def observe(view):
        result = original(view)
        if change == 'revision':
            session._environment_revision += 1
        elif change == 'cancel':
            session.cancel_event.set()
        return result
    session.runner._observe = observe
    session._invalidate_scene = lambda reason: None
    result = session.execute(dict(object_id='upper', scene_version=1,
                                  destination=dict(type='on_object', label='green')))
    assert not result['ok'] and not calls
    assert result['error'] == {'missing': 'destination_not_observed',
                               'revision': 'stale_scene', 'cancel': 'aborted'}[change]


def test_semantic_move_passes_required_support_into_its_grasp_observation(monkeypatch):
    apple, green = pair()
    session, seen, calls = execution_session([[apple, green], [], []], [apple, clipped(green)])
    session.runner.has_held_recovery = False
    session.snapshot = lambda: dict(held_object=None, editor={})
    session._scene['objects'][0]['object_id'] = 'upper'
    executor = SkillExecutor(session, originals=[green.summary()])
    observed = []
    def fresh(label, purpose, support_label=None):
        observed.append((label, support_label))
        return apple, {'object_id': 'upper'}
    monkeypatch.setattr(executor, '_fresh_target', fresh)
    monkeypatch.setattr(executor, '_completed_move', lambda *a, **kw: None)
    monkeypatch.setattr(session, 'execute', lambda *a, **kw: {'ok': True})
    result = executor.execute(dict(skill='move', label='apple',
        destination=dict(type='goal', index=0)),
        [dict(label='apple', destination=dict(type='on_object', label='green'))])
    assert result['ok'] and observed == [('apple', 'green')]


def test_curved_support_is_rejected_before_any_grasp_observation():
    session = fresh_session()
    with pytest.raises(ValueError, match='平顶底座'):
        SkillExecutor(session)._fresh_target('red', 'recover_queued_target', support_label='apple')
    assert not session.observations


def test_direct_curved_support_request_does_not_scan_or_grasp():
    apple, green = pair()
    session, seen, calls = execution_session([[], [], []], [apple, green])
    result = session.execute(dict(object_id='upper', scene_version=1,
                                  destination=dict(type='on_object', label='apple')))
    assert not result['ok'] and result['error'] == 'invalid_destination'
    assert '平顶底座' in result['detail']
    assert not seen and not calls
