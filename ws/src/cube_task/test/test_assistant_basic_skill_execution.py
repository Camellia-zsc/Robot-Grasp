"""Independent offline tests of actual pick-only and measured held placement."""
import copy
import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.core.types import MotionResult, PickResult
from test_task_fakes import make_det
from test_task_skills import _setup


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('skill execution tests must not access any API/network')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def test_pick_only_stops_after_lift_and_keeps_measured_grasp():
    _cfg, _clock, world, arm, _log, skill, grasp, _place = _setup()
    result = skill.pick_only(grasp)
    assert result.ok and result.stage == 'holding' and result.detail['holding']
    assert result.grasp_width == pytest.approx(.03)
    assert [s['stage'] for s in result.detail['steps']] == ['open', 'pre', 'descend', 'close', 'lift']
    assert len(arm.gripper_cmds()) == 2
    assert world.held is not None and not world.placed
    np.testing.assert_array_equal(arm.get_tcp_pose().value, grasp.T_lift)


def test_pick_only_lift_failure_retains_grasp_without_hidden_release():
    _cfg, _clock, world, arm, _log, skill, grasp, _place = _setup()
    arm.fail_fn = lambda kind, target, linear, index: (
        MotionResult(False, 'rejected') if index == 2 else None)
    result = skill.pick_only(grasp)
    assert not result.ok and result.error == 'rejected'
    assert result.detail['holding'] and world.held is not None
    assert len(arm.gripper_cmds()) == 2 and len(arm.motions()) == 3


def test_pick_only_requires_width_even_with_legacy_verification_disabled():
    _cfg, _clock, _world, arm, _log, skill, grasp, _place = _setup()
    skill.verify = False
    def lose_feedback(kind, target, linear, index):
        if index == 1:
            arm.no_width = True
    arm.fail_fn = lose_feedback
    result = skill.pick_only(grasp)
    assert not result.ok and result.error == 'no_gripper_feedback'
    assert result.detail['holding'] and result.detail['fault']
    assert len(arm.motions()) == 2


def test_real_pick_then_place_to_goal_uses_saved_grip_and_no_second_close(tmp_path):
    goals = [{'label': 'red', 'zone_name': 'zone_red'}]
    unchanged = copy.deepcopy(goals)
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False) as session:
        scene = session.prepare()
        executor = SkillExecutor(session, originals=scene['objects'])
        frames = session.camera.frames_rendered
        picked = executor.execute({'id': 'p', 'skill': 'pick', 'label': 'red'}, goals)
        assert picked['ok'] and picked['holding'], picked
        assert session.camera.frames_rendered > frames
        context = session.runner._held_recovery_context
        assert context is not None and context['width'] > session.runner.skill.empty_width
        grip = context['grasp'].T_grasp.copy()
        target_center = context['det'].center_base.copy()
        assert context['zone'] == 'held'
        assert not session.runner._placed_labels
        assert session.snapshot()['held_object'] is not None
        stages = []
        session.on_event = lambda kind, value: stages.append(value.get('stage')) if kind == 'stage' else None
        placed = executor.execute({'id': 'q', 'skill': 'place', 'label': 'red',
                                   'destination': {'type': 'goal', 'index': 0}}, goals)
        assert placed['ok'] and placed['camera_verified'] and placed['truth']['success'], placed
        assert not placed['holding'] and not session.runner.has_held_recovery
        assert 'close' not in stages and 'descend' not in stages
        np.testing.assert_array_equal(context['grasp'].T_grasp, grip)
        np.testing.assert_array_equal(context['det'].center_base, target_center)
        assert goals == unchanged
        verified = executor.execute({'id': 'v', 'skill': 'verify', 'goal_index': 0}, goals)
        assert verified['ok'] and verified['camera_verified'] and verified['truth']['success'], verified


def test_held_observation_is_stationary_and_second_pick_is_refused(tmp_path):
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False) as session:
        session.prepare()
        executor = SkillExecutor(session)
        assert executor.execute({'id': 'p', 'skill': 'pick', 'label': 'red'}, [])['ok']
        joints = session.arm.get_joints().value.copy()
        frames = session.camera.frames_rendered
        observed = executor.execute({'id': 'o', 'skill': 'observe', 'label': 'yellow'}, [])
        assert observed['ok'] and observed['holding']
        np.testing.assert_array_equal(session.arm.get_joints().value, joints)
        assert session.camera.frames_rendered > frames
        refused = executor.execute({'id': 'p2', 'skill': 'pick', 'label': 'yellow'}, [])
        assert not refused['ok'] and refused['error'] == 'holding_object'
        np.testing.assert_array_equal(session.arm.get_joints().value, joints)


def test_dynamic_free_space_search_has_many_configured_candidates(tmp_path):
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False) as session:
        session.prepare()
        executor = SkillExecutor(session)
        target = next(d for d in session._observed_detections if d.label == 'red' and d.valid)
        points = executor._candidates(target, {'type': 'auto_free', 'away_from': ['yellow']},
                                      session._observed_detections)
        assert len(points) > 20
        assert len({tuple(p['xy']) for p in points}) == len(points)
        session._placement['table_bounds']['x'] = [.15, .35]
        changed = executor._candidates(target, {'type': 'auto_free'}, session._observed_detections)
        assert changed and changed != points
        assert all(.15 <= p['xy'][0] <= .35 for p in changed)


def test_real_pick_auto_free_and_repick_original_object_without_final_goal_mutation(tmp_path):
    goals = [{'label': 'red', 'zone_name': 'zone_green'}]
    unchanged = copy.deepcopy(goals)
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False) as session:
        original_scene = session.prepare()
        executor = SkillExecutor(session, originals=original_scene['objects'])
        assert executor.execute({'id': 'p', 'skill': 'pick', 'label': 'red'}, goals)['ok']
        staged = executor.execute({'id': 's', 'skill': 'place', 'label': 'red',
                                   'destination': {'type': 'auto_free', 'away_from': ['yellow']}}, goals)
        assert staged['ok'] and staged['truth']['success'] and staged['camera_verified'], staged
        assert staged['resolved_destination']['type'] == 'point'
        assert not session.runner.has_held_recovery and goals == unchanged
        repicked = executor.execute({'id': 'p2', 'skill': 'pick', 'label': 'red'}, goals)
        assert repicked['ok'] and repicked['holding'], repicked
        finished = executor.execute({'id': 's2', 'skill': 'place', 'label': 'red',
                                     'destination': {'type': 'goal', 'index': 0}}, goals)
        assert finished['ok'] and finished['truth']['actual_zone'] == 'zone_green', finished
        assert goals == unchanged
        restored = executor.execute({'id': 'r', 'skill': 'move', 'label': 'red',
                                     'destination': {'type': 'original', 'label': 'red'}}, goals)
        assert restored['ok'] and restored['camera_verified'] and restored['truth']['success'], restored
        original = next(obj for obj in original_scene['objects'] if obj['label'] == 'red' and obj['valid'])
        fresh = next(d for d in session._last_fresh if d.label == 'red' and d.valid)
        np.testing.assert_allclose(fresh.center_base[:2], original['center_base'][:2], atol=.015)
        # A cube's square silhouette has equivalent axes every quarter turn.
        assert abs(np.remainder(fresh.yaw - original['yaw'] + np.pi / 4, np.pi / 2) - np.pi / 4) < np.deg2rad(10)
        assert goals == unchanged


def test_cancelled_step_sends_no_commands(tmp_path):
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False) as session:
        session.prepare()
        executor = SkillExecutor(session)
        cancelled = threading.Event()
        cancelled.set()
        calls = session.arm.inner.n_calls
        result = executor.execute({'id': 'p', 'skill': 'pick', 'label': 'red'}, [], cancel_event=cancelled)
        assert not result['ok'] and result['error'] == 'aborted'
        assert session.arm.inner.n_calls == calls


@pytest.mark.parametrize('evaluation', ['missing', 'wrong', 'empty'])
def test_verify_never_finishes_on_unknown_false_or_empty_evidence(evaluation):
    goal = {'label': 'red', 'zone_name': 'zone_red'}
    measured = make_det('red', .3, .1)
    items = [] if evaluation == 'empty' else [{'goal': goal, 'label': 'red', 'ok': True}]
    session = SimpleNamespace(
        _operation_lock=threading.RLock(), _check_open=lambda: None,
        cancel_event=SimpleNamespace(is_set=lambda: False, external=None),
        runner=SimpleNamespace(has_held_recovery=False),
        verify_goals=lambda goals: {'ok': True, 'items': copy.deepcopy(items)},
        _observed_detections=[measured],
        _evaluation_object=lambda det: None if evaluation == 'missing' else object(),
        world=SimpleNamespace(held=None, zone_of=lambda obj: SimpleNamespace(name='zone_green')))
    result = SkillExecutor(session).execute({'id': 'v', 'skill': 'verify', 'goal_index': 0}, [goal])
    assert not result['ok']


def test_fresh_one_local_target_cannot_resolve_two_known_same_label_objects():
    measured = make_det('red', .3, .1)
    other = make_det('red', .4, -.1)
    scene = {'objects': [dict(measured.summary(), object_id='a'),
                         dict(other.summary(), object_id='b')]}
    session = SimpleNamespace(scene=scene, observe=lambda **kwargs: scene,
                              _last_fresh=[measured], _target_partial=lambda *args: False,
                              _observation_revision=1, _environment_revision=1)
    with pytest.raises(ValueError, match='歧义'):
        SkillExecutor(session)._fresh_target('red', 'test')


def test_failed_auto_free_move_chooses_another_point_until_scene_intervention(monkeypatch):
    placement = {'table_bounds': {'x': [.15, .5], 'y': [-.3, .3]},
                 'bounds': {'x': [.1, .6], 'y': [-.4, .4], 'r_min': .1, 'r_max': .7},
                 'margin': .002}
    target = make_det('red', .3, 0.)
    attempts = []
    def fail_move(decision, **kwargs):
        attempts.append(copy.deepcopy(decision['destination']))
        return {'ok': False, 'controller_ok': False, 'error': 'placement_not_observed'}
    session = SimpleNamespace(
        _operation_lock=threading.RLock(), _check_open=lambda: None,
        cancel_event=SimpleNamespace(is_set=lambda: False, external=None),
        _placement=placement, _environment_revision=0,
        scene={'placement': placement, 'scene_version': 1},
        _observed_detections=[target], snapshot=lambda: {},
        runner=SimpleNamespace(has_held_recovery=False, nominal={}), execute=fail_move)
    executor = SkillExecutor(session)
    executor._fresh_target = lambda *args: (target, {'object_id': 'measured-red'})
    def failed_late_place(self, action, goals, protected, cancel):
        candidate = self._candidates(target, action['destination'], [target])[0]
        attempts.append(copy.deepcopy(candidate))
        self._remember_failed_free(target.label, candidate)
        return {'ok': False, 'holding': False, 'error': 'placement_not_observed'}
    monkeypatch.setattr(SkillExecutor, '_pick', lambda *args: {'ok': True, 'holding': True})
    monkeypatch.setattr(SkillExecutor, '_place', failed_late_place)
    monkeypatch.setattr('cube_task.assistant.obstacle_recovery._checked_placement',
                        lambda *args, **kwargs: {'xy': [0., 0.]})
    step = {'id': 'm', 'skill': 'move', 'label': 'red', 'destination': {'type': 'auto_free'}}
    assert not executor.execute(step, [])['ok']
    assert not executor.execute(step, [])['ok']
    assert attempts[0] != attempts[1]
    # The exclusion survives executor reconstruction during a task resume.
    resumed = SkillExecutor(session)
    resumed._fresh_target = executor._fresh_target
    assert not resumed.execute(step, [])['ok']
    assert attempts[2] not in attempts[:2]
    session._environment_revision += 1
    assert not resumed.execute(step, [])['ok']
    assert attempts[3] == attempts[0]


def test_failed_auto_free_held_place_keeps_grip_and_skips_previous_point():
    placement = {'table_bounds': {'x': [.15, .5], 'y': [-.3, .3]},
                 'bounds': {'x': [.1, .6], 'y': [-.4, .4], 'r_min': .1, 'r_max': .7},
                 'margin': .002}
    target = make_det('red', .3, 0.)
    context = {'det': target, 'width': .03}
    attempts = []
    def plan(det, neighbors, **options):
        attempts.append(options['placement_zone'].center[:2].tolist())
        return {'det': target}, ''
    failed = PickResult(False, stage='inspect_place', error='placement_changed',
                        detail={'holding': True, 'steps': []})
    session = SimpleNamespace(
        _operation_lock=threading.RLock(), _check_open=lambda: None,
        cancel_event=SimpleNamespace(is_set=lambda: False, external=None),
        _placement=placement, _environment_revision=0,
        scene={'placement': placement, 'scene_version': 1},
        snapshot=lambda: {'held_object': 'red'}, _recovery_evaluation=None,
        _publish_snapshot=lambda: None,
        _finish_attempt=lambda *args: {'ok': False, 'holding': True},
        runner=SimpleNamespace(has_held_recovery=True, nominal={},
            _held_recovery_context=context, plan_held_transfer=plan,
            execute_held_transfer=lambda bundle: (failed, '')))
    executor = SkillExecutor(session)
    executor._stationary = lambda: []
    executor._options = lambda destination: dict(
        placement_zone=SimpleNamespace(name='temporary', center=np.r_[destination['xy'], 0.]))
    step = {'id': 'p', 'skill': 'place', 'label': 'red', 'destination': {'type': 'auto_free'}}
    assert not executor.execute(step, [])['ok']
    assert not executor.execute(step, [])['ok']
    assert attempts[0] != attempts[1]
    assert session.runner._held_recovery_context is context
