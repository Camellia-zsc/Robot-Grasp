"""Bounded relocation uses measured geometry and pays its restoration obligation."""
import copy
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant import obstacle_recovery as recovery
from cube_task.assistant.destinations import placement_metadata
from cube_task.assistant.simulation import SimulationSession
from cube_task.core.config import load_config
from cube_task.core.types import Detection
from cube_task.perception.grasp import GraspPlanner, PlacePlanner
from cube_task.task.runner import TaskRunner


def det(label, xy, yaw=0):
    return Detection(label, 1., center_base=np.array([*xy, .05]), top_z=.05, height=.05,
                     length=.05, width=.05, yaw=yaw, extra={'object_id': label+'-1'})


class TransactionSession:
    def __init__(self, failures=None):
        self._environment_revision = 0
        self.cancel_event = threading.Event()
        self.calls = []
        self.goal_verifications = []
        self.held = None
        self.failures = failures or {}
        self._scene = {'scene_version': 1, 'objects': [
            dict(det('banana', [.49, 0]).summary(), object_id='banana-1'),
            dict(det('yellow', [.54, -.05], .2).summary(), object_id='yellow-1')]}
        self.plan = {'goal': {'label': 'banana', 'zone_name': None,
                              'destination': {'type': 'region', 'name': 'pick_area'}},
                     'obstacle_label': 'yellow', 'original_xy': [.54, -.05],
                     'original_yaw': .2,
                     'temporary_destination': {'type': 'point', 'xy': [.2, -.18]},
                     'environment_revision': 0}
        self._clearance_proposals = {'clear_restore_yellow': {'id': 'clear_restore_yellow',
                                                             '_clearance': copy.deepcopy(self.plan)}}

    @property
    def scene(self):
        return copy.deepcopy(self._scene)

    def snapshot(self):
        return {'held_object': self.held}

    def verify_goals(self, goals):
        self.goal_verifications.append(copy.deepcopy(goals))
        return {'ok': True, 'items': [], 'message': '目标仍在指定区域'}

    def execute(self, decision, cancel_event=None):
        self.calls.append(copy.deepcopy(decision))
        call = len(self.calls)
        if call in self.failures:
            fail = self.failures[call]
            if isinstance(fail, Exception):
                raise fail
            self.held = 'held' if fail.get('holding') else None
            return copy.deepcopy(fail)
        obj = next(d for d in self._scene['objects'] if d['object_id'] == decision['object_id'])
        xy = decision.get('destination', {}).get('xy', [.3, .1])
        obj['center_base'][:2] = xy
        obj['yaw'] = decision.get('_restore_yaw', obj['yaw'])
        self._scene['scene_version'] += 1
        for observed in self._scene['objects']:
            observed['object_id'] = observed['label']+'-'+str(self._scene['scene_version'])
        return {'ok': True, 'label': obj['label'], 'stage': 'done', 'holding': False,
                'destination': copy.deepcopy(decision.get('destination')),
                'verified_revision': self._environment_revision}


def test_transaction_rebinds_each_step_and_preserves_exact_original_goal():
    session = TransactionSession()
    result = recovery.execute_clearance(session, {'id': 'clear_restore_yellow'})
    assert result['ok'] and result['pending_restore'] is None
    assert [c['object_id'] for c in session.calls] == ['yellow-1', 'banana-2', 'yellow-3']
    assert [c['scene_version'] for c in session.calls] == [1, 2, 3]
    assert session.calls[1]['destination'] == session.plan['goal']['destination']
    assert session.calls[2]['destination'] == {'type': 'point', 'xy': [.54, -.05]}
    assert session.calls[0]['_restore_yaw'] == session.calls[2]['_restore_yaw'] == .2
    assert result['label'] == 'banana'  # Do not report restored yellow as original goal.
    assert result['target_result']['ok']
    assert [s['step'] for s in result['clearance_steps']] == ['move_obstacle', 'original_task', 'restore_obstacle']
    assert not session._clearance_proposals
    assert not recovery.execute_clearance(session, {'id': 'clear_restore_yellow'})['ok']
    assert len(session.calls) == 3


@pytest.mark.parametrize('failure', [{'ok': False, 'stage': 'grasp_miss', 'error': 'grasp_miss'},
                                     OSError('adapter disconnected')])
def test_failed_main_still_restores_obstacle_and_does_not_claim_goal_success(failure):
    session = TransactionSession({2: failure})
    result = recovery.execute_clearance(session, 'clear_restore_yellow')
    assert not result['ok'] and result['pending_restore'] is None
    assert len(session.calls) == 3
    assert session.calls[2]['object_id'].startswith('yellow-')
    assert not result['target_result']['ok']


def test_restore_failure_is_visible_even_when_original_target_succeeded():
    session = TransactionSession({3: {'ok': False, 'error': 'unreachable'}})
    result = recovery.execute_clearance(session, 'clear_restore_yellow')
    assert not result['ok'] and result['stage'] == 'obstacle_restore'
    assert result['target_result']['ok']
    assert result['pending_restore']['label'] == 'yellow'
    assert result['pending_restore']['destination']['xy'] == [.54, -.05]


def test_restore_requires_measured_yaw_and_position_after_execution():
    session = TransactionSession()
    execute = session.execute

    def wrong_yaw(decision, **kwargs):
        result = execute(decision, **kwargs)
        if len(session.calls) == 3:
            next(o for o in session._scene['objects'] if o['label'] == 'yellow')['yaw'] += .5
        return result

    session.execute = wrong_yaw
    result = recovery.execute_clearance(session, 'clear_restore_yellow')
    assert not result['ok'] and result['error'] == 'pending_restore'


def test_cancel_or_held_feedback_prevents_further_restore_motion():
    for state in ('cancel', 'held'):
        session = TransactionSession()
        execute = session.execute

        def interrupted(decision, **kwargs):
            result = execute(decision, **kwargs)
            if len(session.calls) == 1:
                if state == 'cancel':
                    session.cancel_event.set()
                else:
                    session.held = 'held-target'
            return result

        session.execute = interrupted
        result = recovery.execute_clearance(session, 'clear_restore_yellow')
        assert not result['ok'] and result['pending_restore'] is not None
        assert len(session.calls) == (1 if state == 'cancel' else 2)


def test_manual_change_expires_unexecuted_proposal():
    session = TransactionSession()
    session._environment_revision = 1
    result = recovery.execute_clearance(session, 'clear_restore_yellow')
    assert not result['ok'] and result['error'] == 'stale_clearance_option'
    assert not session.calls


def proposal_session():
    cfg = load_config(use_overlays=False)
    banana, yellow = det('banana', [.49, 0]), det('yellow', [.54, -.05], .2)
    scene = {'scene_version': 1, 'objects': [], 'placement': placement_metadata(
        cfg, [[.12, -.25], [.4, -.25], [.4, .25], [.12, .25]])}
    return SimpleNamespace(_observed_detections=[banana, yellow], _placement=scene['placement'],
                           scene=scene, _environment_revision=0,
                           runner=SimpleNamespace(zones={}))


def test_proposal_only_moves_a_neighbor_whose_removal_clears_the_target(monkeypatch):
    session = proposal_session()
    checks = []

    def grasp(_session, target, neighbors):
        candidate = copy.deepcopy(target)
        blocked = target.label == 'banana' and any(d.label == 'yellow' for d in neighbors)
        candidate.extra['grasp'] = {'blockers': [{'label': 'yellow'}]} if blocked else {}
        return candidate, SimpleNamespace(feasible=not blocked, reason='blocked_by_neighbor' if blocked else '')

    def placement(_session, target, destination, neighbors, desired_yaw=None):
        checks.append((target.label, copy.deepcopy(destination), desired_yaw))
        xy = destination.get('xy', [.3, .1])
        return {'xy': xy, 'yaw': target.yaw}

    monkeypatch.setattr(recovery, '_grasp', grasp)
    monkeypatch.setattr(recovery, '_checked_placement', placement)
    goal = {'label': 'banana', 'zone_name': None, 'destination': {'type': 'region', 'name': 'pick_area'}}
    options = recovery.propose_clearance(session, session._observed_detections[0], goal)
    assert len(options) == 1
    assert options[0]['id'] == 'clear_restore_yellow'
    assert '黄色方块' in options[0]['description']
    plan = options[0]['_clearance']
    assert plan['goal'] == goal
    assert np.linalg.norm(np.array(plan['temporary_destination']['xy']) - plan['original_xy']) >= .14
    assert len(checks) == 3  # Temporary placement, original task, restoration all checked.
    assert checks[2] == ('yellow', {'type': 'point', 'xy': [.54, -.05]}, .2)
    assert len(session._clearance_proposals) == 1


def test_no_unnecessary_relocation_when_narrow_grasp_already_works(monkeypatch):
    session = proposal_session()
    monkeypatch.setattr(recovery, '_grasp', lambda s, d, n: (d, SimpleNamespace(feasible=True)))
    assert recovery.propose_clearance(session, session._observed_detections[0],
                                     {'label': 'banana'}) == []


def test_unsafe_temporary_places_never_become_model_choices(monkeypatch):
    session = proposal_session()
    target = session._observed_detections[0]

    def grasp(s, d, neighbors):
        candidate = copy.deepcopy(d)
        blocked = d.label == 'banana' and len(neighbors) > 1
        candidate.extra['grasp'] = {'blockers': [{'label': 'yellow'}]}
        return candidate, SimpleNamespace(feasible=not blocked, reason='blocked_by_neighbor')

    monkeypatch.setattr(recovery, '_grasp', grasp)
    monkeypatch.setattr(recovery, '_checked_placement', lambda *a, **k: None)
    goal = {'label': 'banana', 'zone_name': None, 'destination': {'type': 'region', 'name': 'pick_area'}}
    assert recovery.propose_clearance(session, target, goal) == []
    assert not session._clearance_proposals


def test_real_strict_placer_refuses_occupied_point_and_full_footprint_beyond_edge():
    cfg = load_config(use_overlays=False)
    target = det('red', [.3, .16])
    obstacle = det('green', [.3, 0])
    runner = SimpleNamespace(cfg=cfg, arm=SimpleNamespace(guard=SimpleNamespace(x=(.08,.6), y=(-.4,.4), r_min=.1, r_max=.62)), nominal={'red': (.05, .05), 'green': (.05, .05)},
                             placer=PlacePlanner(cfg), planner=GraspPlanner(cfg, 'L1'),
                             zones={}, release_lower={}, free_tilt=0.)
    runner._footprint = lambda d: TaskRunner._footprint(runner, d)
    runner._strict_place_filter = lambda *a: TaskRunner._strict_place_filter(runner, *a)
    runner._floors = lambda *a: {'pre_grasp': .1, 'lift': .1, 'pre_place': .1, 'retreat': .1}
    runner._adjust_strict_release = lambda *a: (True, '')
    session = SimpleNamespace(runner=runner, cfg=cfg)
    session._placement = placement_metadata(cfg, [[.12, -.25], [.4, -.25], [.4, .25], [.12, .25]])
    session._placement_options = lambda d: SimulationSession._placement_options(session, d)
    assert recovery._checked_placement(session, target, {'type': 'point', 'xy': [.3, 0]}, [target, obstacle]) is None
    assert recovery._checked_placement(session, target, {'type': 'point', 'xy': [.3, -.15]}, [target, obstacle]) is not None
    assert recovery._checked_placement(session, target, {'type': 'point', 'xy': [.58, 0]}, [target, obstacle]) is None


def test_virtual_relocation_transforms_all_measured_geometry_and_drops_old_neighbors():
    original = det('yellow', [.5, 0])
    original.extra.update(_points_xy=np.array([[.51, 0]]),
                          _points_base=np.array([[.51, 0, .04]]),
                          _placement_points_base=np.array([[.51, 0, .04]]),
                          cap_center_xy=[.51, 0], stem_xy=[.51, 0],
                          _grasp_neighbors=[{'label': 'banana'}], _neighbor_xy=[[.4, 0]],
                          grasp={'blocked': True})
    moved = recovery._translated(original, [.2, .1], math.pi/2)
    for key in ('_points_xy', '_points_base', '_placement_points_base'):
        np.testing.assert_allclose(moved.extra[key][0][:2], [.2, .11])
        np.testing.assert_allclose(original.extra[key][0][:2], [.51, 0])
    for key in ('cap_center_xy', 'stem_xy'):
        np.testing.assert_allclose(moved.extra[key], [.2, .11])
    assert not {'_grasp_neighbors', '_neighbor_xy', 'grasp'} & moved.extra.keys()


def test_failed_first_move_with_held_blocker_cannot_use_old_table_xy_as_restoration():
    session = TransactionSession({1: {'ok': False, 'stage': 'inspect_place',
                                      'error': 'placement_changed', 'holding': True}})
    result = recovery.execute_clearance(session, 'clear_restore_yellow')
    assert not result['ok']
    assert result['holding']
    assert result['pending_restore']['label'] == 'yellow'
    assert result['target_result'] is None
    assert len(session.calls) == 1  # Must not issue another pick while holding.


def test_original_target_is_reverified_if_user_changes_scene_during_restoration():
    session = TransactionSession()
    execute = session.execute

    def changed_during_restore(decision, **kwargs):
        result = execute(decision, **kwargs)
        if len(session.calls) == 3:
            session._environment_revision += 1
        return result

    def target_missing(goals):
        session.goal_verifications.append(copy.deepcopy(goals))
        return {'ok': False, 'items': [], 'message': '香蕉已移出抓取区'}

    session.execute = changed_during_restore
    session.verify_goals = target_missing
    result = recovery.execute_clearance(session, 'clear_restore_yellow')
    assert not result['ok'] and result['stage'] == 'clearance_final_verify'
    assert result['pending_restore'] is None
    assert result['target_result']['ok']  # Preserve which earlier step succeeded.
    assert result['final_goal_verification']['ok'] is False
    assert session.goal_verifications == [[session.plan['goal']]]
    assert len(session.calls) == 3


def test_unchanged_transaction_does_not_repeat_destination_scan():
    session = TransactionSession()
    result = recovery.execute_clearance(session, 'clear_restore_yellow')
    assert result['ok'] and not session.goal_verifications
    assert 'final_goal_verification' not in result


def test_changed_scene_that_still_meets_goal_can_pass_final_local_confirmation():
    session = TransactionSession()
    execute = session.execute

    def changed_during_restore(decision, **kwargs):
        result = execute(decision, **kwargs)
        if len(session.calls) == 3:
            session._environment_revision += 1
        return result

    session.execute = changed_during_restore
    result = recovery.execute_clearance(session, 'clear_restore_yellow')
    assert result['ok']
    assert session.goal_verifications == [[session.plan['goal']]]
    assert result['verified_revision'] == 1
