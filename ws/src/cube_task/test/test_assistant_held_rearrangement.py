"""Held rearrangements preserve approval, fresh identities and unfinished goals."""
import copy
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cube_task.assistant.destinations import placement_metadata, validate_destination
from cube_task.assistant.held_recovery import (
    _center, _shift_candidates, execute_held_recovery,
)
from cube_task.core.types import Detection, PickResult
from cube_task.perception.grasp import GraspPlanner
from cube_task.assistant.obstacle_recovery import _checked_placement
from test_assistant_batch import RecoveryClient, create
from test_assistant_recovery_approval import ConfirmedSession, pending as pending_approval
from test_runner_local_release import scene as runner_scene


def measured(label, xy):
    return Detection(label, 1., center_base=np.r_[xy, .05], top_z=.05,
                     height=.05, length=.05, width=.05, yaw=0.)


class TransactionSession:
    def __init__(self):
        geometry, _target, _grasp, _place = runner_scene()
        self.runner = SimpleNamespace(zones=geometry.zones,
            execute_held_transfer=Mock(return_value=(PickResult(
                True, stage='done', grasp_width=.05, detail={'holding': False}), '')))
        self._environment_revision = self._observation_revision = 1
        self._recovery_target = measured('red', [.2, .15])
        self._recovery_goal = {'label': 'red', 'zone_name': 'zone_red'}
        self._recovery_evaluation = object()
        self._active_placement = ('original-placement',)
        self._observed_detections = [measured('green', [.4, 0.])]
        self._last_fresh = []
        self.cancel_event = threading.Event()
        self.held = 'red'
        self.version = 1
        self.actions, self.observations, self.events = [], [], []
        self.blocker_valid = True
        self.fail_action = None
        self.hold_on_failure = False
        self.lose_target_after_shift = False
        self.cancel_after = None
        self.final_verified = True
        self.table = {'green': measured('green', [.4, 0.])}
        self.cfg = geometry.cfg
        self._placement = placement_metadata(self.cfg, [[.12, -.25], [.4, -.25],
                                                        [.4, .25], [.12, .25]])
        self.staging = {'type': 'point', 'xy': [.2, .15]}
        self._held_proposals = {'stage_rearrange_green': {
            'bundle': {}, 'staging': self.staging, 'goal': copy.deepcopy(self._recovery_goal),
            'obstacle': 'green', 'protected_goals': [{'label': 'green', 'zone_name': 'zone_red'}],
            'revision': 1}}

    @property
    def scene(self):
        objects = []
        for label, detection in self.table.items():
            item = detection.summary()
            item['object_id'] = label + '-' + str(self.version)
            objects.append(item)
        return {'scene_version': self.version, 'objects': objects,
                'zones': [{'name': zone.name, 'center': zone.center.tolist(),
                           'size': zone.size, 'yaw': zone.yaw} for zone in self.runner.zones.values()],
                'placement': self._placement}

    def _placement_options(self, destination):
        from cube_task.assistant.simulation import SimulationSession
        return SimulationSession._placement_options(self, destination)

    def _finish_attempt(self, pick_result, why, _object_id, _zone, target,
                        _evaluation, destination):
        if not pick_result.ok:
            return {'ok': False, 'holding': True, 'message': why or '暂放未完成'}
        self.held = None
        self.table[target.label] = measured(target.label, destination['xy'])
        if self.cancel_after == 'stage':
            self.cancel_event.set()
        return {'ok': True, 'holding': False, 'message': '暂放通过新相机核验'}

    def observe(self, target=None, purpose='search', require_grasp=False):
        assert not self.cancel_event.is_set(), 'cancelled transaction must not observe again'
        self.version += 1
        self.observations.append(purpose)
        self._last_fresh = copy.deepcopy(list(self.table.values()))
        if purpose == 'inspect_blocker' and not self.blocker_valid:
            for detection in self._last_fresh:
                if detection.label == 'green':
                    detection.valid = False
                    detection.reject_reason = 'touches_image_border'
        if purpose == 'rearrange_target' and target['label'] == 'red' and self.lose_target_after_shift:
            self._last_fresh = [d for d in self._last_fresh if d.label != 'red']
        self._observed_detections = copy.deepcopy(self._last_fresh)
        return self.scene

    def execute(self, decision, cancel_event=None):
        assert not self.cancel_event.is_set(), 'cancelled transaction must not move again'
        if cancel_event is not None and cancel_event.is_set():
            return {'ok': False, 'aborted': True, 'holding': bool(self.held),
                    'message': '任务已停止'}
        destination = decision.get('destination')
        if destination is not None:
            try:
                destination = validate_destination(destination, self.scene)
            except ValueError:
                return {'ok': False, 'holding': False, 'error': 'invalid_destination',
                        'message': '放置目标字段不符合约定'}
        label = decision['object_id'].split('-')[0]
        self.actions.append(copy.deepcopy(decision))
        if self.fail_action == label:
            if self.hold_on_failure:
                self.held = label
                self.table.pop(label, None)
            return {'ok': False, 'holding': self.hold_on_failure, 'label': label,
                    'message': '本步执行失败'}
        xy = (destination['xy'] if destination else
              self.runner.zones[decision['zone_name']].center[:2])
        self.table[label] = measured(label, xy)
        # Normal execute overwrites these; the outer transaction must restore them.
        self._recovery_target = copy.deepcopy(self.table[label])
        self._recovery_goal = {'label': label, 'destination': destination}
        if self.cancel_after == label:
            self.cancel_event.set()
        return {'ok': True, 'holding': False, 'label': label,
                'message': '本步相机核验成功', 'verified_revision': 1,
                'truth': {'success': True, 'object_name': 'evaluated-' + label},
                'camera_verified': True, 'controller_ok': True}

    def verify_goals(self, goals):
        self.checked_goals = copy.deepcopy(goals)
        return {'ok': self.final_verified, 'items': []}

    def snapshot(self):
        return {'held_object': self.held}

    def _emit(self, kind, fields):
        self.events.append((kind, copy.deepcopy(fields)))

    def _publish_snapshot(self):
        pass


@pytest.fixture
def transaction(monkeypatch):
    session = TransactionSession()
    # State-machine tests keep motion deterministic. Candidate geometry is
    # exercised independently below with the real strict placement planners.
    monkeypatch.setattr('cube_task.assistant.held_recovery._checked_placement',
                        lambda *args: None)
    monkeypatch.setattr('cube_task.assistant.held_recovery._shift_candidates',
        lambda *args: [{'destination': {'type': 'point', 'xy': [.44, 0.]},
                       'origin': [.4, 0.], 'zone': 'zone_red'}])
    return session


def execute(session, confirmed=True):
    return execute_held_recovery(session, 'stage_rearrange_green', confirmed=confirmed)


@pytest.mark.parametrize('confirmation', [False, None, 1, 'yes'])
def test_no_motion_or_consumption_without_explicit_human_confirmation(transaction, confirmation):
    with pytest.raises(ValueError, match='人工同意'):
        execute(transaction, confirmation)
    transaction.runner.execute_held_transfer.assert_not_called()
    assert not transaction.actions and transaction._held_proposals


def test_completed_transaction_regrasps_original_named_goal_and_preserves_other_goals(transaction):
    original_target = transaction._recovery_target
    original_evaluation = transaction._recovery_evaluation
    result = execute(transaction)
    assert result['ok'], result
    assert [action['object_id'].split('-')[0] for action in transaction.actions] == ['green', 'red']
    assert transaction.actions[-1].get('zone_name') == 'zone_red'
    assert transaction.actions[-1].get('destination') is None
    assert transaction._recovery_goal == {'label': 'red', 'zone_name': 'zone_red'}
    np.testing.assert_array_equal(transaction._recovery_target.center_base, original_target.center_base)
    assert transaction._recovery_evaluation is original_evaluation
    assert {'label': 'green', 'zone_name': 'zone_red'} in transaction.checked_goals
    assert result['rearranged_objects'][0]['approved_final_destination']['xy'] == [.44, 0.]
    assert not result.get('pending_restore')  # Same-region relocation was explicitly approved.


@pytest.mark.parametrize('when', ['stage', 'green'])
def test_cancel_after_staging_or_rearrangement_preserves_obligations_and_starts_no_next_move(
        transaction, when):
    transaction.cancel_after = when
    result = execute(transaction)
    assert not result['ok'] and result['aborted']
    pending = result['pending_rearrangement']
    assert pending['target_staged']
    assert pending['goal'] == {'label': 'red', 'zone_name': 'zone_red'}
    assert len(transaction.actions) == (0 if when == 'stage' else 1)
    assert len(pending['adjustment_attempts']) == (0 if when == 'stage' else 1)
    assert transaction._recovery_goal == pending['goal']


def test_invalid_fresh_obstacle_never_becomes_a_grasp_input(transaction):
    transaction.blocker_valid = False
    result = execute(transaction)
    assert not result['ok'] and result['pending_rearrangement']['target_staged']
    assert '新相机图像' in result['message']
    assert not transaction.actions


def test_failed_staging_keeps_held_target_and_never_starts_observation_or_regrasp(transaction):
    transaction.runner.execute_held_transfer.return_value = (PickResult(
        False, error='rejected', detail={'holding': True}), '暂放路径拒绝')
    result = execute(transaction)
    assert not result['ok'] and result['holding']
    assert result['pending_rearrangement']['target_location_status'] == 'held'
    assert not result['pending_rearrangement']['target_staged']
    assert not transaction.observations and not transaction.actions


def test_stale_proposal_cannot_start_a_held_release(transaction):
    transaction._environment_revision += 1
    with pytest.raises(ValueError, match='过期'):
        execute(transaction)
    transaction.runner.execute_held_transfer.assert_not_called()
    assert not transaction.actions


def test_current_capacity_skips_unneeded_obstacle_move(transaction, monkeypatch):
    monkeypatch.setattr('cube_task.assistant.held_recovery._checked_placement',
                        lambda *args: {'xy': [.35, 0.], 'yaw': 0.})
    shift = Mock(side_effect=AssertionError('a directly feasible goal needs no obstacle shift'))
    monkeypatch.setattr('cube_task.assistant.held_recovery._shift_candidates', shift)
    result = execute(transaction)
    assert result['ok'], result
    assert [action['object_id'].split('-')[0] for action in transaction.actions] == ['red']
    assert not result['rearranged_objects']
    shift.assert_not_called()


def test_external_cancel_during_candidate_planning_prevents_next_observation_or_pick(
        transaction, monkeypatch):
    external = threading.Event()

    def cancel_during_planning(*_args):
        external.set()
        return [{'destination': {'type': 'point', 'xy': [.44, 0.]},
                 'origin': [.4, 0.], 'zone': 'zone_red'}]

    monkeypatch.setattr('cube_task.assistant.held_recovery._shift_candidates',
                        cancel_during_planning)
    result = execute_held_recovery(transaction, 'stage_rearrange_green',
                                   cancel_event=external, confirmed=True)
    assert not result['ok'] and result['aborted']
    assert not transaction.actions
    assert transaction.observations == ['confirm_staged_target', 'inspect_blocker']


@pytest.mark.parametrize('failed', ['green', 'red'])
def test_failed_step_does_not_complete_goal_or_continue_transaction(transaction, failed):
    transaction.fail_action = failed
    result = execute(transaction)
    assert not result['ok'] and result['pending_rearrangement']['target_staged']
    assert len(transaction.actions) == (1 if failed == 'green' else 2)
    adjustments = result['pending_rearrangement']['adjustment_attempts']
    assert len(adjustments) == 1
    assert adjustments[0]['status'] == ('unconfirmed' if failed == 'green' else 'verified')


def test_final_verification_must_protect_previously_completed_goals(transaction):
    transaction.final_verified = False
    result = execute(transaction)
    assert not result['ok'] and result['pending_rearrangement']['adjustment_attempts']
    assert {'label': 'green', 'zone_name': 'zone_red'} in transaction.checked_goals
    assert result['final_goal_verification']['ok'] is False
    assert result['pending_rearrangement']['target_location_status'] != 'confirmed_staging'


def test_failed_regrasp_reports_target_is_held_instead_of_still_at_staging(transaction):
    transaction.fail_action = 'red'
    transaction.hold_on_failure = True
    result = execute(transaction)
    assert not result['ok'] and result['holding']
    assert result['pending_rearrangement']['target_location_status'] == 'held'


def test_lost_original_target_does_not_reuse_the_obstacles_success_metadata(transaction):
    transaction.lose_target_after_shift = True
    result = execute(transaction)
    assert not result['ok'] and result['pending_rearrangement']['target_staged']
    assert [action['object_id'].split('-')[0] for action in transaction.actions] == ['green']
    assert result['label'] == 'red'
    assert result.get('camera_verified') is not True
    assert (result.get('truth') or {}).get('object_name') != 'evaluated-green'


def geometric_session():
    runner, target, _grasp, _place = runner_scene()
    runner.planner = GraspPlanner(runner.cfg, 'Sandbox')
    obstacle = measured('green', [.4, 0.])
    placement = placement_metadata(runner.cfg, [[.12, -.25], [.4, -.25], [.4, .25], [.12, .25]])
    session = SimpleNamespace(runner=runner, cfg=runner.cfg, _placement=placement,
        cancel_event=threading.Event(),
        _observed_detections=copy.deepcopy([target, obstacle]),
        scene={'placement': placement, 'zones': [{'name': 'zone_red', 'center': [.4, 0., 0.],
                                               'size': [.2, .2], 'yaw': 0.}]})
    from cube_task.assistant.simulation import SimulationSession
    session._placement_options = lambda destination: SimulationSession._placement_options(session, destination)
    return session, target, obstacle


def test_same_region_shift_candidates_pass_real_grasp_support_and_placement_geometry():
    session, target, obstacle = geometric_session()
    candidates = _shift_candidates(session, obstacle, target,
                                    {'type': 'named_region', 'name': 'zone_red'}, [])
    assert candidates
    assert candidates[0]['zone'] == 'zone_red'
    assert np.linalg.norm(np.array(candidates[0]['destination']['xy']) - obstacle.center_base[:2]) > .015
    assert candidates[0]['predicted'].valid


def test_direct_capacity_query_handles_cached_detection_copies_without_self_collision():
    session, target, obstacle = geometric_session()
    obstacle.center_base[:2] = [.47, .07]
    session._observed_detections = copy.deepcopy([target, obstacle])
    peers = [d for d in session._observed_detections if d.label != target.label] + [target]
    checked = _checked_placement(session, target,
                                  {'type': 'named_region', 'name': 'zone_red'}, peers)
    assert checked is not None


def test_same_region_shift_cannot_violate_a_completed_exact_point_goal():
    session, target, obstacle = geometric_session()
    protected = [{'label': 'green', 'destination': {'type': 'point', 'xy': [.4, 0.]}}]
    assert not _shift_candidates(session, obstacle, target,
        {'type': 'named_region', 'name': 'zone_red'}, protected)


def test_pick_area_region_uses_its_configured_polygon_center():
    session, _target, _obstacle = geometric_session()
    np.testing.assert_allclose(_center(session, {'type': 'region', 'name': 'pick_area'}), [.26, 0.])


@pytest.mark.parametrize('cancel', [False, True])
def test_pending_transaction_stops_batch_and_retains_its_record_after_approval(cancel):
    class PendingTransactionSession(ConfirmedSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.started = threading.Event()
            self.finish = threading.Event()

        def execute_recovery(self, chosen, cancel_event=None, confirmed=False):
            assert confirmed is True
            self.repairs.append(chosen)
            self.started.set()
            if cancel:
                assert self.finish.wait(3)
            self.holding = False
            return {'ok': False, 'aborted': cancel_event.is_set(), 'holding': False,
                    'message': '目标已暂放，重排尚未完成', 'stage': 'held_rearrangement',
                    'pending_rearrangement': {'goal': {'label': 'apple', 'zone_name': 'zone_red'},
                        'target_staged': True, 'target_location_status': 'confirmed_staging',
                        'staging_destination': {'type': 'point', 'xy': [.24, -.08]},
                        'adjustment_attempts': []}}

    ctl = create(PendingTransactionSession, RecoveryClient())
    try:
        ctl.submit('依次抓苹果和梨')
        approval = pending_approval(ctl)
        ctl.approve_recovery({'request_id': approval['request_id'],
                             'scene_version': approval['scene_version'], 'approved': True})
        assert ctl._session.started.wait(2)
        if cancel:
            ctl.stop()
            ctl._session.finish.set()
        assert ctl.wait_idle(3)
        state = ctl.snapshot()
        assert not state['last_result']['ok']
        assert state['last_result']['pending_rearrangement']['target_staged']
        assert state['last_result']['recovery_history']
        assert len(ctl._session.executions) == 1 and len(ctl._session.repairs) == 1
        assert state['batch']['status'] == ('stopped' if cancel else 'failed')
        assert state['batch']['items'][1]['status'] == 'cancelled'
        if cancel:
            assert '尚未完成' in state['stage']
    finally:
        ctl._session.finish.set()
        ctl.close()
