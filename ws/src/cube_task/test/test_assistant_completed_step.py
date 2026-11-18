"""Human-completed semantic moves are sensed and skipped before any grasp."""
import copy
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.destinations import placement_metadata
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.core.types import Zone
from test_task_fakes import make_cfg, make_det


GOALS = [{'label': 'red', 'zone_name': 'zone_red'}]


class CameraSession:
    def __init__(self, measured):
        self.cfg = make_cfg()
        self.cfg.set('perception.table.expected_z', 0.)
        self._placement = placement_metadata(self.cfg, [[.12, -.25], [.4, -.25],
                                                       [.4, .25], [.12, .25]])
        self.runner = SimpleNamespace(has_held_recovery=False,
            class_to_zone={'red': 'zone_red', 'pear': 'zone_green'}, nominal={},
            zones={'zone_red': Zone('zone_red', np.array([.17, -.35, 0.]), (.2, .2)),
                   'zone_green': Zone('zone_green', np.array([.17, .35, 0.]), (.2, .2))},
            skill=SimpleNamespace(execution_guard=None))
        self._operation_lock = threading.RLock()
        self.cancel_event = threading.Event()
        self._environment_revision = self._observation_revision = 0
        self._interventions = []
        self.holding = None
        self.camera_measurement = copy.deepcopy(measured)
        self._last_fresh = [copy.deepcopy(measured)]
        self._observed_detections = copy.deepcopy(self._last_fresh)
        self.observations, self.commands, self.views = [], [], []
        self.audit_ok = True
        self.truth_xy = measured.center_base[:2].copy()
        self.evaluation = SimpleNamespace(name='independent-object', bottom_z=lambda: 0.)
        self.world = SimpleNamespace(held=None, zone_of=lambda obj: next(
            (z for z in self.runner.zones.values() if z.contains_xy(*self.truth_xy)), None))
        self._publish()

    def _check_open(self):
        pass

    def _publish(self):
        self.scene = dict(scene_version=self._environment_revision,
            placement=copy.deepcopy(self._placement),
            zones=[dict(name=z.name, center=z.center.tolist(), size=list(z.size), yaw=z.yaw)
                   for z in self.runner.zones.values()],
            objects=[dict(d.summary(), object_id='measured-'+str(i))
                     for i, d in enumerate(self._observed_detections)])

    def snapshot(self):
        return dict(held_object=self.holding, scene_version=self._environment_revision,
                    editor={'intervention_pending': len(self._interventions)})

    def observe(self, **kwargs):
        self.observations.append(copy.deepcopy(kwargs))
        self._last_observation_purpose = kwargs.get('purpose', 'search')
        self._last_fresh = [copy.deepcopy(self.camera_measurement)]
        self.truth_xy = self.camera_measurement.center_base[:2].copy()
        self._observation_revision = self._environment_revision
        self._observed_detections = copy.deepcopy(self._last_fresh)
        self._publish()
        return copy.deepcopy(self.scene)

    def execute(self, decision, **kwargs):
        self.commands.append(copy.deepcopy(decision))
        return dict(ok=True, stage='done', holding=False, camera_verified=True,
                    controller_ok=True, truth={'success': True})

    def _ensure_destination_view(self, destination):
        self.views.append(copy.deepcopy(destination))

    def _target_partial(self, *args):
        return False

    def _evaluation_object(self, measured):
        return self.evaluation if self.audit_ok else None

    def _verify_destination(self, obj, destination):
        return obj is self.evaluation and self.audit_ok


def move(destination=None, label='red'):
    return dict(id='m', skill='move', label=label,
                destination=destination or {'type': 'goal', 'index': 0})


def test_initially_completed_goal_sends_no_grasp_or_extra_observation():
    session = CameraSession(make_det('red', .17, -.35, size=.05))
    result = SkillExecutor(session).execute(move(), GOALS, expected_revision=0)
    assert result['ok'] and result['skipped'] and result['camera_verified']
    assert result['truth']['success'] and result['verified_revision'] == 0
    assert result['items'][0]['goal'] == GOALS[0]
    assert not session.commands and not session.observations


@pytest.mark.parametrize('destination,xy', [
    ({'type': 'named_region', 'name': 'zone_green'}, [.17, .35]),
    ({'type': 'region', 'name': 'pick_area'}, [.26, .10]),
    ({'type': 'point', 'xy': [.30, .12]}, [.30, .12]),
])
def test_side_pear_in_destination_is_recoverable_not_already_completed(destination, xy):
    measured = make_det('pear', *xy, size=.085)
    measured.extra['resting_pose'] = 'side'
    session = CameraSession(measured)
    assert not SkillExecutor(session)._at_destination(measured, destination, None)
    measured.extra['resting_pose'] = 'upright'
    assert SkillExecutor(session)._at_destination(measured, destination, None)


def test_named_zone_completion_cannot_skip_physical_pear_posture_audit():
    measured = make_det('pear', .17, .35, size=.085)
    measured.extra['resting_pose'] = 'upright'
    session = CameraSession(measured)
    audited = []
    session._physics = SimpleNamespace(resting_status=lambda name:
        (audited.append(name) or {'success': False, 'reason': 'pear_not_upright'}))
    result = SkillExecutor(session).check_completed(
        dict(id='pear', skill='move', label='pear',
             destination={'type': 'zone', 'name': 'zone_green'}),
        [{'label': 'pear', 'zone_name': 'zone_green'}])
    assert audited == ['independent-object']
    assert result['skipped'] and not result['ok']
    assert result['error'] == 'completion_not_verified'


def test_upright_round_pear_at_zone_edge_has_no_fictitious_rotated_square_corners():
    from cube_task.assistant.batch import whole_footprint_in_zone
    measured = make_det('pear', .1586, .3067, size=.085, yaw=-.5648)
    measured.extra['resting_pose'] = 'upright'
    session = CameraSession(measured)
    assert whole_footprint_in_zone(measured.summary(), session.scene, 'zone_green')
    # The complete circular belly, including clearance, still has to fit.
    measured.center_base[1] = .285
    assert not whole_footprint_in_zone(measured.summary(), session.scene, 'zone_green')


def test_independent_zone_audit_rejects_actual_far_side_outside_even_if_camera_center_is_inside():
    from cube_task.assistant.catalog import make_catalog_object
    from cube_task.assistant.simulation import SimulationSession
    pear = make_catalog_object('pear', 'audit-pear', [.3044, -.1197])
    zone = Zone('narrow', np.array([.274116, -.1197, 0.]), (.12, .20))
    session = SimpleNamespace(world=SimpleNamespace(held=None),
        runner=SimpleNamespace(zones={'narrow': zone}),
        _physics=SimpleNamespace(resting_status=lambda name: {'success': True}))
    assert not SimulationSession._verify_zone(session, pear, 'narrow')
    pear.place_upright(.274116, -.1197, 0., 0.)
    assert SimulationSession._verify_zone(session, pear, 'narrow')


def test_verify_wrapper_keeps_analytic_round_fruit_zone_audit():
    measured = make_det('pear', .17, .35, size=.085)
    session = CameraSession(measured)
    goal = {'label': 'pear', 'zone_name': 'zone_green'}
    session.verify_goals = lambda goals: {'ok': True, 'items': [
        {'ok': True, 'goal': goal, 'verified_revision': 0}]}
    session.world.zone_of = lambda obj: None
    session._verify_zone = lambda obj, name: name == 'zone_green'
    result = SkillExecutor(session).execute({'id': 'verify', 'skill': 'verify'}, [goal])
    assert result['ok'] and result['truth']['success']


def test_intervention_completed_target_is_observed_at_destination_without_grasp_planning():
    session = CameraSession(make_det('red', .3, 0., size=.05))
    session._environment_revision = 1
    session.camera_measurement = make_det('red', .17, -.35, size=.05)
    result = SkillExecutor(session).execute(move(), GOALS, expected_revision=1)
    assert result['ok'] and result['skipped']
    assert len(session.observations) == 1 and not session.commands
    query = session.observations[0]
    assert query['target']['xy'] == pytest.approx([.17, -.35])
    assert query['target']['label'] == 'red' and not query['target'].get('require_grasp')
    assert query['purpose'] == 'verify_place'


def test_current_observe_already_completed_an_intermediate_move_not_final_goal():
    session = CameraSession(make_det('red', .17, .35, size=.05))
    result = SkillExecutor(session).execute(move({'type': 'zone', 'name': 'zone_green'}), GOALS)
    assert result['ok'] and result['skipped']
    assert result['items'][0]['goal']['zone_name'] == 'zone_green'
    assert not session.commands and not session.observations


@pytest.mark.parametrize('destination,xy', [
    ({'type': 'region', 'name': 'pick_area'}, [.26, .12]),
    ({'type': 'point', 'xy': [.30, .12]}, [.30, .12]),
])
def test_completed_region_and_point_are_skipped(destination, xy):
    session = CameraSession(make_det('red', *xy, size=.05))
    result = SkillExecutor(session).execute(move(destination), GOALS)
    assert result['ok'] and result['skipped'] and not session.commands


@pytest.mark.parametrize('destination,measured', [
    ({'type': 'zone', 'name': 'zone_red'}, make_det('red', .255, -.35, size=.05)),
    ({'type': 'region', 'name': 'pick_area'}, make_det('red', .385, .10, size=.05)),
    ({'type': 'point', 'xy': [.30, .12]}, make_det('red', .33, .12, size=.05)),
])
def test_center_only_or_wrong_point_does_not_skip_and_uses_one_original_target_scan(destination, measured):
    session = CameraSession(measured)
    result = SkillExecutor(session).execute(move(destination), GOALS)
    assert result['ok'] and not result.get('skipped')
    assert len(session.commands) == 1 and len(session.observations) == 1
    assert session.observations[0]['target']['require_grasp']


def test_original_pose_requires_yaw_but_respects_square_axis_symmetry():
    original = make_det('red', .3, .12, size=.05, yaw=0.).summary()
    original['valid'] = True
    session = CameraSession(make_det('red', .3, .12, size=.05, yaw=math.pi/2))
    executor = SkillExecutor(session, originals=[original])
    result = executor.execute(move({'type': 'original', 'label': 'red'}), GOALS)
    assert result['ok'] and result['skipped'] and result['yaw_verified']
    session.camera_measurement = make_det('red', .3, .12, size=.05, yaw=.4)
    session.observe(target={'label': 'red'}, purpose='test-update')
    session.observations.clear()
    result = executor.execute(move({'type': 'original', 'label': 'red'}), GOALS)
    assert not result.get('skipped') and len(session.commands) == 1
    assert session.commands[0]['_restore_yaw'] == 0.


def test_old_complete_measurement_cannot_skip_a_target_moved_out_again():
    session = CameraSession(make_det('red', .17, -.35, size=.05))
    executor = SkillExecutor(session)
    assert executor.execute(move(), GOALS)['skipped']
    session._environment_revision = 1
    session.camera_measurement = make_det('red', .30, 0., size=.05)
    result = executor.execute(move(), GOALS, expected_revision=1)
    assert result['ok'] and not result.get('skipped') and len(session.commands) == 1


def test_held_target_can_never_be_reported_as_already_completed():
    session = CameraSession(make_det('red', .17, -.35, size=.05))
    session.holding = 'red'
    executor = SkillExecutor(session)
    assert executor.check_completed(move(), GOALS, expected_revision=0) is None
    result = executor.execute(move(), GOALS)
    assert not result['ok'] and result['error'] == 'holding_object'
    assert not session.commands and not session.observations


@pytest.mark.parametrize('kind', ['revision', 'pending', 'cancelled'])
def test_completion_precheck_keeps_revision_pending_intervention_and_stop_guards(kind):
    session = CameraSession(make_det('red', .17, -.35, size=.05))
    if kind == 'revision':
        session._environment_revision = 1
    elif kind == 'pending':
        session._interventions.append(('move', {}))
    else:
        session.cancel_event.set()
    result = SkillExecutor(session).check_completed(move(), GOALS, expected_revision=0)
    assert not result['ok'] and result['error'] == ('aborted' if kind == 'cancelled' else 'scene_changed')
    assert not session.commands and not session.observations


def test_scene_change_during_completion_observation_never_reports_success():
    session = CameraSession(make_det('red', .3, 0., size=.05))
    session._observation_revision = -1
    session.camera_measurement = make_det('red', .17, -.35, size=.05)
    observed = session.observe
    def intervention(**kwargs):
        result = observed(**kwargs)
        session._environment_revision += 1
        return result
    session.observe = intervention
    result = SkillExecutor(session).check_completed(move(), GOALS, expected_revision=0)
    assert not result['ok'] and result['error'] == 'scene_changed'
    assert not session.commands


def test_incomplete_geometry_or_raised_support_cannot_count_as_complete():
    measured = make_det('red', .17, -.35, size=.05, table_z=.10)
    session = CameraSession(measured)
    result = SkillExecutor(session).execute(move(), GOALS)
    assert not result.get('skipped') and len(session.commands) == 1


def test_independent_audit_disagreement_returns_failure_without_using_truth_to_grasp():
    session = CameraSession(make_det('red', .17, -.35, size=.05))
    session.audit_ok = False
    result = SkillExecutor(session).execute(move(), GOALS)
    assert not result['ok'] and result['skipped']
    assert result['camera_verified'] and result['truth']['success'] is not True
    assert result['error'] == 'completion_not_verified'
    assert not session.commands and not session.observations


def test_previous_same_revision_source_measurement_avoids_an_extra_destination_probe():
    session = CameraSession(make_det('red', .3, 0., size=.05))
    session._observed_detections[0].extra['_observed_revision'] = 0
    session._last_fresh = []
    result = SkillExecutor(session).execute(move(), GOALS)
    assert result['ok'] and not result.get('skipped')
    assert len(session.observations) == 1
    assert session.observations[0]['target']['require_grasp']


def test_completion_precheck_is_available_for_pick_place_composition_without_executing_it():
    session = CameraSession(make_det('red', .17, -.35, size=.05))
    result = SkillExecutor(session).check_completed(move(), GOALS, expected_revision=0)
    assert result['ok'] and result['skipped'] and not session.commands
    assert session.runner.skill.execution_guard is None


def test_target_newly_found_complete_in_the_grasp_observation_is_not_grasped():
    session = CameraSession(make_det('red', .3, 0., size=.05))
    session.camera_measurement = make_det('red', .17, -.35, size=.05)
    result = SkillExecutor(session).execute(move(), GOALS)
    assert result['ok'] and result['skipped'] and not session.commands
    assert len(session.observations) == 1
    assert session.observations[0]['target']['require_grasp']


def test_older_same_revision_completion_is_remeasured_before_skipping():
    session = CameraSession(make_det('red', .17, -.35, size=.05))
    session._observed_detections[0].extra['_observed_revision'] = 0
    session._last_fresh = []
    session.camera_measurement = make_det('red', .3, 0., size=.05)
    result = SkillExecutor(session).execute(move(), GOALS)
    assert result['ok'] and not result.get('skipped') and len(session.commands) == 1
    assert session.observations[0]['purpose'] == 'verify_place'


def test_invalidated_scene_cannot_reuse_an_old_fresh_completion():
    session = CameraSession(make_det('red', .17, -.35, size=.05))
    session.scene['objects'] = []
    session.camera_measurement = make_det('red', .3, 0., size=.05)
    result = SkillExecutor(session).execute(move(), GOALS)
    assert result['ok'] and not result.get('skipped') and len(session.commands) == 1
    assert session.observations[0]['purpose'] == 'verify_place'


def test_independent_audit_cannot_publish_success_after_pending_interference():
    session = CameraSession(make_det('red', .17, -.35, size=.05))
    def audit(measured):
        session._interventions.append(('move', {}))
        return session.evaluation
    session._evaluation_object = audit
    result = SkillExecutor(session).check_completed(move(), GOALS, expected_revision=0)
    assert not result['ok'] and result['error'] == 'scene_changed'
    assert not session.commands


def test_auto_free_has_no_fixed_postcondition_and_is_never_completion_skipped():
    session = CameraSession(make_det('red', .17, -.35, size=.05))
    result = SkillExecutor(session).check_completed(move({'type': 'auto_free'}), GOALS)
    assert result is None and not session.commands and not session.observations


def test_consumed_pre_action_measurements_cannot_confirm_or_disprove_completion():
    session = CameraSession(make_det('red', .17, -.35, size=.05))
    session._observed_detections[0].extra['_observed_revision'] = 0
    session._last_observation_purpose = 'consumed_for_pick'
    session.camera_measurement = make_det('red', .3, 0., size=.05)
    result = SkillExecutor(session).execute(move(), GOALS)
    assert result['ok'] and not result.get('skipped') and len(session.commands) == 1
    assert session.observations[0]['purpose'] == 'verify_place'


@pytest.mark.parametrize('one_per_class,old_revision,skipped', [
    (True, 0, True), (True, 1, False), (False, 0, False),
])
def test_destination_probe_forgets_only_stale_single_instance_source_memory(
        one_per_class, old_revision, skipped):
    from cube_task.assistant.observation_policy import merge_local_detections
    session = CameraSession(make_det('red', .3, 0., size=.05))
    session.editor = SimpleNamespace(metadata=lambda: {'one_per_class': one_per_class})
    session._environment_revision = 1
    session._observed_detections[0].extra['_observed_revision'] = old_revision
    session.camera_measurement = make_det('red', .17, -.35, size=.05)
    # Force an actual destination probe even if the current-revision source
    # would normally prove negative; this models a consumed pre-action image.
    session._last_observation_purpose = 'consumed_for_pick'
    observe = session.observe
    def retain_unseen_source(**kwargs):
        previous = copy.deepcopy(session._observed_detections)
        observe(**kwargs)
        session._observed_detections = merge_local_detections(previous,
            session._last_fresh, lambda d: False, forget=kwargs.get('forget', ()))
        session._publish()
        return copy.deepcopy(session.scene)
    session.observe = retain_unseen_source
    result = SkillExecutor(session).check_completed(move(), GOALS, expected_revision=1)
    assert bool(result and result.get('skipped')) == skipped
    assert not session.commands
    assert len(session.observations[0]['forget']) == int(one_per_class and old_revision == 0)
