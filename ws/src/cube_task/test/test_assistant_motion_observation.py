"""Evidence survives wrist motion, but never survives a changed object unseen."""
import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.motion_observation import MotionObservation
from cube_task.assistant.simulation import SimulationSession
from test_assistant_targeted_scan import block


def memory():
    session = SimpleNamespace(_environment_revision=0, _interventions=[],
        cancel_event=threading.Event(), _target_partial=lambda d, ref: not d.valid,
        _observed_detections=[], runner=SimpleNamespace(nominal={'pear': [.075, .075]}),
        _placement={'table_bounds': {'x': [.05, .65], 'y': [-.5, .5]}})
    return session, MotionObservation(session)


def test_changing_wrist_pose_does_not_invalidate_base_frame_measurements():
    session, observer = memory()
    target = block('apple')
    target.extra['_capture_flange'] = np.eye(4)
    observer.remember([target], 0)
    target.center_base[:] = 100.  # Evidence owns its immutable capture.
    np.testing.assert_allclose(observer.current('apple')[0].center_base, [.2, .3, .05])


def test_one_moved_object_does_not_invalidate_other_objects():
    session, observer = memory()
    observer.remember([block('apple'), block('pear')], 0)
    session._environment_revision = 1
    observer.invalidate(['pear'])
    assert observer.current('pear') == []
    assert observer.current('apple')[0].extra['_observed_revision'] == 1
    moved = block('pear', xy=(.17, -.35))
    observer.remember([moved], 1)
    assert not observer.invalidated
    np.testing.assert_allclose(observer.current('pear')[0].center_base[:2], [.17, -.35])


def test_crop_and_missing_view_cannot_reconfirm_an_invalidated_object():
    session, observer = memory()
    observer.remember([block('pear')], 0)
    session._environment_revision = 1
    observer.invalidate(['pear'])
    observer.remember([], 1)
    crop = block('pear'); crop.valid = False
    observer.remember([crop], 1)
    assert observer.current('pear') == []


def test_duplicate_current_identities_revoke_previous_unique_evidence():
    _session, observer = memory()
    observer.remember([block('apple')], 0)
    observer.remember([block('apple'), block('apple', xy=(.4, .3))], 0)
    assert observer.current('apple') == []


def test_unclassified_fragments_are_obstacles_without_a_fictitious_moved_identity():
    _session, observer = memory()
    observer.remember([block('unknown')], 0)
    observer.invalidate(['unknown'])
    assert not observer.evidence and not observer.invalidated
    assert not observer.uncertainty_obstacles()


@pytest.mark.parametrize('changed,shifted,stacked',
                         [(False,False,False),(True,False,False),(False,True,False),(False,False,True)])
def test_temporal_identity_requires_unchanged_coincident_body_geometry(changed, shifted, stacked):
    _session, observer = memory()
    observer.remember([block('red')], 0)
    if changed:
        observer.invalidate(['red'])
    fresh = block('apple', xy=(.3,.3) if shifted else (.2,.3))
    if stacked:
        fresh.top_z, fresh.height = .11,.06
    observer.reconcile([fresh], 0)
    assert fresh.valid
    assert fresh.label == ('apple' if changed or shifted or stacked else 'red')


def test_identity_tracking_cannot_turn_an_incomplete_crop_into_a_complete_target():
    _session, observer = memory()
    observer.remember([block('red')], 0)
    crop = block('apple'); crop.width = .02
    observer.reconcile([crop], 0)
    assert not crop.valid and crop.label == 'apple'


def test_a_release_hint_only_labels_new_complete_pixels_and_expires_on_change():
    session, observer = memory()
    hint = block('red'); hint.valid = False; hint.extra['_identity_revision'] = 0
    session._released_observation_hints = {'red':hint}
    assert not observer.current()
    fresh = block('apple')
    observer.reconcile([fresh], 0)
    assert fresh.valid and fresh.label == 'red'
    assert not observer.current()  # Association alone does not publish evidence.
    observer.invalidate(['red'])
    fresh = block('apple');observer.reconcile([fresh], 0)
    assert fresh.label == 'apple'


def test_distant_classification_cannot_rebind_an_unchanged_identity():
    _session, observer = memory()
    observer.remember([block('red')], 0)
    wrong = block('red', xy=(.4,-.3))
    observer.reconcile([wrong], 0)
    assert not wrong.valid
    observer.invalidate(['red'])
    moved = block('red', xy=(.4,-.3))
    observer.reconcile([moved], 0)
    assert moved.valid


def test_a_capture_from_before_an_intervention_cannot_complete_a_new_goal():
    session, observer = memory()
    session._environment_revision = 1
    observer.invalidate(['pear'])
    observer.remember([block('pear', xy=(.17, -.35))], 0)
    assert observer.current('pear') == []


@pytest.mark.parametrize('pending', [True, False])
def test_cancel_or_queued_intervention_prevents_evidence_reuse(pending):
    session, observer = memory()
    observer.remember([block('apple')], 0)
    if pending:
        session._interventions.append({})
    else:
        session.cancel_event.set()
    assert observer.current() == []


@pytest.mark.parametrize('reused', [True, False])
def test_motion_sample_rejects_old_pixels_at_the_current_wrist_pose(reused):
    session, observer = memory()
    session._prepared = True
    session.clock = SimpleNamespace(now=lambda: 10.)
    old = SimpleNamespace(depth=np.ones((2,2)), stamp=10.)
    returned = old if reused else SimpleNamespace(depth=np.ones((2,2)), stamp=9.99)
    session.camera = SimpleNamespace(_last=old, get_frame=lambda **kw: returned)
    session.arm = SimpleNamespace(get_flange_pose=lambda: SimpleNamespace(value=np.eye(4)))
    assert observer.capture(force=True) is False
    assert observer.stats['frames'] == 0 and not observer.evidence


@pytest.mark.parametrize('stage,held,masked', [('',True,True), ('lift',False,True),
                                            ('',False,False), ('retreat',True,False)])
def test_motion_capture_keeps_held_body_out_of_obstacles_between_pick_and_place(
        stage, held, masked, monkeypatch):
    from cube_task.core.types import CameraIntrinsics, Frame
    session, observer = memory()
    target, obstacle = block('pear'), block('yellow')
    grasp = SimpleNamespace(target=target)
    frame = Frame(np.zeros((4,5,3), np.uint8), np.ones((4,5), np.float32),
                  CameraIntrinsics(100.,100.,2.,2.,5,4), 10.)
    session._prepared, session._stage = True, stage
    session._recovery_target = target
    session.clock = SimpleNamespace(now=lambda: 10.)
    session.camera = SimpleNamespace(_last=None, get_frame=lambda **kw: frame)
    session.arm = SimpleNamespace(get_flange_pose=lambda: SimpleNamespace(value=np.eye(4)),
                                  get_tcp_pose=lambda: SimpleNamespace(value=np.eye(4)))
    session.runner.ext = SimpleNamespace(parent_frame='flange', T=np.eye(4))
    session.runner.T_flange_tcp = np.eye(4)
    session.runner.zones = {}
    session.runner.skill = SimpleNamespace(active_grasp=grasp, _read_width=lambda: .05)
    session.runner._held_recovery_context = dict(det=target, grasp=grasp) if held else None
    session.runner.perception = SimpleNamespace(process=lambda *a: SimpleNamespace(
        detections=[target, obstacle], warnings=[]))
    session.editor = SimpleNamespace(metadata=lambda: dict(one_per_class=True))
    session._scan_stats = dict(frames=0)
    session._cache_observation = session._publish_scene = session._emit = lambda *a, **kw: None
    masks = []
    def mask(*args, **kwargs):
        masks.append(args)
        return frame
    monkeypatch.setattr('cube_task.assistant.carried_observation.mask_carried_depth', mask)
    assert observer.capture(force=True)
    assert bool(masks) is masked
    assert bool(observer.current('pear')) is not masked
    assert observer.current('yellow')


def test_axis_symmetric_upright_pear_keeps_complementary_cloud_despite_pca_yaw():
    _session, observer = memory()
    pear = block('pear', top=.11, height=.11)
    points = np.array([[.18+i*.003,.28+j*.003,.04+k*.003]
                       for i in range(5) for j in range(5) for k in range(2)])
    pear.extra.update(resting_pose='upright', _placement_points_base=points, _cloud_views=2)
    observer.remember([pear], 0)
    new = copy.deepcopy(pear)
    new.yaw = 1.2
    new.extra.pop('_placement_points_base')
    new.extra['_cloud_views'] = 1
    observer.remember([new], 0)
    assert observer.current('pear')[0].extra['_cloud_views'] >= 2


def test_unobserved_moved_body_becomes_exclusion_volume_never_a_position():
    session, observer = memory()
    observer.invalidate(['pear'])
    volume = observer.uncertainty_obstacles()[0]
    assert not volume.valid and volume.label == 'unobserved_pear'
    assert volume.length == pytest.approx(.6) and volume.width == pytest.approx(1.)
    assert volume.top_z >= .095
    assert observer.current() == []
    assert observer.uncertainty_obstacles(exclude=['pear']) == []


def test_flat_long_fruit_does_not_use_its_length_as_unknown_obstacle_height():
    _session, observer = memory()
    observer.invalidate(['banana'])
    volume = observer.uncertainty_obstacles()[0]
    assert volume.top_z == pytest.approx(.06)  # 40 mm body plus uncertainty.


def test_live_publish_keeps_execution_status_and_stage():
    session = object.__new__(SimulationSession)
    session._status, session._stage = 'executing', 'lift'
    session._scene_version = 0
    session._placement = {}
    session._scan_stats = {}
    session.runner = SimpleNamespace(class_to_zone={})
    session._zones = lambda: []
    session._publish_snapshot = session._emit = lambda *a, **k: None
    session.backend = 'mujoco'
    scene = session._publish_scene([], [], [], 1., preserve_status=True)
    assert session._status == 'executing' and session._stage == 'lift'
    assert scene['observed_at'] == 1.


def test_lift_revokes_source_evidence_before_any_object_motion():
    session, observer = memory()
    target = block('apple')
    observer.remember([target, block('green')], 0)
    session._motion_observer = observer
    session._recovery_target = target
    session._publish_snapshot = session._emit = lambda *a, **k: None
    observer.frames.append(('measured background', np.eye(4), 0))
    SimulationSession._on_stage(session, 'lift')
    assert observer.current('apple') == []
    assert len(observer.current('green')) == 1
    assert len(observer.frames) == 1
    observer.invalidate(['green'])  # An external move still revokes old rays.
    assert not observer.frames


def test_invalidated_source_cannot_fall_back_to_legacy_same_pose_cache():
    session = SimpleNamespace(_motion_observer=SimpleNamespace(source=lambda ref: None))
    assert not SimulationSession._reusable_target(session, {'label': 'apple'})


def test_human_completed_target_is_skipped_from_incremental_evidence_without_scan():
    from cube_task.assistant.skill_execution import SkillExecutor
    from test_assistant_completed_step import CameraSession, move, GOALS
    measured = block('red', xy=(.17, -.35))
    session = CameraSession(measured)
    session._environment_revision = 1
    session._last_observation_purpose = 'consumed_for_pick'
    observer = session._motion_observer = MotionObservation(session)
    observer.remember([measured], 1)
    executor = SkillExecutor(session)
    result = executor._completed_move(move(), {'type': 'named_region', 'name': 'zone_red'}, None, None)
    assert result['ok'] and result['skipped'] and result['camera_verified']
    assert session.observations == [] and session.commands == []


def test_intervention_notice_alone_does_not_prove_a_target_is_finished():
    from cube_task.assistant.skill_execution import SkillExecutor
    from test_assistant_completed_step import CameraSession
    session = CameraSession(block('red', xy=(.17, -.35)))
    observer = session._motion_observer = MotionObservation(session)
    observer.remember(session._last_fresh, 0)
    session._environment_revision = 1
    observer.invalidate(['red'])
    assert SkillExecutor(session)._current_measurement('red') is None


def test_named_destination_retries_prior_release_refusal_with_strict_zone(tmp_path, monkeypatch):
    with SimulationSession(realtime=False, log_dir=tmp_path) as session:
        scene = session.prepare()
        obj = next(o for o in scene['objects'] if o['valid'] and o['label'] == 'red')
        session.runner._tracks = [dict(label='red', xy=obj['center_base'][:2],
            blacklisted=True, reason='unreachable release: y needs workspace margin')]
        calls = []
        def plan(det, **options):
            assert not session.runner._tracks
            assert options['placement_zone'].name == 'zone_red'
            np.testing.assert_allclose(options['placement_polygon'],
                                       session.runner.zones['zone_red'].corners_xy())
            calls.append(det)
            return None, 'diagnostic_plan_only'
        monkeypatch.setattr(session.runner, 'pick_detection', plan)
        monkeypatch.setattr(session, '_finish_attempt', lambda *a: {'ok': False})
        session.execute(dict(action='pick', object_id=obj['object_id'],
            scene_version=scene['scene_version'], zone_name='zone_red'))
        assert len(calls) == 1


def test_enlarged_cards_match_compiled_physics_and_survive_rebuild(tmp_path):
    pytest.importorskip('mujoco')
    with SimulationSession(realtime=False, backend='mujoco', log_dir=tmp_path) as session:
        for _ in range(2):
            for zone in session.world.zones:
                assert tuple(zone.size) == (.30, .30)
                assert tuple(session.runner.zones[zone.name].size) == (.30, .30)
                assert tuple(session.info['zones'][zone.name].size) == (.30, .30)
                assert tuple(session._physics.mujoco_world.zones[zone.name].size) == (.30, .30)
            session._physics._build()


def test_streaming_scene_updates_dashboard_without_replacing_motion_status():
    from cube_task.assistant.server import DashboardController
    controller = object.__new__(DashboardController)
    controller._lock = threading.RLock()
    controller._stage = '移动到放置区上方'
    controller._status = 'executing'
    events = []
    controller._event = lambda *a: events.append(a)
    scene = dict(incremental=True, scene_version=8, objects=[{'label': 'apple'}])
    controller._on_sim_event('scene', scene)
    assert controller._scene == scene and controller._scene is not scene
    assert controller._stage == '移动到放置区上方' and not events
    controller._on_sim_event('motion_observation', {'message': '观测更新'})
    assert controller._stage == '移动到放置区上方' and not events
