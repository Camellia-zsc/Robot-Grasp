"""Cropped source measurements can aim views, never authorize manipulation."""
import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.core.types import CameraIntrinsics
from test_assistant_active_observation import fake_executor
from test_assistant_targeted_scan import block, scan_session


HINT = dict(label='pear', valid=False, center_base=[.3196, -.1542, .1334],
            top_z=.1334, height=.1334, length=.0753, width=.0347,
            reject_reason='touches_image_border', view=1)


def test_partial_current_measurement_is_only_an_optional_aiming_hint():
    session = SimpleNamespace(scene=dict(objects=[copy.deepcopy(HINT)]))
    old = dict(label='pear', valid=True, center_base=[.1, .1, .1])
    executor = SkillExecutor(session, originals=[old])
    assert executor._reference('pear') == old
    hint = executor._reference('pear', allow_partial=True)
    assert hint['center_base'] == HINT['center_base']
    assert hint['valid'] is False and hint['_observation_hint_only'] is True
    assert session.scene['objects'] == [HINT]


def test_ambiguous_partial_instances_do_not_invent_a_source_identity():
    duplicate = dict(HINT, center_base=[.45, -.1, .14])
    session = SimpleNamespace(scene=dict(objects=[HINT, duplicate]))
    assert SkillExecutor(session)._reference('pear', allow_partial=True) == dict(label='pear')


def test_unexplained_misclassified_crop_can_aim_missing_target_search_only():
    crop = dict(HINT, label='apple')
    apple = dict(label='apple', valid=True, center_base=[.16, -.35, .06])
    session = SimpleNamespace(scene={'objects': [crop, apple]},
                              editor=SimpleNamespace(metadata=lambda: {'one_per_class': True}))
    executor = SkillExecutor(session)
    assert executor._reference('red') == {'label': 'red'}
    hint = executor._reference('red', allow_partial=True)
    assert hint['valid'] is False and hint['_observation_hint_only']
    assert hint['center_base'] == crop['center_base']
    assert session.scene['objects'][0]['label'] == 'apple'
    session.scene['objects'].append(dict(crop, center_base=[.45, .2, .1]))
    assert executor._reference('red', allow_partial=True) == {'label': 'red'}


def test_moved_upper_retains_camera_aiming_hint_for_previously_hidden_support():
    upper = dict(label='pear', valid=True, center_base=[.3, -.1, .17],
                 support_observed=True, support_label='yellow', support_z=.05,
                 support_polygon=[[.28, -.12], [.31, -.12], [.31, -.09], [.28, -.09]])
    executor = SkillExecutor(SimpleNamespace(scene={'objects': []}), originals=[upper])
    hint = executor._reference('yellow', allow_partial=True)
    assert hint['valid'] is False and hint['_observation_hint_only'] is True
    np.testing.assert_allclose(hint['center_base'], [.295, -.105, .05])
    assert executor._reference('yellow') == {'label': 'yellow'}


def test_conflicting_support_patches_do_not_invent_a_base_location():
    upper = dict(label='pear', support_observed=True, support_label='yellow', support_z=.05,
                 support_polygon=[[.28, -.12], [.31, -.12], [.31, -.09]])
    other = dict(upper, support_polygon=[[.48, .12], [.51, .12], [.51, .15]])
    executor = SkillExecutor(SimpleNamespace(scene={'objects': [other]}), originals=[upper])
    assert executor._reference('yellow', allow_partial=True) == {'label': 'yellow'}


def test_source_label_with_location_precedes_its_different_placement_destination():
    executor = fake_executor()
    session = executor.session
    session.runner.has_held_recovery = False
    session.snapshot = lambda: {'held_object': None}
    session.scene['objects'] = [copy.deepcopy(HINT)]
    calls = []
    session.observe = lambda **kwargs: calls.append(copy.deepcopy(kwargs)) or session.scene
    session._ensure_destination_view = lambda destination: pytest.fail('must inspect measured source')
    result = executor._observe(dict(skill='observe', label='pear', mode='active',
        destination=dict(type='goal', index=0)), [dict(label='pear', zone_name='zone_green')])
    assert result['ok']
    target = calls[0]['target']
    assert target['center_base'] == HINT['center_base']
    assert target['top_z'] == HINT['top_z']
    assert target['_observe_source'] and target['_observation_hint_only']
    assert target['valid'] is False


def test_unknown_source_location_uses_the_explicit_region_as_search_fallback():
    executor = fake_executor()
    session = executor.session
    session.runner.has_held_recovery = False
    session.snapshot = lambda: {'held_object': None}
    observations, destinations = [], []
    session.observe = lambda **kwargs: observations.append(kwargs) or session.scene
    session._ensure_destination_view = destinations.append
    executor._observation_focus = lambda step, goals: (np.array([.49, 0.]), None)
    result = executor._observe(dict(skill='observe', label='yellow', mode='active',
        destination=dict(type='goal', index=0)), [dict(label='yellow', zone_name='zone_yellow')])
    assert result['ok']
    assert observations[0]['target'] == dict(xy=[.49, 0.], label='yellow')
    assert len(destinations) == 1


@pytest.mark.parametrize('complete', [False, True])
def test_source_scan_uses_new_views_even_when_complete_silhouette_lacks_grasp_views(complete):
    pear = block('pear', xy=(.335, -.17), top=.157, height=.106)
    session, seen = scan_session([[], [pear], [pear]])
    session.runner.views = [[0.] * 6 for _ in range(3)]
    session._source_observation_views = lambda reference: [1, 2]
    guarded = []
    session._safe_observation_joints = lambda q, detections: guarded.append(q) or True
    scene = session.observe(target=dict(HINT, valid=complete, _observe_source=True,
                                       _observation_hint_only=not complete),
                            purpose='skill_observe')
    assert seen == [1, 2] and len(guarded) == 2
    assert session._scan_stats == dict(local_scans=1, full_scans=0, frames=2)
    assert any(o['label'] == 'pear' and o['valid'] for o in scene['objects'])


def test_fresh_grasp_request_for_complete_but_unusable_source_selects_local_source_views():
    pear = block('pear', top=.157, height=.106)
    scene = {'objects':[pear.summary()]}
    calls = []
    session = SimpleNamespace(scene=scene, _last_fresh=[pear], _environment_revision=1,
        _observation_revision=1, _target_partial=lambda d,r:False,
        _reusable_target=lambda r:False,
        observe=lambda **kw:calls.append(kw) or scene)
    result, _summary = SkillExecutor(session)._fresh_target('pear','skill_pick')
    assert result is pear
    assert calls[0]['target']['_observe_source'] is True
    assert not calls[0]['target'].get('_observation_hint_only')


def test_source_view_route_is_rechecked_from_current_joints_before_dispatch():
    pear = block('pear', top=.157, height=.106)
    session, seen = scan_session([[], [pear], [pear]])
    session.runner.views = [[0.] * 6 for _ in range(3)]
    session._source_observation_views = lambda reference: [1, 2]
    checks = iter([True, False])
    session._safe_observation_joints = lambda q, detections: next(checks)
    session.observe(target=dict(HINT, _observe_source=True, _observation_hint_only=True),
                    purpose='skill_observe')
    assert seen == [1]  # A preplanned second viewpoint cannot bypass a changed route.


def test_unreachable_new_source_views_fall_back_to_two_local_fixed_views():
    partial = block('pear', top=.13, height=.13)
    partial.valid = False
    session, seen = scan_session([[partial], [partial], [partial]])
    session._source_observation_views = lambda reference: []
    session._safe_observation_joints = lambda q, detections: True
    session.runner.views = [[0.] * 6 for _ in range(3)]
    session.observe(target=dict(HINT, _observe_source=True, _observation_hint_only=True),
                    purpose='skill_observe')
    assert seen == [0, 1]
    assert not any(o['valid'] for o in session.scene['objects'])


def test_all_new_source_routes_rejected_at_dispatch_do_not_publish_camera_success():
    from cube_task.assistant.simulation import _SourceObservationUnavailable
    session, seen = scan_session([[], [], []])
    session.runner.views = [[0.] * 6 for _ in range(3)]
    session._source_observation_views = lambda reference: [1, 2]
    session._safe_observation_joints = lambda q, detections: False
    with pytest.raises(_SourceObservationUnavailable, match='未取得新的相机证据'):
        session.observe(target=dict(HINT, _observe_source=True, _observation_hint_only=True),
                        purpose='skill_observe')
    assert seen == [] and session._last_fresh == []
    assert session._observation_revision == -1


def test_executor_reports_no_safe_source_view_without_claiming_fresh_measurement():
    from cube_task.assistant.simulation import _SourceObservationUnavailable
    executor = fake_executor()
    session = executor.session
    session.runner.has_held_recovery = False
    session.snapshot = lambda: {'held_object': None}
    session.scene['objects'] = [copy.deepcopy(HINT)]
    def unavailable(**kwargs):
        raise _SourceObservationUnavailable('未取得新的相机证据')
    session.observe = unavailable
    result = executor._observe(dict(skill='observe', label='pear', mode='active'), [])
    assert result['ok'] is False and result['camera_verified'] is False
    assert result['error'] == 'no_safe_observation_path'


def source_camera_session():
    session = object.__new__(SimulationSession)
    session.cancel_event = threading.Event()
    session._interventions = []
    session._environment_revision = 1
    session._observed_detections = [block('pear', top=.133, height=.133)]
    session._base_view_count = 6
    session._view_labels = ['fixed'] * 6
    session.cfg = {'perception.table.expected_z': 0.}
    session.camera = SimpleNamespace(intrinsics=lambda: CameraIntrinsics(
        554., 554., 319.5, 239.5, 640, 480))
    calls, poses = [], {}
    def ik(pose, q_seed):
        calls.append(pose.copy())
        q = np.full(6, len(calls) * .05)
        poses[tuple(q)] = pose.copy()
        return q, 0., 0.
    session.arm = SimpleNamespace(kin=SimpleNamespace(ik=ik, fk=lambda q: poses[tuple(q)]),
        get_joints=lambda: SimpleNamespace(value=[0.] * 6))
    session.runner = SimpleNamespace(ext=SimpleNamespace(parent_frame='flange', T=np.eye(4)),
        T_flange_tcp=np.eye(4), views=[[0.] * 6 for _ in range(6)],
        _at_view=-1, _at_observe=False)
    session._safe_observation_joints = lambda q, detections: True
    return session, calls


def test_source_view_proposals_are_bounded_reused_and_independent_of_object_class():
    session, calls = source_camera_session()
    first = session._source_observation_views(HINT)
    assert first == [7, 8, 9, 10] and len(calls) == 4
    assert len(session.runner.views) == 11
    second = session._source_observation_views(dict(HINT, label='other_object'))
    assert second == first and len(session.runner.views) == 11
    assert len(calls) == 8  # No motion API exists on this proposal-only harness.


def test_source_view_proposals_require_guarded_full_paths_and_add_no_unsafe_slots():
    session, _calls = source_camera_session()
    session._safe_observation_joints = lambda q, detections: False
    assert session._source_observation_views(HINT) == []
    assert len(session.runner.views) == 6


def test_source_view_requires_actual_fk_coverage_in_addition_to_requested_camera_pose():
    session, _calls = source_camera_session()
    wrong = np.eye(4)
    wrong[:3, 3] = [2., 2., 2.]
    session.arm.kin.fk = lambda q: wrong
    session._safe_observation_joints = lambda q, detections: pytest.fail('cropped solution is not a route')
    assert session._source_observation_views(HINT) == []
    assert len(session.runner.views) == 6


@pytest.mark.parametrize('bad', [dict(top_z=float('nan')), dict(length=None),
                                  dict(center_base=[float('inf'), 0., .1]),
                                  dict(top_z=-.1)])
def test_invalid_aiming_geometry_does_not_create_camera_commands(bad):
    session, calls = source_camera_session()
    assert session._source_observation_views(dict(HINT, **bad)) == []
    assert calls == []


def test_cancelled_source_search_never_creates_executable_views():
    session, calls = source_camera_session()
    session.cancel_event.set()
    assert session._source_observation_views(HINT) == []
    assert not calls


@pytest.mark.parametrize('available', [True, False])
def test_curved_source_observation_waits_for_body_anchor_before_authorizing_grasp(monkeypatch, available):
    pear = block('pear', top=.1125, height=.1125)
    pear.extra['resting_pose'] = 'upright'
    session, seen = scan_session([[pear], [pear], [pear]])
    session._placement_cloud = lambda *args: np.ones((100,3))
    monkeypatch.setattr('cube_task.assistant.obstacle_recovery._grasp',
        lambda s,d,n:(copy.deepcopy(d),SimpleNamespace(feasible=True,target=copy.deepcopy(d))))
    fits = []
    def anchor(grasp):
        fits.append(True)
        return dict(ok=available and len(seen)>=3,error='sphere_surface_coverage_unconfirmed')
    monkeypatch.setattr('cube_task.assistant.held_anchor.source_anchor',anchor)
    scene = session.observe(target=dict(label='pear',xy=[.2,.3],require_grasp=True),purpose='recover_queued_target')
    assert seen == [0,1,2] and fits
    measured = next(o for o in scene['objects'] if o['label']=='pear')
    assert measured['valid'] is available
    if not available:assert 'source_anchor_unconfirmed' in measured['reject_reason']


def test_auto_free_move_keeps_semantic_choice_until_after_actual_pick(monkeypatch):
    executor=fake_executor()
    executor._held=lambda:None
    executor.session.snapshot=lambda:{}
    source=block('pear'); obj=dict(object_id='pear-source')
    original=dict(type='auto_free',away_from=['yellow'])
    executor._resolve=lambda *a:(copy.deepcopy(original),None)
    executor._completed_move=lambda *a,**k:None
    executor._fresh_target=lambda *a,**k:(source,obj)
    executor._candidates=lambda *a:[dict(type='point',xy=[.2,.1])]
    executor._check_plan=lambda:None
    executor._stopped=lambda *a:False
    executor.session._observed_detections=[]
    monkeypatch.setattr('cube_task.assistant.obstacle_recovery._checked_placement',lambda *a:object())
    calls=[]
    executor._pick=lambda *a:(calls.append('pick') or dict(ok=True,holding=True))
    def place(step,*args):
        calls.append('place')
        assert step['destination']==original
        return dict(ok=True,holding=False,resolved_destination=dict(type='point',xy=[.3,.1]))
    executor._place=place
    result=executor._move(dict(id='m',skill='move',label='pear',destination=original),[],[],None)
    assert result['ok'] and calls==['pick','place']
    assert result['resolved_destination']['xy']==[.3,.1]
