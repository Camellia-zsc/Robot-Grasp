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


def test_partial_source_scan_uses_new_views_and_stops_after_two_complete_fruit_views():
    pear = block('pear', xy=(.335, -.17), top=.157, height=.106)
    session, seen = scan_session([[], [pear], [pear]])
    session.runner.views = [[0.] * 6 for _ in range(3)]
    session._source_observation_views = lambda reference: [1, 2]
    guarded = []
    session._safe_observation_joints = lambda q, detections: guarded.append(q) or True
    scene = session.observe(target=dict(HINT, _observe_source=True, _observation_hint_only=True),
                            purpose='skill_observe')
    assert seen == [1, 2] and len(guarded) == 2
    assert session._scan_stats == dict(local_scans=1, full_scans=0, frames=2)
    assert any(o['label'] == 'pear' and o['valid'] for o in scene['objects'])


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
