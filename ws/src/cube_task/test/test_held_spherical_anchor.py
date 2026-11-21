"""A visible spherical part can correct translation without inventing an axis."""
import copy

import numpy as np

from cube_task.assistant.held_anchor import (apply_translation_anchor, held_anchor,
                                           seating_requires_feedback, source_anchor)
from cube_task.assistant.spherical_anchor import validate_anchor_pair
from cube_task.core.transforms import inv_T, top_down_T
from cube_task.core.types import Detection, GraspPlan, PickResult


def sphere_cloud(center, radius, seed):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(2500, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return np.asarray(center)+radius*directions+rng.normal(0., .00002, directions.shape)


def measured_grasp():
    center = np.array([.4, 0., .0425])
    points = sphere_cloud(center, .0425, 21)
    target = Detection('pear', 1., center_base=np.array([.4, 0., .112]),
        top_z=.112, height=.112, length=.085, width=.085, yaw=0.,
        extra={'resting_pose': 'upright', '_placement_points_base': points,
               '_stack_support': {'top_z': .05}})
    pose = top_down_T(.4, 0., .09, .5)
    return GraspPlan(target, pose.copy(), pose.copy(), pose.copy(), .1, 0., .048, .5)


def test_live_sphere_translation_calibrates_future_rigid_mapping_and_retains_command():
    grasp = measured_grasp()
    command = grasp.T_grasp.copy()
    source = source_anchor(grasp)
    assert source['ok']
    np.testing.assert_allclose(source['center'], [.4, 0., .0425], atol=.0001)
    local = np.array([.004, -.009, .061])
    fits = []
    for seed, position in ((32, [.3, .1, .2]), (43, [.32, .12, .22])):
        tcp = top_down_T(*position, 1.)
        center = (tcp @ np.r_[local, 1.])[:3]
        raw = Detection('pear', .9, valid=False, reject_reason='partial_object',
                        extra={'_points_base': sphere_cloud(center, .0425, seed)})
        fits.append(held_anchor([raw], grasp, source, tcp))
    pair = validate_anchor_pair(*fits, 1., 2.)
    assert pair['ok'] and pair['translation_only']
    updated = apply_translation_anchor(grasp, source, pair, 7)
    assert updated['ok'] and updated['changed'] and updated['translation_only']
    np.testing.assert_array_equal(grasp.T_grasp[:3, :3], command[:3, :3])
    np.testing.assert_allclose((inv_T(grasp.T_grasp) @ np.r_[source['center'], 1.])[:3],
                               local, atol=.0001)
    np.testing.assert_array_equal(grasp.target.extra['_executed_grasp_T'], command)
    assert not np.shares_memory(grasp.target.extra['_executed_grasp_T'], grasp.T_grasp)


def test_ambiguous_held_semantic_instances_do_not_update_grip():
    grasp = measured_grasp()
    source = source_anchor(grasp)
    tcp = top_down_T(.3, .1, .2, 1.)
    center = (tcp @ inv_T(grasp.T_grasp) @ np.r_[source['center'], 1.])[:3]
    raw = Detection('pear', .9, extra={'_points_base': sphere_cloud(center, .0425, 45)})
    assert not held_anchor([raw, copy.deepcopy(raw)], grasp, source, tcp)['ok']


def fragment_scene():
    grasp = measured_grasp()
    source = source_anchor(grasp)
    tcp = top_down_T(.3, .1, .2, 1.)
    center = (tcp @ inv_T(grasp.T_grasp) @ np.r_[source['center'], 1.])[:3]
    points = sphere_cloud(center, .0425, 45)
    split = points[:, 0] < center[0]
    fragments = []
    for index, cloud in enumerate((points[split], points[~split])):
        mask = np.zeros((2, 2), dtype=bool)
        mask[index, 0] = True
        fragments.append(Detection('pear', .9, valid=False, mask=mask,
                                  extra={'_points_base': cloud}))
    grasp.target.extra['_held_translation_anchor'] = {'center_tcp': (inv_T(tcp)@np.r_[center, 1.])[:3]}
    return grasp, source, tcp, center, fragments


def test_tracked_disjoint_fragments_must_jointly_explain_one_spherical_anchor():
    grasp, source, tcp, center, fragments = fragment_scene()
    fit = held_anchor(fragments, grasp, source, tcp)
    assert fit['ok'] and min(fit['fragment_inlier_fractions']) > .95
    np.testing.assert_allclose(fit['center_base'], center, atol=.0001)
    del grasp.target.extra['_held_translation_anchor']
    assert not held_anchor(fragments, grasp, source, tcp)['ok']


def test_two_different_spheres_or_overlapping_masks_cannot_merge_as_one_held_body():
    grasp, source, tcp, _center, fragments = fragment_scene()
    fragments[1].extra['_points_base'] = fragments[1].extra['_points_base'] + [.045, 0., 0.]
    assert not held_anchor(fragments, grasp, source, tcp)['ok']
    grasp, source, tcp, _center, fragments = fragment_scene()
    fragments[1].mask = fragments[0].mask.copy()
    assert not held_anchor(fragments, grasp, source, tcp)['ok']


def test_invalid_or_unbounded_translation_never_mutates_the_grip_reference():
    grasp = measured_grasp()
    original = grasp.T_grasp.copy()
    source = source_anchor(grasp)
    for pair in ({'ok': False}, {'ok': True, 'translation_only': False},
                 {'ok': True, 'translation_only': True, 'center': [10., 0., 0.]},
                 {'ok': True, 'translation_only': True, 'center': [0., np.nan, 0.]}):
        assert not apply_translation_anchor(grasp, source, pair, 0)['ok']
        np.testing.assert_array_equal(grasp.T_grasp, original)


def test_small_seating_change_avoids_captures_but_missing_original_feedback_cannot():
    grasp = measured_grasp()
    result = PickResult(False, detail={'steps': [{'stage': 'close', 'ok': True, 'width': .0464}]})
    assert seating_requires_feedback(grasp, result, .0385)
    assert not seating_requires_feedback(grasp, result, .0459)
    assert seating_requires_feedback(grasp, PickResult(False), .0385)
    grasp.target.extra['_closed_grip_width'] = .0464
    assert seating_requires_feedback(grasp, PickResult(False), .0385)
    assert not seating_requires_feedback(grasp, PickResult(False), .0459)
    grasp.target.extra['resting_pose'] = 'side'
    assert not seating_requires_feedback(grasp, result, .0385)


def anchored_release_session(monkeypatch, *, clouds=True, second_change=False, duplicate_stamp=False,
                             stale_frames=False):
    from types import SimpleNamespace
    from test_assistant_release_phases import release_session
    session, _target, _grasp, place, _now, moves, _calls = release_session(monkeypatch)
    grasp = measured_grasp()
    target = grasp.target
    session._active_placement = (target, None, place.zone, {})
    session._observed_detections = [target]
    now = [2.]
    session.clock = SimpleNamespace(now=lambda: now[0])
    session.arm.get_tcp_pose = lambda: SimpleNamespace(value=np.eye(4), stamp=now[0])
    session.camera._last.stamp = 1.
    captures, replans = [], []
    local = np.array([.005, -.004, .052])
    def capture():
        captures.append(1)
        now[0] = 2.+.05*len(captures)
        session.camera._last.stamp = (2. if duplicate_stamp else now[0]-(1. if stale_frames else 0.))
        if len(captures) == 2 and second_change:
            session._environment_revision += 1
        if not clouds:
            return []
        return [Detection('pear', .9, center_base=local.copy(), valid=False,
                          extra={'_points_base': sphere_cloud(local, .0425, 100+len(captures))})]
    session._stationary_recovery_observation = capture
    def replan(*args, **kw):
        replans.append(args[1].T_grasp.copy())
        return place, ''
    session.runner.replan_release = replan
    result = PickResult(False, detail={'holding': True,
        'steps': [{'stage': 'close', 'ok': True, 'width': .0464}]})
    return session, grasp, place, result, captures, replans, moves, local


def test_seated_stack_cannot_replan_without_observed_held_anchor(monkeypatch):
    session, grasp, place, result, _captures, replans, moves, _local = anchored_release_session(monkeypatch, clouds=False)
    assert session._before_release(grasp, place, .0385, result) is None
    assert not replans and not moves and result.detail['holding']
    assert 'held_anchor_unconfirmed' in result.detail['placement_planning_reason']


def test_two_fresh_anchor_captures_precede_release_replanning(monkeypatch):
    session, grasp, place, result, captures, replans, moves, local = anchored_release_session(monkeypatch)
    original = grasp.T_grasp.copy()
    assert session._before_release(grasp, place, .0385, result) is place
    assert len(captures) == 2 and len(replans) == 1 and not moves
    assert not np.array_equal(replans[0], original)
    source = source_anchor(grasp)
    np.testing.assert_allclose((inv_T(grasp.T_grasp) @ np.r_[source['center'], 1.])[:3], local, atol=.0001)


def test_edit_during_anchor_capture_invalidates_reference_before_replan(monkeypatch):
    session, grasp, place, result, captures, replans, moves, _local = anchored_release_session(monkeypatch, second_change=True)
    original = grasp.T_grasp.copy()
    assert session._before_release(grasp, place, .0385, result) is None
    assert len(captures) == 2 and not replans and not moves
    np.testing.assert_array_equal(grasp.T_grasp, original)


def test_reused_frame_cannot_supply_anchor_confirmation(monkeypatch):
    session, grasp, place, result, captures, replans, moves, _local = anchored_release_session(monkeypatch, duplicate_stamp=True)
    original = grasp.T_grasp.copy()
    assert session._before_release(grasp, place, .0385, result) is None
    assert len(captures) == 2 and not replans and not moves
    np.testing.assert_array_equal(grasp.T_grasp, original)


def test_before_open_remeasures_small_drift_without_moving_the_planned_reference(monkeypatch):
    session, grasp, place, result, captures, replans, moves, local = anchored_release_session(monkeypatch)
    source = source_anchor(grasp)
    apply_translation_anchor(grasp, source,
        dict(ok=True, translation_only=True, center=local+[.001, 0., 0.]), 1)
    reference = grasp.T_grasp.copy()
    ok, reason = session._before_open(grasp, place, result)
    assert ok and not reason and len(captures) == 2 and not replans and not moves
    np.testing.assert_array_equal(grasp.T_grasp, reference)
    assert result.detail['held_anchor_before_open']['capture_stamps'] == [2.05, 2.1]


def test_before_open_large_new_slip_updates_reference_but_refuses_old_release(monkeypatch):
    session, grasp, place, result, captures, replans, moves, local = anchored_release_session(monkeypatch)
    source = source_anchor(grasp)
    apply_translation_anchor(grasp, source,
        dict(ok=True, translation_only=True, center=local+[.006, 0., 0.]), 1)
    reference = grasp.T_grasp.copy()
    ok, reason = session._before_open(grasp, place, result)
    assert not ok and reason and len(captures) == 2 and not replans and not moves
    assert not np.array_equal(grasp.T_grasp, reference) and result.detail['holding']
    np.testing.assert_allclose(grasp.target.extra['_held_translation_anchor']['center_tcp'], local, atol=.0001)


def test_before_open_duplicate_frames_or_edit_cannot_authorize_release(monkeypatch):
    for kwargs in ({'duplicate_stamp': True}, {'second_change': True}, {'clouds': False},
                   {'stale_frames': True}):
        session, grasp, place, result, _captures, replans, moves, local = anchored_release_session(monkeypatch, **kwargs)
        source = source_anchor(grasp)
        apply_translation_anchor(grasp, source, dict(ok=True, translation_only=True, center=local), 1)
        reference = grasp.T_grasp.copy()
        ok, reason = session._before_open(grasp, place, result)
        assert not ok and 'held_anchor_unconfirmed' in reason and not replans and not moves
        np.testing.assert_array_equal(grasp.T_grasp, reference)


def test_noncalibrated_grip_does_not_trigger_redundant_release_captures(monkeypatch):
    session, grasp, place, result, captures, replans, moves, _local = anchored_release_session(monkeypatch)
    result.grasp_width = .0464
    assert session._before_open(grasp, place, result) == (True, '')
    assert not captures and not replans and not moves


def test_held_continuation_without_original_close_step_still_requires_anchor(monkeypatch):
    session, grasp, place, result, captures, replans, moves, _local = anchored_release_session(monkeypatch, clouds=False)
    grasp.target.extra['_closed_grip_width'] = .0464
    result.detail['steps'] = []
    assert session._before_release(grasp, place, .0385, result) is None
    assert captures and not replans and not moves and result.detail['holding']


def test_new_seating_change_after_hover_planning_cannot_open_without_anchor(monkeypatch):
    session, grasp, place, result, captures, replans, moves, _local = anchored_release_session(monkeypatch)
    session.runner.skill._verify_held = lambda *args: .0385
    ok, why = session._before_open(grasp, place, result)
    assert not ok and 'held_anchor_unconfirmed' in why
    assert result.grasp_width == .0385 and not captures and not moves and not replans


def test_carry_feedback_precedes_first_transport_and_stable_grip_reuses_it(monkeypatch):
    session, grasp, place, result, captures, replans, moves, local = anchored_release_session(monkeypatch)
    result.grasp_width = .0385
    route = [place.T_pre]
    session._plan_carried_entry = lambda *args: (route, '')
    assert session._before_carry(grasp, place, result) == (True, '', route)
    assert len(captures) == 2 and not replans and not moves
    np.testing.assert_allclose(grasp.target.extra['_held_translation_anchor']['center_tcp'], local, atol=.0001)
    # The same unchanged grip need not repeat a stationary pair at each waypoint.
    assert session._before_carry(grasp, place, result) == (True, '', None)
    assert len(captures) == 2


def test_carry_without_visible_anchor_never_approves_old_reference(monkeypatch):
    session, grasp, place, result, captures, replans, moves, _local = anchored_release_session(monkeypatch, clouds=False)
    result.grasp_width = .0385
    original = grasp.T_grasp.copy()
    ok, reason, route = session._before_carry(grasp, place, result)
    assert not ok and reason.startswith('held_anchor_unconfirmed')
    assert route is None
    assert captures and not replans and not moves and result.detail['holding']
    np.testing.assert_array_equal(grasp.T_grasp, original)
