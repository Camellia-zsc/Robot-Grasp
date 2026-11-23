"""Exercise the assistant's real L1 perception -> safe motion seam."""
from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.core.types import PickResult


@pytest.fixture
def session(tmp_path):
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False) as sim:
        sim.prepare()
        yield sim


def _red(scene):
    return next(o for o in scene['objects'] if o['label'] == 'red' and o['valid'])


def _decision(scene):
    return {'action': 'pick', 'object_id': _red(scene)['object_id'],
            'scene_version': scene['scene_version'], 'zone_name': 'zone_red'}


def test_prepare_uses_camera_multiview_and_publishes_actual_kinematics(session):
    scene = session.scene
    assert {o['label'] for o in scene['objects'] if o['valid']} == {'red', 'green', 'yellow'}
    assert {o['view'] for o in scene['objects']} == {0, 1, 2}
    assert scene['class_to_zone']['red'] == 'zone_red'
    assert session.cfg.get('robot.backend') == session.cfg.get('camera.backend') == 'sim'
    assert session.camera.frames_rendered >= 3 * session.runner.n_avg
    for measured in scene['objects']:
        truth = next(o for o in session.world.objects if o.label == measured['label'])
        np.testing.assert_allclose(measured['center_base'][:2], truth.center_xy(), atol=0.004)
        assert measured['object_id'] != truth.name
    snapshot = session.snapshot()
    json.dumps(snapshot, allow_nan=False)
    np.testing.assert_allclose(snapshot['joints'], session.arm.get_joints().value)
    np.testing.assert_allclose(snapshot['tcp_xyz'], session.arm.get_tcp_pose().value[:3, 3])
    frames = session.arm.kin.fk_all(snapshot['joints'])
    np.testing.assert_allclose(snapshot['link_points'][1:-1], [T[:3, 3] for T in frames])
    image = cv2.imdecode(np.frombuffer(session.frame_jpeg(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.shape[:2] == (480, 640)
    # Each pick-area and placement-zone view is independently inspectable.
    # Reading these frozen images must not move the arm or render extra frames.
    views = session.observation_views()
    assert len(views) == len(session.runner.views) == 6
    assert [v['id'] for v in views] == [v['view_id'] for v in views] == list(range(6))
    assert [v['stamp'] for v in views] == sorted(v['stamp'] for v in views)
    assert {label for v in views for label in v['detected_labels']} == {'red', 'green', 'yellow'}
    before = (session.camera.frames_rendered, session.clock.now(), session.arm.inner.n_calls)
    for view in views:
        frame = session.observation_frame_jpeg(view['id'])
        decoded = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[:2] == (view['height'], view['width']) == (480, 640)
        assert view['sim_time'] == pytest.approx(view['stamp'] - session._started_at)
    assert session.frame_jpeg() == session.observation_frame_jpeg(5)
    assert session.observation_frame_jpeg(100) is None
    assert (session.camera.frames_rendered, session.clock.now(), session.arm.inner.n_calls) == before
    views[0]['detected_labels'].clear()
    views.clear()
    assert len(session.observation_views()) == 6
    snapshot['objects'].clear()
    scene['objects'].clear()
    assert len(session.snapshot()['objects']) == len(session.scene['objects']) == 3


def test_pick_moves_red_truth_and_expires_observation_ids(session):
    old_scene = session.scene
    scene = session.observe()
    assert scene['scene_version'] > old_scene['scene_version']
    calls = session.arm.inner.n_calls
    refused = session.execute(_decision(old_scene))
    assert refused['error'] == 'stale_scene' and session.arm.inner.n_calls == calls
    bad_zone = dict(_decision(scene), zone_name='invented_zone')
    assert session.execute(bad_zone)['error'] == 'invalid_zone'
    assert session.arm.inner.n_calls == calls
    held_snapshots = []
    scan_progress = []

    def watch(kind, payload):
        if kind == 'stage':
            snapshot = session.snapshot()
            held_snapshots.extend(o for o in snapshot['objects'] if o['held'])
        elif kind == 'scan_view':
            scan_progress.append(payload)

    session.on_event = watch
    result = session.execute(_decision(scene))
    assert result['ok'] and result['score'] == {'correct': 1, 'wrong': 0, 'remaining': 2}
    red = next(o for o in session.world.objects if o.label == 'red')
    assert session.world.zone_of(red).name == 'zone_red'
    assert result['controller_ok']
    assert result['truth'] == {'success': True, 'object_name': red.name,
                               'expected_zone': 'zone_red', 'actual_zone': 'zone_red'}
    assert session.world.held is None
    assert held_snapshots and all(o['id'] == red.name for o in held_snapshots)
    red_snapshot = next(o for o in session.snapshot()['objects'] if o['label'] == 'red')
    assert red_snapshot['placed'] and not red_snapshot['held']
    assert _red(session.scene)['in_zone']
    assert _red(session.scene)['zone'] == 'zone_red'
    assert result['scene'] == session.scene
    # Source and destination are checked locally; unrelated views are unnecessary.
    assert 0 < len(scan_progress) < 6
    assert all(1 <= step['total'] <= 6 and step['label'] and step['message']
               for step in scan_progress)
    calls = session.arm.inner.n_calls
    assert session.execute(_decision(scene))['error'] == 'stale_scene'
    assert session.arm.inner.n_calls == calls


def test_red_can_be_observed_and_picked_again_from_yellow_then_green(session):
    """Placed cubes remain camera-measured targets for later user instructions."""
    for zone_name in ('zone_yellow', 'zone_green'):
        scene = session.scene
        red = _red(scene)
        if zone_name == 'zone_green':
            assert red['in_zone'] and red['zone'] == 'zone_yellow'
            assert red['view'] >= 3
        result = session.execute(dict(_decision(scene), zone_name=zone_name))
        assert result['ok'], result
        assert result['truth']['actual_zone'] == zone_name
        assert result['truth']['success']
        assert len([o for o in session.scene['objects'] if o['valid'] and o['label'] == 'red']) == 1
        assert len(session.world.objects) == 3
    assert _red(session.scene)['zone'] == 'zone_green'
    assert 'zone_yellow' not in session.runner._placed


@pytest.mark.parametrize('requested_zone,expected_success',
                         [('zone_red', False), ('zone_green', True)])
def test_requested_placement_is_independently_checked_against_truth(
        session, monkeypatch, requested_zone, expected_success):
    """A successful controller result alone cannot prove the requested placement."""
    red = next(obj for obj in session.world.objects if obj.label == 'red')
    green_zone = session.world.zone('zone_green')

    def report_success(det, zone_name=None, neighbors=(), *,
                       placement_zone=None, placement_polygon=None, exact_xy=False):
        # Model a controller that releases in green, regardless of its requested
        # destination; the independent evaluator must detect any mismatch.
        session._motion_observer.invalidate([det.label])  # Real lift revokes the source track.
        red.place_upright(*green_zone.center[:2], 0.0, green_zone.floor_z)
        return PickResult(ok=True, stage='done', label=det.label, zone=zone_name), ''

    monkeypatch.setattr(session.runner, 'pick_detection', report_success)
    result = session.execute(dict(_decision(session.scene), zone_name=requested_zone))
    assert result['controller_ok'] and result['ok'] is expected_success
    assert result['truth'] == {'success': expected_success, 'object_name': red.name,
                               'expected_zone': requested_zone, 'actual_zone': 'zone_green'}
    # Competition scoring still uses class defaults. Explicit user destinations
    # are judged against the user's request, even when that score says "wrong".
    assert result['score'] == {'correct': 0, 'wrong': 1, 'remaining': 2}
    if not expected_success:
        assert result['error'] == 'placement_not_verified'


def test_ambiguous_truth_identity_cannot_count_as_verified_success(session, monkeypatch):
    red = next(obj for obj in session.world.objects if obj.label == 'red')
    duplicate = red.copy()
    duplicate.name = 'red_decoy'
    session.world.add_object(duplicate)
    red_zone = session.world.zone('zone_red')

    def report_success(det, zone_name=None, neighbors=(), *,
                       placement_zone=None, placement_polygon=None, exact_xy=False):
        session._motion_observer.invalidate([det.label])
        red.place_upright(*red_zone.center[:2], 0.0, red_zone.floor_z)
        return PickResult(ok=True, stage='done', label=det.label, zone=zone_name), ''

    monkeypatch.setattr(session.runner, 'pick_detection', report_success)
    result = session.execute(_decision(session.scene))
    assert result['controller_ok'] and not result['ok']
    assert result['truth'] == {'success': False, 'object_name': None,
                               'expected_zone': 'zone_red', 'actual_zone': None}


def test_stop_is_nonblocking_and_preserves_held_cube_while_worker_runs(session):
    reached = threading.Event()
    release = threading.Event()
    outcome = {}

    def pause_before_move(kind, payload):
        if kind == 'stage' and payload['stage'] == 'lift':
            reached.set()
            release.wait(15.0)

    session.on_event = pause_before_move

    def work():
        try:
            outcome['result'] = session.execute(_decision(session.scene))
        except BaseException as exc:
            outcome['exception'] = exc

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        assert reached.wait(30.0), 'worker did not reach the lift after grasping'
        start = time.monotonic()
        before = session.snapshot()
        assert any(o['held'] for o in before['objects'])
        session.stop()
        assert time.monotonic() - start < 0.5
    finally:
        release.set()
        worker.join(20.0)
    assert not worker.is_alive() and 'exception' not in outcome
    result = outcome['result']
    assert not result['ok'] and result['aborted'] and result['holding']
    after = session.snapshot()
    assert after['status'] == 'stopped'
    np.testing.assert_allclose(after['joints'], before['joints'], atol=0)
    assert after['objects'] == before['objects']
    # The frozen observation remains visible but cannot authorize a new grasp.
    assert session.scene['objects']
    assert not session._reusable_target(_red(session.scene))


def test_external_cancel_is_shared_with_safe_arm_and_cannot_issue_commands(session):
    calls = session.arm.inner.n_calls
    non_pick = dict(_decision(session.scene), action='clarify')
    assert session.execute(non_pick)['error'] == 'invalid_action'
    assert session.arm.inner.n_calls == calls
    cancel = threading.Event()
    cancel.set()
    calls = session.arm.inner.n_calls
    result = session.execute(_decision(session.scene), cancel_event=cancel)
    assert result['aborted'] and not result['ok']
    assert session.cancel_event is session.arm.abort_event is session.arm.inner.abort_event
    assert session.cancel_event.is_set()
    assert session.arm.inner.n_calls == calls
    assert session.snapshot()['status'] == 'stopped'
    session.set_speed(4)
    assert session.snapshot()['speed'] == 4
    with pytest.raises(ValueError):
        session.set_speed(float('nan'))
