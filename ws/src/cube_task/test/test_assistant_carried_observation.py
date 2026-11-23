"""One capture supplies separate environment segmentation and grip surfaces."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.carried_observation import CarriedDetections
from cube_task.assistant.simulation import SimulationSession
from cube_task.core.types import CameraIntrinsics, Detection, Frame, Stamped
from cube_task.core.clock import SimClock


def stationary_session(monkeypatch, *, held=True, feedback=True, tcp_stamp=2.):
    frame = Frame(np.zeros((4, 5, 3), np.uint8), np.ones((4, 5), np.float32),
                  CameraIntrinsics(100., 100., 2., 2., 5, 4), 2.)
    filtered = replace(frame, depth=frame.depth * .5)
    raw = Detection('pear', .8, valid=False)
    environment = Detection('apple', .99)
    target = Detection('red', .99)
    grasp = SimpleNamespace(target=target)
    context = dict(det=target, grasp=grasp)
    session = object.__new__(SimulationSession)
    session._environment_revision = 0
    session.clock = SimClock()
    session.clock.sleep(1.99)
    session.arm = SimpleNamespace(get_flange_pose=lambda: Stamped(np.eye(4), 2.),
                                  get_tcp_pose=lambda: Stamped(np.eye(4), tcp_stamp))
    session.camera = SimpleNamespace(_last=frame)
    processed, masks, cached = [], [], []
    def process(capture, *args):
        processed.append(capture)
        return SimpleNamespace(detections=[raw] if capture is frame else [environment])
    session.runner = SimpleNamespace(_last_stamp=1., n_avg=1, frame_timeout=.5,
        T_flange_tcp=np.eye(4), ext=SimpleNamespace(parent_frame='flange', T=np.eye(4)),
        perception=SimpleNamespace(process=process), zones={}, nominal={'red': (.05, .05)},
        _held_recovery_context=context if held else None, has_held_recovery=held,
        _held_transfer_feedback=lambda det: (context if feedback else None, .05, ''),
        skill=SimpleNamespace(feedback_max_age=.25, _read_width=lambda: .05, _healthy=lambda: feedback))
    session._scan_stats = dict(frames=0)
    session._cache_observation = lambda *args, **kwargs: cached.append(args)
    session._emit = lambda *args: None
    captures = []
    def capture(*args, **kwargs):
        captures.append(1)
        session.clock.sleep(.01)
        return frame
    def mask(capture, pose, source, tcp, opening, **kwargs):
        masks.append((capture, source, opening))
        return filtered
    monkeypatch.setattr('cube_task.camera.utils.capture_averaged', capture)
    monkeypatch.setattr('cube_task.assistant.carried_observation.mask_carried_depth', mask)
    return session, grasp, frame, filtered, raw, environment, processed, masks, cached, captures


def test_one_physical_capture_keeps_raw_visibility_and_calibration_separate(monkeypatch):
    session, grasp, frame, filtered, raw, environment, processed, masks, cached, captures = stationary_session(monkeypatch)
    observed = session._stationary_recovery_observation()
    assert isinstance(observed, CarriedDetections) and observed == [environment]
    assert observed.anchor_detections == [raw]
    assert processed == [frame, filtered] and len(captures) == 1
    assert masks == [(frame, grasp, .05)]
    assert session.camera._last is frame and cached[0][2] is frame
    assert session._scan_stats['frames'] == 1 and session.runner._last_stamp == 2.


def test_first_move_callback_filters_before_a_recovery_context_exists(monkeypatch):
    session, grasp, _frame, _filtered, raw, environment, _processed, masks, _cached, _captures = stationary_session(monkeypatch, held=False)
    observed = session._stationary_recovery_observation(grasp=grasp, width=.05)
    assert observed == [environment] and observed.anchor_detections == [raw]
    assert len(masks) == 1


@pytest.mark.parametrize('held,feedback,stamp', [(False, True, 2.), (True, False, 2.),
                                                (True, True, 1.)])
def test_unconfirmed_or_old_grip_never_removes_environment_samples(monkeypatch, held, feedback, stamp):
    session, _grasp, _frame, _filtered, raw, _environment, processed, masks, _cached, captures = stationary_session(
        monkeypatch, held=held, feedback=feedback, tcp_stamp=stamp)
    assert session._stationary_recovery_observation() == [raw]
    assert not masks and len(processed) == len(captures) == 1
