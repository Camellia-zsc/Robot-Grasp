"""A wide upper body needs a side view of the measured support surface."""
import copy
from types import SimpleNamespace

import numpy as np

from cube_task.assistant.stacking import camera_confirmed
from cube_task.core.types import CameraIntrinsics
from cube_task.sim.scenarios import look_at
from test_assistant_stack_view_hints import joint_session
from test_assistant_targeted_scan import block


GOAL = dict(label='pear', destination=dict(type='on_object', label='yellow'))


def side_view_session():
    session = joint_session()
    upper = block('pear', xy=(.3194, -.1592), top=.1518, height=.1518)
    upper.length, upper.width = .0747, .0424
    upper.valid, upper.reject_reason = False, 'touches_image_border'
    support = dict(label='yellow', xy=[.3361, -.1634], top_z=.04815,
                   length=.04668, width=.04583, yaw=-.4747)
    source = block('pear', xy=(.51, -.01), top=.1124, height=.1124)
    source.extra['_stack_support'] = support
    session._active_placement = (source, copy.deepcopy(GOAL['destination']), 'on_object_yellow', {})
    session.on_event = None
    session._view_labels = ['initial']
    session.camera.intrinsics = lambda: CameraIntrinsics(
        width=640, height=480, fx=415.69, fy=415.69, cx=319.5, cy=239.5)
    gaze = np.r_[support['xy'], support['top_z']+.06]
    radial = gaze[:2]/np.linalg.norm(gaze[:2])
    angle = np.radians(-45.)
    direction = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])@radial
    eye = np.r_[gaze[:2]-.16*direction, gaze[2]+.12]
    expected = look_at(eye, gaze)
    selected = np.array([.385, 1.744, -.845, -1.064, 1.152, 2.290])
    attempted, sweeps = [], []
    def solve(pose, **_kwargs):
        attempted.append(pose.copy())
        return ((selected.copy(), 0., 0.) if np.allclose(pose, expected, atol=1e-8)
                else (None, 1., 1.))
    session.arm.kin.ik = solve
    session.arm.kin.fk = lambda _q: expected.copy()
    session.arm.get_joints = lambda: SimpleNamespace(value=np.zeros(6))
    session._safe_observation_joints = lambda q, neighbors: sweeps.append((q.copy(), neighbors)) or True
    session._stack_joint_view = lambda *_args: None
    return session, upper, expected, selected, attempted, sweeps


def test_occluded_wide_body_offers_a_distinct_reachable_side_view_of_saved_support():
    session, upper, expected, selected, attempted, sweeps = side_view_session()
    view = session._stack_verification_view([GOAL], [upper], 0)
    assert view == session._base_view_count
    np.testing.assert_allclose(session.runner.views[view], selected)
    assert any(np.allclose(pose, expected, atol=1e-8) for pose in attempted)
    assert len(sweeps) == 1
    assert any(d.label == 'yellow' and not d.valid for d in sweeps[0][1])
    assert not camera_confirmed(session, 'pear', GOAL['destination'])


def test_side_view_never_uses_a_solution_whose_actual_camera_clips_the_stack():
    session, upper, _expected, _selected, _attempted, sweeps = side_view_session()
    session.arm.kin.fk = lambda _q: look_at(np.array([2., 2., .3]), np.array([2., 1.9, 0.]))
    assert session._stack_verification_view([GOAL], [upper], 0) is None
    assert not sweeps
