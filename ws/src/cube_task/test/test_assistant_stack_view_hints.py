"""Occluded support memory may aim a camera, but cannot confirm a placement."""
from types import SimpleNamespace
import copy
import threading

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.stacking import camera_confirmed
from test_assistant_targeted_scan import block
from test_assistant_observation_policy import INTR, camera


GOAL = dict(label='apple', destination=dict(type='on_object', label='green'))


def remembered_session():
    session = object.__new__(SimulationSession)
    upper = block('apple', top=.06, height=.06)
    upper.extra['_stack_support'] = dict(label='green', xy=[.18, .14], top_z=.05,
                                         length=.05, width=.05, yaw=.3)
    session._active_placement = (upper, copy.deepcopy(GOAL['destination']), 'on_object_green', {})
    session._observed_detections = []
    session._last_fresh = []
    session._observation_revision = session._environment_revision = 0
    session._interventions = []
    session.world = SimpleNamespace(held=None)
    session.cfg = SimpleNamespace(get=lambda *a: 0.)
    return session


def test_saved_measured_support_can_aim_camera_but_is_never_fresh_evidence():
    session = remembered_session()
    base = session._stack_support_hint(GOAL)
    assert base.label == 'green' and not base.valid
    np.testing.assert_allclose(base.center_base, [.18, .14, .05])
    assert not session._observed_detections and not session._last_fresh
    assert not camera_confirmed(session, 'apple', GOAL['destination'])


@pytest.mark.parametrize('change', ['upper', 'destination', 'support', 'nan', 'missing'])
def test_support_hint_must_belong_to_matching_goal_and_have_finite_geometry(change):
    session = remembered_session()
    upper, destination, _, _ = session._active_placement
    if change == 'upper':
        upper.label = 'pear'
    elif change == 'destination':
        destination['label'] = 'red'
    elif change == 'support':
        upper.extra['_stack_support']['label'] = 'red'
    elif change == 'nan':
        upper.extra['_stack_support']['top_z'] = np.nan
    else:
        upper.extra.pop('_stack_support')
    assert session._stack_support_hint(GOAL) is None


def test_current_complete_base_precedes_saved_aiming_hint():
    session = remembered_session()
    base = block('green', xy=(.22, .14))
    session._observed_detections = [base]
    assert session._stack_support_hint(GOAL) is base


def joint_session():
    session = remembered_session()
    session.cancel_event = threading.Event()
    session.camera = SimpleNamespace(intrinsics=lambda: INTR)
    session.runner = SimpleNamespace(views=[[0., .5, -.4, 0., 1., 0.]],
        ext=SimpleNamespace(parent_frame='flange', T=np.eye(4)), T_flange_tcp=np.eye(4))
    session._base_view_count = 1
    session._view_order = lambda target: [0]
    session.arm = SimpleNamespace(kin=SimpleNamespace(
        within_limits=lambda q: True, fk=lambda q: camera(.2, .3)))
    return session


def test_joint_fallback_changes_a_taught_pose_and_checks_full_sweep():
    session = joint_session()
    checks = []
    session._safe_observation_joints = lambda q, neighbors: checks.append(q.copy()) or True
    upper, lower = block('apple', top=.11), block('green')
    q = session._stack_joint_view(upper, lower, [upper, lower], 0)
    assert len(checks) == 1
    np.testing.assert_array_equal(q, checks[0])
    assert np.linalg.norm(q-session.runner.views[0]) > .03
    assert session.runner.views == [[0., .5, -.4, 0., 1., 0.]]


def test_joint_fallback_skips_attempted_camera_poses_and_rechecks_each_new_sweep():
    session = joint_session()
    checks = []
    session._safe_observation_joints = lambda q, neighbors: checks.append(q.copy()) or True
    upper, lower = block('apple', top=.11), block('green')
    attempted = []
    for _ in range(3):
        q = session._stack_joint_view(upper, lower, [upper, lower], 0, attempted)
        assert q is not None and all(np.max(np.abs(q - old)) >= .03 for old in attempted)
        attempted.append(q)
    assert len(checks) == 3
    for planned, checked in zip(attempted, checks):
        np.testing.assert_array_equal(planned, checked)


def test_joint_fallback_can_look_from_above_and_both_sides():
    session = joint_session()
    session.arm.kin.within_limits = lambda q: q[1] > .5 and abs(abs(q[0]) - .12) < 1e-6
    session._safe_observation_joints = lambda q, neighbors: q[0] > 0
    upper, lower = block('apple', top=.11), block('green')
    q = session._stack_joint_view(upper, lower, [upper, lower], 0)
    assert q is not None and q[1] > .5 and q[0] == pytest.approx(.12)


def test_displaced_stack_does_not_require_covering_old_support_to_aim_camera():
    session = joint_session()
    references, obstacles = [], []
    session._view_order = lambda ref: references.append(ref) or [0]
    session._safe_observation_joints = lambda q, neighbors: obstacles.extend(neighbors) or True
    upper = block('apple', xy=(.2, .3), top=.11)
    old_base = block('green', xy=(1.3, .3))
    old_base.valid = False
    q = session._stack_joint_view(upper, old_base, [upper, old_base], 0)
    assert q is not None
    assert 'required_support' not in references[0]
    assert any(item is old_base for item in obstacles)
    np.testing.assert_array_equal(old_base.center_base[:2], [1.3, .3])
    assert not old_base.valid and not session._last_fresh


def test_joint_candidates_prefer_new_line_of_sight_without_requiring_axis_rotation():
    session = joint_session()
    session.arm.kin.fk = lambda q: camera(q[0], .3)
    session._safe_observation_joints = lambda *args: True
    upper, lower = block('apple', xy=(.2, .3), top=.11), block('green', xy=(.2, .3))
    previous = np.array([-.15, .18, -.4, 0., 1., 0.])
    q = session._stack_joint_view(upper, lower, [upper, lower], 0, [previous])
    assert q[0] >= .15  # Wider candidates may provide still more angular novelty.
    np.testing.assert_array_equal(session.arm.kin.fk(q)[:3, :3],
                                  session.arm.kin.fk(previous)[:3, :3])


@pytest.mark.parametrize('changed', [False, True])
def test_second_extra_view_tries_different_direction_before_similar_optical_pose(changed):
    session = joint_session()
    session.on_event = None
    session._observed_detections = [block('green')]
    session._view_labels = ['initial']
    chosen = np.ones(6)
    def propose(*args):
        if changed:
            session._environment_revision += 1
        return chosen
    session._stack_joint_view = propose
    session.arm.kin.ik = lambda *a, **kw: pytest.fail('must not exhaust budget on similar optical views')
    result = session._stack_verification_view([GOAL], [block('apple', top=.11)], 0,
                                             [np.zeros(6), np.ones(6) * .1, np.ones(6) * .2])
    if changed:
        assert result is None and len(session.runner.views) == 1
    else:
        assert result == 1
        np.testing.assert_array_equal(session.runner.views[result], chosen)


def test_exact_stack_view_skips_previously_attempted_joint_solution():
    session = joint_session()
    session._observed_detections = [block('green')]
    q = np.ones(6)
    session.arm.kin.ik = lambda *a, **kw: (q.copy(), 0., 0.)
    session.arm.get_joints = lambda: SimpleNamespace(value=np.zeros(6))
    checks, fallbacks = [], []
    session._safe_observation_joints = lambda *args: checks.append(args) or True
    session._stack_joint_view = lambda *args: fallbacks.append(args) or None
    assert session._stack_verification_view([GOAL], [block('apple', top=.11)], 0, [q]) is None
    assert not checks and len(fallbacks) == 1
    np.testing.assert_array_equal(fallbacks[0][-1][0], q)


@pytest.mark.parametrize('reason', ['sweep', 'coverage', 'limits', 'stop', 'revision', 'intervention'])
def test_joint_fallback_rejects_uncovered_unsafe_or_invalidated_candidates(reason):
    session = joint_session()
    checks = []
    session._safe_observation_joints = lambda q, neighbors: checks.append(q.copy()) or reason != 'sweep'
    if reason == 'coverage':
        session.arm.kin.fk = lambda q: camera(10., 10.)
    elif reason == 'limits':
        session.arm.kin.within_limits = lambda q: False
    elif reason == 'stop':
        session.cancel_event.set()
    elif reason == 'revision':
        session._environment_revision += 1
    elif reason == 'intervention':
        session._interventions.append({})
    upper, lower = block('apple', top=.11), block('green')
    assert session._stack_joint_view(upper, lower, [upper, lower], 0) is None
    assert bool(checks) is (reason == 'sweep')


@pytest.mark.parametrize('invalidate', [None, 'revision', 'stop'])
def test_measured_central_stack_uses_safe_fk_fallback_when_optical_ik_is_unreachable(
        tmp_path, monkeypatch, invalidate):
    pytest.importorskip('mujoco')
    # Camera measurements from the failing central-table layout. No simulator
    # object coordinates or private checkpoint file are read by this regression.
    def measured(label, xy, top, length, width, yaw=0., valid=True):
        item = block(label, xy=xy, top=top, height=top)
        item.length, item.width, item.yaw, item.valid = length, width, yaw, valid
        return item
    upper = measured('apple', (.1735, .1273), .1099, .0674, .0484, -.7249, False)
    base = measured('green', (.183, .1365), .051, .0498, .0498, 1.4934)
    fresh = [upper,
        measured('pear', (.2829, .114), .1109, .0751, .0468, -.9626),
        measured('red', (.1914, .0043), .0505, .0494, .049, -.3517),
        measured('yellow', (.2327, -.1058), .0481, .047, .047, -1.4753),
        measured('banana', (.2594, -.1949), .0395, .1205, .0271, -.216, False)]
    with SimulationSession(realtime=False, backend='mujoco', log_dir=tmp_path) as session:
        session.arm.inner.set_joints(session.runner.views[1])
        session._observed_detections = [base] + copy.deepcopy(fresh)
        session._observation_revision = session._environment_revision = 0
        before = session.arm.get_joints().value.copy()
        motions = session.arm.inner.n_calls
        solutions, safe_checks = [], []
        solve = session.arm.kin.ik
        guard = session._safe_observation_joints
        def ik(*args, **kwargs):
            result = solve(*args, **kwargs)
            solutions.append(result[0])
            return result
        def safe(q, obstacles):
            result = guard(q, obstacles)
            safe_checks.append((q.copy(), result))
            if result and invalidate == 'revision':
                session._environment_revision += 1
            elif result and invalidate == 'stop':
                session.cancel_event.set()
            return result
        monkeypatch.setattr(session.arm.kin, 'ik', ik)
        monkeypatch.setattr(session, '_safe_observation_joints', safe)
        selected = session._stack_verification_view([GOAL], fresh, 0)
        assert len(solutions) == 4 and all(q is None for q in solutions)
        assert safe_checks and safe_checks[-1][1]
        if invalidate:
            assert selected is None and len(session.runner.views) == session._base_view_count
        else:
            assert selected == session._base_view_count
            np.testing.assert_allclose(session.runner.views[selected], safe_checks[-1][0])
        np.testing.assert_array_equal(session.arm.get_joints().value, before)
        assert session.arm.inner.n_calls == motions
