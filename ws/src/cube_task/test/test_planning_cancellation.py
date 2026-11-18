"""Planning cancellation must stop computation while retaining the held object."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from cube_task.assistant import placement_observation
from cube_task.assistant.placement_observation import (
    iter_safe_inspection_poses, safe_inspection_poses, safe_inspection_route,
)
from cube_task.core.config import load_config
from cube_task.core.transforms import make_T, top_down_T
from cube_task.core.types import Detection, GraspPlan, Stamped
from cube_task.robot.kinematics import PiperKinematics, flange_to_tcp_from_cfg
from cube_task.robot.safety import SafeArm, SafetyGuard


def banana_grasp():
    target = Detection('banana', 1., center_base=np.array([.4549, -.0063, .0399]),
                       top_z=.0399, height=.0399, length=.1435, width=.0444, yaw=1.2254)
    return GraspPlan(target, top_down_T(.4516, -.0086, .1199, 2.7962),
                     top_down_T(.4516, -.0086, .018, 2.7962),
                     top_down_T(.4516, -.0086, .1812, 2.7962),
                     .051258564783693825, 0., .04125856478369382, 2.7962)


@pytest.mark.parametrize('preset', [False, True])
def test_inspection_stops_after_abort_instead_of_checking_924_candidates(preset):
    event, calls = threading.Event(), []
    if preset:
        event.set()

    def check(*args, **kwargs):
        calls.append(1)
        event.set()
        return False, 'unreachable'

    arm = SimpleNamespace(check_reachable=check, abort_event=event)
    poses, _minimum = safe_inspection_poses(
        arm, banana_grasp(), make_T(np.diag([1., -1., -1.]), [0., 0., .05]),
        np.eye(4), [.26, 0.], 0.)
    assert poses == []
    assert len(calls) == (0 if preset else 1)


@pytest.mark.parametrize('cancel_stage', ['before', 'reachable', 'path'])
def test_cancelled_inspection_route_never_returns_motion_or_continues_preflight(cancel_stage):
    event, calls = threading.Event(), []
    if cancel_stage == 'before':
        event.set()

    def check(*args, **kwargs):
        calls.append('reachable')
        if cancel_stage == 'reachable':
            event.set()
        return True, ''

    def path(*args, **kwargs):
        calls.append('path')
        event.set()
        return True, ''

    arm = SimpleNamespace(check_reachable=check, abort_event=event,
                          _check_linear_path=path, _kin=None)
    grasp = banana_grasp()
    route, why = safe_inspection_route(arm, grasp, top_down_T(.3, 0., .25, 0.),
                                      top_down_T(.31, 0., .25, 0.), [], .05)
    assert route == [] and why == 'aborted'
    assert calls == {'before': [], 'reachable': ['reachable'],
                     'path': ['reachable', 'path']}[cancel_stage]


def test_ik_cancellation_interrupts_a_seed_before_trying_more(monkeypatch):
    event = threading.Event()
    kin = PiperKinematics()
    target = kin.fk(np.zeros(6))
    target[:3, 3] += [.1, .1, .1]
    original, calls = kin._jacobian_from, []

    def jacobian(frames):
        calls.append(1)
        event.set()
        return original(frames)

    monkeypatch.setattr(kin, '_jacobian_from', jacobian)
    q, _ep, _eo = kin.ik(target, q_seed=np.zeros(6), cancelled=event.is_set)
    assert q is None and len(calls) == 1


def test_inspection_iterator_does_not_collect_twelve_reachable_poses_before_yielding():
    calls, progress = [], []

    def check(*args, **kwargs):
        calls.append(1)
        return True, ''

    iterator = iter_safe_inspection_poses(
        SimpleNamespace(check_reachable=check), banana_grasp(),
        make_T(np.diag([1., -1., -1.]), [0., 0., .05]), np.eye(4), [.26, 0.], 0.,
        progress=progress.append)
    assert calls == []
    assert next(iterator).shape == (4, 4)
    assert len(calls) == 1
    iterator.close()
    assert len(calls) == 1 and progress[-1]['reachable'] == 1


def test_inspection_deadline_can_interrupt_visibility_scoring(monkeypatch):
    event, scored, checks = threading.Event(), [], []
    original = placement_observation._inspection_visibility_score

    def score(*args, **kwargs):
        scored.append(1)
        event.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(placement_observation, '_inspection_visibility_score', score)
    arm = SimpleNamespace(check_reachable=lambda *a, **k: checks.append(1) or (True, ''))
    poses, _minimum = safe_inspection_poses(
        arm, banana_grasp(), make_T(np.diag([1., -1., -1.]), [0., 0., .05]),
        np.eye(4), [.26, 0.], 0., cancelled=event.is_set)
    assert poses == [] and checks == [] and scored == [1]


def test_inspection_reachability_budget_stops_rejected_search():
    calls = []
    arm = SimpleNamespace(check_reachable=lambda *a, **k: calls.append(1) or (False, 'unreachable'))
    poses, _minimum = safe_inspection_poses(
        arm, banana_grasp(), make_T(np.diag([1., -1., -1.]), [0., 0., .05]),
        np.eye(4), [.26, 0.], 0., max_checks=2)
    assert poses == [] and len(calls) == 2


def test_vectorized_visibility_counts_only_wholly_visible_support_groups():
    points = np.array([[0., 0., 1.], [.1, 0., 1.],
                       [0., 0., -1.], [.1, 0., 1.],
                       [.1, .1, 1.], [.2, .1, 1.], [.3, .1, 1.]])
    score = placement_observation._inspection_visibility_score(
        np.eye(4), points, np.array([2, 2, 3]), np.array([0, 2, 4]), None, None)
    assert score == (2, 2.5)


def planning_arm():
    cfg = load_config(use_overlays=False)
    kin = PiperKinematics()
    inner = SimpleNamespace(get_joints=lambda: Stamped(np.zeros(6), 0.))
    return SafeArm(inner, SafetyGuard(cfg), kin, flange_to_tcp_from_cfg(cfg))


def test_planning_scope_deadline_does_not_latch_stop_and_restores_on_exception():
    arm = planning_arm()
    with pytest.raises(RuntimeError):
        with arm.planning_scope(lambda: False):
            with arm.planning_scope(lambda: True):
                assert arm.check_reachable(top_down_T(.3, 0., .05, 0.)) == (False, 'aborted')
                assert not arm.abort_event.is_set()
            assert not arm._planning_cancelled()
            raise RuntimeError('exercise finally cleanup')
    assert not arm._planning_cancelled() and not arm.abort_event.is_set()


@pytest.mark.parametrize('linear', [False, True])
def test_safe_arm_propagates_scope_cancellation_into_ik_steps(monkeypatch, linear):
    arm = planning_arm()
    event, calls = threading.Event(), []
    original = arm.kin._jacobian_from

    def jacobian(frames):
        calls.append(1)
        event.set()
        return original(frames)

    monkeypatch.setattr(arm.kin, '_jacobian_from', jacobian)
    target = top_down_T(.3, .01, .05, 0.)
    with arm.planning_scope(event.is_set):
        if linear:
            ok, why = arm._check_linear_path(top_down_T(.3, 0., .05, 0.), target,
                                              np.zeros(6))
        else:
            ok, why = arm.check_reachable(target)
    assert not ok and why == 'aborted' and len(calls) == 1
    assert not arm.abort_event.is_set()


def test_safe_arm_preserves_injected_kinematics_without_cancellation_parameter():
    arm = planning_arm()
    calls = []

    def old_ik(target, q_seed=None):
        calls.append(1)
        return np.zeros(6), 0., 0.

    arm._kin = SimpleNamespace(ik=old_ik)
    assert arm.check_reachable(top_down_T(.3, 0., .05, 0.)) == (True, '')
    assert calls == [1]
