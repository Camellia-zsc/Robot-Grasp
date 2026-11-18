"""Camera-grounded inspection while carrying an object; never releases it."""
from __future__ import annotations

import copy
import math

import numpy as np

from ..core.extrinsics import camera_pose_in_base
from ..core.transforms import inv_T
from ..core.types import PickResult
from ..perception.geometry import points_in_polygon, project_points
from ..task.skills import _Lost, _StepFailed
from .placement_observation import (iter_safe_inspection_poses, safe_inspection_route,
                                    inspection_history, remember_inspection)


def measurements_changed(before, after):
    """New complete objects or material measured movement count as information."""
    def valid(items):
        return [d for d in items if d.get('valid') and d.get('center_base') is not None]
    previous, current = valid(before), valid(after)
    if sorted(d.get('label') for d in previous) != sorted(d.get('label') for d in current):
        return True
    for item in current:
        matches = [d for d in previous if d.get('label') == item.get('label')]
        if not matches or min(np.linalg.norm(np.asarray(d['center_base']) - item['center_base'])
                              for d in matches) > .015:
            return True
    return False


def _coverage(session, polygon):
    """Measured visible table samples are progress evidence, never a release plan."""
    low, high = np.min(polygon, axis=0), np.max(polygon, axis=0)
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 9), np.linspace(low[1], high[1], 9))
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    inside = points_in_polygon(xy, polygon, margin=1e-8)
    result = np.zeros(len(xy), bool)
    frame = session.camera._last
    if frame is None or frame.depth is None:
        return result
    intr = frame.intrinsics
    pose = camera_pose_in_base(session.runner.ext, session.arm.get_flange_pose().value,
                               session.runner.T_flange_tcp)
    table = float(session.cfg.get('perception.table.expected_z', 0.) or 0.)
    points = np.column_stack((xy, np.full(len(xy), table + .001)))
    uv = project_points(points, pose, intr)
    T_cb = inv_T(pose)
    expected = points @ T_cb[2, :3] + T_cb[2, 3]
    indices = np.flatnonzero(inside & np.all(np.isfinite(uv), axis=1) & (expected > 0)
        & (uv[:, 0] >= 8) & (uv[:, 0] <= intr.width - 9)
        & (uv[:, 1] >= 8) & (uv[:, 1] <= intr.height - 9))
    pixels = np.rint(uv[indices]).astype(int)
    depth = np.asarray(frame.depth)[pixels[:, 1], pixels[:, 0]]
    result[indices] = np.isfinite(depth) & (depth > 0) & (depth >= expected[indices] - .008)
    return result


def inspect_held(session, xy, polygon, stationary, check_current):
    """Acquire at most two useful guarded views, retaining the measured grasp."""
    from .simulation import _PlacementBudget, _PlacementInterrupted

    context = session.runner._held_recovery_context
    target = context['det']
    context, width, why = session.runner._held_transfer_feedback(target)
    if context is None:
        return dict(ok=False, error='held_feedback_unavailable', message=why,
                    changed=False, holding=bool(session.snapshot().get('held_object')))
    grasp = copy.deepcopy(context['grasp'])
    budget = _PlacementBudget(session.arm, session.cancel_event)
    result = PickResult(False, label=target.label, grasp_width=width,
                        detail={'holding': True, 'steps': []})
    views, checked, last_reason = 0, 0, 'no_safe_observation_path'
    before = copy.deepcopy(session.scene.get('objects', []))
    try:
        neighbors = budget.run(stationary)
        check_current()
        remember_inspection(session, grasp, session.arm.get_tcp_pose().value)
        covered = _coverage(session, polygon)
        revision = session._environment_revision
        if getattr(session, '_skill_coverage_revision', None) != revision:
            session._skill_coverage_revision = revision
            session._skill_observation_coverage = {}
        coverage_memory = session._skill_observation_coverage
        focus_key = tuple(np.round(np.asarray(polygon, float), 4).ravel())
        covered |= coverage_memory.get(focus_key, np.zeros_like(covered))
        if len(coverage_memory) >= 32 and focus_key not in coverage_memory:
            coverage_memory.pop(next(iter(coverage_memory)))
        coverage_memory[focus_key] = covered.copy()
        ext = session.runner.ext
        T_fc = ext.T if ext.parent_frame == 'flange' else session.runner.T_flange_tcp @ ext.T
        table = float(session.cfg.get('perception.table.expected_z', 0.) or 0.)
        session._on_stage('inspect_place')
        session._emit('active_observation', {'message': '保持夹持，正在选择能看清目标区域的安全视角'})

        def progress(fields):
            if fields.get('checked', 0) and fields['checked'] % 16 == 0:
                session._emit('placement_progress', dict(fields,
                    message='正在检查局部观察视角，仍保持夹持'))

        poses = iter_safe_inspection_poses(session.arm, grasp, T_fc,
            session.runner.T_flange_tcp, xy, table, table_z=table,
            free_tilt_deg=session.runner.free_tilt,
            nominal=session.runner.nominal.get(target.label), intrinsics=session.camera.intrinsics(),
            cancelled=budget.cancelled, progress=progress,
            exclude_poses=inspection_history(session, grasp))
        sentinel = object()
        while checked < 12 and views < 2:
            pose = budget.run(next, poses, sentinel)
            if pose is sentinel:
                break
            checked += 1
            current = session.arm.get_tcp_pose().value.copy()
            rotation = np.trace(current[:3, :3].T @ pose[:3, :3])
            angle = math.acos(float(np.clip((rotation - 1.) / 2., -1., 1.)))
            if np.linalg.norm(current[:3, 3] - pose[:3, 3]) < .005 and angle < math.radians(3):
                continue
            route, last_reason = budget.run(safe_inspection_route,
                session.arm, grasp, current, pose, neighbors, width, table_z=table,
                nominal=session.runner.nominal.get(target.label), neighbor_nominal=session.runner.nominal,
                free_tilt_deg=session.runner.free_tilt, cancelled=budget.cancelled)
            if not route:
                continue
            for waypoint in route:
                session._apply_interventions()
                check_current()
                if session.cancel_event.is_set():
                    raise _PlacementInterrupted()
                current_context, current_width, feedback_why = session.runner._held_transfer_feedback(target)
                if current_context is None or abs(current_width - width) > 1e-6:
                    return dict(ok=False, changed=False, views=views,
                        holding=bool(session.snapshot().get('held_object')),
                        error='held_feedback_changed', fault=True,
                        reason=feedback_why or 'measured_opening_changed',
                        message='夹持反馈发生变化，已停止尚未执行的观察路径并保留现场')
                session.runner.skill._move(result, 'inspect_place', waypoint, True)
                session.runner.skill._verify_held(result, 'dropped_in_transit')
            neighbors = budget.run(stationary)
            check_current()
            remember_inspection(session, grasp, session.arm.get_tcp_pose().value)
            views += 1
            fresh_coverage = _coverage(session, polygon)
            changed = bool(np.any(fresh_coverage & ~covered) or
                           measurements_changed(before, session.scene.get('objects', [])))
            covered |= fresh_coverage
            coverage_memory[focus_key] = covered.copy()
            session._emit('active_observation', {
                'message': '已从安全新视角获得局部观测，保持夹持', 'views': views, 'changed': changed})
            if changed:
                return dict(ok=True, changed=True, views=views, holding=True,
                    stage='observed', observation_mode='active_local', camera_verified=True,
                    reason='new_measured_coverage', message='已换到受检新视角，获得新的局部测量')
        if views:
            return dict(ok=True, changed=False, views=views, holding=True, stage='observed',
                observation_mode='active_local', camera_verified=True,
                reason='no_new_measured_coverage', message='已完成安全换视角，尚未获得更多有效依据')
        return dict(ok=False, changed=False, views=0, holding=True,
            error='no_safe_observation_path', reason=last_reason,
            message='未找到通过完整路径检查的局部观察视角，仍保持夹持')
    except _PlacementInterrupted:
        stopped = session.cancel_event.is_set()
        return dict(ok=False, changed=False, views=views, holding=True,
            aborted=stopped, error='aborted' if stopped else 'observation_planning_budget_exceeded',
            reason='planning_interrupted', message='观察已停止，保持当前夹持' if stopped else
                '本轮主动观察计算已达预算，仍保持夹持，需选择其他步骤')
    except _StepFailed as exc:
        return dict(ok=False, changed=False, views=views,
            holding=bool(session.snapshot().get('held_object')), error=exc.error,
            diagnostics=exc.detail, fault=exc.fault, message='局部观察运动未完成，已保留现场')
    except _Lost as exc:
        session.runner._held_recovery_context = None
        return dict(ok=False, changed=True, holding=False, error=exc.stage,
            message='局部观察时夹持反馈丢失，已停止后续动作')
