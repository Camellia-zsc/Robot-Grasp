"""Actual RGB-D, FK/IK and independent scoring across the recovery adapter."""
import copy

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.core.types import MotionResult, PickResult


def select(sim, label='yellow'):
    scene = sim.scene
    target = next(o for o in scene['objects'] if o['label'] == label and o['valid'])
    return {'scene_version': scene['scene_version'], 'action': 'pick',
            'object_id': target['object_id'], 'zone_name': 'zone_red'}


def choose(context, option):
    return {'scene_version': context['scene_version'], 'action': 'recover',
            'option_id': option, 'message': '保持原目标，使用本地恢复方案'}


def test_held_failure_uses_stationary_camera_then_verified_placement(tmp_path, monkeypatch):
    with SimulationSession(seed=0, realtime=False, log_dir=tmp_path) as sim:
        sim.prepare()
        original_move = sim.arm.move_tcp
        reject = {'release', 'recover_up'}

        def blocked(T, linear=False, **kwargs):
            if sim._stage in reject:
                reject.remove(sim._stage)
                return MotionResult(ok=False, error='rejected', detail='test: local path blocked')
            return original_move(T, linear=linear, **kwargs)

        monkeypatch.setattr(sim.arm, 'move_tcp', blocked)
        result = sim.execute(select(sim))
        assert not result['ok'] and result['holding'] and result['fault']
        goal = {'label': 'yellow', 'zone_name': 'zone_red'}
        joints = sim.arm.get_joints().value.copy()
        calls, frames = sim.arm.inner.n_calls, sim.camera.frames_rendered
        context = sim.prepare_recovery(result, goal, 1, [])
        assert context['observation_mode'] == 'stationary_with_last_scan'
        assert context['options'][0]['id'] == 'place_held_alternate'
        assert sim.arm.inner.n_calls == calls
        np.testing.assert_allclose(sim.arm.get_joints().value, joints, atol=1e-10)
        assert sim.camera.frames_rendered > frames
        assert any(v['label'] == '恢复时静止观测' for v in sim.observation_views())
        assert context['goal'] == goal
        with pytest.raises(ValueError):
            sim.prepare_recovery(result, dict(goal, zone_name='zone_green'), 1, [])
        repaired = sim.execute_recovery(choose(context, 'place_held_alternate'))
        assert repaired['ok'] and repaired['truth']['success']
        assert repaired['truth']['actual_zone'] == 'zone_red'
        assert not repaired['holding'] and sim.world.held is None
        with pytest.raises(ValueError):
            sim.execute_recovery(choose(context, 'place_held_alternate'))


def test_empty_grasp_reobserves_same_target_and_rejects_stale_recovery(tmp_path, monkeypatch):
    with SimulationSession(seed=0, realtime=False, log_dir=tmp_path) as sim:
        sim.prepare()
        pick = sim.runner.pick_detection
        monkeypatch.setattr(sim.runner, 'pick_detection', lambda *args, **kwargs:
                            (PickResult(ok=False, stage='grasp_miss', error='grasp_miss',
                                        label='yellow', zone='zone_red'), ''))
        result = sim.execute(select(sim))
        assert not result['ok'] and not result['holding']
        monkeypatch.setattr(sim.runner, 'pick_detection', pick)
        goal = {'label': 'yellow', 'zone_name': 'zone_red'}
        context = sim.prepare_recovery(result, goal, 1, [])
        assert context['observation_mode'] == 'target_search'
        assert context['options'][0]['id'] == 'reobserve_retry'
        sim.observe()
        calls = sim.arm.inner.n_calls
        with pytest.raises(ValueError, match='已变化'):
            sim.execute_recovery(choose(context, 'reobserve_retry'))
        assert sim.arm.inner.n_calls == calls
        context = sim.prepare_recovery(result, goal, 1, [])
        repaired = sim.execute_recovery(choose(context, 'reobserve_retry'))
        assert repaired['ok'] and repaired['truth']['actual_zone'] == 'zone_red'
