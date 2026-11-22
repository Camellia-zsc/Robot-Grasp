"""Placement feedback reports geometry independently of motion completion."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.catalog import make_catalog_object
from cube_task.core.types import Detection, PickResult, Zone


@pytest.fixture
def session():
    session = SimulationSession.__new__(SimulationSession)
    session.cancel_event = threading.Event()
    session._physics = None
    session._last_fresh = []
    session._scene = {'scene_version': 2, 'objects': []}
    session._feedback = {'grasp_plan': {}, 'checks': []}
    zone = Zone('zone_red', np.array([.17, -.35, 0]), (.2, .2), label='red')
    session._environment_revision = session._observation_revision = 0
    session.world = SimpleNamespace(held=None, score=Mock(return_value={}),
                                    zone_of=Mock(return_value=zone))
    session.runner = SimpleNamespace(class_to_zone={'apple': 'zone_red'},
                                     zones={'zone_red': zone})
    session.logger = Mock()
    session._invalidate_scene = Mock()
    session.observe = Mock()
    session._publish_snapshot = Mock()
    session._emit = Mock()
    return session


def test_placement_feedback_passes_when_correctly_placed_but_retreat_failed(session):
    motion_result = PickResult(ok=False, stage='retreat', error='rejected',
                               label='apple', zone='zone_red')
    summary = session._finish_attempt(
        motion_result, '', 'v1-object-1', 'zone_red',
        Detection('apple', 1., center_base=np.array([.17, -.35, .06]), height=.06),
        make_catalog_object('apple', 'catalog-apple-1', [.17, -.35]))

    assert summary['truth']['success'] is True
    assert summary['controller_ok'] is False
    assert summary['ok'] is False
    assert summary['stage'] == 'retreat' and summary['error'] == 'rejected'
    assert session._feedback['checks'][-1] == {
        'stage': 'placement', 'ok': True, 'message': '已独立核验物品落入指定区域'}


def test_placement_feedback_fails_when_motion_finished_outside_target_zone(session):
    session.world.zone_of.return_value = None
    motion_result = PickResult(ok=True, stage='done', label='apple', zone='zone_red')
    summary = session._finish_attempt(
        motion_result, '', 'v1-object-1', 'zone_red',
        Detection('apple', 1., center_base=np.array([.17, -.35, .06]), height=.06),
        make_catalog_object('apple', 'catalog-apple-1', [.4, .1]))

    assert summary['truth']['success'] is False
    assert summary['truth']['actual_zone'] is None
    assert summary['controller_ok'] is True
    assert summary['ok'] is False
    assert summary['error'] == 'placement_not_verified'
    assert session._feedback['checks'][-1] == {
        'stage': 'placement', 'ok': False, 'message': '落位核验未通过'}
