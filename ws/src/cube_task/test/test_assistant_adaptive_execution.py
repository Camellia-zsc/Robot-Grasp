"""Real camera/IK execution for adaptive grasps, interventions and local scans.

Only initial experimental placement and independent assertions read world truth.
All task coordinates, clutter and replacement plans use production perception.
Online intent is replaced by a single local response and networking is disabled.
"""
import copy
import json
from pathlib import Path
import socket
import threading

import numpy as np
import pytest

from cube_task.assistant.server import DashboardController
from cube_task.assistant.simulation import SimulationSession
from cube_task.sim.world import SimObject


FIXTURES = Path(__file__).with_name('fixtures')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('adaptive execution regression must not access a network/API')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def pick(sim, label, zone='zone_red', destination=None):
    scene = sim.scene
    matches = [obj for obj in scene['objects'] if obj['valid'] and obj['label'] == label]
    assert len(matches) == 1, scene['objects']
    decision = dict(action='pick', scene_version=scene['scene_version'],
                    object_id=matches[0]['object_id'], zone_name=zone if destination is None else None)
    if destination is not None:
        decision['destination'] = destination
    return decision


def recorded_objects():
    fixture = json.loads((FIXTURES / 'assistant_banana_neighbor_scene.json').read_text(encoding='utf-8'))
    return [SimObject(item['id'], item['label'], item['shape'], item['size'],
                       item['color_rgb'][::-1], item['transform']) for item in fixture['objects']]


def collect(events):
    return lambda kind, fields: events.append((kind, copy.deepcopy(fields)))


def save_report(tmp_path, name, **data):
    path = tmp_path / (name + '.json')
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def save_camera_evidence(sim, tmp_path):
    for metadata in sim.observation_views():
        jpeg = sim.observation_frame_jpeg(metadata['view_id'])
        if jpeg:
            (tmp_path / ('observation-%d.jpg' % metadata['view_id'])).write_bytes(jpeg)


def test_exact_banana_neighbor_scene_returns_to_pick_area_without_full_rescan(tmp_path):
    events = []
    with SimulationSession(realtime=False, log_dir=tmp_path, on_event=collect(events)) as sim:
        sim.world.objects = recorded_objects()
        sim.prepare()
        assert sim._scan_stats['full_scans'] == 1 and sim._scan_stats['frames'] == 6
        before = {obj.name: obj.T_base_obj.copy() for obj in sim.world.objects if obj.label != 'banana'}
        result = sim.execute(pick(sim, 'banana', destination={'type': 'region', 'name': 'pick_area'}))
        save_report(tmp_path, 'exact_banana', result=result, scans=sim._scan_stats,
                    observations=[fields for kind, fields in events if kind == 'observation_complete'])
        assert result['ok'] and result['controller_ok'] and result['truth']['success'], result
        assert not result['holding'] and sim.world.held is None
        assert sim._scan_stats['full_scans'] == 1
        confirm = [fields for kind, fields in events
                   if kind == 'observation_complete' and fields['purpose'] == 'confirm_pick']
        assert len(confirm) == 1 and len(confirm[0]['views']) < 6, confirm
        assert any(obj['valid'] and obj['label'] == 'banana' for obj in sim.scene['objects'])
        for obj in sim.world.objects:
            if obj.name in before:
                np.testing.assert_array_equal(obj.T_base_obj, before[obj.name])


def queue_from_another_thread(sim, object_id, xy):
    """Exercise the same nonblocking queue boundary as an external UI request."""
    answers, failures = [], []

    def send():
        try:
            answers.append(sim.intervene_scene(dict(action='move', object_id=object_id,
                             xy=list(xy), scene_version=sim.snapshot()['scene_version'])))
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=send)
    worker.start()
    worker.join(3.)
    assert not worker.is_alive(), 'intervention waited for the entire execution lock'
    assert not failures, failures
    assert answers and answers[0]['queued']


def test_target_moved_during_pre_aborts_old_grasp_then_reobserves_and_recovers(tmp_path):
    events, injected = [], []
    with SimulationSession(realtime=False, log_dir=tmp_path) as sim:
        sim.prepare()
        before = {obj.name: obj.T_base_obj.copy() for obj in sim.world.objects if obj.label != 'red'}
        red = next(obj for obj in sim.world.objects if obj.label == 'red')
        moved_xy = [.29, .20]  # Empty supported tabletop, >10 cm from the old target.

        def event(kind, fields):
            events.append((kind, copy.deepcopy(fields)))
            if kind == 'stage' and fields['stage'] == 'pre' and not injected:
                injected.append(True)
                queue_from_another_thread(sim, red.name, moved_xy)

        sim.on_event = event
        failed = sim.execute(pick(sim, 'red'))
        assert injected and any(kind == 'scene_intervention_applied' for kind, _ in events)
        assert not failed['ok'] and failed['error'] == 'scene_changed', failed
        assert not failed['holding'] and sim.world.held is None
        assert not any(kind == 'skill_step' and fields.get('stage') == 'close'
                       for kind, fields in events), events
        np.testing.assert_allclose(red.center_xy(), moved_xy, atol=1e-8)
        context = sim.prepare_recovery(failed, {'label': 'red', 'zone_name': 'zone_red'}, 1, [])
        assert context['observation_mode'] == 'target_search'
        assert any(option['id'] == 'reobserve_retry' for option in context['options']), context
        recovered = sim.execute_recovery(dict(action='recover', option_id='reobserve_retry',
                              scene_version=context['scene_version'], message='按新测位置恢复原抓取任务'))
        save_report(tmp_path, 'moved_target_recovery', initial=failed, recovered=recovered,
                    scans=sim._scan_stats)
        assert recovered['ok'] and recovered['truth']['success'], recovered
        assert sim._scan_stats['full_scans'] == 1
        for obj in sim.world.objects:
            if obj.name in before:
                np.testing.assert_array_equal(obj.T_base_obj, before[obj.name])


def test_neighbor_moved_into_planned_slot_during_transit_causes_local_new_slot(tmp_path):
    events, initial_slot = [], []
    with SimulationSession(realtime=False, log_dir=tmp_path) as sim:
        sim.prepare()
        green = next(obj for obj in sim.world.objects if obj.label == 'green')
        untouched = {obj.name: obj.T_base_obj.copy() for obj in sim.world.objects
                     if obj.label not in ('red', 'green')}

        def event(kind, fields):
            events.append((kind, copy.deepcopy(fields)))
            if kind == 'stage' and fields['stage'] == 'transit' and not initial_slot:
                # Stage is announced before transit; the held red object is
                # still safely lifted at its source, away from this empty slot.
                assert sim.world.held is not None and sim.world.held.label == 'red'
                initial_slot.extend(sim._placement_center[:2])
                queue_from_another_thread(sim, green.name, initial_slot)

        sim.on_event = event
        result = sim.execute(pick(sim, 'red'))
        save_camera_evidence(sim, tmp_path)
        save_report(tmp_path, 'occupied_slot_replan', result=result, initial_slot=initial_slot,
                    final_slot=sim._placement_center, scans=sim._scan_stats,
                    independent_world=sim.snapshot(), events=events)
        assert initial_slot and any(kind == 'scene_intervention_applied' for kind, _ in events), events
        assert result['ok'] and result['truth']['success'], result
        np.testing.assert_allclose(green.center_xy(), initial_slot, atol=1e-8)
        assert np.linalg.norm(np.asarray(sim._placement_center)[:2] - initial_slot) > .04
        assert any(kind == 'placement_refined' for kind, _ in events)
        red = next(obj for obj in sim.world.objects if obj.label == 'red')
        assert sim.world.zone_of(red).name == 'zone_red'
        assert np.linalg.norm(red.center_xy() - green.center_xy()) > .05
        assert sim._scan_stats['full_scans'] == 1
        for obj in sim.world.objects:
            if obj.name in untouched:
                np.testing.assert_array_equal(obj.T_base_obj, untouched[obj.name])


class OfflineSettings:
    def public_info(self):
        return dict(configured=True, provider='offline-test', model='one-batch', api_style='responses')


class OneBatchClient:
    def __init__(self):
        self.calls = 0

    def decide(self, instruction, scene, on_delta, cancel_event=None):
        self.calls += 1
        items = [dict(object_id=next(obj['object_id'] for obj in scene['objects']
                                    if obj['valid'] and obj['label'] == label), zone_name=zone)
                 for label, zone in (('red', 'zone_red'), ('green', 'zone_green'))]
        result = dict(action='pick_batch', scene_version=scene['scene_version'], items=items,
                      message='保留同一批计划，依次搬红块和绿块。')
        on_delta(json.dumps(result, ensure_ascii=False))
        return result


class RecordedSession(SimulationSession):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.executions = []
        self.observations = []

    def _emit(self, kind, fields):
        if kind == 'observation_complete' and hasattr(self, 'observations'):
            self.observations.append(copy.deepcopy(fields))
        super()._emit(kind, fields)

    def execute(self, decision, cancel_event=None):
        result = super().execute(decision, cancel_event)
        self.executions.append(copy.deepcopy(result))
        return result


def test_two_real_sequential_tasks_share_one_intent_and_initial_full_scan(tmp_path):
    client = OneBatchClient()
    controller = DashboardController(realtime=False, session_factory=RecordedSession,
                                    client=client, settings=OfflineSettings(), log_dir=tmp_path)
    try:
        assert controller.wait_idle(180), 'initial scan timed out'
        assert controller.snapshot()['status'] == 'ready', controller.snapshot()
        controller.submit('依次把红块放红区、绿块放绿区')
        assert controller.wait_idle(240), 'sequential tasks timed out'
        state = controller.snapshot()
        session = controller._session
        save_report(tmp_path, 'two_object_queue', state=state, scans=session._scan_stats,
                    observations=session.observations)
        assert state['status'] == 'complete', state
        assert client.calls == 1 and state['metrics']['model_requests'] == 1
        assert state['batch']['completed'] == state['batch']['total'] == 2
        assert len(session.executions) == 2 and all(result['ok'] for result in session.executions)
        assert session._scan_stats['full_scans'] == 1
        confirms = [obs for obs in session.observations
                    if obs['purpose'] in ('confirm_pick', 'recover_queued_target')]
        assert len(confirms) == 2 and all(len(obs['views']) < 6 for obs in confirms)
        # Previously prepare + submit + two finish scans alone captured 24
        # processed views. Count inspections too, avoiding hidden scan costs.
        assert session._scan_stats['frames'] < 24, session._scan_stats
        assert session.world.held is None
    finally:
        controller.close()
