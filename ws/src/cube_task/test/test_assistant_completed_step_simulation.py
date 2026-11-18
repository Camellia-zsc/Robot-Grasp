"""Real RGB-D replay of a task completed by a manual scene intervention."""
import copy
import json
import socket
import time

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor


def test_real_human_completed_goal_is_locally_seen_without_regrasp(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('completion replay must not call an online API')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    events = []
    with SimulationSession(seed=0, log_dir=tmp_path, realtime=False,
            on_event=lambda kind, value: events.append((kind, copy.deepcopy(value)))) as session:
        initial = session.prepare()
        executor = SkillExecutor(session, originals=initial['objects'])
        target = next(o for o in session.world.objects if o.label == 'red')
        zone = session.runner.zones['zone_red']
        initial_revision = session._environment_revision
        session.intervene_scene(dict(action='move', object_id=target.name,
            xy=zone.center[:2].tolist(), yaw=0.,
            scene_version=session.snapshot()['scene_version']))
        assert session._environment_revision == initial_revision + 1
        assert any(kind == 'scene_intervention_applied' for kind, value in events)
        before = copy.deepcopy(session._scan_stats)
        events.clear()

        def no_regrasp(*args, **kwargs):
            raise AssertionError('an already completed target must not be regrasped')
        monkeypatch.setattr(session, 'execute', no_regrasp)
        result = executor.execute(dict(id='m', skill='move', label='red',
            destination={'type': 'goal', 'index': 0}),
            [{'label': 'red', 'zone_name': 'zone_red'}],
            expected_revision=session._environment_revision)
        assert result['ok'] and result['skipped'] and result['camera_verified'], result
        assert result['truth']['success'] and not result['holding'], result
        assert session._scan_stats['full_scans'] == before['full_scans']
        assert session._scan_stats['local_scans'] == before['local_scans'] + 1
        assert session._scan_stats['frames'] - before['frames'] <= 2
        assert not any(kind == 'stage' and value.get('stage') in
                       ('descend', 'close', 'lift', 'release') for kind, value in events)
        (tmp_path / 'human-completed-goal.json').write_text(json.dumps(dict(
            result=result, scan_delta={k: session._scan_stats[k] - before[k]
                                      for k in before}), ensure_ascii=False, indent=2),
            encoding='utf-8')
        counts = dict(snapshots=0, intervention_checks=0)
        publish, apply = session._publish_snapshot, session._apply_interventions
        def counted_publish():
            counts['snapshots'] += 1
            return publish()
        def counted_apply():
            counts['intervention_checks'] += 1
            return apply()
        monkeypatch.setattr(session, '_publish_snapshot', counted_publish)
        monkeypatch.setattr(session, '_apply_interventions', counted_apply)
        session._last_snapshot_wall = time.monotonic() - 1.
        started = time.perf_counter()
        for _ in range(200):
            session._tick()
        benchmark = dict(ticks=200, elapsed_seconds=time.perf_counter() - started, **counts)
        assert counts['intervention_checks'] == 200
        (tmp_path / 'tick-after.json').write_text(json.dumps(benchmark, indent=2), encoding='utf-8')
        print('Completion replay evidence:', tmp_path)
        print('Tick benchmark:', json.dumps(benchmark))
