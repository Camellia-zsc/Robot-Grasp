"""Validate actual model recovery once, then keep the normal dashboard available.

The one-shot guarded-motion refusal below is test instrumentation only. It is
removed before this script publishes the final ready marker. No keys are read
here; the normal model client owns the existing local connection configuration.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--local-prefill', action='store_true',
                    help='first two placements already verified online; repeat locally without paid calls')
args = parser.parse_args()
sys.path.insert(0, str(ROOT / 'ws' / 'src' / 'cube_task'))

from cube_task.assistant.server import DashboardController, DashboardServer
from cube_task.core.types import MotionResult


def wait_task(controller, limit=300):
    started = time.monotonic()
    previous = None
    lengths, recoveries, failures = set(), [], []
    while time.monotonic() - started < limit:
        state = controller.snapshot()
        stage = (state['status'], state['stage'])
        if stage != previous:
            print(json.dumps({'elapsed': round(time.monotonic() - started, 1),
                              'status': stage[0], 'stage': stage[1]}, ensure_ascii=False),
                  flush=True)
            previous = stage
        replies = [m for m in state['messages'] if m['role'] == 'assistant']
        if replies:
            lengths.add((replies[-1]['id'], len(replies[-1]['content'])))
        if state.get('last_result') and not state['last_result'].get('ok'):
            if not failures:
                failures.append(state['last_result'])
        if state['recovery']['status'] != 'idle':
            if not recoveries or recoveries[-1] != state['recovery']['status']:
                recoveries.append(state['recovery']['status'])
        if not state['busy']:
            return {'elapsed_wall_s': round(time.monotonic() - started, 2),
                    'stream_lengths_seen': sorted(lengths), 'recovery_states': recoveries,
                    'initial_failures': failures, 'state': state}
        time.sleep(.1)
    raise TimeoutError('dashboard task did not finish')


controller = DashboardController(realtime=False,
                                 log_dir=ROOT / 'local_validation' / 'capacity_recovery')
server = DashboardServer(('127.0.0.1', 8767), controller)
threading.Thread(target=server.serve_forever, daemon=True).start()
reports = []
restore_move = None
try:
    initial = wait_task(controller)
    assert initial['state']['status'] == 'ready', initial['state']['error']
    sim = controller._session
    for label, instruction in [('red', '把红色方块放到红色放置区'),
                               ('green', '把绿色方块放到红色放置区')]:
        if args.local_prefill:
            scene = sim.scene
            target = next(o for o in scene['objects'] if o['label'] == label and o['valid'])
            result = sim.execute({'scene_version': scene['scene_version'], 'action': 'pick',
                                  'object_id': target['object_id'], 'zone_name': 'zone_red'})
            assert result['ok'], result
            with controller._lock:
                controller._scene = sim.scene
            reports.append({'instruction': instruction, 'method': 'local measured replay',
                            'result': result})
            continue
        controller.submit(instruction)
        report = wait_task(controller)
        assert report['state']['last_result']['ok'], report['state']['last_result']
        reports.append({'instruction': instruction, **report})
    previous = {o.name: o.T_base_obj.copy() for o in sim.world.objects if o.label != 'yellow'}
    restore_move = sim.arm.move_tcp
    rejected = {'release', 'recover_up'}

    def reject_once(T, linear=False, **kwargs):
        if sim._stage in rejected:
            rejected.remove(sim._stage)
            return MotionResult(ok=False, error='rejected',
                                detail='验证注入：本次放置路径被拒绝，需保持夹持并另选空位')
        return restore_move(T, linear=linear, **kwargs)

    sim.arm.move_tcp = reject_once
    instruction = '把黄色方块放到红色放置区'
    controller.submit(instruction)
    report = wait_task(controller)
    reports.append({'instruction': instruction, **report})
    assert report['state']['last_result']['ok'], report['state']['last_result']
    assert report['state']['recovery']['status'] == 'succeeded'
    assert report['state']['recovery']['attempt'] == 1
    assert not rejected, 'failure injection did not exercise held placement recovery'
    assert report['initial_failures'] and report['initial_failures'][0]['holding']
    for obj in sim.world.objects:
        assert sim.world.zone_of(obj).name == 'zone_red'
        if obj.name in previous:
            np.testing.assert_allclose(obj.T_base_obj, previous[obj.name], atol=1e-8)
    metrics = controller.snapshot()['metrics']
    expected = ({'requests': 1, 'successes': 1, 'model_requests': 2} if args.local_prefill else
                {'requests': 3, 'successes': 3, 'model_requests': 4})
    assert metrics == expected, metrics
    assert controller.snapshot()['messages'][-1]['kind'] == 'recovery'
    (ROOT / 'local_validation' / 'three_cubes_red_observation.jpg').write_bytes(
        sim.observation_frame_jpeg(5))
    print('Three cubes in red zone and real model-guided held recovery verified.', flush=True)
finally:
    if restore_move is not None:
        controller._session.arm.move_tcp = restore_move
    controller.realtime = True
    if controller._session is not None:
        controller._session.clock.realtime = True
    controller.set_speed(1)
    (ROOT / 'local_validation' / 'capacity_recovery_online.json').write_text(json.dumps(
        {'verification_pacing': 'unpaced L1, restored realtime afterward',
         'failure_injection': 'one guarded release/recover_up refusal, removed afterward',
         'runs': reports}, ensure_ascii=False, indent=2), encoding='utf-8')

print('Verified normal dashboard ready: http://127.0.0.1:8767', flush=True)
(ROOT / 'local_validation' / 'recovery_dashboard_ready.json').write_text(json.dumps(
    {'url': 'http://127.0.0.1:8767', 'validation_complete': True,
     'instrumentation_removed': True, 'metrics': controller.snapshot()['metrics']},
    ensure_ascii=False, indent=2), encoding='utf-8')
try:
    threading.Event().wait()
except KeyboardInterrupt:
    pass
finally:
    server.shutdown()
    server.server_close()
    controller.close()
