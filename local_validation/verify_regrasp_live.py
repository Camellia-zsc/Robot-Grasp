"""Two real model requests through the fixed local dashboard (no credential access)."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

BASE = 'http://127.0.0.1:8766'
OUT = Path(__file__).resolve().parent


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as response:
        return json.load(response)


def post(path, payload):
    request = urllib.request.Request(BASE + path, data=json.dumps(
        payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def wait_ready():
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        state = get('/api/state')
        if not state['busy']:
            assert state['status'] == 'ready', state.get('error')
            return state
        time.sleep(.25)
    raise TimeoutError('initial preparation')


def run(instruction, expected_zone, image_name):
    post('/api/command', {'instruction': instruction})
    started = time.monotonic()
    last_stage = None
    images, joints, stream_lengths, camera_times = set(), set(), set(), set()
    last_image = 0
    while time.monotonic() - started < 360:
        state = get('/api/state')
        stage = (state['status'], state['stage'])
        if stage != last_stage:
            print(json.dumps({'elapsed': round(time.monotonic() - started, 1),
                              'status': stage[0], 'stage': stage[1]},
                             ensure_ascii=False), flush=True)
            last_stage = stage
        sim = state.get('simulation', {})
        joints.add(tuple(round(q, 4) for q in sim.get('joints', [])))
        camera_times.add(state.get('camera', {}).get('sim_time'))
        replies = [m for m in state['messages'] if m['role'] == 'assistant']
        if replies:
            stream_lengths.add(len(replies[-1]['content']))
        if time.monotonic() - last_image >= .6:
            with urllib.request.urlopen(BASE + '/api/camera.jpg?mode=live', timeout=15) as response:
                content = response.read()
                if response.status == 200 and content:
                    images.add(hashlib.sha256(content).hexdigest())
            last_image = time.monotonic()
        if not state['busy']:
            break
        time.sleep(.2)
    else:
        raise TimeoutError(instruction)
    result = state.get('last_result') or {}
    assert result.get('ok'), {'error': state['error'], 'result': result}
    assert result['truth']['success'] and result['truth']['actual_zone'] == expected_zone
    red = [o for o in state['scene']['objects'] if o['label'] == 'red' and o['valid']]
    assert len(red) == 1 and red[0]['zone'] == expected_zone, red
    assert len(images) > 3, 'live camera did not change'
    assert len(state['observation_views']) == 6
    seen = [v for v in state['observation_views'] if 'red' in v['detected_labels']]
    assert seen
    chosen = next((v for v in seen if '放置区' in v['label']), seen[0])
    with urllib.request.urlopen(BASE + '/api/camera.jpg?mode=observation&view=' +
                                str(chosen['id']), timeout=15) as response:
        (OUT / image_name).write_bytes(response.read())
    report = {'instruction': instruction,
              'elapsed_wall_s': round(time.monotonic() - started, 2),
              'distinct_live_frames': len(images), 'distinct_joint_states': len(joints),
              'distinct_camera_times': len(camera_times),
              'stream_lengths_seen': sorted(stream_lengths),
              'measured_red': red, 'observation_views': state['observation_views'],
              'reply': replies[-1] if replies else None,
              'result': result, 'metrics': state['metrics']}
    print(json.dumps({'verified': expected_zone, 'live_frames': len(images)},
                     ensure_ascii=False), flush=True)
    return report


wait_ready()
reports = []
try:
    post('/api/speed', {'speed': 4})
    reports.append(run('把红色方块抓到黄色放置区里', 'zone_yellow',
                       'regrasp_yellow_observation.jpg'))
    reports.append(run('把红色方块抓到绿色放置区里', 'zone_green',
                       'regrasp_green_observation.jpg'))
finally:
    post('/api/speed', {'speed': 1})
    (OUT / 'regrasp_live_online.json').write_text(json.dumps(
        {'base_url': BASE, 'runs': reports}, ensure_ascii=False, indent=2), encoding='utf-8')
assert len(reports) == 2
print('Both real online regrasp tasks and live camera checks passed.', flush=True)
