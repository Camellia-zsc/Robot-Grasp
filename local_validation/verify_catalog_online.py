"""Fill a three-fruit scene; --execute makes one normal, potentially paid request.

Start the dashboard first. This script uses its public loopback HTTP interface,
does not read credentials, and never bypasses the normal model or motion checks.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-url', default='http://127.0.0.1:8768')
parser.add_argument('--execute', action='store_true', help='call the configured online model')
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('catalog_online_validation.json'))
args = parser.parse_args()


def call(path, data=None):
    payload = None if data is None else json.dumps(data).encode()
    request = urllib.request.Request(args.base_url.rstrip('/') + path, data=payload,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def wait_idle(limit=240):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        state = call('/api/state')
        if not state['busy']:
            if state.get('error'):
                raise RuntimeError(state['error'])
            return state
        time.sleep(0.2)
    raise TimeoutError('dashboard operation did not complete')


state = wait_idle()


def edit(action, **fields):
    global state
    call('/api/scene-edit', dict(scene_version=state['simulation']['scene_version'],
                                action=action, **fields))
    state = wait_idle()


edit('clear')
for label, xy, yaw in [('banana', [.26, -.12], -.4),
                       ('pear', [.26, .07], .6), ('apple', [.35, .05], .3)]:
    edit('add', catalog_id=label, xy=xy, yaw=yaw)
record = {'mode': 'kinematic', 'setup_objects': state['simulation']['objects'],
          'paid_request_requested': args.execute}
if args.execute:
    call('/api/command', {'instruction': '把香蕉放到黄色区域'})
    deadline = time.monotonic() + 240
    previous = None
    observed = None
    reply_lengths = set()
    while time.monotonic() < deadline:
        state = call('/api/state')
        stage = (state['status'], state['stage'])
        if stage != previous:
            print(json.dumps({'status': stage[0], 'stage': stage[1]}, ensure_ascii=False), flush=True)
            previous = stage
        if state['status'] == 'thinking':
            observed = state['scene']
        for message in state['messages']:
            if message['role'] == 'assistant':
                reply_lengths.add(len(message['content']))
        if not state['busy']:
            break
        time.sleep(.2)
    else:
        raise TimeoutError('online task did not complete')
    record.update(result=state.get('last_result'), observation=observed,
                  feedback=state['simulation'].get('feedback'), metrics=state['metrics'],
                  public_replies=[m['content'] for m in state['messages'] if m['role'] == 'assistant'],
                  stream_lengths_seen=sorted(reply_lengths), error=state['error'])
args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
if args.execute:
    result = record.get('result') or {}
    print(json.dumps({'ok': result.get('ok'), 'message': result.get('message'),
                      'metrics': state['metrics'], 'output': str(args.output)}, ensure_ascii=False))
    if not result.get('ok'):
        raise SystemExit(1)
