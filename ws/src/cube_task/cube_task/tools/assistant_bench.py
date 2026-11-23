"""Replay representative conversational tasks through the real dashboard controller."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
import time

import numpy as np

from ..assistant.catalog import make_catalog_object
from ..assistant.server import DashboardController
from ..assistant.simulation import SimulationSession


LABELS = ('red', 'green', 'yellow', 'apple', 'banana', 'pear')
INSTRUCTIONS = {
    1: ('把所有物品放到各自默认区域',),
    2: ('把梨子放到黄色方块上面', '把黄色方块放到黄色区域'),
    3: ('把梨子放到黄色方块上面', '把所有物品放到各自默认区域'),
    4: ('把所有六个物品全部放回抓取区',),
    5: ('把抓取区的香蕉和梨子都放到红色区域，依次完成',),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def random_objects(session, seed):
    """Generate a reproducible nonoverlapping tabletop scene, independently of planning."""
    random = np.random.default_rng(seed)
    for _ in range(100):
        objects = []
        for label in random.permutation(LABELS):
            for _ in range(300):
                xy = random.uniform([.18, -.24], [.48, .24])
                yaw = random.uniform(-np.pi, np.pi)
                candidate = make_catalog_object(str(label), 'bench-' + label, xy, yaw)
                try:
                    session.editor._check_placement(candidate, objects)
                except ValueError:
                    continue
                # A small additional setup gap permits ordinary independent
                # grasps while retaining random positions and orientations.
                if any(np.linalg.norm(candidate.center_xy() - other.center_xy())
                       < .5 * max(candidate.aabb()[1][:2] - candidate.aabb()[0][:2])
                       + .5 * max(other.aabb()[1][:2] - other.aabb()[0][:2]) + .02
                       for other in objects):
                    continue
                objects.append(candidate)
                break
            else:
                break
        if len(objects) == len(LABELS):
            return objects
    raise RuntimeError('Could not generate all six objects in the declared random workspace')


def stacking_objects(session):
    """Six objects with a reachable empty base and the other objects nearby."""
    layout = dict(red=(.12, -.37, 0.), apple=(.22, -.32, 0.),
                  green=(.12, .37, 0.), banana=(.54, 0., np.pi / 2),
                  yellow=(.25, 0., 0.), pear=(.35, .14, .3))
    objects = []
    for label, (x, y, yaw) in layout.items():
        obj = make_catalog_object(label, 'bench-' + label, [x, y], yaw)
        session.editor._check_placement(obj, objects)
        objects.append(obj)
    return objects


class BenchSession(SimulationSession):
    def __init__(self, *, stacking=False, intervention=False, **kwargs):
        super().__init__(backend='mujoco', **kwargs)
        if intervention:
            layout = dict(red=(.44, -.2, 0.), apple=(.32, -.2, 0.),
                          green=(.12, .37, 0.), yellow=(.54, 0., 0.),
                          banana=(.25, -.04, 0.), pear=(.32, .13, .3))
            objects = []
            for label, (x, y, yaw) in layout.items():
                obj = make_catalog_object(label, 'bench-' + label, [x, y], yaw)
                self.editor._check_placement(obj, objects)
                objects.append(obj)
            self.world.objects = objects
        else:
            self.world.objects = stacking_objects(self) if stacking else random_objects(self, self.seed)
        self.bench_intervention = intervention
        self._physics.apply_scene_changes()
        self._remember_physics()
        for zone in self.world.zones:
            assert tuple(zone.size) == tuple(self.runner.zones[zone.name].size)
            assert tuple(zone.size) == tuple(self._physics.mujoco_world.zones[zone.name].size)
            assert min(zone.size) >= .30

        invalidate = self._motion_observer.invalidate
        self._invalidations_saved = 0
        def record_invalidation(labels, **options):
            import traceback
            labels = tuple(labels)
            if self._invalidations_saved < 6 and labels:
                self._invalidations_saved += 1
                path = self.logger.dir / ('evidence-invalidation-%d.pkl' % self._invalidations_saved)
                path.write_bytes(pickle.dumps(dict(labels=labels, options=options,
                    evidence={label:copy.deepcopy(self._motion_observer.evidence.get(label))
                              for label in labels}, frame=getattr(self.camera, '_last', None),
                    pose=self.camera.current_T_base_camera(), revision=self._environment_revision,
                    stage=self._stage, stack=[(Path(t.filename).name,t.name,t.lineno)
                        for t in traceback.extract_stack(limit=5)])))
            return invalidate(labels, **options)
        self._motion_observer.invalidate = record_invalidation

        # A rejected sweep needs the exact camera/grip inputs for an offline
        # diagnosis. Never pickle the controller, model client or credentials.
        from ..assistant import placement_observation
        original = getattr(placement_observation.safe_inspection_route, '_bench_original',
                           placement_observation.safe_inspection_route)
        self._route_refusals = 0
        def record_route(arm, grasp, current, destination, neighbors, opening, **options):
            route, reason = original(arm, grasp, current, destination, neighbors, opening, **options)
            if not route and self._route_refusals < 6:
                self._route_refusals += 1
                path = self.logger.dir / ('route-refusal-%d.pkl' % self._route_refusals)
                path.write_bytes(pickle.dumps(dict(grasp=grasp, current=current,
                    destination=destination, neighbors=neighbors, opening=opening,
                    joints=arm.get_joints().value, reason=reason, frame=self.camera._last,
                    camera_pose=self.camera.current_T_base_camera())))
            return route, reason
        record_route._bench_original = original
        placement_observation.safe_inspection_route = record_route
        replan = self.runner.replan_release
        def record_release(det, grasp, previous, width, neighbors, **options):
            visibility = options.get('observation_filter')
            counts = dict(checked=0, accepted=0)
            if visibility is not None:
                def measured(*args, **kwargs):
                    value = visibility(*args, **kwargs)
                    counts['checked'] += 1
                    counts['accepted'] += bool(value)
                    return value
                options['observation_filter'] = measured
            started = time.monotonic()
            result = replan(det, grasp, previous, width, neighbors, **options)
            with (self.logger.dir/'release-trace.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(label=det.label, ok=result[0] is not None,
                    why=result[1], wall_seconds=time.monotonic()-started, visibility=counts))+'\n')
            if result[0] is None and not (self.logger.dir/'release-input.pkl').exists():
                from ..core.extrinsics import camera_pose_in_base
                (self.logger.dir/'release-input.pkl').write_bytes(pickle.dumps(dict(
                    det=det, grasp=grasp, previous=previous, width=width, neighbors=neighbors,
                    frame=self.camera._last, joints=self.arm.get_joints().value,
                    pose=camera_pose_in_base(self.runner.ext, self.arm.get_flange_pose().value,
                                            self.runner.T_flange_tcp),
                    options={k:v for k,v in options.items() if k in
                             ('placement_zone','placement_polygon','exact_xy')},
                    observations=self._motion_observer.support_views())))
            return result
        self.runner.replan_release = record_release


def run_phase(controller, instruction, output, timeout):
    output.mkdir(parents=True, exist_ok=True)
    session = controller._session
    before = copy.deepcopy(session._scan_stats)
    motion_before = copy.deepcopy(session._motion_observer.stats)
    metrics_before = copy.deepcopy(controller.snapshot()['metrics'])
    event_start = controller._next_id
    started = time.monotonic()
    approvals = []
    seen_approvals = set()
    last_plan = None
    last_checkpoint = 0.
    saved_grips = set()
    drops = []
    intervention = {}
    planned = {}
    observation_counts = dict(stationary_frames=0, inspection_motion_steps=0)
    diagnostic_views = 0
    picked_labels = []
    listener = session.on_event
    def record_event(kind, payload):
        nonlocal diagnostic_views
        if kind == 'decision' and payload.get('action') in ('pick', 'pick_only'):
            planned.update(label=payload['label'], xy=payload.get('place', {}).get('xy'))
            picked_labels.append(payload['label'])
        if (getattr(session, 'bench_intervention', False) and not intervention
                and kind == 'skill_step' and payload.get('stage') == 'lift'
                and planned.get('label') in ('banana', 'pear')):
            first = planned['label']
            other = 'pear' if first == 'banana' else 'banana'
            zone = session.runner.zones['zone_red']
            # The human deliberately occupies the planner's provisional slot.
            # Use UI catalog dimensions only to keep the edit inside the card.
            margin = .055 if other == 'pear' else .085
            half = np.asarray(zone.size)/2-margin
            intended = zone.center[:2] if planned['xy'] is None else planned['xy']
            xy = np.clip(np.asarray(intended), zone.center[:2]-half, zone.center[:2]+half)
            snap = session.snapshot()
            obj = next(o for o in snap['objects'] if o['label'] == other)
            queued = controller.intervene_scene(dict(action='move', object_id=obj['id'],
                xy=xy.tolist(), yaw=0., environment_revision=snap['environment_revision']))
            intervention.update(first=first, moved=other, xy=xy.tolist(),
                                original_slot=planned['xy'], queued=queued)
        if kind == 'scene_intervention_applied':
            intervention['applied'] = True
        if kind == 'scene_intervention_rejected':
            intervention.update(applied=False, error=payload.get('message'))
        if kind == 'scene' and intervention.get('applied') and session.snapshot().get('held_object'):
            from ..assistant.batch import whole_footprint_in_zone
            observed = [o for o in payload.get('objects', ())
                        if o.get('label') == intervention['moved'] and o.get('valid')]
            if len(observed) == 1 and whole_footprint_in_zone(observed[0], payload, 'zone_red'):
                intervention['camera_updated_while_holding'] = True
        if kind == 'placement_refined' and intervention.get('applied'):
            intervention['final_slot'] = payload.get('position')
        if kind == 'recovery_observation':
            observation_counts['stationary_frames'] += 1
            if intervention and diagnostic_views < 8 and session.snapshot().get('held_object'):
                diagnostic_views += 1
                from ..core.extrinsics import camera_pose_in_base
                (output/('stationary-view-%d.pkl' % diagnostic_views)).write_bytes(pickle.dumps(dict(
                    frame=session.camera._last, detections=session._last_fresh,
                    grasp=getattr(session.runner.skill, 'active_grasp', None),
                    tcp=session.arm.get_tcp_pose().value,
                    pose=camera_pose_in_base(session.runner.ext, session.arm.get_flange_pose().value,
                                            session.runner.T_flange_tcp),
                    revision=session._environment_revision,
                    invalidated=sorted(session._motion_observer.invalidated))))
        if kind == 'skill_step' and payload.get('stage') == 'inspect_place':
            observation_counts['inspection_motion_steps'] += 1
        if kind == 'skill_step' and payload.get('stage') in ('close', 'lift', 'transit', 'inspect_place'):
            tcp = session.arm.get_tcp_pose().value
            objects = {o.label: o.T_base_obj.tolist() for o in session.world.objects}
            with (output / 'motion-diagnostics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=payload, tcp=tcp.tolist(), objects=objects),
                                        ensure_ascii=False)+'\n')
        if kind in ('grasp_anchor_refined', 'placement_changed', 'placement_refined',
                    'physics_event', 'inspection_motion_incomplete'):
            with (output / 'diagnostics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(event=kind, **payload), ensure_ascii=False)+'\n')
        if kind == 'physics_event' and payload.get('kind') == 'dropped':
            drops.append(copy.deepcopy(payload))
        if listener is not None:
            listener(kind, payload)
    session.on_event = record_event
    controller.submit(instruction)
    timed_out = False
    while not controller.wait_idle(1.):
        state = controller.snapshot()
        task = state['skill_task']
        if time.monotonic() - last_checkpoint > 5.:
            write_json(output / 'progress-state.json', state)
            last_checkpoint = time.monotonic()
            context = getattr(session.runner, '_held_recovery_context', None)
            key = (task.get('plan_number'), len(task.get('history', ())))
            if context is not None and key not in saved_grips:
                saved_grips.add(key)
                # Private diagnostic evidence, never part of model context.
                (output / ('grip-%s-%s.pkl' % key)).write_bytes(pickle.dumps(
                    dict(grasp=copy.deepcopy(context['grasp']), width=context['width'])))
            if task.get('history') and task['history'][-1].get('ok') is not True:
                (output / ('source-%s-%s.pkl' % key)).write_bytes(pickle.dumps(
                    copy.deepcopy(session._last_fresh)))
        plan = task.get('plan_number')
        if plan != last_plan:
            last_plan = plan
            print(json.dumps(dict(instruction=instruction, plan=plan, stage=state['stage'],
                                  steps=task.get('steps', [])),
                             ensure_ascii=False), flush=True)
        approval = state['recovery_approval']
        if approval.get('status') == 'pending' and approval['request_id'] not in seen_approvals:
            seen_approvals.add(approval['request_id'])
            approvals.append(copy.deepcopy(approval))
            # Isolated benchmark scenes pre-authorize temporary tabletop
            # rearrangement. The production approval mechanism still runs.
            controller.approve_recovery(dict(request_id=approval['request_id'],
                scene_version=approval['scene_version'], approved=True))
        if time.monotonic() - started > timeout:
            timed_out = True
            controller.stop()
            controller.wait_idle(30.)
            break
    state = controller.snapshot()
    session.on_event = listener
    goals = state['skill_task']['goals']
    truth = []
    for goal in goals:
        obj = next((o for o in session.world.objects if o.label == goal['label']), None)
        valid = (session._verify_destination(obj, goal['destination']) if goal.get('destination')
                 else session._verify_zone(obj, goal['zone_name']))
        truth.append(dict(goal=goal, ok=bool(valid)))
    history = state['skill_task']['history']
    events = [e for e in state['events'] if e.get('id', 0) > event_start]
    summary = dict(instruction=instruction, complete=state['status'] == 'complete',
        all_requested_goals=(sorted(g['label'] for g in goals) == sorted(LABELS)
                             if '所有' in instruction else True),
        physical_success=bool(truth) and all(item['ok'] for item in truth),
        goals=goals, truth=truth, timed_out=timed_out,
        wall_seconds=round(time.monotonic() - started, 2),
        scans={key: session._scan_stats[key] - before[key] for key in before},
        zone_sizes={name:list(zone.size) for name,zone in session.runner.zones.items()},
        motion_observation={key: session._motion_observer.stats[key]-motion_before[key]
                            for key in motion_before},
        observation_work=observation_counts, picked_labels=picked_labels,
        model_requests=state['metrics']['model_requests'] - metrics_before['model_requests'],
        approvals=len(approvals), plans=state['skill_task']['plan_number'],
        failed_steps=[h for h in history if not h.get('ok') and not h.get('deferred')],
        deferred_steps=[h for h in history if h.get('deferred')],
        manipulation_steps=sum(h.get('skill') in ('pick', 'place', 'move')
                               and h.get('skipped') is not True and not h.get('deferred')
                               and h.get('manipulation_started') is not False for h in history),
        drops=len(drops), unsupported_drops=sum(not d.get('support') for d in drops),
        drop_events=drops,
        message=state['stage'], history=history)
    if getattr(session, 'bench_intervention', False):
        moved = intervention.get('moved')
        extra_grasps = [label for label in picked_labels if label == moved]
        summary['intervention'] = dict(intervention, redundant_grasps=len(extra_grasps),
            passed=bool(intervention.get('applied') and not extra_grasps
                        and intervention.get('camera_updated_while_holding')
                        and summary['complete'] and summary['physical_success']
                        and {g['label'] for g in goals} == {'banana', 'pear'}
                        and all(g.get('zone_name') == 'zone_red' for g in goals)))
    write_json(output / 'summary.json', summary)
    write_json(output / 'dashboard-state.json', state)
    write_json(output / 'approvals.json', approvals)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def run_case(case, seed, output, timeout):
    directory = output / ('case%d-seed%d' % (case, seed))
    directory.mkdir(parents=True, exist_ok=True)
    def session_factory(**kwargs):
        return BenchSession(stacking=case in (2, 3), intervention=case == 5, **kwargs)
    controller = DashboardController(seed=seed, realtime=False, session_factory=session_factory,
                                     log_dir=directory / 'logs', backend='mujoco')
    try:
        if not controller.wait_idle(180.):
            raise RuntimeError('Scene preparation did not finish')
        initial = controller.snapshot()
        if initial['status'] != 'ready':
            raise RuntimeError(initial['error'])
        write_json(directory / 'initial-state.json', initial)
        print(json.dumps(dict(case=case, seed=seed, ready=True,
            model=initial['model'], observed=[(o['label'], o['valid'])
                                            for o in initial['scene']['objects']]),
                         ensure_ascii=False), flush=True)
        phases = []
        for number, instruction in enumerate(INSTRUCTIONS[case], 1):
            phase = run_phase(controller, instruction, directory / ('phase%d' % number), timeout)
            phases.append(phase)
            if not (phase['complete'] and phase['physical_success'] and phase['all_requested_goals']):
                break
        result = dict(case=case, seed=seed, phases=phases,
            complete=len(phases) == len(INSTRUCTIONS[case]) and all(
                p['complete'] and p['physical_success'] and p['all_requested_goals'] for p in phases))
        if case == 5:
            result['complete'] = result['complete'] and all(
                p.get('intervention', {}).get('passed') for p in phases)
        write_json(directory / 'result.json', result)
        return result
    finally:
        controller.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default='1,2,3,4')
    parser.add_argument('--seeds', default='7,19')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1200.)
    args = parser.parse_args()
    cases = [int(v) for v in args.cases.split(',')]
    seeds = [int(v) for v in args.seeds.split(',')]
    if any(case not in INSTRUCTIONS for case in cases):
        parser.error('cases must contain 1, 2, 3, 4, or 5')
    results = []
    for case in cases:
        for seed in seeds:
            results.append(run_case(case, seed, args.output, args.timeout))
            write_json(args.output / 'results.json', results)
    return 0 if all(result['complete'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
