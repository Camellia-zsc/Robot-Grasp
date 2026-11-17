"""Restore the recorded held fixture without turning simulator truth into detections.

World geometry/joints initialize the simulator only. The planner's grasp and
observations are reconstructed exclusively from the recorded camera/log data.
The returned session must be closed by its caller. No server or API is touched.
"""
import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType
import numpy as np

BASE = Path(__file__).parent
PKG = BASE.parent
sys.path.insert(0, str(PKG)); os.environ['CUBE_TASK_ROOT'] = str(PKG)
from cube_task.assistant.simulation import SimulationSession
from cube_task.core.transforms import inv_T, top_down_T
from cube_task.core.types import Detection, GraspPlan, PlacePlan
from cube_task.sim.world import SimObject


def detection(summary):
    d = Detection(**{k: v for k, v in summary.items() if k in Detection.__dataclass_fields__})
    d.center_base = np.asarray(d.center_base, float)
    if 'view' in summary:
        d.extra['view'] = summary['view']
    return d


def reconstruct_session(log_dir=None, recorded=None, realtime=False):
    """Return a ready independent session at the exact recorded held state.

    ``recorded`` may be an already loaded sanitized snapshot. The default is
    fixtures/assistant_held_banana_scene.json, containing no chat/API settings.
    Saved camera summaries have rounded values and lack raw clouds; full-size
    public nominal geometry remains active, so this never claims an exact raw
    RGB-D cache reconstruction. All world transforms and joints are exact.
    """
    if recorded is None:
        recorded = json.loads((BASE / 'fixtures/assistant_held_banana_scene.json').read_text(encoding='utf-8'))
    state = recorded['simulation']
    s = SimulationSession(seed=state['seed'], realtime=realtime,
                          log_dir=Path(log_dir or tempfile.mkdtemp(prefix='held-banana-fixture-')))
    try:
        s.world.held = None
        s.world.objects = [SimObject(o['id'], o['label'], o['shape'], o['size'],
                            list(reversed(o['color_rgb'])), o['transform'])
                           for o in state['objects']]
        s.arm.inner.set_joints(state['joints'])
        s.arm.inner._width = s.arm.inner._cmd_width = state['gripper_width']
        s.world.held = next(o for o in s.world.objects if o.name == state['held_object'])
        s.world.held_width = state['gripper_width']
        tcp = s.arm.get_tcp_pose().value.copy()
        s.world._T_tcp_obj = inv_T(tcp) @ s.world.held.T_base_obj
        s.world.update_tcp(tcp, state['gripper_width'])
        np.testing.assert_allclose(tcp, state['tcp_transform'], atol=1e-10)
        for obj, original in zip(s.world.objects, state['objects']):
            np.testing.assert_allclose(obj.T_base_obj, original['transform'], atol=1e-10)
        command = next(e for e in reversed(recorded['events']) if e.get('kind') == 'decision'
                       and e.get('action') == 'pick' and e.get('label') == 'banana')
        target = detection(command['target'])
        grasp_pose = top_down_T(*command['grasp_xyz'], command['closing_angle'])
        grasp = GraspPlan(target, grasp_pose.copy(), grasp_pose, grasp_pose.copy(),
                          command['open_width'], command['close_width'],
                          command['expected_width'], command['closing_angle'])
        place = PlacePlan('zone_yellow', tcp.copy(), top_down_T(.48, .01, .018, math.pi / 4),
                          tcp.copy(), .072)
        measured = [detection(o) for o in recorded['scene']['objects']]
        neighbors = [d for d in measured if not (d.label == target.label
                     and np.linalg.norm(d.center_base[:2] - target.center_base[:2]) < .03)]
        s._prepared = True
        s._scene = copy.deepcopy(recorded['scene'])
        s._scene_version = recorded['scene']['scene_version']
        s._detections = MappingProxyType({o['object_id']: d for o, d in
                                         zip(recorded['scene']['objects'], measured)})
        s._observed_detections = copy.deepcopy(measured)
        s._recovery_target = copy.deepcopy(target)
        s._recovery_goal = {'label': 'banana', 'zone_name': 'zone_yellow'}
        s._recovery_neighbors = copy.deepcopy(neighbors)
        s._recovery_evaluation = s.world.held
        s._recovery_context = None
        s._active_placement = (target, None, 'zone_yellow', {})
        s._environment_revision = s._active_revision = 2
        s._observation_revision = 2
        s._scan_stats = copy.deepcopy(state['observation_stats'])
        s.runner._held_recovery_context = {
            'det': copy.deepcopy(target), 'grasp': copy.deepcopy(grasp),
            'place': copy.deepcopy(place), 'width': state['gripper_width'],
            'zone': 'zone_yellow', 'avoid': [(0.49, 0.), (.48, -.01), (.48, .01)]}
        s._status, s._stage = 'error', 'inspect_lift'
        s._publish_snapshot()
        return s
    except BaseException:
        s.close()
        raise
