"""Local scans stop on the actual postcondition, not the first visible object."""
import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.core.types import Detection, PerceptionResult, PickResult, Zone


def block(label, xy=(.2, .3), top=.05, height=.05):
    return Detection(label, 1., center_base=np.r_[xy, top], top_z=top,
                     height=height, length=.05, width=.05, yaw=0.)


def scan_session(frames):
    session = object.__new__(SimulationSession)
    session._closed, session._prepared = False, True
    session._operation_lock = threading.RLock()
    session.cancel_event = threading.Event()
    session._environment_revision = session._observation_revision = 1
    session._observed_detections, session._interventions = [], []
    session._scan_stats = dict(local_scans=0, full_scans=0, frames=0)
    session._scene = dict(objects=[], zones=[])
    session.world = SimpleNamespace(held=None)
    session.arm = SimpleNamespace(get_flange_pose=lambda: SimpleNamespace(value=np.eye(4)))
    session.clock = SimpleNamespace(now=lambda: 10.)
    session._publish_snapshot = lambda: None
    session._emit = session._emit_scan_view = lambda *a, **kw: None
    session._cache_observation = lambda *a, **kw: None
    session._placement_cloud = session._stable_cloud = lambda *a: None
    session._view_order = lambda target: [0, 1, 2]
    seen = []
    def observe(view):
        seen.append(view)
        return 'ok', PerceptionResult(copy.deepcopy(frames[view]), frame_stamp=10.), SimpleNamespace(stamp=10.)
    session.runner = SimpleNamespace(_observe=observe, view_merge=.04,
        _partial=lambda d: False, ext=SimpleNamespace(parent_frame='flange', T=np.eye(4)),
        T_flange_tcp=np.eye(4), _last_obs_failure='')
    def publish(detections, warnings, diagnostics, stamp):
        session._scene['objects'] = [d.summary() for d in detections]
        return copy.deepcopy(session._scene)
    session._publish_scene = publish
    return session, seen


STACK_GOAL = dict(label='pear', destination=dict(type='on_object', label='green'))


def test_stack_verification_seeks_visible_support_then_stops():
    pear, green = block('pear', top=.16, height=.11), block('green')
    session, seen = scan_session([[pear], [pear, green], []])
    session.observe(target=dict(xy=[.2, .3], label='pear', verify_goals=[STACK_GOAL]),
                    purpose='verify_place')
    assert seen == [0, 1]
    from cube_task.assistant.stacking import camera_confirmed
    assert camera_confirmed(session, 'pear', STACK_GOAL['destination'])


def test_stack_verification_stops_after_one_view_with_support_evidence():
    pear, green = block('pear', top=.16, height=.11), block('green')
    session, seen = scan_session([[pear, green], [], []])
    session.observe(target=dict(xy=[.2, .3], verify_goals=[STACK_GOAL]), purpose='verify_place')
    assert seen == [0]


def test_stack_verification_does_not_turn_missing_support_into_full_scan():
    pear = block('pear', top=.16, height=.11)
    session, seen = scan_session([[pear], [pear], []])
    session.observe(target=dict(xy=[.2, .3], verify_goals=[STACK_GOAL]), purpose='verify_place')
    assert seen == [0, 1]


def test_batch_final_verification_searches_once_until_all_goals_are_seen():
    red, green = block('red', xy=(.16, .3)), block('green', xy=(.24, .3))
    session, seen = scan_session([[red], [green], []])
    zone = Zone('zone_red', np.array([.2, .3, 0.]), (.2, .2), label='red')
    session.runner.zones = {'zone_red': zone}
    session._scene['zones'] = [dict(name=zone.name, center=zone.center.tolist(),
                                   size=list(zone.size), yaw=zone.yaw)]
    session._physics = None
    session._evaluation_object = lambda d: None
    result = session.verify_goals([dict(label=c, zone_name='zone_red') for c in ('red', 'green')])
    assert result['ok']
    assert seen == [0, 1]
    assert session._scan_stats['local_scans'] == 1


def fresh_session():
    from test_assistant_completed_step import CameraSession
    session = CameraSession(block('red', xy=(.2, 0.)))
    session.clock = SimpleNamespace(now=lambda: 10.)
    session.runner.max_frame_age = 1.
    session.arm = SimpleNamespace(get_flange_pose=lambda: SimpleNamespace(value=np.eye(4)))
    session.runner.planner = SimpleNamespace(plan=lambda d: SimpleNamespace(feasible=True))
    session._last_observation_purpose = 'skill_observe'
    session._last_fresh[0].extra.update(_capture_stamp=10., _capture_flange=np.eye(4))
    session._reusable_target = lambda reference: SimulationSession._reusable_target(session, reference)
    return session


def test_explicit_observe_then_grasp_reuses_same_fresh_complete_frame():
    session = fresh_session()
    det, target = SkillExecutor(session)._fresh_target('red', 'recover_queued_target')
    assert det.label == target['label'] == 'red'
    assert not session.observations
    assert session._last_observation_purpose == 'recover_queued_target'


@pytest.mark.parametrize('change', ['old', 'moved_camera', 'revision', 'pending', 'consumed', 'verify', 'ambiguous', 'infeasible'])
def test_changed_or_unsafe_observation_requires_a_new_grasp_image(change):
    session = fresh_session()
    if change == 'old':
        session.clock = SimpleNamespace(now=lambda: 12.)
    elif change == 'moved_camera':
        pose = np.eye(4)
        pose[0, 3] = .02
        session.arm = SimpleNamespace(get_flange_pose=lambda: SimpleNamespace(value=pose))
    elif change == 'revision':
        session._environment_revision += 1
    elif change == 'pending':
        session._interventions.append({'action': 'move'})
    elif change == 'consumed':
        session._last_observation_purpose = 'consumed_for_pick'
    elif change == 'verify':
        session._last_observation_purpose = 'verify_place'
    elif change == 'ambiguous':
        session._last_fresh.append(block('red', xy=(.3, 0.)))
    elif change == 'infeasible':
        session.runner.planner.plan = lambda d: SimpleNamespace(feasible=False)
    SkillExecutor(session)._fresh_target('red', 'recover_queued_target')
    assert len(session.observations) == 1


def test_real_explicit_local_observation_is_not_repeated_before_grasp(tmp_path):
    with SimulationSession(realtime=False, log_dir=tmp_path) as session:
        session.prepare()
        executor = SkillExecutor(session)
        result = executor.execute(dict(id='see', skill='observe', label='red'), [],
            expected_revision=session._environment_revision)
        assert result['ok']
        before = copy.deepcopy(session._scan_stats)
        motions = session.arm.inner.n_calls
        target, _ = executor._fresh_target('red', 'recover_queued_target')
        assert target.label == 'red'
        assert session._scan_stats == before
        assert session.arm.inner.n_calls == motions
        session._last_observation_purpose = 'verify_place'
        executor._fresh_target('red', 'recover_queued_target')
        assert session._scan_stats['frames'] == before['frames'] + 1
        assert session.arm.inner.n_calls == motions
        before = copy.deepcopy(session._scan_stats)
        # Freshness expires even without a human edit. The next image is
        # captured in place when this camera view still covers the target.
        session.clock.sleep(session.runner.max_frame_age + .1)
        executor._fresh_target('red', 'recover_queued_target')
        assert session._scan_stats['frames'] == before['frames'] + 1
        assert session.arm.inner.n_calls == motions


def test_planning_refusal_does_not_advance_physics_or_scan(tmp_path, monkeypatch):
    with SimulationSession(realtime=False, log_dir=tmp_path) as session:
        session.prepare()
        target = next(d for d in session._observed_detections if d.label == 'red')
        obj = session._evaluation_object(target)
        def unexpected(*args, **kwargs):
            raise AssertionError('a planning refusal must not settle or reobserve')
        session._physics = SimpleNamespace(settle=unexpected,
            resting_status=lambda name: dict(success=True, posture_ok=True))
        monkeypatch.setattr(session, 'observe', unexpected)
        monkeypatch.setattr(session, '_verify_destination', lambda *a: False)
        monkeypatch.setattr(session, '_publish_snapshot', lambda: None)
        result = session._finish_attempt(None, 'no_clear_supported_release',
            'target', 'desktop_point', target, obj, dict(type='point', xy=[.3, 0.]))
        session._physics = None
        assert result['stage'] == 'plan' and not result['ok']


def test_missing_stack_image_is_reported_separately_from_physical_failure(tmp_path, monkeypatch):
    from cube_task.assistant import stacking
    with SimulationSession(realtime=False, log_dir=tmp_path) as session:
        session._last_fresh = []
        monkeypatch.setattr(session, 'observe', lambda **kw: session.scene)
        monkeypatch.setattr(session, '_verify_destination', lambda *a: True)
        monkeypatch.setattr(stacking, 'audit', lambda *a: dict(success=True, stable=True, contact=True))
        monkeypatch.setattr(stacking, 'camera_confirmed', lambda *a: False)
        result = session._finish_attempt(PickResult(ok=True, stage='done', label='pear'), '',
            'target', 'on_object_green', block('pear', top=.11, height=.11),
            session.world.objects[0], dict(type='on_object', label='green'))
        assert result['controller_ok'] and result['truth']['success']
        assert not result['ok'] and not result['camera_verified']
        assert result['error'] == 'placement_not_observed'
        assert '物理检查通过' in result['message'] and '支撑关系' in result['message']
        assert '没有位于' not in result['message']


@pytest.mark.parametrize('support_after', [1, 2, 3, None])
def test_stack_fallback_stops_on_support_and_has_a_three_view_budget(support_after):
    from cube_task.assistant.stacking import camera_confirmed
    pear, green = block('pear', top=.16, height=.11), block('green')
    session, seen = scan_session([[pear], [pear], []])
    session.runner.views = [np.full(6, n).tolist() for n in range(3)]
    proposals = []
    observe = session.runner._observe
    def capture(view):
        if view != 2:
            return observe(view)
        seen.append(view)
        visible = [pear, green] if len(proposals) == support_after else [pear]
        return 'ok', PerceptionResult(copy.deepcopy(visible), frame_stamp=10.), SimpleNamespace(stamp=10.)
    def propose(goals, fresh, revision, attempted):
        proposals.append([q.copy() for q in attempted])
        session.runner.views[2] = np.full(6, 1 + len(proposals)).tolist()
        return 2
    session.runner._observe = capture
    session._stack_verification_view = propose
    session.observe(target=dict(xy=[.2, .3], verify_goals=[STACK_GOAL]), purpose='verify_place')
    count = support_after or 3
    assert seen == [0, 1] + [2] * count and len(proposals) == count
    assert [len(a) for a in proposals] == list(range(2, count + 2))
    assert len(session.runner.views) == 3
    assert session._scan_stats == dict(local_scans=1, full_scans=0, frames=2 + count)
    assert camera_confirmed(session, 'pear', STACK_GOAL['destination']) is (support_after is not None)


def test_reused_stack_slot_keeps_complementary_captures_until_verification_finishes():
    from cube_task.assistant.stacking import camera_confirmed
    pear, green = block('pear', top=.16, height=.11), block('green')
    partial = block('pear', top=.16, height=.16)
    partial.valid = False
    session, seen = scan_session([[partial], [partial], []])
    session.runner.views = [np.full(6, n).tolist() for n in range(3)]
    session._stack_verification_view = lambda *args: 2
    frames = iter([[partial], [partial], [green], [pear]])
    def capture(view):
        seen.append(view)
        return 'ok', PerceptionResult(copy.deepcopy(next(frames)), frame_stamp=10.), SimpleNamespace(stamp=10.)
    captures = []
    def placement_cloud(det, views, fresh):
        captures.append([(d.label, d.extra['_capture_index']) for items in views.values() for d in items])
    session.runner._observe = capture
    session._placement_cloud = placement_cloud
    session.observe(target=dict(xy=[.2, .3], verify_goals=[STACK_GOAL]), purpose='verify_place')
    assert seen == [0, 1, 2, 2]
    assert captures and all(len(items) == 4 for items in captures)
    assert camera_confirmed(session, 'pear', STACK_GOAL['destination'])


def route_session(height):
    session = object.__new__(SimulationSession)
    session.world = SimpleNamespace(held=None)
    session.cancel_event = threading.Event()
    session._interventions = []
    samples = []
    def fk(q):
        samples.append(float(q[0]))
        pose = np.diag([1., -1., -1., 1.])
        pose[:3, 3] = [q[0], .3, height(float(q[0]))]
        return pose
    session.arm = SimpleNamespace(kin=SimpleNamespace(fk=fk), table_z=0.,
        get_joints=lambda: SimpleNamespace(value=np.zeros(6)),
        gripper_width=lambda: SimpleNamespace(value=.06),
        guard=SimpleNamespace(check_tcp=lambda *a, **kw: (True, '')))
    session.runner = SimpleNamespace(T_flange_tcp=np.eye(4))
    return session, samples


def test_observation_route_allows_low_start_far_from_objects_but_checks_every_segment():
    session, samples = route_session(lambda x: .10 + min(x / .10, 1.) * .20)
    assert session._safe_observation_joints([.3, 0, 0, 0, 0, 0], [block('pear', top=.16)])
    assert len(samples) >= 26
    assert np.max(np.diff(samples)) <= np.deg2rad(1.)


@pytest.mark.parametrize('reason', ['low_middle', 'workspace_middle', 'unknown_geometry', 'pending', 'holding'])
def test_observation_route_rejects_unsafe_or_unmeasured_sweep(reason):
    session, _ = route_session(lambda x: .12 if reason == 'low_middle' and .15 < x < .25 else .3)
    obstacle = block('unknown', top=.16)
    if reason == 'workspace_middle':
        session.arm.guard.check_tcp = lambda pose, *a, **kw: (not .15 < pose[0, 3] < .25, '')
    elif reason == 'unknown_geometry':
        obstacle.height = None
    elif reason == 'pending':
        session._interventions.append({'action': 'move'})
    elif reason == 'holding':
        session.world.held = object()
    assert not session._safe_observation_joints([.3, 0, 0, 0, 0, 0], [obstacle])


def test_observation_route_keeps_tool_clear_of_table_in_empty_space():
    session, _ = route_session(lambda x: .008)
    assert not session._safe_observation_joints([.3, 0, 0, 0, 0, 0], [block('red', xy=(.5, .5))])


def test_observation_route_does_not_fill_the_empty_space_between_open_fingers():
    session,_ = route_session(lambda x:.14)
    obstacle = block('green',top=.16,height=.16)
    obstacle.length = obstacle.width = .015
    assert session._safe_observation_joints([.3,0,0,0,0,0],[obstacle])
    obstacle.center_base[1] += .035  # The same object now enters a real finger.
    assert not session._safe_observation_joints([.3,0,0,0,0,0],[obstacle])


def test_observation_route_preserves_round_body_without_rotated_box_corner_obstacles():
    session,_ = route_session(lambda x:.14)
    obstacle = block('pear',xy=(.2,.39),top=.16,height=.16)
    obstacle.length=obstacle.width=.085
    obstacle.yaw=np.pi/4
    obstacle.extra['resting_pose']='upright'
    assert session._safe_observation_joints([.3,0,0,0,0,0],[obstacle])
    obstacle.center_base[1]-=.02
    assert not session._safe_observation_joints([.3,0,0,0,0,0],[obstacle])


@pytest.mark.parametrize('case',['clear','blocked_second_leg','changed'])
def test_empty_camera_lifts_via_gateway_only_when_both_joint_sweeps_are_checked(case):
    session,_ = route_session(lambda x:.14)
    session._environment_revision = 1
    session._emit = lambda *a: None
    session.arm.get_tcp_pose = lambda:SimpleNamespace(value=np.diag([1.,-1.,-1.,1.]))
    gateway = np.array([.5,0.,0.,0.,0.,0.])
    session.arm._ik_tcp = lambda *a:(gateway,0.,0.)
    checked,moved = [],[]
    def safe(q,ds,*,start_joints=None,escaping=False):
        checked.append((np.asarray(q).copy(),start_joints))
        return bool(q[0]==.5 or (start_joints is not None and case!='blocked_second_leg'))
    def move_joints(q):
        moved.append(q.copy())
        if case=='changed':session._environment_revision+=1
        return SimpleNamespace(ok=True)
    session._safe_observation_joints = safe
    session.arm.move_joints = move_joints
    result = session._prepare_observation_motion(np.array([1.,0.,0.,0.,0.,0.]),[])
    assert result is (case=='clear')
    assert len(moved)==(0 if case=='blocked_second_leg' else 1)
    assert any(start is not None for q,start in checked)


def test_empty_camera_can_use_a_checked_taught_gateway_when_fixed_attitude_ik_fails():
    from cube_task.core.transforms import top_down_T
    session, _ = route_session(lambda x: .3)
    session._environment_revision = 1
    session._emit = lambda *a: None
    session.arm.get_tcp_pose = lambda: SimpleNamespace(value=top_down_T(.2, 0., .1, 0.))
    gateway = np.array([.5, 0., 0., 0., 0., 0.])
    session.runner.views = [gateway]
    session.arm._ik_tcp = lambda *a: (None, 1., 1.)
    session._safe_observation_joints = lambda q, ds, start_joints=None, escaping=False: bool(
        np.allclose(q, gateway) or start_joints is not None)
    moves = []
    session.arm.move_joints = lambda q: moves.append(q) or SimpleNamespace(ok=True)
    assert session._prepare_observation_motion(np.ones(6), [])
    np.testing.assert_array_equal(moves, [gateway])


@pytest.mark.parametrize('case', ['raise', 'lower', 'actual_collision'])
def test_empty_gateway_can_only_leave_an_existing_margin_by_a_clear_upward_exit(case):
    start = .162 if case == 'actual_collision' else .177
    session, _ = route_session(lambda x: start+(.2*x if case != 'lower' else -.05*x))
    obstacle = block('green', xy=(0., .3), top=.16, height=.16)
    obstacle.length = obstacle.width = .2
    q = [.3, 0., 0., 0., 0., 0.]
    assert not session._safe_observation_joints(q, [obstacle])
    assert session._safe_observation_joints(q, [obstacle], escaping=True) is (case == 'raise')


def test_observation_route_never_shrinks_known_object_to_a_short_camera_measurement():
    session, _ = route_session(lambda x: .10)
    obstacle = block('pear', xy=(.2, .375), top=.16)
    obstacle.length = obstacle.width = .01
    assert session._safe_observation_joints([.3, 0, 0, 0, 0, 0], [obstacle])
    session.runner.nominal = {'pear': (.085, .055)}
    assert not session._safe_observation_joints([.3, 0, 0, 0, 0, 0], [obstacle])


def test_physical_final_audit_skips_a_taught_view_whose_joint_sweep_hits_the_stack():
    pear,green = block('pear',top=.16,height=.11),block('green')
    session,seen = scan_session([[pear,green],[pear,green],[]])
    session._physics = object()
    session.runner.views = [[float(i)]*6 for i in range(3)]
    checked = []
    session._safe_observation_joints = lambda q, ds: checked.append(q[0]) or q[0] != 0.
    session.observe(target=dict(xy=[.2,.3],verify_goals=[STACK_GOAL]),purpose='verify_place')
    assert checked == [0.,1.] and seen == [1]


def test_full_refresh_also_checks_known_physical_obstacles_before_camera_motion():
    pear = block('pear', top=.16, height=.11)
    session, seen = scan_session([[pear], [pear], [pear]])
    session._physics = object()
    session._observed_detections = [pear]
    session.runner.views = [[float(i)]*6 for i in range(3)]
    session._base_view_count = 3
    session._cache_lock = threading.Lock()
    checked = []
    session._safe_observation_joints = lambda q, ds: checked.append(q[0]) or q[0] != 0.
    session.observe()
    assert checked == [0., 1., 2.]
    assert seen == [1, 2]


def test_just_released_geometry_blocks_camera_sweep_before_a_fresh_image_exists():
    session,_ = route_session(lambda x:.12)
    session.runner.nominal = {}
    source = block('red',xy=(.5,.5))
    destination = np.eye(4)
    destination[:3,3] = [-.3,-.2,.11]
    assert session._safe_observation_joints([.3,0,0,0,0,0],[source])
    session._remember_released_obstacle(source,PickResult(True,detail={'released_transform':destination}))
    assert not session._safe_observation_joints([.3,0,0,0,0,0],[source])
    assert not session._released_observation_obstacle.valid
    assert '_released_observation_obstacle' not in session.__dict__.get('_scene',{})


def test_cropped_region_verification_can_use_checked_joint_views_without_inventing_a_support():
    session,_ = scan_session([])
    session.arm.kin = object()
    session.runner.nominal = {'pear':(.085,.085)}
    session._source_observation_views = lambda hint: []
    session._stack_verification_view = lambda *a: None
    captured = []
    session._stack_joint_view = lambda *args: captured.append(args) or np.ones(6)
    session._remember_stack_view = lambda q,stacked: 6 if not stacked else 7
    cropped = block('pear',xy=(.17,.3),top=.11,height=.11)
    cropped.valid = False
    view = session._verification_view([dict(label='pear',zone_name='zone_green')],[cropped],1)
    assert view==6 and captured[0][0].width==.085
    assert captured[0][1].label=='_table_aim' and not captured[0][1].valid
    assert session._scene['objects']==[]


def test_verification_keeps_earlier_release_aim_when_later_objects_were_moved():
    session, _ = scan_session([])
    session._stack_verification_view = lambda *a: None
    green = block('green', xy=(.33, -.12))
    green.valid = False
    session._released_observation_hints = {'green': green}
    session._released_observation_obstacle = block('apple', xy=(.27, .04))
    session._source_observation_views = lambda hint: [1] if hint['label'] == 'green' else []
    session.runner.views = [np.zeros(6), np.ones(6)]
    assert session._verification_view([dict(label='green', destination={'type':'region','name':'pick_area'})], [], 1) == 1
    assert not session._scene['objects']


@pytest.mark.parametrize('changed', [False, True])
def test_stack_fallback_reuses_transient_view_and_invalidates_old_arrival(changed):
    session, _ = scan_session([])
    session.arm.kin = SimpleNamespace(ik=lambda *a, **kw: (np.ones(6), 0., 0.))
    session.arm.get_joints = lambda: SimpleNamespace(value=np.zeros(6))
    def checked(*args):
        if changed:
            session._environment_revision += 1
        return True
    session._safe_observation_joints = checked
    session._observed_detections = [block('green')]
    session._base_view_count = 2
    session.runner.views = [np.zeros(6).tolist() for _ in range(3)]
    session._view_labels = ['first', 'second', 'old point']
    session._destination_views = {(.2, .3): 2}
    session.runner._at_observe, session.runner._at_view = True, 2
    upper = block('pear', top=.16, height=.16)
    upper.valid = False
    view = session._stack_verification_view([STACK_GOAL], [upper], 1)
    assert len(session.runner.views) == 3
    if changed:
        assert view is None and session.runner._at_observe
    else:
        assert view == 2
        assert not session.runner._at_observe and not session._destination_views
        session.cfg = SimpleNamespace(get=lambda *a: 0.)
        session.runner._at_observe = True
        assert session._ensure_destination_view(dict(type='point', xy=[.3, .1])) == 2
        assert session._view_labels[2] == '桌面目标点视角'
        assert not session.runner._at_observe and len(session.runner.views) == 3
