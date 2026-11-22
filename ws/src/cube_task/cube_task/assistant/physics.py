"""Actual MuJoCo robot/contact dynamics behind the dashboard geometry view.

The SimWorld is a read/render mirror and an editing surface only. Its grasp
attachment and upright release methods are never used by this adapter. Every
object pose and held state comes from the MuJoCo free bodies and finger contacts.
"""
from __future__ import annotations

import copy
import threading
import time
from pathlib import Path

import numpy as np

from ..core.config import Config, deep_merge
from ..core.transforms import inv_T
from ..robot.kinematics import flange_to_tcp_from_cfg
from ..robot.safety import SafeArm, SafetyGuard
from ..sim.mujoco.arm import MujocoArm
from ..sim.mujoco.ik import MujocoIK
from ..sim.mujoco.scene import build_scene
from ..sim.sim_camera import SimCamera


class PhysicsKinematics:
    """Flange FK/IK in the actual MJCF; snapshot frames use those same bodies."""

    def __init__(self, world, limits=None):
        self._solver = MujocoIK(world.model, body=world.meta['flange_body'], limits=limits)
        self._mj = world._mj
        self._lock = threading.RLock()
        self._bodies = [self._mj.mj_name2id(world.model, self._mj.mjtObj.mjOBJ_BODY, name)
                        for name in ('base_link', 'link1', 'link2', 'link3',
                                     'link4', 'link5', 'link6')]

    @property
    def limits(self):
        return self._solver.limits

    def within_limits(self, q, tol=1e-9):
        return self._solver.within_limits(q, tol)

    def limit_violations(self, q, tol=1e-9):
        return self._solver.limit_violations(q, tol)

    def fk(self, q):
        with self._lock:
            return self._solver.fk(q)

    def fk_all(self, q):
        with self._lock:
            self._solver._set_q(np.asarray(q, float))
            return [self._solver.T_base_world @ self._solver._body_T(int(b))
                    for b in self._bodies]

    def ik(self, target, **kwargs):
        with self._lock:
            # Forward the computation deadline into the iterative solver too.
            return self._solver.solve(target, **kwargs)


class DashboardPhysics:
    """Keep dashboard identities stable while the physical model can be rebuilt.

    ``apply_scene_changes`` is required after editor/list changes. It teleports
    existing free bodies only for explicit human edits; adding/removing objects
    rebuilds the MJCF and preserves the physical joint state. ``sync`` must run
    before rendering or auditing. Normal time advancement synchronizes it too.
    """

    backend = 'mujoco'
    REST_NOSLIP_ITERATIONS = 30
    REST_NOSLIP_TOLERANCE = 1e-10

    def __init__(self, cfg, geometry_world, info, clock, on_event=None):
        self.world = geometry_world
        self.info = copy.deepcopy(info)
        self.clock = clock
        self.on_event = on_event
        self.expected_labels = set()
        self._closed = False
        self._lock = threading.RLock()
        self._previous = {}
        self._supports = []
        self._last_event = {}
        self._settling = False
        self.arm = None
        self.camera = None
        self.mujoco_world = None
        self.cfg = Config(copy.deepcopy(cfg.data), cfg.root, list(cfg.sources))
        # Match the project's configured physical stroke. The scene builder
        # changes both finger joint ranges; forces/contact remain simulated.
        # A separately configured MJCF stroke always takes precedence.
        if self.cfg.get('sim.mujoco.gripper_max_width', None) is None:
            self.cfg.set('sim.mujoco.gripper_max_width',
                         float(self.cfg.get('robot.gripper.max_width', .070)))
        if not self.cfg.get('sim.mujoco.assets_dir', None):
            bundled = Path(__file__).resolve().parents[5] / 'local_assets/mujoco_menagerie/agilex_piper'
            if (bundled / 'piper.xml').is_file():
                self.cfg.set('sim.mujoco.assets_dir', str(bundled))
        # A thin display card must not alter a taught tabletop destination z.
        self.cfg.set('sim.mujoco.card_collision', False)
        # Keep the held body's belly in the wrist camera's depth field instead
        # of imaging only a narrow ring behind its neck. This is a physical
        # mounting change: build_scene derives both the rendered pose and its
        # calibrated extrinsics from this same copied transform.
        offset_key = 'assistant.physics.wrist_camera_flange_z_offset'
        try:
            camera_offset = float(self.cfg.get(offset_key, -.025))
        except (TypeError, ValueError):
            raise ValueError(offset_key + ' must be a finite distance in metres') from None
        if not np.isfinite(camera_offset):
            raise ValueError(offset_key + ' must be a finite distance in metres')
        self.cfg.set(offset_key, camera_offset)
        camera_pose = np.asarray(self.info['T_flange_camera'], float).copy()
        camera_pose[2, 3] += camera_offset
        self.info['T_flange_camera'] = camera_pose
        # Apply once to our scenario copy, never again during scene rebuilds.
        self._build()
        self.clock.add_listener(self._after_advance)
        self.sync(emit=False)

    def _remove_engine_listener(self):
        if self.mujoco_world is None or self.mujoco_world.sim is None:
            return
        listener = self.mujoco_world.sim._on_advance
        self.clock._listeners[:] = [fn for fn in self.clock._listeners if fn != listener]

    def _build(self):
        previous_arm = self.arm
        raw = previous_arm.inner if previous_arm is not None else None
        connected = bool(raw is not None and raw._connected)
        enabled = bool(raw is not None and raw._enabled)
        width = raw._width_meas() if raw is not None else 0.
        q = raw._q_meas() if raw is not None else None
        abort = getattr(previous_arm, 'abort_event', None)
        if q is not None:
            self.cfg.set('robot.poses.home_joints', q.tolist())
        self._remove_engine_listener()
        source = self.world.copy()
        source.held = None
        current_info = copy.deepcopy(self.info)
        current_info['objects'] = [obj.to_dict() for obj in source.objects]
        self.mujoco_world = mw = build_scene(current_info, self.cfg, world=source,
                                            clock=self.clock)
        # Keep the compiled user configuration separate from temporary rest
        # precision. Rebuilding the model never copies the previous mode.
        self._contact_solver_config = (int(mw.model.opt.noslip_iterations),
                                       float(mw.model.opt.noslip_tolerance))
        self._rest_contact_precision = False
        if not hasattr(mw.data, 'qM'):
            raise RuntimeError('当前物理驱动需要 MuJoCo 3.3.7，请安装 requirements-physics.txt')
        self.cfg = Config(deep_merge(self.cfg.data, mw.config_overlay()), self.cfg.root,
                          list(self.cfg.sources))
        # The geometric sensor renders measured physical poses; no graphics
        # context or second dynamics engine is created by its preview thread.
        self.cfg.set('camera.backend', 'sim')
        inner = MujocoArm(self.cfg, self.clock, mw.sim)
        # The driver intersects the MJCF ranges with the controller's joint
        # limits. Planning must use that same intersection before moving.
        kin = PhysicsKinematics(mw, limits=inner.ik.limits)
        ft = flange_to_tcp_from_cfg(self.cfg)
        if previous_arm is None:
            self.arm = SafeArm(inner, SafetyGuard(self.cfg), kin=kin, T_flange_tcp=ft,
                table_check_joint_moves=bool(self.cfg.get('robot.safety.table_check_joint_moves', True)))
        else:
            # TaskRunner/Skill keep the SafeArm instance; only its physical
            # driver and kinematics change when the edited object set changes.
            previous_arm._arm = inner
            previous_arm._kin = kin
            previous_arm._T_ft, previous_arm._T_tf = ft, inv_T(ft)
            self.arm = previous_arm
        self.arm.set_table_z(mw.table_z)
        # Planning budgets charge this worker's computation, not time spent
        # waiting behind the live viewer or another validation process.
        self.arm.planning_time = time.thread_time
        if abort is not None:
            self.arm.set_abort_event(abort)
        if q is not None:
            inner.set_joints(q)
        if width > 0:
            m, d, mj = mw.model, mw.data, mw._mj
            for name, value in (('joint7', width / 2), ('joint8', -width / 2)):
                j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
                d.qpos[m.jnt_qposadr[j]] = value
            inner._grip_target = inner._grip_pos = inner._cmd_width = width
            mj.mj_forward(m, d)
        if connected:
            self.arm.connect()
        if enabled:
            self.arm.enable()
        self.world.table_z = mw.table_z
        for zone in self.world.zones:
            if zone.name in mw.zones:
                zone.center = mw.zones[zone.name].center.copy()
        if self.camera is None:
            self.camera = SimCamera(self.world, self.clock, mw.intrinsics,
                T_flange_camera=mw.T_flange_camera, arm=self.arm,
                noise_std=float(self.cfg.get('sim.noise_std', .0008)),
                dropout=float(self.cfg.get('sim.dropout', .002)),
                seed=int(self.cfg.get('sim.seed', 0)))
        else:
            self.camera._intr = mw.intrinsics
            self.camera.T_flange_camera = mw.T_flange_camera.copy()
        # Rebuilds must not retain the obsolete engine/controller listener.
        if self._after_advance in self.clock._listeners:
            self.clock._listeners.remove(self._after_advance)
            self.clock.add_listener(self._after_advance)
        self._previous.clear()

    def _after_advance(self, _before, _after):
        if not self._closed:
            self.sync()

    def _event(self, kind, obj, *, support=None, reason='', expected=False):
        names = {'grasped': '夹爪接触已夹持', 'released': '夹爪接触已释放',
                 'dropped': '物品从夹爪掉落', 'rolled': '物品发生滚动或位移',
                 'settled': '物品已静止', 'support_lost': '物品失去下方支撑'}
        support_obj = next((o for o in self.world.objects if o.name == support), None)
        event = dict(kind=kind, object=obj.name, label=obj.label, support=support,
            support_label=None if support_obj is None else support_obj.label,
            expected=bool(expected), reason=reason, message=names.get(kind, kind),
            sim_time=self.clock.now())
        self._last_event = event
        if self.on_event is not None:
            self.on_event('physics_event', copy.deepcopy(event))

    def _support_contacts(self, held):
        mw = self.mujoco_world
        by_name = {obj.name: obj for obj in self.world.objects}
        relations = {}
        for contact in mw.data.contact[:int(mw.data.ncon)]:
            if float(contact.dist) > .002:
                continue
            a, b = (int(g) for g in contact.geom)
            for ga, gb in ((a, b), (b, a)):
                upper = mw.geom_object(ga)
                lower = mw.geom_object(gb)
                if upper is None or upper in held:
                    continue
                obj = by_name[upper]
                if lower is None:
                    if gb not in (mw.meta['table_geom'], mw.meta['floor_geom']):
                        continue
                    support = 'table' if gb == mw.meta['table_geom'] else 'floor'
                else:
                    if (lower in held or mw.object_com(upper)[2] <= mw.object_com(lower)[2]
                            or obj.bottom_z() < by_name[lower].bottom_z() + .003):
                        continue
                    support = lower
                relations[(upper, support)] = dict(object=upper, label=obj.label,
                    support=support, support_label=by_name[support].label if support in by_name else None)
        return list(relations.values())

    def sync(self, emit=True):
        """Copy only real physical state into the dashboard world."""
        with self._lock, self.mujoco_world._locked():
            mw = self.mujoco_world
            by_name = {o.name: o for o in self.world.objects}
            held = set(mw.held_objects())
            for rec in mw.objects:
                if rec.name in by_name:
                    by_name[rec.name].T_base_obj = mw.object_pose(rec.name)
            self.world.held = next((by_name[n] for n in held if n in by_name), None)
            self.world._T_tcp_obj = None  # never attach a geometry object to TCP
            raw = self.arm.inner
            self.world.tcp_pose = raw._tcp_meas()
            self.world.gripper_width = raw._width_meas()
            self.world.held_width = self.world.gripper_width if held else None
            self._supports = self._support_contacts(held)
            supports = {r['object']: r['support'] for r in self._supports}
            settling = False
            for rec in mw.objects:
                obj = by_name.get(rec.name)
                if obj is None:
                    continue
                speed, angular = mw.object_speed(rec.name)
                moving = speed > .02 or angular > .5
                settling |= moving and rec.name not in held
                com = mw.object_com(rec.name)
                old = self._previous.get(rec.name)
                support = supports.get(rec.name)
                expected = obj.label in self.expected_labels
                reported_motion = bool(old and old.get('reported_motion'))
                if old and emit:
                    if rec.name in held and not old['held']:
                        self.world.last_grasp = dict(ok=True, object=obj.name,
                            width=self.world.gripper_width, reason='physical_finger_contacts')
                        self._event('grasped', obj, expected=expected)
                    elif old['held'] and rec.name not in held:
                        opened = raw._grip_target > old['width'] + .003
                        kind = 'released' if opened else 'dropped'
                        self.world.last_release = dict(ok=opened, object=obj.name,
                            reason='physical_contacts_lost', support=support)
                        self._event(kind, obj, support=support, expected=expected and opened)
                    elif old['support'] and support is None and moving and rec.name not in held:
                        self._event('support_lost', obj, support=old['support'], expected=expected)
                    elif old['moving'] and not moving and rec.name not in held:
                        self._event('settled', obj, support=support, expected=expected)
                        if not reported_motion and np.linalg.norm(com[:2]-old['anchor'][:2]) > .008:
                            self._event('rolled', obj, support=support, expected=expected)
                    elif (moving and rec.name not in held and not reported_motion
                          and np.linalg.norm(com[:2]-old['anchor'][:2]) > .008):
                        self._event('rolled', obj, support=support, expected=expected)
                        reported_motion = True
                anchor = (old['anchor'] if old and moving else com.copy())
                self._previous[rec.name] = dict(held=rec.name in held, moving=moving,
                    support=support, anchor=anchor, width=self.world.gripper_width,
                    reported_motion=reported_motion if moving else False)
            self._settling = bool(settling)
        return self.world

    def apply_scene_changes(self):
        """Apply explicit editor mutations to real free bodies, never to grasps."""
        with self._lock:
            mw = self.mujoco_world
            previous_supports = copy.deepcopy(self._supports)
            names = {o.name for o in self.world.objects}
            if names != {o.name for o in mw.objects}:
                if mw.held_objects():
                    raise ValueError('夹持期间不能增删物理物品')
                self._build()
            else:
                for obj in self.world.objects:
                    actual = mw.object_pose(obj.name)
                    if not np.allclose(actual, obj.T_base_obj, atol=1e-10, rtol=0):
                        if obj.name in mw.held_objects():
                            raise ValueError('不能直接移动夹爪持有的物品')
                        mw.set_object_pose(obj.name, obj.T_base_obj)
            self.sync(emit=False)
            current_supports = {(r['object'], r['support']) for r in self._supports}
            for relation in previous_supports:
                if (relation['support'] not in ('table', 'floor')
                        and (relation['object'], relation['support']) not in current_supports):
                    obj = next((o for o in self.world.objects if o.name == relation['object']), None)
                    if obj is not None and obj is not self.world.held:
                        self._event('support_lost', obj, support=relation['support'],
                                    reason='support_changed_by_intervention')
        return dict(arm=self.arm, camera=self.camera)

    def settle(self, seconds=.6):
        self.clock.sleep(float(seconds))
        return self.sync()

    def set_rest_contact_precision(self, eligible):
        """Keep released stacks accurate while other bodies are manipulated.

        Called after sync and before the next clock advance. Real contacts only
        select numerical precision; they do not supply a placement or success.
        No body state, forces, friction or contact constraints are modified.
        """
        with self._lock, self.mujoco_world._locked():
            # _support_contacts already excludes contacts involving a held
            # body. An unrelated grasp must not disable a resting stack's
            # numerical precision during execution, observation or inference.
            active = bool(eligible and any(
                relation['support'] not in ('table', 'floor') for relation in self._supports))
            iterations, tolerance = self._contact_solver_config
            if active:
                iterations = max(iterations, self.REST_NOSLIP_ITERATIONS)
                tolerance = min(tolerance, self.REST_NOSLIP_TOLERANCE)
            options = self.mujoco_world.model.opt
            options.noslip_iterations, options.noslip_tolerance = iterations, tolerance
            self._rest_contact_precision = active

    def state(self):
        return dict(backend='mujoco', gravity=self.mujoco_world.model.opt.gravity.tolist(),
                    settling=self._settling, support_relations=copy.deepcopy(self._supports),
                    last_event=copy.deepcopy(self._last_event),
                    grip_model='physical_finger_contacts',
                    gripper_max_width=float(self.mujoco_world.max_width),
                    contact_solver=dict(rest_precision=self._rest_contact_precision,
                        noslip_iterations=int(self.mujoco_world.model.opt.noslip_iterations),
                        noslip_tolerance=float(self.mujoco_world.model.opt.noslip_tolerance)),
                    physics_steps=self.mujoco_world.sim.n_steps)

    def support_status(self, object_id, support_label):
        """Independent physical audit; never supplies target measurements."""
        self.sync()
        mw = self.mujoco_world
        record = next((o for o in mw.objects if o.name == object_id), None)
        supports = [o for o in mw.objects if o.label == support_label and o.name != object_id]
        if record is None or len(supports) != 1:
            return dict(success=False, stable=False, support_label=support_label,
                        reason='support_identity_unconfirmed')
        support = supports[0]
        held = set(mw.held_objects())
        speed, angular = mw.object_speed(record.name)
        base_speed, base_angular = mw.object_speed(support.name)
        stable = bool(speed < .01 and angular < .2 and base_speed < .01 and base_angular < .2)
        contact = any(r['object'] == object_id and r['support'] == support.name
                      for r in self._supports)
        resting = self.resting_status(object_id)
        success = bool(contact and stable and resting['success']
                       and object_id not in held and support.name not in held)
        return dict(success=success, stable=stable, contact=contact,
                    object=object_id, support=support.name, support_label=support_label,
                    linear_speed=speed, angular_speed=angular,
                    support_linear_speed=base_speed, support_angular_speed=base_angular,
                    resting=resting,
                    reason='' if success else (resting['reason'] or
                        'support_contact_or_stability_unconfirmed'))

    def resting_status(self, object_id):
        """Audit a released body's posture and motion, never a planning input.

        The pear's designed resting pose is upright. A banana's geometry is
        already horizontal in its nominal frame and is not a toppled pear.
        A stationary body on its side is not proof of successful pear placement.
        """
        self.sync()
        mw = self.mujoco_world
        record = next((o for o in mw.objects if o.name == object_id), None)
        if record is None:
            return dict(success=False, stable=False, reason='object_identity_unconfirmed')
        speed, angular = mw.object_speed(object_id)
        stable = bool(speed < .01 and angular < .2)
        tilt = float(np.degrees(np.arccos(np.clip(mw.object_pose(object_id)[2, 2], -1., 1.))))
        posture_ok = record.label != 'pear' or tilt <= 10.
        contact = any(r['object'] == object_id for r in self._supports)
        released = object_id not in set(mw.held_objects())
        success = bool(stable and posture_ok and contact and released)
        reason = ('' if success else 'pear_not_upright' if not posture_ok else
                  'object_motion_or_support_unconfirmed')
        return dict(success=success, stable=stable, contact=contact, released=released,
                    posture_ok=posture_ok, tilt_deg=round(tilt, 2),
                    linear_speed=speed, angular_speed=angular, reason=reason)

    def close(self):
        if self._closed:
            return
        self.set_rest_contact_precision(False)
        self._closed = True
        self._remove_engine_listener()
        self.clock._listeners[:] = [fn for fn in self.clock._listeners if fn != self._after_advance]
