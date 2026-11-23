"""A short TCP displacement can still require a long, slow wrist rotation."""
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.sim.mujoco.arm import MujocoArm


@pytest.mark.parametrize('timeout,expected_min', [(None,13.25),(2.,2.)])
@pytest.mark.parametrize('command',['joints','tcp'])
def test_joint_command_default_deadline_covers_planned_rotation_but_explicit_timeout_wins(timeout,expected_min,command):
    arm = object.__new__(MujocoArm)
    arm._motion_gate = lambda: None
    arm._take_fault = lambda kinds: None
    arm._q_meas = lambda: np.zeros(6)
    arm._tcp_meas = lambda: np.eye(4)
    arm._clock = SimpleNamespace(now=lambda:0.)
    arm.ik = SimpleNamespace(within_limits=lambda q:True,fk=lambda q:np.eye(4))
    arm.ik.solve = lambda *args,**kwargs:(np.array([1.4,0.,0.,0.,0.,0.]),0.,0.)
    arm._polish = lambda T,q:q
    arm.command_log=[]
    arm._speed,arm._joint_speed=100.,.2
    arm._timeout_base,arm._timeout_per_m=6.,15.
    limits=[]
    arm._execute=lambda qs,v,T,q,t0,limit: limits.append(limit)
    if command=='joints':
        arm.move_joints([1.4,0.,0.,0.,0.,0.],timeout=timeout)
    else:
        arm.move_tcp(np.eye(4),timeout=timeout)
    if timeout is None:
        assert limits[0]>=expected_min
    else:
        assert limits==[timeout]
