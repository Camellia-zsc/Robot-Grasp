"""Fruit rest on their actual contact geometry, without upright constraints."""
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('mujoco')

from cube_task.assistant.catalog import make_catalog_object
from cube_task.core.config import load_config
from cube_task.core.transforms import rot_y
from cube_task.sim.mujoco.scene import build_scene
from cube_task.sim.scenarios import make_scenario


@pytest.fixture(scope='module')
def fruit_worlds():
    """Use the production fruit definitions and the production MuJoCo builder."""
    worlds = {}
    for label in ('pear', 'apple', 'banana'):
        world, info = make_scenario('L1', 0, mount='eye_in_hand',
                                    difficulty='normal', layout='A')
        obj = make_catalog_object(label, 'resting-'+label, [.65, .35])
        world.objects = [obj]
        info['objects'] = [obj.to_dict()]
        cfg = load_config(use_overlays=False)
        assets = Path(__file__).resolve().parents[4] / 'local_assets/mujoco_menagerie/agilex_piper'
        cfg.set('sim.mujoco.assets_dir', str(assets))
        cfg.set('sim.mujoco.card_collision', False)
        worlds[label] = (obj, build_scene(info, cfg, world=world))
    return worlds


def _release(obj, world, *, tilt=0., gap=0., velocity=0.):
    transform = np.eye(4)
    transform[:3, :3] = rot_y(math.radians(tilt))
    transform[:3, 3] = [.65, .35, 0.]
    obj.T_base_obj = transform
    transform[2, 3] += world.table_z - obj.bottom_z() + gap
    world.set_object_pose(obj.name, transform)
    world.data.qvel[world.object(obj.name).dof_adr] = velocity
    world.forward()
    world.settle(3.)
    obj.T_base_obj = world.object_pose(obj.name)
    return math.degrees(math.acos(np.clip(obj.T_base_obj[2, 2], -1., 1.)))


@pytest.mark.parametrize('tilt,gap,velocity', [(3., 0., 0.), (10., .003, .02),
                                             (20., .003, .02), (0., .010, .02)])
def test_pear_returns_upright_after_gentle_physical_release(fruit_worlds, tilt, gap, velocity):
    pear, world = fruit_worlds['pear']
    final_tilt = _release(pear, world, tilt=tilt, gap=gap, velocity=velocity)
    assert final_tilt < 2., (tilt, gap, velocity, final_tilt)
    assert pear.bottom_z() == pytest.approx(world.table_z, abs=.002)
    assert np.linalg.norm(pear.T_base_obj[:2, 3] - [.65, .35]) < .004
    assert world.object_speed(pear.name)[0] < .002
    assert world.object(pear.name).mass == pytest.approx(.100)
    assert any((world.geom_object(int(a)) == pear.name and int(b) == world.meta['table_geom'])
               or (world.geom_object(int(b)) == pear.name and int(a) == world.meta['table_geom'])
               for a, b in world.data.contact.geom[:world.data.ncon])


def test_large_pear_tilt_still_falls_under_gravity(fruit_worlds):
    pear, world = fruit_worlds['pear']
    assert _release(pear, world, tilt=60.) > 70.
    assert not world.held_objects()
    assert not any(int(kind) == int(world._mj.mjtEq.mjEQ_WELD)
                   for kind in world.model.eq_type)


def test_apple_low_release_does_not_need_a_geometry_change(fruit_worlds):
    apple, world = fruit_worlds['apple']
    _release(apple, world, gap=.003, velocity=.02)
    assert apple.bottom_z() == pytest.approx(world.table_z, abs=.002)
    assert np.linalg.norm(apple.center_xy() - [.65, .35]) < .004
    assert world.object_speed(apple.name)[0] < .002


def test_banana_default_is_lying_flat_and_remains_supported(fruit_worlds):
    banana, world = fruit_worlds['banana']
    _release(banana, world, gap=.003, velocity=.02)
    low, high = banana.aabb()
    assert low[2] == pytest.approx(world.table_z, abs=.002)
    assert high[2] - low[2] < .045
    assert high[0] - low[0] > .150
    assert world.object_speed(banana.name)[0] < .002
