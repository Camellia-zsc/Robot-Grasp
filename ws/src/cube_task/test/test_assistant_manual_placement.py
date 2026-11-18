"""Human layout edits use the table; goal containment belongs to task checks."""
import math

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession


@pytest.fixture
def manual(tmp_path):
    with SimulationSession(realtime=False, log_dir=tmp_path) as sim:
        sim._prepared = True
        sim._status = 'ready'
        sim._invalidate_scene('isolated manual edit test')
        sim._publish_snapshot()
        edit(sim, 'clear')
        yield sim


def edit(sim, action, **fields):
    return sim.edit_scene(dict(scene_version=sim.snapshot()['scene_version'],
                               action=action, **fields))


@pytest.mark.parametrize('xy', [[.26, 0.], [.17, -.35], [.17, .35], [.49, 0.],
                               [.7, .5], [-.2, -.5], [.39, 0.], [.17, -.25]])
def test_manual_add_accepts_pick_zones_other_table_and_crossing_region_edges(manual, xy):
    sim = manual
    calls = sim.arm.inner.n_calls
    edit(sim, 'add', catalog_id='red', xy=xy, yaw=math.pi / 4)
    np.testing.assert_allclose(sim.world.objects[0].center_xy(), xy, atol=1e-10)
    assert sim.arm.inner.n_calls == calls
    assert sim.scene['objects'] == [], 'manual coordinates must not become detections'


def test_manual_move_and_rotate_can_cross_every_task_region_without_moving_robot(manual):
    sim = manual
    edit(sim, 'add', catalog_id='pear', xy=[.26, 0.])
    name = sim.world.objects[0].name
    calls = sim.arm.inner.n_calls
    for xy, yaw in [([.17, -.35], .4), ([.49, 0.], -.7),
                    ([.17, .35], 1.2), ([.75, -.5], 0.)]:
        edit(sim, 'move', object_id=name, xy=xy, yaw=yaw)
        edit(sim, 'move', object_id=name, yaw=yaw + .1)
        np.testing.assert_allclose(sim.world.objects[0].center_xy(), xy, atol=1e-10)
    assert sim.arm.inner.n_calls == calls


def test_manual_region_boundary_straddling_does_not_count_as_task_completion(manual):
    sim = manual
    edit(sim, 'add', catalog_id='red', xy=[.39, 0.])
    obj = sim.world.objects[0]
    destination = {'type': 'region', 'name': 'pick_area'}
    assert not sim._verify_destination(obj, destination)
    edit(sim, 'move', object_id=obj.name, xy=[.26, 0.])
    assert sim._verify_destination(sim.world.objects[0], destination)
    assert sim.editor.metadata()['pick_polygon'] == sim.info['pick_polygon']


def test_table_boundary_error_does_not_claim_a_pick_area_restriction(manual):
    before = manual.snapshot()['objects']
    with pytest.raises(ValueError, match='桌面') as error:
        edit(manual, 'add', catalog_id='red', xy=[99., 0.])
    assert '抓取区' not in str(error.value)
    assert manual.snapshot()['objects'] == before
