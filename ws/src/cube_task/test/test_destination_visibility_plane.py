"""A table destination is observed at its own floor, not the source pedestal."""
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.placement_observation import candidate_visibility_filter
from cube_task.core.transforms import top_down_T
from cube_task.core.types import CameraIntrinsics, Detection, Frame, PickResult


def pedestal_capture():
    intr = CameraIntrinsics(200., 200., 180., 120., 361, 241)
    depth = np.zeros((241, 361), np.float32)
    # Only this portion of the real table at z=0 was observed. At the actual
    # destination it covers the entire target/finger footprint. Projecting the
    # same XY onto the 50 mm source pedestal wrongly samples outside it.
    depth[100:141, 280:321] = .5
    frame = Frame(np.zeros((241, 361, 3), np.uint8), depth, intr, 2.)
    camera = top_down_T(0., 0., .5, -np.pi / 2)
    target = Detection('red', 1., center_base=np.array([.2, 0., .11]),
                       top_z=.11, height=.06, length=.05, width=.05, yaw=0.,
                       extra={'table_z': .05})
    grasp = SimpleNamespace(target=target, T_grasp=top_down_T(.2, 0., .08, 0.),
                            expected_width=.03, closing_angle=0.)
    return frame, camera, target, grasp


def test_pedestal_to_table_uses_destination_plane_and_keeps_source_reference():
    frame, camera, target, grasp = pedestal_capture()
    before = frame.depth.copy()
    wrong_plane = candidate_visibility_filter(target, grasp, frame, camera, max_open_width=.05)
    assert not wrong_plane(.3, 0., 0.)
    table = candidate_visibility_filter(target, grasp, frame, camera,
                                        max_open_width=.05, landing_z=0.)
    assert table(.3, 0., 0.)
    assert target.extra['table_z'] == .05 and target.height == .06
    np.testing.assert_array_equal(frame.depth, before)


@pytest.mark.parametrize('depth', [0., .3, float('nan')])
def test_unknown_or_foreground_at_real_destination_is_not_visible_support(depth):
    frame, camera, target, grasp = pedestal_capture()
    frame.depth[112:129, 296:305] = depth
    frame.meta['carried_self_masked_pixels'] = 150
    table = candidate_visibility_filter(target, grasp, frame, camera,
                                        max_open_width=.05, landing_z=0.)
    assert not table(.3, 0., 0.)


@pytest.mark.parametrize('height', [float('nan'), float('inf'), 'bad'])
def test_unknown_destination_height_cannot_create_support_visibility(height):
    frame, camera, target, grasp = pedestal_capture()
    with pytest.raises((TypeError, ValueError)):
        candidate_visibility_filter(target, grasp, frame, camera, landing_z=height)


@pytest.mark.parametrize('support_height', [None, .12])
def test_release_callback_supplies_actual_destination_height(monkeypatch, support_height):
    from test_assistant_release_phases import release_session
    session, target, grasp, place, _now, _moves, _calls = release_session(monkeypatch)
    target.extra['table_z'] = .05
    if support_height is not None:
        target.extra['_stack_support'] = dict(top_z=support_height)
    planes = []
    def visible(*args, **kwargs):
        planes.append(kwargs.get('landing_z'))
        return lambda *a, **k: True
    monkeypatch.setattr('cube_task.assistant.placement_observation.candidate_visibility_filter', visible)
    assert session._before_release(grasp, place, .05, PickResult(False, detail={'holding': True})) is place
    assert planes and all(z == (0. if support_height is None else support_height) for z in planes)
    assert target.extra['table_z'] == .05
