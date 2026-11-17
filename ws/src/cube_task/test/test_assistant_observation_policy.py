"""Local scene updates must not resurrect objects from unrefreshed views."""
from types import SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.observation_policy import (
    measurement_visible, merge_local_detections, rank_views, unique_target,
)
from cube_task.core.transforms import make_T
from cube_task.core.types import CameraIntrinsics, Detection, Frame


INTR = CameraIntrinsics(200., 200., 100., 100., 200, 200)


def det(label='banana', xy=(0., 0.), valid=True):
    return Detection(label, .99, center_base=np.r_[xy, .05], top_z=.05,
                     height=.05, length=.08, width=.03, yaw=0., valid=valid)


def camera(x=0., y=0., z=1.):
    # Optical z points down, optical y toward negative base y.
    return make_T(np.diag([1., -1., -1.]), [x, y, z])


def frame(depth=1.):
    return Frame(np.zeros((200, 200, 3), np.uint8),
                 np.full((200, 200), depth, np.float32), INTR, 1.)


def test_target_last_view_precedes_other_covering_views():
    cameras = [camera(0.), camera(.1), camera(.8)]
    assert rank_views(cameras, INTR, [0., 0.], preferred=1) == [1, 0, 2]


def test_a_preferred_view_cannot_outrank_one_that_covers_target():
    assert rank_views([camera(.8), camera()], INTR, [0., 0.], preferred=0) == [1, 0]


def test_view_ranking_rejects_nonfinite_target():
    with pytest.raises(ValueError):
        rank_views([camera()], INTR, [np.nan, 0.])


def test_target_search_ends_only_for_one_reliable_fresh_target():
    target = det()
    assert unique_target([target], 'banana') is target
    assert unique_target([det(valid=False)], 'banana') is None
    assert unique_target([target], 'banana', partial=lambda d: True) is None
    assert unique_target([target, det(xy=(.04, 0.))], 'banana') is None
    assert unique_target([target, det(xy=(.3, 0.))], 'banana', [0., 0.]) is target
    assert unique_target([det(xy=(.3, 0.))], 'banana', [0., 0.]) is None


def test_visible_missing_target_is_deleted_even_when_saved_in_an_old_view():
    original = det()
    original.extra['view'] = 4
    neighbor = det('yellow', (.8, 0.))
    measured = frame()
    scene = merge_local_detections(
        [original, neighbor], [], lambda d: measurement_visible(d, measured, camera()))
    assert scene == [neighbor]
    assert original.extra['view'] == 4  # Never mutate source measurements.


def test_fresh_geometry_supersedes_stale_measurement_regardless_of_class():
    original, corrected = det('pear'), det('banana', (.005, 0.))
    assert merge_local_detections([original], [corrected], lambda d: False) == [corrected]


def test_forgetting_picked_target_does_not_delete_same_class_elsewhere():
    original, duplicate, neighbor = det(), det(xy=(.01, 0.)), det(xy=(.2, 0.))
    close_other = det('yellow', (.035, 0.))
    assert merge_local_detections([original, duplicate, neighbor, close_other], [],
                                  lambda d: False, forget=[original]) == [neighbor, close_other]


@pytest.mark.parametrize('case', ['occluded', 'depth_hole', 'zero', 'cropped', 'behind', 'no_depth'])
def test_unobserved_old_object_remains_an_obstacle(case):
    target, image, T = det(), frame(), camera()
    if case == 'occluded':
        image.depth[:] = .5
    elif case == 'depth_hole':
        image.depth[:] = np.nan
    elif case == 'zero':
        image.depth[:] = 0.
    elif case == 'cropped':
        target.center_base[0] = .46
    elif case == 'behind':
        T = camera(z=-1.)
    elif case == 'no_depth':
        image.depth = None
    assert not measurement_visible(target, image, T)
    assert merge_local_detections([target], [],
                                  lambda d: measurement_visible(d, image, T)) == [target]


def test_target_depth_itself_does_not_count_as_foreground_occlusion():
    assert measurement_visible(det(), frame(.95), camera())


def test_missing_geometry_cannot_claim_camera_coverage():
    target = det()
    target.width = None
    assert not measurement_visible(target, frame(), camera())
    assert not measurement_visible(SimpleNamespace(center_base=None), frame(), camera())
