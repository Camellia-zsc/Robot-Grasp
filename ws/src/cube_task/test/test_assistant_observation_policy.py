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


def test_monitored_complete_body_survives_a_new_crop_even_with_covered_old_footprint():
    original = det('pear')
    crop = det('pear', (.005, 0.), valid=False)
    assert merge_local_detections([original], [crop], lambda d: True,
                                  preserve_incomplete=True) == [original, crop]
    # Genuine disappearance still revokes the old geometry.
    assert merge_local_detections([original], [], lambda d: True,
                                  preserve_incomplete=True) == []


def test_fresh_geometry_supersedes_stale_measurement_regardless_of_class():
    original, corrected = det('pear'), det('banana', (.005, 0.))
    assert merge_local_detections([original], [corrected], lambda d: False) == [corrected]


def test_forgetting_picked_target_does_not_delete_same_class_elsewhere():
    original, duplicate, neighbor = det(), det(xy=(.01, 0.)), det(xy=(.2, 0.))
    close_other = det('yellow', (.035, 0.))
    assert merge_local_detections([original, duplicate, neighbor, close_other], [],
                                  lambda d: False, forget=[original]) == [neighbor, close_other]


def test_confirmed_singleton_rebind_removes_distant_ghost_but_keeps_other_obstacles():
    upright = det('pear', (.30, -.12))
    fallen = det('pear', (.35, -.20), valid=False)
    fallen.reject_reason = 'stable_grasp_not_observed'
    neighbor = det('yellow', (.2, .1))
    # Caller has complete fresh camera evidence of this one-per-class object.
    assert merge_local_detections([upright, neighbor], [fallen], lambda d: False,
        singleton_labels={'pear'}) == [neighbor, fallen]


def test_singleton_rebind_requires_one_unambiguous_fresh_measurement():
    old = det('pear', (.1, .1))
    a, b = det('pear', (.3, .1)), det('pear', (.4, .1))
    assert merge_local_detections([old], [a, b], lambda d: False,
        singleton_labels={'pear'}) == [old, a, b]
    assert merge_local_detections([old], [], lambda d: False,
        singleton_labels={'pear'}) == [old]


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


@pytest.mark.parametrize('visible,complete,forgotten', [(False,False,False), (True,False,False),
                                                       (False,True,False), (False,False,True)])
def test_monitored_scene_crop_keeps_complete_source_only_when_it_cannot_disprove_it(visible,complete,forgotten):
    old, fresh = det(), det(valid=complete)
    kept = merge_local_detections([old], [fresh], lambda d: visible,
                                 preserve_incomplete=True, forget=[old] if forgotten else [])
    assert any(item is old for item in kept) is (not complete and not forgotten)
    assert any(item is fresh for item in kept)


def test_monitored_visible_source_with_no_replacement_detection_is_removed():
    old = det()
    assert not merge_local_detections([old], [], lambda d: True,
                                      preserve_incomplete=True)


def test_cropped_wrong_class_cannot_erase_an_unchanged_complete_identity():
    old, crop = det(), det(label='apple', valid=False)
    kept = merge_local_detections([old], [crop], lambda d: True, preserve_incomplete=True)
    assert old in kept and crop in kept
    crop.valid = True
    assert merge_local_detections([old], [crop], lambda d: True,
                                  preserve_incomplete=True) == [crop]


def test_new_upper_body_does_not_erase_its_occluded_known_support():
    base, upper = det(), det(label='pear')
    upper.top_z, upper.height = .16, .11
    kept = merge_local_detections([base], [upper], lambda d: False,
                                  preserve_incomplete=True)
    assert base in kept and upper in kept


def test_merged_upper_and_support_silhouette_does_not_erase_known_lower_body():
    base, merged = det(label='yellow'), det(label='pear')
    base.extra['catalog_complete'] = True
    # The stacked body and base touch in the image. Segmentation estimates
    # the taller silhouette from the tabletop rather than its support plane.
    merged.top_z, merged.height = .124, .124
    kept = merge_local_detections([base], [merged], lambda d: False,
                                  preserve_incomplete=True)
    assert any(d is base for d in kept)
    assert any(d is merged for d in kept)


@pytest.mark.parametrize('depth,absent', [(.95,False),(1.,True),(.8,False),(float('nan'),False)])
def test_segmentation_miss_needs_actual_background_depth_before_forgetting(depth, absent):
    from cube_task.assistant.observation_policy import measurement_absent
    target = det()
    target.extra['_points_base'] = np.array([[i*.001,j*.001,.05]
                                           for i in range(4) for j in range(4)])
    assert measurement_absent(target, frame(depth), camera()) is absent


def test_old_edge_surface_rays_cannot_erase_a_support_whose_core_is_still_visible():
    from cube_task.assistant.observation_policy import measurement_absent
    target = det()
    target.extra['_points_base'] = np.array([[.032+i*.001,j*.001,.05]
                                           for i in range(4) for j in range(4)])
    image = frame(1.)
    image.depth[100,100] = .95
    assert not measurement_absent(target, image, camera())
