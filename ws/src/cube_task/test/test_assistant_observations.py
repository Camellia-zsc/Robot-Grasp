"""Regression for a clipped pear/green silhouette becoming a second pear."""
import numpy as np
import pytest

from cube_task.assistant.catalog import make_catalog_object
from cube_task.assistant.observations import merge_observations
from cube_task.assistant.simulation import SimulationSession
from cube_task.core.transforms import make_T, rot_z
from cube_task.core.types import Detection
from cube_task.sim.world import SimObject
from cube_task.task.views import tag_view


def box_cloud(center, width=.03, step=.002):
    axes = [np.arange(-width / 2, width / 2 + step / 2, step) + v for v in center]
    return np.column_stack([a.ravel() for a in np.meshgrid(*axes, indexing='ij')]).astype(np.float32)


def observation(label, xy, cloud, valid=True):
    return Detection(label, .99 if valid else .64, center_base=np.array([*xy, .08]),
                     n_points=0 if cloud is None else len(cloud), valid=valid,
                     reject_reason='' if valid else 'touches_image_border',
                     extra={'_points_base': cloud})


def consolidate(*views):
    return merge_observations({i: tag_view(dets, i) for i, dets in enumerate(views)},
                              partial=lambda d: not d.valid or d.extra.get('incomplete', False))


def pear_view(xy, height, count):
    det = observation('pear', xy, None)
    det.height = det.top_z = height
    det.length, det.width = .075, .055
    det.n_points = count
    return det


def test_pear_neck_visibility_wins_over_a_denser_low_view():
    low = pear_view([.198, -.011], .102, 19455)
    whole = pear_view([.197, -.002], .114, 17746)
    objects, _ = consolidate([low], [whole])
    assert len(objects) == 1 and objects[0] is whole


@pytest.mark.parametrize('case', ['partial', 'invalid', 'tiny_gain', 'narrow', 'different_label', 'missing_height'])
def test_pear_view_preference_requires_reliable_more_complete_geometry(case):
    low = pear_view([.198, -.011], .102, 19455)
    candidate = pear_view([.197, -.002], .114, 17746)
    if case == 'partial':
        candidate.extra['incomplete'] = True
    elif case == 'invalid':
        candidate.valid = False
    elif case == 'tiny_gain':
        candidate.height = candidate.top_z = .105
    elif case == 'narrow':
        candidate.width = .033
    elif case == 'different_label':
        candidate.label = 'apple'
    elif case == 'missing_height':
        candidate.height = None
    objects, _ = consolidate([low], [candidate])
    assert len(objects) == 1 and objects[0] is low


def test_a_taller_nearby_pear_is_still_a_separate_target():
    low = pear_view([.198, -.011], .102, 19455)
    neighbor = pear_view([.27, -.002], .114, 17746)
    objects, _ = consolidate([low], [neighbor])
    assert objects == [low, neighbor]


def test_a_bridging_view_cannot_replace_two_separate_targets_with_the_same_detection():
    first = pear_view([.20, 0.], .100, 20000)
    second = pear_view([.26, 0.], .110, 20000)
    bridge = pear_view([.23, 0.], .130, 18000)
    objects, _ = merge_observations(
        {0: tag_view([first, second], 0), 1: tag_view([bridge], 1)}, dist=.04,
        partial=lambda d: not d.valid)
    assert len(objects) == 2 and objects[0] is first and objects[1] is second


@pytest.mark.parametrize('reverse', [False, True])
def test_mixed_candidate_is_explained_by_other_views_not_a_third_object(reverse):
    pear = observation('pear', [.20, 0], box_cloud([.20, 0, .08]))
    green = observation('green', [.32, -.08], box_cloud([.32, -.08, .04]))
    mixed = observation('pear', [.245, -.02],
                        np.vstack([pear.extra['_points_base'], green.extra['_points_base']]), False)
    views = ([pear, green], [mixed])
    objects, diagnostics = consolidate(*(reversed(views) if reverse else views))
    assert len(objects) == 2 and all(d.valid for d in objects)
    assert len(diagnostics) == 1
    assert diagnostics[0]['supporting_labels'] == ['green', 'pear']
    assert diagnostics[0]['support_ratio'] == 1.0
    assert not mixed.valid and mixed.reject_reason == 'touches_image_border'


@pytest.mark.parametrize('valid', [False, True])
def test_a_neighbor_of_the_same_class_is_not_removed(valid):
    anchor = observation('pear', [.20, 0], box_cloud([.20, 0, .08]))
    neighbor = observation('pear', [.26, 0], box_cloud([.26, 0, .08]), valid)
    objects, diagnostics = consolidate([anchor], [neighbor])
    assert len(objects) == 2 and any(d is neighbor for d in objects)
    assert not diagnostics


@pytest.mark.parametrize('case', ['same_view', 'missing', 'nan', 'partial_anchor', 'no_anchor'])
def test_unreliable_or_same_view_support_does_not_erase_evidence(case):
    cloud = box_cloud([.20, 0, .08])
    anchor = observation('pear', [.20, 0], cloud)
    candidate = observation('pear', [.26, 0], cloud.copy(), False)
    views = ([anchor], [candidate])
    if case == 'same_view':
        views = ([anchor, candidate],)
    elif case == 'missing':
        candidate.extra.clear()
    elif case == 'nan':
        candidate.extra['_points_base'][0, 0] = np.nan
    elif case == 'partial_anchor':
        anchor.extra['incomplete'] = True
    elif case == 'no_anchor':
        views = ([], [candidate])
    objects, diagnostics = consolidate(*views)
    assert any(d is candidate for d in objects)
    assert not diagnostics


def test_an_empty_space_inside_the_measured_envelope_is_not_surface_support():
    cloud = box_cloud([.20, 0, .08], width=.16, step=.004)
    surface = cloud[np.max(np.abs(cloud - [.20, 0, .08]), axis=1) > .075]
    anchor = observation('pear', [.20, 0], surface)
    candidate = observation('apple', [.25, 0], box_cloud([.25, 0, .08], width=.012), False)
    objects, diagnostics = consolidate([anchor], [candidate])
    assert any(d is candidate for d in objects)
    assert not diagnostics


def test_a_small_separate_unknown_cluster_cannot_hide_under_the_two_percent_allowance():
    cloud = box_cloud([.20, 0, .08])
    anchor = observation('pear', [.20, 0], cloud)
    unknown = box_cloud([.30, 0, .08], width=.004)
    assert len(unknown) < .02 * len(cloud)
    candidate = observation('pear', [.25, 0], np.vstack([cloud, unknown]), False)
    objects, diagnostics = consolidate([anchor], [candidate])
    assert any(d is candidate for d in objects)
    assert not diagnostics


def test_surface_support_cannot_be_borrowed_from_a_different_objects_envelope():
    cloud = box_cloud([.20, 0, .08], width=.16, step=.004)
    shell = cloud[np.max(np.abs(cloud - [.20, 0, .08]), axis=1) > .075]
    enclosing = observation('pear', [.20, 0], shell)
    small = observation('green', [.266, 0], box_cloud([.266, 0, .08], width=.010))
    unknown = box_cloud([.252, 0, .08], width=.004)
    candidate = observation('pear', [.257, 0], np.vstack([small.extra['_points_base'], unknown]), False)
    objects, diagnostics = merge_observations(
        {0: tag_view([enclosing, small], 0), 1: tag_view([candidate], 1)},
        dist=.001, partial=lambda d: not d.valid)
    assert any(d is candidate for d in objects)
    assert not diagnostics


def captured_layout():
    blocks = [
        ('yellow', [0.3534206706, -0.1732924437], -2.0425141151, 0.0480740264),
        ('green', [0.3134764400, -0.0829820053], 2.6726933888, 0.0509264656),
        ('red', [0.3232283665, 0.0972086836], 0.4233755780, 0.0504616839),
    ]
    colors = {'yellow': (34, 198, 224), 'green': (54, 164, 60), 'red': (46, 40, 200)}
    objects = [SimObject(label, 'unavailable-truth', 'box', [size] * 3, colors[label],
                         make_T(rot_z(yaw), [*xy, 0.0]))
               for label, xy, yaw, size in blocks]
    for label, xy, yaw in [
        ('pear', [0.2115300883, -0.0026151822], 0.0),
        ('banana', [0.2103082311, 0.1549317021], 0.7891919447),
        ('apple', [0.1950230755, -0.3850442161], -0.8824257212),
    ]:
        obj = make_catalog_object(label, label, xy, yaw)
        obj.label = 'unavailable-truth'
        objects.append(obj)
    return objects


def test_actual_six_view_scan_does_not_publish_a_second_pear(tmp_path):
    with SimulationSession(realtime=False, log_dir=tmp_path) as session:
        session.world.objects = captured_layout()
        scene = session.prepare()
        # Include invalid detections: the old test's valid-only assertion missed this bug.
        pears = [o for o in scene['objects'] if o['label'] == 'pear']
        assert len(pears) == 1, pears
        assert pears[0]['valid']
        assert {o['label'] for o in scene['objects'] if o['valid']} == {
            'red', 'green', 'yellow', 'apple', 'banana', 'pear'}
        assert len(session.observation_views()) == 6
        assert any(d['label'] == 'pear' for d in scene['observation_diagnostics'])
        # The model's selectable table and executor must use the same consolidated set.
        assert len(session._detections) == len(scene['objects'])
        assert sum(d.label == 'pear' for d in session._detections.values()) == 1
        assert pears[0]['height'] > .11  # Preserve the more complete neck view.
        np.testing.assert_allclose(pears[0]['center_base'][:2], [.197, -.002], atol=.003)
