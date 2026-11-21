"""A measured sphere centre is a translation anchor, never a body-axis pose."""
import copy

import numpy as np
import pytest

from cube_task.assistant.spherical_anchor import fit_spherical_anchor, validate_anchor_pair


def belly_points(radius=.0425, count=1000, noise=.00004, seed=5):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-1.2, 1.2, count)
    z = rng.uniform(-.5, -.05, count)
    radial = np.sqrt(1 - z*z)
    normals = np.column_stack((radial*np.cos(angle), radial*np.sin(angle), z))
    center = np.array([.34, -.17, .10])
    points = center + radius*normals + rng.normal(0, noise, (count, 3))
    return points, center


@pytest.mark.parametrize('radius', [.0425, .031])
def test_fixed_radius_center_is_recovered_from_a_partial_belly_without_mutating_input(radius):
    points, center = belly_points(radius)
    before = points.copy()
    points.setflags(write=False)
    result = fit_spherical_anchor(points, radius, center + [.012, -.008, .009])
    assert result['ok'], result
    assert np.linalg.norm(np.array(result['center']) - center) < .0002
    assert result['inlier_fraction'] > .95
    assert result['normal_eigenvalues'][0] >= .015
    assert result['hypothesis_spread'] < .0005
    assert 'orientation' not in result
    np.testing.assert_array_equal(points, before)


def test_fused_body_cloud_can_include_neck_outliers_but_raw_neck_dominated_cloud_is_rejected():
    body, center = belly_points(count=800)
    rng = np.random.default_rng(6)
    outliers = center + rng.uniform(-.055, .055, (800, 3))
    fused = fit_spherical_anchor(np.vstack((body, outliers[:200])), .0425, center)
    assert fused['ok'], fused
    assert np.linalg.norm(np.array(fused['center']) - center) < .0002
    dominated = fit_spherical_anchor(np.vstack((body[:500], outliers[:500])), .0425, center)
    assert not dominated['ok'] and 'center' not in dominated


def test_only_neck_geometry_cannot_be_claimed_as_the_large_belly_sphere():
    rng = np.random.default_rng(7)
    normals = rng.normal(size=(1000, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    points = normals * [.027, .027, .030] + [.34, -.17, .135]
    result = fit_spherical_anchor(points, .0425, [.34, -.17, .10])
    assert not result['ok'] and 'center' not in result


def test_extremely_narrow_surface_patch_has_no_acceptable_three_axis_coverage():
    rng = np.random.default_rng(9)
    yz = rng.uniform(-.06, .06, (800, 2))
    normals = np.column_stack((np.sqrt(1 - np.sum(yz*yz, axis=1)), yz))
    points = np.array([.34, -.17, .10]) + .0425 * normals
    result = fit_spherical_anchor(points, .0425, [.35, -.17, .10])
    assert not result['ok'] and 'center' not in result


@pytest.mark.parametrize('points,radius,prior', [
    (np.zeros((5, 3)), .0425, [0., 0., 0.]),
    (np.zeros((100, 2)), .0425, [0., 0., 0.]),
    (np.full((100, 3), np.nan), .0425, [0., 0., 0.]),
    (np.zeros((100, 3)), -1., [0., 0., 0.]),
    (np.zeros((100, 3)), .0425, [np.nan, 0., 0.]),
])
def test_bad_geometry_and_parameters_produce_no_actionable_anchor(points, radius, prior):
    result = fit_spherical_anchor(points, radius, prior)
    assert not result['ok'] and 'center' not in result


def anchor(center):
    return dict(ok=True, center=list(center), radius=.0425, rmse=.0001,
                inlier_fraction=.9, normal_eigenvalues=[.03, .25, .72],
                normal_condition=24., uncertainty_2sigma=.0001, hypothesis_spread=.0001)


def test_two_distinct_captures_only_produce_a_consistent_translation_anchor():
    first, second = anchor([.004, .001, .056]), anchor([.0042, .0009, .0561])
    before = copy.deepcopy((first, second))
    result = validate_anchor_pair(first, second, 10., 10.05)
    assert result['ok'] and result['translation_only']
    np.testing.assert_allclose(result['center'], [.0041, .00095, .05605])
    assert 0 < result['center_change'] < .001
    assert 'orientation' not in result
    assert (first, second) == before


@pytest.mark.parametrize('change', ['same_stamp', 'backward_stamp', 'nonfinite_stamp',
                                  'failed_fit', 'moved_center', 'different_radius'])
def test_reused_or_disagreeing_frames_cannot_update_the_translation_anchor(change):
    first, second = anchor([0., 0., .056]), anchor([0., 0., .056])
    stamp = 10.05
    if change == 'same_stamp': stamp = 10.
    if change == 'backward_stamp': stamp = 9.
    if change == 'nonfinite_stamp': stamp = float('nan')
    if change == 'failed_fit': second = dict(ok=False, error='no_fit')
    if change == 'moved_center': second['center'][0] = .002
    if change == 'different_radius': second['radius'] = .031
    result = validate_anchor_pair(first, second, 10., stamp)
    assert not result['ok'] and 'center' not in result


@pytest.mark.parametrize('field,value', [
    ('rmse', -.001), ('inlier_fraction', 1.1),
    ('normal_eigenvalues', [.1]), ('normal_condition', -1.),
    ('uncertainty_2sigma', -.001), ('hypothesis_spread', -.001),
    ('hypothesis_spread', float('nan')),
])
def test_invalid_quality_cannot_become_valid_by_averaging_two_captures(field, value):
    first, second = anchor([0., 0., .056]), anchor([0., 0., .056])
    second[field] = value
    result = validate_anchor_pair(first, second, 10., 10.05)
    assert not result['ok'] and 'center' not in result
