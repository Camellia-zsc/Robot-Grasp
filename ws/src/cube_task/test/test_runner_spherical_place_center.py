"""A partial visible bounding box cannot replace a fitted source sphere center."""
import numpy as np
import pytest

from cube_task.core.types import Detection
from test_runner_stack_opening import stacked_scene


def biased_source_scene():
    parts = stacked_scene()
    runner, target, grasp, previous, support, zone, polygon = parts
    anchor = np.array([.2133, .15, .0425])
    target.extra['_held_translation_anchor'] = dict(
        source_center=anchor.tolist(), translation_only=True,
        pair=dict(ok=True, translation_only=True))
    # The wrist/fused visible box misses one flank by 13.3 mm. Its full cloud
    # still constrains collision, but cannot determine the body's center.
    target.extra['_placement_points_base'] = np.array([
        [x, y, z] for x in np.linspace(.18, .22, 4)
        for y in np.linspace(.13, .17, 4) for z in np.linspace(.025, .11, 4)])
    for pose in (previous.T_pre, previous.T_release, previous.T_retreat):
        pose[:2, 3] = [.408, .004]
    return parts, anchor


def real_replan(parts, extra=(), *, exact_xy=False):
    runner, target, grasp, previous, support, zone, polygon = parts
    return runner.replan_release(target, grasp, previous, .048,
        [target, support, *extra], placement_zone=zone, placement_polygon=polygon,
        exact_xy=exact_xy, observation_filter=lambda *args, **kwargs: True)


def test_real_stack_replan_lands_source_sphere_on_support_center_despite_biased_bbox():
    parts, anchor = biased_source_scene()
    runner, target, grasp, _previous, support, _zone, _polygon = parts
    candidate, why = real_replan(parts)
    assert candidate is not None, why
    rotation = candidate.T_release[:3, :3] @ grasp.T_grasp[:3, :3].T
    sphere_center = candidate.T_release[:3, 3] + rotation @ (anchor - grasp.T_grasp[:3, 3])
    assert np.linalg.norm(sphere_center[:2]-support.center_base[:2]) < .001
    np.testing.assert_allclose(runner.placer.last_info['xy'], support.center_base[:2], atol=.001)
    assert candidate.open_width == pytest.approx(.095002)


def test_anchor_center_keeps_full_cloud_projection_and_actual_3d_finger_checks():
    parts, _anchor = biased_source_scene()
    runner = parts[0]
    original = runner._strict_place_filter
    body_checks = []
    def factory(*args, **kwargs):
        allowed = original(*args, **kwargs)
        def checked(*values, **options):
            if options.get('held_projection') is not None:
                body_checks.append(options)
            return allowed(*values, **options)
        return checked
    runner._strict_place_filter = factory
    candidate, why = real_replan(parts)
    assert candidate is not None, why
    assert body_checks
    assert len(body_checks[-1]['held_projection']) >= 8+len(parts[1].extra['_placement_points_base'])
    assert body_checks[-1]['gripper_pose'] is not None
    assert body_checks[-1]['gripper_open_width'] == pytest.approx(candidate.open_width)


def test_stack_prefers_measured_support_center_over_a_previous_offset_release():
    parts, _anchor = biased_source_scene()
    runner, target, grasp, _previous, support, _zone, _polygon = parts
    anchor = np.array([.2, .15, .0425])
    target.extra['_held_translation_anchor']['source_center'] = anchor.tolist()
    candidate, why = real_replan(parts)
    assert candidate is not None, why
    rotation = candidate.T_release[:3, :3] @ grasp.T_grasp[:3, :3].T
    actual_center = candidate.T_release[:3, 3] + rotation @ (anchor - grasp.T_grasp[:3, 3])
    assert np.linalg.norm(actual_center[:2]-support.center_base[:2]) < .001
    np.testing.assert_allclose(runner.placer.last_info['xy'], support.center_base[:2], atol=.001)


def test_biased_source_anchor_does_not_bypass_a_measured_tall_neighbor():
    parts, _anchor = biased_source_scene()
    neighbor = Detection('unknown', 1., center_base=np.array([.457, 0., .2]),
        height=.2, top_z=.2, length=.008, width=.008, yaw=0.)
    candidate, why = real_replan(parts, [neighbor], exact_xy=True)
    assert candidate is None and why
