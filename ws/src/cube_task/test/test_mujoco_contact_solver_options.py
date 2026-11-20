"""Contact solver precision reaches the compiled model and remains configurable."""
from pathlib import Path

import pytest

pytest.importorskip('mujoco')

from cube_task.sim.mujoco.scene import build_scene
from cube_task.sim.scenarios import make_scenario


ASSETS = Path(__file__).resolve().parents[4] / 'local_assets/mujoco_menagerie/agilex_piper'


def build(options=None):
    world, info = make_scenario('L1', 1)
    return build_scene(info, world=world, assets_dir=str(ASSETS),
                       cfg={'sim': {'mujoco': options or {}}})


def test_manipulation_contact_solver_defaults_and_explicit_overrides():
    default = build().model.opt
    assert default.noslip_iterations == 5
    assert default.noslip_tolerance == pytest.approx(1e-6, rel=1e-12, abs=0.)
    configured = build({'noslip_iterations': 7, 'noslip_tolerance': 1e-8}).model.opt
    assert configured.noslip_iterations == 7
    assert configured.noslip_tolerance == pytest.approx(1e-8, rel=1e-12, abs=0.)


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf')])
def test_invalid_contact_solver_tolerance_fails_before_compilation(value):
    with pytest.raises(ValueError, match='noslip_tolerance'):
        build({'noslip_tolerance': value})
