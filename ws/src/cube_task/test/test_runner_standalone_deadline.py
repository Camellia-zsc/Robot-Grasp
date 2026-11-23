"""A new dashboard action must not inherit the previous action's expired clock."""
import pytest

from test_task_fakes import Rig, make_det


@pytest.mark.parametrize('method', ['pick_detection', 'pick_only_detection'])
def test_new_standalone_pick_after_long_idle_gets_a_fresh_budget(method):
    rig = Rig('L1')
    runner = rig.runner()
    target = rig.world.add(make_det('red', .3, .05))
    runner._deadline = rig.clock.now() + 60.
    rig.clock.sleep(4000.)  # Human approval / overnight idle, not robot work.
    result, why = getattr(runner, method)(target, budget_s=120.)
    assert result is not None and result.ok, why
    assert rig.arm.motions()


@pytest.mark.parametrize('method', ['pick_detection', 'pick_only_detection'])
def test_fresh_budget_still_expires_during_this_actions_planning(method):
    rig = Rig('L1')
    runner = rig.runner()
    target = rig.world.add(make_det('red', .3, .05))
    original = runner.planner.plan
    def slow_plan(*args, **kwargs):
        rig.clock.sleep(125.)
        return original(*args, **kwargs)
    runner.planner.plan = slow_plan
    result, why = getattr(runner, method)(target, budget_s=120.)
    assert result is None and 'timeout' in why
    assert not rig.arm.commands


@pytest.mark.parametrize('method', ['pick_detection', 'pick_only_detection'])
def test_new_budget_cannot_clear_an_explicit_stop(method):
    rig = Rig('L1')
    runner = rig.runner()
    runner._deadline = rig.clock.now() - 1.
    runner.operator.abort_event.set()
    result, why = getattr(runner, method)(make_det('red', .3, .05))
    assert result is None and 'aborted' in why
    assert not rig.arm.commands
