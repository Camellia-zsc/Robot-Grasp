"""Completed goals are rechecked using fresh local images only after changes."""
from types import SimpleNamespace

import numpy as np

from cube_task.assistant.simulation import SimulationSession
from cube_task.core.types import Detection, Zone


def verifier(objects):
    session = object.__new__(SimulationSession)
    zone = Zone('zone_red', np.array([.17, -.35, 0]), (.2, .2), label='red')
    session.runner = SimpleNamespace(zones={'zone_red': zone})
    session._physics = None
    session._evaluation_object = lambda d: None
    session._environment_revision = session._observation_revision = 1
    scene = {'zones': [dict(name=zone.name, center=zone.center.tolist(),
                           size=list(zone.size), yaw=0)]}
    calls = []
    def observe(**request):
        calls.append(request)
        session._last_fresh = list(objects)
        session._observation_revision = session._environment_revision
        return scene
    session.observe = observe
    return session, calls


def block(label, xy):
    return Detection(label, 1., center_base=np.r_[xy, .05], top_z=.05,
                     height=.05, length=.05, width=.05, yaw=0)


def test_same_destination_goals_share_one_fresh_local_view():
    session, calls = verifier([block('red', [.13, -.35]), block('green', [.21, -.35])])
    result = session.verify_goals([dict(label=c, zone_name='zone_red') for c in ('red', 'green')])
    assert result['ok'] and len(calls) == 1
    assert all(i['ok'] for i in result['items'])


def test_old_cached_goal_never_substitutes_a_missing_fresh_object():
    session, calls = verifier([])
    session._scene = {'objects': [block('red', [.17, -.35]).summary()]}
    result = session.verify_goals([dict(label='red', zone_name='zone_red')])
    assert not result['ok'] and len(calls) == 1
    assert all(call['purpose'] == 'verify_place' for call in calls)
    assert calls[0]['target']['verify_goals'] == [dict(label='red', zone_name='zone_red')]


def test_object_moved_out_of_goal_during_later_task_fails_final_verification():
    session, _ = verifier([block('red', [.29, .2])])
    result = session.verify_goals([dict(label='red', zone_name='zone_red')])
    assert not result['ok']
